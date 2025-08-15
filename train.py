import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import datasets
import diffusers
import accelerate
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
import os
import math
import random
import numpy as np
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable cudnn benchmark for better performance
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f"🌱 Set all seeds to {seed}")

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

@dataclass
class TrainingConfig:
    # Image parameters
    image_size = 64  # Increased for more compute
    train_batch_size = 256  # Much larger batch for RTX 4090
    eval_batch_size = 128
    num_epochs = 10
    gradient_accumulation_steps = 1  # Reduced since batch is larger
    seed = 0
    
    # Transformer specific configs
    patch_size = 8  # Larger patches for efficiency
    hidden_size = 768  # Doubled model size
    num_layers = 12  # More layers
    num_heads = 12  # More attention heads
    mlp_ratio = 4.0
    dropout = 0.1
    
    # Optimizer parameters
    muon_lr = 0.01
    adamw_lr_ratio = 0.1
    weight_decay = 0.1
    grad_clip = 1.0
    
    # Training parameters
    max_steps = None  # Will be calculated from epochs
    eval_every = 500
    warmup_ratio = 0.05
    
    # Technical
    use_amp = True
    mixed_precision = 'bf16'  # Better for RTX 4090
    compile_model = True  # Enable torch.compile for speed
    
    def __post_init__(self):
        # max_steps will be set based on dataset size and epochs
        if self.max_steps is not None:
            self.warmup_steps = int(self.max_steps * self.warmup_ratio)
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"

config = TrainingConfig()

def transform(dataset):
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((config.image_size, config.image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: 2 * (x - 0.5)),
    ])
    images = [preprocess(image) for image in dataset["image"]]
    return {"images": images}

def get_dataloader():
    cifar10_dataset = datasets.load_dataset('cifar10', split='train')
    cifar10_dataset.reset_format()
    cifar10_dataset.set_transform(transform)
    return torch.utils.data.DataLoader(
        cifar10_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=8,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=4,  # Prefetch batches
    ), cifar10_dataset

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(t.device)  # Move to same device as input tensor
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # Use RMSNorm for better performance
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # Use SiLU activation like in the LLM
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention with residual connection and dropout
        normalized_x = self.modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(normalized_x, normalized_x, normalized_x)[0]
        x = x + self.dropout(gate_msa.unsqueeze(1) * attn_out)
        
        # MLP with residual connection and dropout
        mlp_out = self.mlp(self.modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + self.dropout(gate_mlp.unsqueeze(1) * mlp_out)
        
        return x

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiT(nn.Module):
    def __init__(self, input_size=32, patch_size=4, in_channels=1, hidden_size=384, 
                 depth=6, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Add input dropout
        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)
        ])
        
        # Use RMSNorm in final layer
        self.final_layer = nn.Sequential(
            nn.RMSNorm(hidden_size, eps=1e-6),
            nn.Linear(hidden_size, patch_size * patch_size * self.out_channels, bias=False)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize pos_embed with sin-cos embedding
        pos_embed = self.get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.n_patches**0.5))
        self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear with improved initialization
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP with smaller std
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize all linear layers properly
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init_weights)

        # Zero-out adaLN modulation layers (critical for training stability)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers (critical for diffusion training)
        nn.init.constant_(self.final_layer[-1].weight, 0)
        if hasattr(self.final_layer[-1], 'bias') and self.final_layer[-1].bias is not None:
            nn.init.constant_(self.final_layer[-1].bias, 0)

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size, cls_token=False, extra_tokens=0):
        grid_h = torch.arange(grid_size, dtype=torch.float32)
        grid_w = torch.arange(grid_size, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing='ij')
        grid = torch.stack(grid, dim=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token and extra_tokens > 0:
            pos_embed = torch.cat([torch.zeros([extra_tokens, embed_dim]), pos_embed], dim=0)
        return pos_embed

    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        emb = torch.cat([emb_h, emb_w], dim=1)
        return emb

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float64)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega

        pos = pos.reshape(-1)
        out = torch.einsum('m,d->md', pos, omega)
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        emb = torch.cat([emb_sin, emb_cos], dim=1)
        return emb

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1]**0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t):
        # Patch embedding with scaling (like in transformers)
        x = self.x_embedder(x) * math.sqrt(self.x_embedder.proj.out_channels)
        x = x + self.pos_embed
        x = self.input_dropout(x)
        
        # Timestep embedding
        t = self.t_embedder(t)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, t)
        
        # Final layer with adaptive normalization
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = self.modulate(self.final_layer[0](x), shift, scale)
        x = self.final_layer[1](x)
        x = self.unpatchify(x)
        return x

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def get_model():
    # Using DiT (Diffusion Transformer) for unconditional generation
    return DiT(
        input_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=3,  # RGB channels for CIFAR-10
        hidden_size=config.hidden_size,
        depth=config.num_layers,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout
    )

def setup_muon_optimizer(model: nn.Module, config: TrainingConfig):
    """Setup Muon optimizer with hybrid approach for DiT"""
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        # Use Muon for 2D weight matrices (attention, MLP weights)
        if (param.ndim == 2 and 
            'pos_embed' not in name and 
            'norm' not in name and 
            'adaLN_modulation' not in name and
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    
    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")
    
    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(
        adamw_params, 
        lr=config.muon_lr * config.adamw_lr_ratio, 
        weight_decay=config.weight_decay
    )
    
    return [muon_optimizer, adamw_optimizer]

def get_noise_scheduler():
    return diffusers.DDPMScheduler(num_train_timesteps=200)

def get_lr_schedulers(optimizers, config: TrainingConfig, total_steps: int):
    """Create learning rate schedulers for all optimizers"""
    schedulers = []
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    for optimizer in optimizers:
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)
    
    return schedulers

def train_loop(config: TrainingConfig, model, noise_scheduler, train_dataloader):
    """Optimized training loop with Muon optimizer and mixed precision"""
    print(f"\n🚀 Training DiT with Muon optimizer")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Calculate total steps based on epochs
    steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.num_epochs
    print(f"  📊 Training for {config.num_epochs} epochs ({total_steps:,} steps)")
    
    # Compile model for better performance
    if config.compile_model:
        print("🔥 Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")
    
    # Setup optimizers and schedulers
    optimizers = setup_muon_optimizer(model, config)
    schedulers = get_lr_schedulers(optimizers, config, total_steps)
    
    # Mixed precision scaler - use bfloat16 for RTX 4090
    scaler = GradScaler() if config.use_amp and config.mixed_precision == 'fp16' else None
    use_autocast = config.use_amp
    autocast_dtype = torch.bfloat16 if config.mixed_precision == 'bf16' else torch.float16
    
    # Training loop
    model.train()
    step = 0
    best_loss = float('inf')
    
    pbar = tqdm(total=total_steps, desc="Training DiT")
    
    for epoch in range(config.num_epochs):
        print(f"\n📅 Epoch {epoch + 1}/{config.num_epochs}")
        for batch in train_dataloader:
                
            clean_images = batch['images'].to(device)
            noise = torch.randn_like(clean_images)
            batch_size = clean_images.shape[0]
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, 
                (batch_size,), device=device
            )
            
            # Add noise to images
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps.cpu())
            noisy_images = noisy_images.to(device)
            
            # Forward pass with gradient accumulation
            if use_autocast:
                with autocast(dtype=autocast_dtype):
                    noise_pred = model(noisy_images, timesteps)
                    loss = F.mse_loss(noise_pred, noise)
                    loss = loss / config.gradient_accumulation_steps
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                noise_pred = model(noisy_images, timesteps)
                loss = F.mse_loss(noise_pred, noise)
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
            
            # Optimizer step after accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if scaler is not None:
                    # Unscale gradients for clipping
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    
                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    
                    # Step optimizers
                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    
                    # Step schedulers
                    for scheduler in schedulers:
                        scheduler.step()
                    
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
            
            # Logging
            if step % 100 == 0:
                current_loss = loss.item() * config.gradient_accumulation_steps
                lr = optimizers[0].param_groups[0]["lr"]
                
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'lr': f'{lr:.2e}',
                    'step': step
                })
                
                if current_loss < best_loss:
                    best_loss = current_loss
            
            step += 1
            if step % 100 == 0:
                pbar.update(100)
            
            if step >= total_steps:
                break
        
        if step >= total_steps:
            break
    
    pbar.close()
    print(f"  🏆 Best loss: {best_loss:.4f}")
    return model

def main():
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Optimize GPU settings
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        
        # Enable TensorFloat-32 for RTX 4090
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("🚀 Enabled TF32 for faster training")
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Load data
    train_dataloader, cifar10_dataset = get_dataloader()
    print(f"📊 Dataset: {len(cifar10_dataset)} samples, {len(train_dataloader)} batches")
    
    # Initialize model
    model = get_model()
    noise_scheduler = get_noise_scheduler()
    
    # Test model shapes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_image = cifar10_dataset[0]["images"].unsqueeze(0).to(device)
    model_test = model.to(device)
    
    print(f"Input shape: {sample_image.shape}")
    with torch.no_grad():
        test_output = model_test(sample_image, torch.tensor([0]).to(device))
        print(f'Output shape: {test_output.shape}')
    
    print(f"\n📋 Training Configuration:")
    print(f"   Architecture: {config.hidden_size}d, {config.num_layers}L, {config.num_heads}H")
    print(f"   Training: {config.max_steps} steps, batch size {config.train_batch_size}")
    print(f"   Patch size: {config.patch_size}x{config.patch_size}")
    print(f"   Muon LR: {config.muon_lr}, AdamW LR: {config.muon_lr * config.adamw_lr_ratio}")
    
    # Train model
    import time
    start_time = time.time()
    
    trained_model = train_loop(config, model, noise_scheduler, train_dataloader)
    
    training_time = time.time() - start_time
    print(f"  ⏱️ Training completed in {training_time/60:.1f} minutes")
    
    # Save model and config
    os.makedirs("cifar10_diffusion_ckpt", exist_ok=True)
    torch.save(trained_model.state_dict(), "cifar10_diffusion_ckpt/dit_model.pth")
    torch.save(config, "cifar10_diffusion_ckpt/config.pth")
    print(f"💾 Model saved to cifar10_diffusion_ckpt/")
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"⏱️ Total time: {training_time/60:.1f} minutes")

if __name__ == "__main__":
    main()