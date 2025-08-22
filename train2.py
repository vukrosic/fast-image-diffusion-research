import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import math

# Minimal UNet for 32x32 images
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x, t):
        h = self.norm1(F.relu(self.conv1(x)))
        h = h + self.time_mlp(F.silu(t))[:, :, None, None]
        h = self.norm2(F.relu(self.conv2(h)))
        return h + self.residual(x)

class MinimalUNet(nn.Module):
    def __init__(self, in_ch=3, ch=64, ch_mult=(1, 2, 2, 2), time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Encoder
        self.down = nn.ModuleList()
        channels = [ch]
        now_ch = ch
        
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            self.down.append(ConvBlock(now_ch if i > 0 else in_ch, out_ch, time_dim))
            channels.append(out_ch)
            now_ch = out_ch
            if i != len(ch_mult) - 1:
                self.down.append(nn.Conv2d(now_ch, now_ch, 3, stride=2, padding=1))
        
        # Middle
        self.mid = ConvBlock(now_ch, now_ch, time_dim)
        
        # Decoder
        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            self.up.append(ConvBlock(now_ch + channels[i], out_ch, time_dim))
            now_ch = out_ch
            if i != 0:
                self.up.append(nn.ConvTranspose2d(now_ch, now_ch, 4, stride=2, padding=1))
        
        self.final = nn.Conv2d(now_ch, in_ch, 1)
        
    def forward(self, x, t):
        t = self.time_mlp(t)
        
        # Encoder
        saved = []
        for i, layer in enumerate(self.down):
            if isinstance(layer, ConvBlock):
                x = layer(x, t)
                saved.append(x)
            else:
                x = layer(x)
        
        # Middle
        x = self.mid(x, t)
        
        # Decoder
        j = 0
        for i, layer in enumerate(self.up):
            if isinstance(layer, ConvBlock):
                x = torch.cat([x, saved[-(j+1)]], dim=1)
                x = layer(x, t)
                j += 1
            else:
                x = layer(x)
        
        return self.final(x)

# DDPM Implementation
class DDPM:
    def __init__(self, model, device, T=1000):
        self.model = model
        self.device = device
        self.T = T
        
        # Pre-compute beta schedule (linear)
        self.betas = torch.linspace(1e-4, 0.02, T).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        
    def q_sample(self, x0, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise, noise
    
    def p_losses(self, x0, t):
        """Calculate loss"""
        noise = torch.randn_like(x0)
        xt, noise = self.q_sample(x0, t, noise)
        pred_noise = self.model(xt, t)
        return F.mse_loss(pred_noise, noise)
    
    @torch.no_grad()
    def p_sample(self, xt, t):
        """Single denoising step"""
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = (1.0 / torch.sqrt(self.alphas[t])).reshape(-1, 1, 1, 1)
        
        pred_noise = self.model(xt, t)
        mean = sqrt_recip_alphas_t * (xt - betas_t * pred_noise / sqrt_one_minus_alpha_bar_t)
        
        if t[0] > 0:
            noise = torch.randn_like(xt)
            std = torch.sqrt(betas_t)
            return mean + std * noise
        else:
            return mean
    
    @torch.no_grad()
    def sample(self, batch_size=16):
        """Generate samples"""
        x = torch.randn(batch_size, 3, 32, 32).to(self.device)
        
        for t in reversed(range(self.T)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t_batch)
        
        return x

# Training script
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    batch_size = 128  # Large batch for 4090
    lr = 2e-4
    epochs = 100  # Should be enough for decent results
    T = 1000
    
    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale to [-1, 1]
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=4, pin_memory=True, persistent_workers=True)
    
    # Model
    model = MinimalUNet(ch=128).to(device)  # Increased channels for better quality
    model = torch.compile(model, mode="reduce-overhead")  # PyTorch 2.0+
    ddpm = DDPM(model, device, T)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Mixed precision for speed
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    model.train()
    step = 0
    
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        losses = []
        
        for x, _ in pbar:
            x = x.to(device)
            
            # Random timesteps
            t = torch.randint(0, T, (x.shape[0],), device=device)
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                loss = ddpm.p_losses(x, t)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)  # More efficient
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            losses.append(loss.item())
            step += 1
            
            # Update progress bar
            if len(losses) > 0:
                pbar.set_postfix({'loss': np.mean(losses[-100:])})
            
            # Generate samples every 1000 steps
            if step % 1000 == 0:
                model.eval()
                samples = ddpm.sample(4)
                model.train()
                # Save or display samples here
                print(f"Step {step}: Generated samples shape: {samples.shape}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'ddpm_cifar10_epoch_{epoch+1}.pt')
    
    return model, ddpm

# Quick inference function
@torch.no_grad()
def generate_images(model_path, num_samples=16):
    device = torch.device('cuda')
    model = MinimalUNet(ch=128).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    ddpm = DDPM(model, device)
    model.eval()
    
    samples = ddpm.sample(num_samples)
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    return samples

if __name__ == "__main__":
    # Train the model
    model, ddpm = train()
    
    # Generate samples
    model.eval()
    samples = ddpm.sample(16)
    print(f"Generated {samples.shape[0]} samples")