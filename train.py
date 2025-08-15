import torch
import torchvision
import datasets
import diffusers
import accelerate
from tqdm.auto import tqdm
import os
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 32
    train_batch_size = 64
    eval_batch_size = 32
    num_epochs = 1
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmpup_steps = 500
    mixed_precision = 'fp16'
    seed = 0

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
    mnist_dataset = datasets.load_dataset('mnist', split='train')
    mnist_dataset.reset_format()
    mnist_dataset.set_transform(transform)
    return torch.utils.data.DataLoader(
        mnist_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
    ), mnist_dataset

def get_model():
    # Using UNet2DModel for unconditional generation
    return diffusers.UNet2DModel(
        sample_size=config.image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D", 
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

def get_noise_scheduler():
    return diffusers.DDPMScheduler(num_train_timesteps=200)

def get_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

def get_lr_scheduler(optimizer, train_dataloader):
    return diffusers.optimization.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmpup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

def train_loop(
    config,
    model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    lr_scheduler,
    max_steps=None,
):
    accelerator = accelerate.Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            if max_steps is not None and step >= max_steps:
                break
            clean_images = batch['images']

            noise = torch.randn(clean_images.shape).to(clean_images.device)
            batch_size = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (batch_size,), device=clean_images.device)

            # Fix: move timesteps to cpu for scheduler
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps.cpu())

            with accelerator.accumulate(model):
                # Changed: Transformer2DModel returns tensor directly, not dict
                noise_pred = model(noisy_images, timestep=timesteps).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

    accelerator.unwrap_model(model)

def main():
    torch.manual_seed(config.seed)
    train_dataloader, mnist_dataset = get_dataloader()
    model = get_model()
    noise_scheduler = get_noise_scheduler()
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to("cuda")
    optimizer = get_optimizer(model)
    lr_scheduler = get_lr_scheduler(optimizer, train_dataloader)

    # Optionally, test model input/output shape
    sample_image = mnist_dataset[0]["images"].unsqueeze(0)
    print("Input shape:", sample_image.shape)
    # Changed: use timestep as tensor for transformer
    print('Output shape:', model(sample_image, timestep=torch.tensor([0])).sample.shape)

    # Train
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, max_steps=None)

    # Save model and scheduler
    os.makedirs("mnist_diffusion_ckpt", exist_ok=True)
    model.save_pretrained("mnist_diffusion_ckpt/transformer")
    print("Model saved to mnist_diffusion_ckpt/")

if __name__ == "__main__":
    main()