import torch
import torchvision
import datasets
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def transform(dataset):
    """Transform function to preprocess images"""
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: 2 * (x - 0.5)),
    ])
    # CIFAR-10 uses "img" key instead of "image"
    images = [preprocess(image) for image in dataset["img"]]
    labels = dataset["label"]  # CIFAR-10 labels (0-9)
    return {"images": images, "labels": labels}

def display_cifar_samples(num_samples=10):
    """Download and display CIFAR-10 samples with their labels"""
    # Load CIFAR-10 dataset
    print("Downloading CIFAR-10 dataset...")
    cifar10_dataset = datasets.load_dataset('cifar10', split='train')
    
    # Apply transform
    cifar10_dataset.set_transform(transform)
    
    # Select random samples
    indices = np.random.choice(len(cifar10_dataset), num_samples, replace=False)
    
    # Create subplot grid
    rows = (num_samples + 4) // 5  # Calculate rows needed for 5 columns
    fig, axes = plt.subplots(rows, 5, figsize=(15, 3*rows))
    
    # Flatten axes array for easier indexing
    if rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if num_samples == 1 else axes
    
    # Display images
    for i, idx in enumerate(indices):
        sample = cifar10_dataset[int(idx)]
        image = sample["images"]
        label = sample["labels"]
        
        # Convert tensor to numpy and denormalize
        image_np = image.numpy().transpose(1, 2, 0)
        image_np = (image_np + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        image_np = np.clip(image_np, 0, 1)  # Clip values to valid range
        
        # Display image
        axes[i].imshow(image_np)
        axes[i].set_title(f"{CIFAR10_CLASSES[label]} ({label})")
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Displayed {num_samples} CIFAR-10 samples with their labels")

if __name__ == "__main__":
    display_cifar_samples(10)
