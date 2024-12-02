import matplotlib.pyplot as plt
import numpy as np
import torch
from dynaconf import Dynaconf

from nvit.model import ViT, ViTConfig


def visualize_patches(img: torch.Tensor, local_patches: torch.Tensor, global_patches: torch.Tensor) -> None:
    """Visualize the local and global patches"""
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img.permute(1, 2, 0).cpu().numpy())
    plt.title("Original Image")

    # Local patches
    plt.subplot(1, 3, 2)
    patch_grid = make_grid(local_patches)
    plt.imshow(patch_grid.cpu().numpy())
    plt.title("Local Patches")

    # Global patches
    plt.subplot(1, 3, 3)
    patch_grid = make_grid(global_patches)
    plt.imshow(patch_grid.cpu().numpy())
    plt.title("Global Patches")

    plt.tight_layout()
    plt.show()

def make_grid(patches: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Create a grid visualization of patches"""
    n = int(np.sqrt(patches.size(0)))
    grid = patches.view(n, n, patches.size(-2), patches.size(-1))
    grid = torch.cat([torch.cat([p for p in row], dim=1) for row in grid], dim=0)
    if normalize:
        grid = (grid - grid.min()) / (grid.max() - grid.min())
    return grid

def debug_model(settings_path: str = "settings.yaml", secrets_path: str = "secrets.yaml") -> None:
    """Debug model by passing sample data through it"""
    # Load settings
    settings = Dynaconf(settings_files=[settings_path], secrets=secrets_path, envvar_prefix="NVIT")

    # Initialize model config from settings
    config = ViTConfig(
        image_size=settings.model.image_size,
        n_layer=settings.model.n_layer,
        n_head=settings.model.n_head,
        n_embd=settings.model.n_embd,
        use_nViT=settings.model.use_nViT,
        dropout=settings.model.dropout,
        bias=settings.model.bias,
        channels=3,
        num_classes=settings.model.num_classes,
        local_patch_size=settings.model.local_patch_size,
        global_patch_size=settings.model.global_patch_size,
        kohonen_nodes=settings.model.kohonen_nodes,
        kohonen_alpha=settings.model.kohonen_alpha,
        use_kohonen=settings.model.use_kohonen,
        reconstruction_weight=settings.model.reconstruction_weight,
        map_balance_weight=settings.model.map_balance_weight,
    )

    # Initialize model
    model = ViT(config)
    print("\nModel configuration:")
    print(f"Local patch size: {config.local_patch_size}")
    print(f"Global patch size: {config.global_patch_size}")
    print(f"Using Kohonen: {config.use_kohonen}")
    print(f"Number of Kohonen nodes: {config.kohonen_nodes}")

    # Move to CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Create sample input
    batch_size = 2
    sample_input = torch.randn(
        batch_size,
        3,
        config.image_size,
        config.image_size,
        device=device,
    )

    # Forward pass
    print(f"\nRunning model on {device}")
    print(f"Input shape: {sample_input.shape}")

    with torch.no_grad():
        # Get intermediate representations for visualization
        local_patches = model.local_patch_embed(sample_input[0:1])
        global_patches = model.global_patch_embed(sample_input[0:1])

        # Run full forward pass
        logits, aux_losses = model(sample_input)

    print("\nOutput shapes:")
    print(f"Logits: {logits.shape}")
    print(f"Local patches: {local_patches.shape}")
    print(f"Global patches: {global_patches.shape}")

    print("\nAuxiliary losses:")
    for loss_name, loss_value in aux_losses.items():
        print(f"{loss_name}: {loss_value.item():.4f}")

    print(f"\nNumber of parameters: {model.num_params:,}")

    # Visualize patches
    try:
        visualize_patches(
            sample_input[0],
            local_patches[0].permute(1, 2, 0),
            global_patches[0].permute(1, 2, 0),
        )
    except Exception as e:
        print(f"\nVisualization failed: {e}")

if __name__ == "__main__":
    debug_model()
