"""Debugging utilities for the NViT model."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from dynaconf import Dynaconf
from loguru import logger
from PIL import Image
from torch.nn import functional as F  # noqa: N812
from torchvision import transforms

from nvit.model import ViT, ViTConfig

LOGGER = logger.bind(name="nvit.debug")


def extract_patches(img: torch.Tensor, patch_size: int, stride: int, padding: int = 0) -> torch.Tensor:
    """Extract raw patches from image before embedding.

    :param img: Input image tensor [B, C, H, W]
    :param patch_size: Size of patches to extract
    :param stride: Stride between patches
    :param padding: Padding to apply before extraction
    :return: Patches tensor of shape [N, C, patch_size, patch_size] where N is number of patches
    """
    LOGGER.debug(f"Input image shape: {img.shape}")

    # Add padding if specified
    if padding > 0:
        img = F.pad(img, (padding, padding, padding, padding), mode="reflect")

    # Extract patches using unfold
    # First unfold height dimension
    patches = img.unfold(2, patch_size, stride)
    # Then unfold width dimension
    patches = patches.unfold(3, patch_size, stride)

    # Reshape to [B, C, num_patches_h, num_patches_w, patch_size, patch_size]
    B, C, _, _, H, W = patches.shape

    # Reshape to [N, C, patch_size, patch_size] where N = B * num_patches_h * num_patches_w
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, patch_size, patch_size)

    LOGGER.debug(f"Output patches shape: {patches.shape}")

    return patches


def visualize_patches(img: torch.Tensor, local_patches: torch.Tensor, global_patches: torch.Tensor) -> None:
    """Visualize the original image and its decomposition into patches.
    
    :param img: Original image tensor [B, C, H, W]
    :param local_patches: Local patch tensor (not used, kept for API compatibility)
    :param global_patches: Global patch tensor (not used, kept for API compatibility)
    """
    # Ensure img has a batch dimension
    if img.dim() == 3:
        img = img.unsqueeze(0)  # Add batch dimension

    # Extract raw patches
    local_raw = extract_patches(img, patch_size=8, stride=8)
    global_raw = extract_patches(img, patch_size=16, stride=8, padding=4)

    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    img_show = img[0].permute(1, 2, 0).cpu()
    img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())
    plt.imshow(img_show.to(dtype=torch.float32))
    plt.title(f"Original Image {tuple(img[0].shape)}")

    # Local patches - arrange in 4x4 grid
    plt.subplot(1, 3, 2)
    local_grid = torchvision.utils.make_grid(local_raw, nrow=4, normalize=True, padding=2)
    plt.imshow(local_grid.permute(1, 2, 0).cpu())
    plt.title(f"Local Patches (8x8) - {local_raw.shape[0]} patches")

    # Global patches - arrange in 4x4 grid
    plt.subplot(1, 3, 3)
    global_grid = torchvision.utils.make_grid(global_raw, nrow=4, normalize=True, padding=2)
    plt.imshow(global_grid.permute(1, 2, 0).cpu())
    plt.title(f"Global Patches (16x16) - {global_raw.shape[0]} patches")

    plt.tight_layout()


def make_patch_grid(patches: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Arrange patches in a grid for visualization.
    
    :param patches: Tensor of patches [N, C, H, W]
    :param grid_size: Size of grid (e.g., 4 for 4x4 grid)
    :return: Grid tensor ready for visualization
    """
    LOGGER.debug(f"Input patches shape: {patches.shape}")

    from torchvision.utils import make_grid

    # Don't normalize individual patches
    grid = make_grid(
        patches[:grid_size**2],
        nrow=grid_size,
        padding=2,
        normalize=False,
    )

    # Convert to [H, W, C] for matplotlib
    grid = grid.permute(1, 2, 0)

    LOGGER.debug(f"Final grid shape: {grid.shape}")
    return grid.cpu()


def visualize_kohonen_maps(model: ViT, sample_input: torch.Tensor) -> None:
    """Visualize the Kohonen map activations.

    :param model: ViT model instance
    :param sample_input: Input tensor to visualize
    """
    if not model.config.use_kohonen:
        LOGGER.info("Kohonen maps not enabled")
        return

    plt.figure(figsize=(15, 5))

    # Get local and global patches
    with torch.no_grad():
        local_patches = model.local_patch_embed(sample_input)
        global_patches = model.global_patch_embed(sample_input)

        # Reshape patches
        local_patches = local_patches.flatten(2).transpose(1, 2)
        global_patches = global_patches.flatten(2).transpose(1, 2)

        # Get Kohonen map activations
        local_repr, local_indices = model.local_kohonen(local_patches)
        global_repr, global_indices = model.global_kohonen(global_patches)

    # Plot local map activations
    plt.subplot(1, 3, 1)
    map_size = int(np.sqrt(model.config.kohonen_nodes // 2))
    activation_map = torch.zeros(map_size, map_size)
    unique_indices, counts = torch.unique(local_indices[0], return_counts=True)  # Only use first batch
    for idx, count in zip(unique_indices, counts):
        activation_map[idx // map_size, idx % map_size] = count
    # Normalize to [0, 1]
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
    plt.imshow(activation_map.cpu().numpy(), cmap="viridis")
    plt.title("Local Map Activations")
    plt.colorbar()

    # Plot global map activations
    plt.subplot(1, 3, 2)
    activation_map = torch.zeros(map_size, map_size)
    unique_indices, counts = torch.unique(global_indices[0], return_counts=True)  # Only use first batch
    for idx, count in zip(unique_indices, counts):
        activation_map[idx // map_size, idx % map_size] = count
    # Normalize to [0, 1]
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
    plt.imshow(activation_map.cpu().numpy(), cmap="viridis")
    plt.title("Global Map Activations")
    plt.colorbar()

    # Plot combined representation similarity
    plt.subplot(1, 3, 3)
    combined = model.combine_representations(local_repr, global_repr)
    # Take only the first batch and reshape if needed
    batch_repr = combined[0]  # Shape: (16, 128)

    # Compute pairwise similarities between patches
    similarity_matrix = F.cosine_similarity(
        batch_repr.unsqueeze(0),  # Shape: (1, 16, 128)
        batch_repr.unsqueeze(1),  # Shape: (16, 1, 128)
        dim=2,  # Compare along the embedding dimension
    )  # Result should be (16, 16)

    # Normalize to [0, 1]
    similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min() + 1e-8)

    plt.imshow(similarity_matrix.to(dtype=torch.float32).cpu().numpy(), cmap="viridis")
    plt.title(f"Combined Representation\nSimilarity Matrix ({similarity_matrix.shape[0]}x{similarity_matrix.shape[1]})")
    plt.colorbar()

    LOGGER.info(
        "Shapes for similarity computation:\n"
        f"Local repr: {local_repr.shape}\n"
        f"Global repr: {global_repr.shape}\n"
        f"Combined repr: {combined.shape}\n"
        f"Similarity matrix: {similarity_matrix.shape}",
    )

    plt.tight_layout()
    plt.show(block=False)


def make_grid(patches: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """Create a grid visualization of patches.

    :param patches: Input tensor of patches (B, H, W, C) or (H, W, C)
    :param normalize: Whether to normalize the output grid
    :return: Grid visualization of patches
    """
    # Remove batch dimension if present
    if patches.dim() == 4:
        patches = patches.squeeze(0)  # (H, W, C)

    LOGGER.debug(f"Initial patches shape: {patches.shape}")

    # First, rearrange to (C, H, W) format
    patches = patches.permute(2, 0, 1)  # Now (128, 4, 4)
    C, H, W = patches.shape

    grid_size = int(np.sqrt(C))  # Calculate grid size
    LOGGER.debug(f"Grid size calculated: {grid_size}x{grid_size}")

    # Reshape patches to be (grid_size, grid_size, H, W)
    patches = patches[:grid_size * grid_size]  # Ensure we have a square number of patches
    patches = patches.view(grid_size, grid_size, H, W)
    LOGGER.debug(f"After reshape: {patches.shape}")

    # Rearrange into final grid
    patches = patches.permute(0, 2, 1, 3)  # (grid_size, H, grid_size, W)
    LOGGER.debug(f"After permute: {patches.shape}")
    patches = patches.reshape(grid_size * H, grid_size * W)
    LOGGER.debug(f"Final shape: {patches.shape}")

    if normalize:
        patches = (patches - patches.min()) / (patches.max() - patches.min() + 1e-8)

    return patches


def debug_model(settings_path: str = "settings.yaml", secrets_path: str = "secrets.yaml") -> None:
    """Debug model by passing sample data through it.
    
    :param settings_path: Path to settings YAML file
    :param secrets_path: Path to secrets YAML file
    """
    # Load settings
    settings = Dynaconf(settings_files=[settings_path], secrets=secrets_path, envvar_prefix="NVIT")

    # Initialize model config from settings
    config = ViTConfig(
        image_size=settings.model.image_size,
        n_layer=settings.model.n_layer,
        n_head=settings.model.n_head,
        n_embd=settings.model.n_embd,
        use_nvit=settings.model.use_nvit,
        flash_attn=settings.model.flash_attn,
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

    # Move to CUDA and convert to bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = model.to(device).to(dtype)
    LOGGER.info("Using device: {device} with dtype: {dtype}", device=device, dtype=dtype)

    # Create sample input and convert to bfloat16
    sample_input = load_image("cat.png")
    sample_input = sample_input.to(device).to(dtype)
    batch_size = 2048
    sample_input = sample_input.repeat(batch_size, 1, 1, 1)

    # Full forward pass
    LOGGER.info("Running forward pass on {device}", device=device)
    LOGGER.info("Input shape: {input_shape}", input_shape=sample_input.shape)

    with torch.no_grad():
        # Get intermediate representations for visualization
        local_patches = model.local_patch_embed(sample_input[0:1])
        global_patches = model.global_patch_embed(sample_input[0:1])
        LOGGER.info("Patch embeddings shapes:")
        LOGGER.info("Local patches: {local_patches_shape}", local_patches_shape=local_patches.shape)
        LOGGER.info("Global patches: {global_patches_shape}", global_patches_shape=global_patches.shape)

        # Run full forward pass
        logits, aux_losses = model(sample_input)
        LOGGER.info("Forward pass outputs:")
        LOGGER.info("Logits shape: {logits_shape}", logits_shape=logits.shape)
        LOGGER.info("\nAuxiliary losses:")
        for loss_name, loss_value in aux_losses.items():
            LOGGER.info("{loss_name}: {loss_value:.4f}", loss_name=loss_name, loss_value=loss_value.item())

    LOGGER.info("Number of parameters: {num_params:,}", num_params=model.num_params)

    # Visualize patches and Kohonen maps
    visualize_patches(
        sample_input[0].to(dtype=torch.float32),
        local_patches[0].to(dtype=torch.float32),
        global_patches[0].to(dtype=torch.float32),
    )
    if config.use_kohonen:
        visualize_kohonen_maps(model, sample_input)

    LOGGER.info("Debug complete - model successfully processed forward pass")


def load_image(image_path: str) -> torch.Tensor:
    """Load an image and convert it to a tensor.
    
    :param image_path: Path to the image file
    :return: Image tensor [1, C, H, W]
    """
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
    ])
    # Keep as float32 initially - will be converted to bfloat16 later if needed
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor


def visualize_patches_with_image(image_path: str) -> None:
    """Load an image and visualize its patches.
    
    :param image_path: Path to the image file
    """
    img = load_image(image_path)

    # Extract raw patches
    local_raw = extract_patches(img, patch_size=8, stride=8)
    global_raw = extract_patches(img, patch_size=16, stride=8, padding=4)

    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    img_show = img[0].permute(1, 2, 0).cpu()
    plt.imshow(img_show)
    plt.title(f"Original Image {tuple(img[0].shape)}")

    # Local patches - arrange in 4x4 grid
    plt.subplot(1, 3, 2)
    local_grid = make_patch_grid(local_raw, grid_size=4)
    plt.imshow(local_grid)
    plt.title(f"Local Patches (8x8) - {local_raw.shape[0]} patches")

    # Global patches - arrange in 4x4 grid
    plt.subplot(1, 3, 3)
    global_grid = make_patch_grid(global_raw, grid_size=4)
    plt.imshow(global_grid)
    plt.title(f"Global Patches (16x16) - {global_raw.shape[0]} patches")

    plt.tight_layout()
    plt.show(block=False)


if __name__ == "__main__":
    import time
    start_time = time.time()
    debug_model()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
