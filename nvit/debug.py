import torch
from dynaconf import Dynaconf
from nvit.model import ViT, ViTConfig

def debug_model(settings_path: str = "settings.yaml", secrets_path: str = "secrets.yaml") -> None:
    """Debug model by passing sample data through it"""
    # Load settings
    settings = Dynaconf(settings_files=[settings_path], secrets=secrets_path, envvar_prefix="NVIT")
    
    # Initialize model config from settings
    config = ViTConfig(
        image_size=settings.model.image_size,
        patch_size=settings.model.patch_size,
        n_layer=settings.model.n_layer,
        n_head=settings.model.n_head,
        n_embd=settings.model.n_embd,
        use_nViT=1,
        dropout=settings.model.dropout,
        bias=settings.model.bias,
        channels=3,
        num_classes=settings.model.num_classes
    )
    
    # Initialize model
    model = ViT(config)
    
    print(model.sz)
    
    # Move to CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create sample input
    batch_size = 2
    sample_input = torch.randn(
        batch_size, 
        3, 
        config.image_size, 
        config.image_size,
        device=device
    )
    
    # Forward pass
    print(f"Running model on {device}")
    print(f"Input shape: {sample_input.shape}")
    
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {model.num_params:,}")

if __name__ == "__main__":
    debug_model()
