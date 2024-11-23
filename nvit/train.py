import os
import gc
import sys
import time
import math
import signal
import logging
import functools
from pathlib import Path
from dataclasses import asdict
from datetime import timedelta
from dataclasses import dataclass
from contextlib import nullcontext
from collections import OrderedDict
from typing import Union, cast, Any, Optional, Dict, Tuple

import torchvision
import torchvision.transforms as transforms

import torch
import torch.distributed as dist
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Subset
from torch.nn import ModuleDict, functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    LinearLR, 
    SequentialLR, 
    ReduceLROnPlateau,
    _LRScheduler
)

import wandb
import psutil
import numpy as np
from tqdm import tqdm
from dynaconf import Dynaconf

from nvit.model import ViTConfig, ViT


@dataclass
class Model:
    config: ViTConfig
    transformer: ModuleDict
    module: ViT


class MockDPPModel(ViT):
    @classmethod
    def no_sync(cls) -> nullcontext:
        return nullcontext()


class Trainer:
    
    @functools.cached_property
    def ddp(self) -> bool:
        return int(os.environ.get('RANK', -1)) != -1
    
    @classmethod
    @functools.cache
    def get_module(cls, model: ViT) -> Model:
        """Get model components whether model is DDP or not"""
        if isinstance(model, DDP):
            transformer = model.module.transformer
            config = model.module.config
            module = model.module
        else:
            transformer = model.transformer
            config = model.config
            module = model
        return Model(config, transformer, module)
    
    @property
    def model(self) -> ViT:
        ...
    
    @model.setter
    def model(self, value: ViT) -> None:
        self._model = value
        self.get_module.cache_clear()
        
    @model.getter
    def model(self) -> ViT:
        return self.get_module(self._model).module
    
    def __init__(self, settings_path: str = "settings.yaml", secrets_path: str = "secrets.yaml"):
        self.settings = Dynaconf(
            envvar_prefix="NVIT",
            settings_files=[settings_path],
            secrets=secrets_path,
            load_dotenv=True
        )
        self._model: ViT | DDP
        self.device: str
        self.optimizer: torch.optim.Optimizer
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.ctx: Any
        
        self.early_stopping_counter: Optional[int] = None
        self.last_artifact_version: Optional[str] = None
        self.best_val_loss: Optional[float] = None
        self.ddp_rank: Optional[int] = None
        self.ddp_local_rank: Optional[int] = None
        self.ddp_world_size: Optional[int] = None
        
        self.iter_num: int = 0
        self.finished: bool = False
        self.master_process: bool = True
        self.seed_offset: int = 0
        self.last_metrics: Dict[str, float] = {}
        
        # Print all available cuda devices
        print(f"Available CUDA devices: {torch.cuda.device_count()}")
        print(f"CUDA device properties: {torch.cuda.get_device_properties(0)}")
        
        # Setup signal handlers
        if self.master_process:
            signal.signal(signal.SIGINT, self.handle_signal)
            signal.signal(signal.SIGTERM, self.handle_signal)
        
        self.prep_folder()
        self.setup_logging()
        self.setup_distributed()
        self.setup_device_context()
        self.initialize_model()
        
        self.optimizer = self.model.configure_optimizers(
            self.settings.optimizer.weight_decay,
            self.settings.optimizer.learning_rate,
            (self.settings.optimizer.beta1, self.settings.optimizer.beta2),
            self.device
        )
        
        self.scheduler: Optional[_LRScheduler] = None
        self.scaler: Optional[GradScaler] = None
        
        # Setup AMP scaler if using mixed precision
        if self.settings.system.use_amp and self.settings.system.dtype in ['float16', 'bfloat16']:
            self.scaler = GradScaler()
    
    def setup_logging(self) -> None:
        """Setup logging configuration"""
        log_level = logging.INFO if self.master_process else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(Path(self.settings.data.out_dir) / 'training.log')
            ] if self.master_process else [logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def cleanup(self) -> None:
        """Cleanup resources in correct order"""
        try:
            # First save final checkpoint if needed
            if self.master_process and self.iter_num > 0:
                self.save_checkpoint(
                    self.iter_num,
                    self.last_metrics,  # Use stored metrics instead of placeholder
                    torch.get_rng_state()
                )
            
            # Then cleanup DDP if it was used
            if self.ddp:
                try:
                    dist.barrier()
                    dist.destroy_process_group()
                except Exception as e:
                    self.logger.error(f"Error during DDP cleanup: {e}")
            
            # Finally cleanup wandb
            if wandb.run is not None:
                wandb.finish()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def validate_only(self) -> Dict[str, float]:
        """Run validation only mode"""
        self.logger.info("Running in validation-only mode")
        self.train_loader, self.val_loader = self.get_data_loaders()
        
        if self.settings.training.init_from != "resume":
            raise ValueError("Must provide a checkpoint to run validation-only mode")
            
        metrics = self.validate()
        self.logger.info(f"Validation metrics: {metrics}")
        return metrics

    def prep_folder(self) -> None:
        """Prepare output folder"""
        if self.master_process:
            if not os.path.exists(self.settings.data.out_dir):
                os.makedirs(self.settings.data.out_dir, mode=0o777, exist_ok=True)
        
    def setup_distributed(self) -> None:
        """Initialize distributed training settings"""
        if not self.ddp:
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
            return
        
        # Add debug logging
        self.logger.info("Initializing distributed training...")
        self.logger.info(f"Environment variables: RANK={os.environ.get('RANK')}, "
                        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
                        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
        
        self.ddp_rank = int(os.environ['RANK'])
        self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
        self.ddp_world_size = int(os.environ['WORLD_SIZE'])

        # Initialize process group with timeout
        dist.init_process_group(
            backend=self.settings.system.backend,
            init_method='env://',  # Add explicit init method
            timeout=timedelta(minutes=30)  # Add timeout
        )
        
        self.device = f'cuda:{self.ddp_local_rank}'
        torch.cuda.set_device(self.device)
        self.master_process = self.ddp_rank == 0
        self.seed_offset = self.ddp_rank

        # Add synchronization point
        dist.barrier()
        
        if self.master_process:
            self.logger.info(f"Distributed training initialized with {self.ddp_world_size} GPUs")

        assert self.settings.training.gradient_accumulation_steps % self.ddp_world_size == 0
        self.settings.training.gradient_accumulation_steps //= self.ddp_world_size

    def setup_device_context(self) -> None:
        """Setup device type and context for training"""
        device_type = self.settings.system.device
        ptdtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16
        }[self.settings.system.dtype]
        
        self.ctx = (
            nullcontext() if device_type == 'cpu' or not self.settings.system.use_amp 
            else autocast(device_type=device_type, dtype=ptdtype)
        )

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Initialize and return training and validation data loaders"""
        try:
            trainset = None
            valset = None
            
            # Get base transforms
            train_transform_list = []
            val_transform_list = []
            
            if self.settings.data.dataset.lower() == 'imagenet':
                # Base transforms for ImageNet
                train_transform_list.extend([
                    transforms.RandomResizedCrop(self.settings.model.image_size),
                    transforms.RandomHorizontalFlip(),
                ])
                val_transform_list.extend([
                    transforms.Resize(256),
                    transforms.CenterCrop(self.settings.model.image_size),
                ])
                
                # Normalization values for ImageNet
                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                
            elif self.settings.data.dataset.lower() in ['cifar10', 'cifar100']:
                # Base transforms for CIFAR
                train_transform_list.extend([
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize(self.settings.model.image_size),
                    transforms.RandomHorizontalFlip(),
                ])
                val_transform_list.extend([
                    transforms.Resize(self.settings.model.image_size),
                ])
                
                # Normalization values for CIFAR
                normalize = transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)
                )
            else:
                raise ValueError(f"Unsupported dataset: {self.settings.data.dataset}")

            # Add augmentations if enabled
            if self.settings.data.augmentation.enabled:
                if self.settings.data.augmentation.color_jitter:
                    train_transform_list.append(
                        transforms.ColorJitter(
                            brightness=self.settings.data.augmentation.color_jitter,
                            contrast=self.settings.data.augmentation.color_jitter,
                            saturation=self.settings.data.augmentation.color_jitter,
                        )
                    )
                
                if self.settings.data.augmentation.random_affine:
                    train_transform_list.append(
                        transforms.RandomAffine(
                            degrees=15,
                            translate=(0.1, 0.1)
                        )
                    )
                
                if self.settings.data.augmentation.cutout:
                    train_transform_list.append(
                        transforms.RandomErasing(
                            p=0.5,
                            scale=(0.02, 0.33),
                            ratio=(0.3, 3.3),
                        )
                    )
                    
                if self.settings.data.augmentation.auto_augment:
                    if self.settings.data.dataset.lower() == 'imagenet':
                        train_transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
                    else:
                        train_transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))

            # Add final transforms
            train_transform_list.extend([transforms.ToTensor(), normalize])
            val_transform_list.extend([transforms.ToTensor(), normalize])

            # Create transform compositions
            train_transform = transforms.Compose(train_transform_list)
            val_transform = transforms.Compose(val_transform_list)
            
            # Create datasets
            if self.settings.data.dataset.lower() == 'imagenet':
                trainset = torchvision.datasets.ImageNet(
                    root='./data',
                    split='train',
                    transform=train_transform,
                    download=self.master_process # Only download on main node
                )
                valset = torchvision.datasets.ImageNet(
                    root='./data', 
                    split='val',
                    transform=val_transform,
                    download=self.master_process # Only download on main node
                )
            else:  # CIFAR10 or CIFAR100
                dataset_class = (torchvision.datasets.CIFAR10 
                               if self.settings.data.dataset.lower() == 'cifar10'
                               else torchvision.datasets.CIFAR100)
                
                trainset = dataset_class(
                    root='./data',
                    train=True,
                    download=self.master_process, # Only download on main node
                    transform=train_transform
                )
                valset = dataset_class(
                    root='./data',
                    train=False,
                    download=self.master_process, # Only download on main node
                    transform=val_transform
                )

            if trainset is None or valset is None:
                raise ValueError(f"Dataset {self.settings.data.dataset} not properly initialized")

            # Create samplers for distributed training
            train_sampler = DistributedSampler(
                trainset,
                num_replicas=self.ddp_world_size if self.ddp else 1,
                rank=self.ddp_rank if self.ddp else 0,
                shuffle=True,
                seed=self.settings.training.seed if hasattr(self.settings.training, 'seed') else 42
            ) if self.ddp else None

            val_sampler = DistributedSampler(
                valset,
                num_replicas=self.ddp_world_size if self.ddp else 1,
                rank=self.ddp_rank if self.ddp else 0,
                shuffle=False
            ) if self.ddp else None

            # Create data loaders with appropriate samplers
            train_loader = DataLoader(
                trainset, 
                batch_size=self.settings.training.batch_size,
                shuffle=(train_sampler is None),  # Don't shuffle if using sampler
                sampler=train_sampler,
                num_workers=self.settings.data.num_workers,
                pin_memory=True if self.settings.system.device == 'cuda' else False,
                drop_last=True  # Recommended for DDP to avoid uneven batch sizes
            )
            
            val_loader = DataLoader(
                valset, 
                batch_size=self.settings.training.batch_size,
                shuffle=False,  # Don't shuffle validation
                sampler=val_sampler,
                num_workers=self.settings.data.num_workers,
                pin_memory=True if self.settings.system.device == 'cuda' else False,
                drop_last=False
            )
            
            return train_loader, val_loader
            
        except Exception as e:
            self.logger.error(f"Error loading datasets: {e}")
            raise
    
    def load_from_wandb(self, artifact_name: str) -> None:
        """Load model from wandb artifact"""
        if self.settings.wandb.mode != "online":
            raise ValueError("Wandb must be enabled and online to load from artifacts")
        
        api = wandb.Api()
        artifact = api.artifact(artifact_name, type='model')
        artifact_dir = artifact.download()
        
        checkpoint_path = Path(artifact_dir) / 'checkpoint_best.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found in artifact: {checkpoint_path}")
        
        self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load checkpoint from path"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model = ViT(ViTConfig(**checkpoint['model_args']))
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.iter_num = checkpoint['iter_num']
            
            # Restore RNG state
            torch.set_rng_state(checkpoint['rng_state_pytorch'])
            np.random.set_state(checkpoint['rng_state_numpy'])
            
            self.logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
            self.logger.info(f"Resuming from iteration {self.iter_num}")
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            raise

    def initialize_model(self) -> None:
        """Initialize the model architecture"""
        model_args = {
            'image_size': self.settings.model.image_size,
            'patch_size': self.settings.model.patch_size,
            'n_layer': self.settings.model.n_layer,
            'n_head': self.settings.model.n_head,
            'n_embd': self.settings.model.n_embd,
            'use_nViT': self.settings.model.use_nViT,
            'dropout': self.settings.model.dropout,
            'bias': self.settings.model.bias,
            'num_classes': self.settings.model.num_classes
        }

        if self.settings.training.init_from == 'scratch':
            self.model = ViT(ViTConfig(**model_args))
        elif self.settings.training.init_from == 'resume':
            ckpt_path = Path(self.settings.data.checkpoint_dir) / self.settings.data.checkpoint_file
            self.load_checkpoint(ckpt_path)
        elif self.settings.training.init_from == 'wandb':
            self.load_from_wandb(self.settings.wandb.artifact_name)
        else:
            raise ValueError(f"Invalid init_from value: {self.settings.training.init_from}")
        
        self.model.to(self.device)
        
        # First wrap with DDP if using distributed training
        if self.ddp:
            self.logger.info(f"DDP: {self.ddp}, wrapping the model")
            self.model = cast(ViT, DDP(
                self.model, 
                device_ids=[self.ddp_local_rank],
                static_graph=False,  # Required for torch.compile compatibility
                broadcast_buffers=False,
                find_unused_parameters=False
            ))
            
        # Then compile if enabled
        if self.settings.system.compile:
            self.logger.info("Compiling model with torch.compile()")
            self.model = cast(ViT, torch.compile(self.model))

        # Verify model wrapping
        if self.ddp:
            assert isinstance(self.model, (DDP, torch._dynamo.eval_frame.OptimizedModule)), \
                f"Model should be DDP or compiled DDP, but got {type(self.model)}"

    def normalize_matrices(self) -> None:
        """Normalize model matrices if using nViT"""
        if not self.settings.model.use_nViT:
            return

        def justnorm(x: torch.Tensor, idim: int = -1) -> torch.Tensor:
            dtype = x.dtype
            x = x.float()
            return (x / x.norm(p=2, dim=idim, keepdim=True)).to(dtype=dtype)

        model_obj = self.get_module(cast(ViT, self.model))
        
        # Normalize transformer blocks
        for block in model_obj.transformer.h:
            block.query.weight.data.copy_(justnorm(block.query.weight.data, 1))
            block.key.weight.data.copy_(justnorm(block.key.weight.data, 1))
            block.value.weight.data.copy_(justnorm(block.value.weight.data, 1))
            block.att_c_proj.weight.data.copy_(justnorm(block.att_c_proj.weight.data, 0))
            block.c_fc.weight.data.copy_(justnorm(block.c_fc.weight.data, 1))
            block.mlp_c_proj.weight.data.copy_(justnorm(block.mlp_c_proj.weight.data, 0))

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """Estimate loss on train and validation sets"""
        out = {}
        self.model.eval()
        for split, loader in [('train', self.train_loader), ('val', self.val_loader)]:
            losses = torch.zeros(self.settings.training.eval_iters)
            for k, (X, y) in enumerate(loader):
                if k >= self.settings.training.eval_iters:
                    break
                X, y = X.to(self.device), y.to(self.device)
                with self.ctx:
                    logits = self.model(X)
                    loss = F.cross_entropy(logits, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def setup_wandb(self) -> None:
        """Initialize wandb logging"""
        if not self.master_process or self.settings.wandb.mode not in ["online", "offline"]:
            return
        
        if self.settings.wandb.mode == "online":
            print(f"Logging to wandb with key: {self.settings.get('wandb_api_key', os.getenv('WANDB_API_KEY'))[:5]}...")
            wandb.login(key=self.settings.get('wandb_api_key', os.getenv('WANDB_API_KEY')))  # loaded from secrets.yaml or environment variable NVIT_WANDB_API_KEY, or WANDB_API_KEY
        
        wandb_config = {
            "model_config": asdict(self.model.config),
            "training": self.settings.training,
            "optimizer": self.settings.optimizer,
            "system": self.settings.system,
        }

        wandb.init(
            mode=self.settings.wandb.mode,
            project=self.settings.wandb.project,
            name=f"{self.settings.wandb.run_name}_{time.strftime('%Y%m%d_%H%M%S')}",
            config=wandb_config,
        )
        
        # Only watch the model if not using DDP or torch.compile
        if not self.ddp and not self.settings.system.compile:
            wandb.watch(
                self.model,
                log="all",
                log_freq=self.settings.training.log_interval,
                log_graph=True,
            )
        else:
            # For DDP/compiled models, watch with reduced monitoring
            wandb.watch(
                self.model,
                log="parameters",  # Only log parameters, not gradients
                log_freq=self.settings.training.log_interval,
                log_graph=False,  # Disable graph logging
            )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb with proper grouping"""
        if not self.master_process or wandb.run is None:
            return

        # Group metrics by prefix (e.g., 'train/', 'val/', 'optimizer/')
        grouped_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            grouped_metrics[key] = value

        wandb.log(grouped_metrics, step=step)

    @torch.no_grad()
    def compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
        """
        Compute top-1 and top-5 accuracy
        """
        maxk = min(5, logits.size(1))  # top-5 or less if num_classes < 5
        batch_size = targets.size(0)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        top1_acc = correct[0].float().sum().item() * 100.0 / batch_size
        top5_acc = correct[:maxk].float().sum().item() * 100.0 / batch_size

        return top1_acc, top5_acc

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Full validation loop with detailed metrics
        """
        self.model.eval()
        val_loss = 0.0
        top1_acc = 0.0
        top5_acc = 0.0
        num_batches = 0

        for X, y in self.val_loader:
            X, y = X.to(self.device), y.to(self.device)
            
            with self.ctx:
                logits = self.model(X)
                loss = F.cross_entropy(logits, y)
                
            batch_top1, batch_top5 = self.compute_accuracy(logits, y)
            
            val_loss += loss.item()
            top1_acc += batch_top1
            top5_acc += batch_top5
            num_batches += 1

        metrics = {
            "val/loss": val_loss / num_batches,
            "val/top1_accuracy": top1_acc / num_batches,
            "val/top5_accuracy": top5_acc / num_batches
        }
        
        self.model.train()
        return metrics

    def save_checkpoint(self, iter_num: int, metrics: Dict[str, float], rng_state_pytorch: torch.Tensor) -> None:
        """Save model checkpoint with improved metadata and wandb artifact handling"""
        if not self.master_process:
            return
        
        tcheckpointsaving_begin = time.time()
        raw_model = self.get_module(self.model).module
        
        # Format timestamp properly
        timestamp = time.strftime('%d_%m_%Y-%Hh%Mm')
        
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': asdict(self.model.config),
            'iter_num': iter_num,
            'metrics': metrics,
            'config': self.settings.to_dict(),
            'rng_state_pytorch': rng_state_pytorch,
            'rng_state_numpy': np.random.get_state(),
            'timestamp': timestamp
        }

        # Save latest checkpoint locally
        latest_path = Path(self.settings.data.out_dir) / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)

        # Handle best checkpoint
        current_val_loss = metrics['val/loss']
        is_best = current_val_loss < getattr(self, 'best_val_loss', float('inf'))
        
        if is_best:
            self.best_val_loss = current_val_loss
            
            # Save best checkpoint locally
            best_path = Path(self.settings.data.out_dir) / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            
            # Save to wandb if enabled
            if wandb.run is not None:
                # Create artifact name with timestamp
                artifact_name = f"model-{self.settings.wandb.run_name}-{'nvit' if self.settings.model.use_nViT else 'vit'}-{timestamp}"
                
                # Create a wandb Artifact
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="model",
                    metadata={
                        "iter_num": iter_num,
                        "val_loss": current_val_loss,
                        "metrics": metrics,
                        "timestamp": timestamp,
                        "using_nvit": self.settings.model.use_nViT
                    }
                )
                
                # Add file to artifact
                artifact.add_file(str(best_path))
                
                # Log artifact to wandb
                wandb.log_artifact(artifact)
                
                # Delete old version if it exists
                if hasattr(self, 'last_artifact_version') and self.last_artifact_version is not None:
                    try:
                        api = wandb.Api()
                        run = wandb.run
                        
                        if run is None:
                            raise ValueError("Wandb run is not initialized")
                        
                        old_artifact_name = f"{run.entity}/{run.project}/{self.last_artifact_version}"
                        artifact = api.artifact(old_artifact_name)
                        artifact.delete()
                    except Exception as e:
                        self.logger.info(f"Failed to delete old artifact: {e}")
                
                # Store new artifact name
                self.last_artifact_version = artifact_name

        self.logger.info(f"Checkpoint saving time: {time.time()-tcheckpointsaving_begin:.2f} sec")

    def should_stop_early(self, val_loss: float) -> bool:
        """Check if training should stop early"""
        if not hasattr(self, 'early_stopping_counter') or self.early_stopping_counter is None:
            self.early_stopping_counter = 0
            self.best_val_loss = float('inf')
        
        if self.best_val_loss is None:
            self.best_val_loss = float('inf')
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        return (self.early_stopping_counter >= self.settings.training.early_stopping_patience 
                if hasattr(self.settings.training, 'early_stopping_patience') else False)

    def evaluate(self) -> Dict[str, float]:
        """Periodic evaluation with improved metrics"""
        rng_state_pytorch = torch.get_rng_state()
        
        # Get detailed validation metrics
        val_metrics = self.validate()
        train_loss = self.estimate_loss()['train']
        
        metrics = {
            "train/loss": train_loss,  # Changed from "train" to "train/loss"
            "val/loss": val_metrics['val/loss'],
            "val/top1_accuracy": val_metrics['val/top1_accuracy'],
            "val/top5_accuracy": val_metrics['val/top5_accuracy'],
            "optimizer/learning_rate": self.get_lr(self.iter_num) if self.settings.optimizer.decay_lr else self.settings.optimizer.learning_rate,
            "training/global_step": self.iter_num,
        }
        
        # Store metrics for later use
        self.last_metrics = metrics.copy()
        
        # Add model statistics
        if hasattr(self.model, 'parameters'):
            metrics["model/gradient_norm"] = self.compute_gradient_norm()
        metrics["model/parameter_norm"] = self.compute_parameter_norm()
        
        # Log metrics to wandb
        if self.master_process:
            self.log_metrics(metrics)
        
        # Check for early stopping
        if self.should_stop_early(metrics['val/loss']):  # Updated key
            self.logger.info("Early stopping triggered!")
            self.mark_training_finished()
        
        # Save checkpoint
        if self.settings.training.always_save_checkpoint and self.iter_num > 0:
            self.save_checkpoint(self.iter_num, metrics, rng_state_pytorch)
        
        return metrics

    def compute_gradient_norm(self) -> float:
        """Compute total gradient norm for all parameters"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def compute_parameter_norm(self) -> float:
        """Compute total parameter norm"""
        total_norm = 0.0
        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def get_memory_usage(self) -> Dict[str, float]:
        if not self.settings.system.log_memory:
            return {}
        
        """Get current memory usage statistics"""
        memory_stats = {
            "ram_used_gb": psutil.Process().memory_info().rss / (1024 * 1024 * 1024),
            "ram_percent": psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            memory_stats.update({
                "cuda_used_gb": torch.cuda.memory_allocated() / (1024 * 1024 * 1024),
                "cuda_cached_gb": torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
            })
        
        return memory_stats

    def train(self) -> None:
        """Main training loop"""
        try:
            tlaunch = time.time()
            
            # Add debug logging for data loading
            self.logger.info("Setting up data loaders...")
            self.train_loader, self.val_loader = self.get_data_loaders()
            self.logger.info("Data loaders initialized successfully")
            
            # Initialize wandb only on master process
            if self.master_process:
                self.setup_wandb()
            
            # Ensure model is properly wrapped with DDP
            if self.ddp:
                self.logger.info(f"Model type before DDP wrap: {type(self.model)}")
                if not isinstance(self.model, DDP):
                    self.model = cast(ViT, DDP(
                        self.model,
                        device_ids=[self.ddp_local_rank],
                        output_device=self.ddp_local_rank,
                        static_graph=False,
                        broadcast_buffers=False,
                        find_unused_parameters=False
                    ))
                    self.model = cast(ViT, torch.compile(self.model))
                self.logger.info(f"Model type after DDP wrap: {type(self.model)}")
            
            # Add synchronization point before training
            if self.ddp:
                dist.barrier()
            
            # Initialize progress bar
            pbar = tqdm(total=self.settings.training.max_iters, 
                       initial=self.iter_num,
                       desc="Training",
                       disable=not self.master_process)
            
            postfix: Dict[str, Union[str, float, int]] = OrderedDict({
                "loss": "+inf", 
                "lr": f"{self.settings.optimizer.learning_rate:.4e}", 
                "time_ms": "0.000"
            })

            # Initialize training state
            local_iter_num = 0
            t0 = time.time()
            
            # Initialize stats file if starting from scratch
            if self.master_process and self.settings.training.init_from == 'scratch':
                stat_fname = Path(self.settings.data.out_dir) / "stat"
                with open(stat_fname, "w") as f:
                    resstr = f"{0:.6e} {0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0:.4e} {0.0:.4e}"
                    resstr = resstr + self.get_hparams_str() + "\n"
                    f.write(resstr)
            
            if self.iter_num == 0 and self.settings.training.eval_only:
                self.evaluate()

            # Calculate total epochs based on max_iters and dataset size
            current_epoch = math.floor(self.iter_num / len(self.train_loader))

            while (local_iter_num < self.settings.training.max_iters_per_launch
                   and self.iter_num < self.settings.training.max_iters
                   and time.time() - tlaunch < self.settings.training.time_limit_seconds
                   and not self.finished):
                
                # Set epoch for distributed sampler
                if self.ddp:
                    self.train_loader.sampler.set_epoch(current_epoch)  # type: ignore
                
                # Set random seed for reproducibility
                local_seed = 100 * self.iter_num + self.seed_offset
                np.random.seed(local_seed)
                torch.manual_seed(local_seed)
                torch.cuda.manual_seed(local_seed)

                # Learning rate scheduling
                lr = self.get_lr(self.iter_num) if self.settings.optimizer.decay_lr else self.settings.optimizer.learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                    
                if self.iter_num % self.settings.training.eval_interval == 0 and self.master_process:
                    losses = self.evaluate()
                    
                    # Write statistics
                    self.write_statistics(self.iter_num, lr, losses)

                # Training step
                for X, y in self.train_loader:
                    if self.settings.system.device == 'cuda':
                        X = X.pin_memory().to(self.device, non_blocking=True)
                        y = y.pin_memory().to(self.device, non_blocking=True)
                    else:
                        X, y = X.to(self.device), y.to(self.device)

                    loss = torch.tensor(torch.inf, device=self.device)
                    for micro_step in range(self.settings.training.gradient_accumulation_steps):
                        if isinstance(self._model, DDP) and micro_step < self.settings.training.gradient_accumulation_steps - 1:
                            context = self._model.no_sync()
                        else:
                            context = nullcontext()
                        
                        with context, self.ctx:
                            logits = self.model(X)
                            loss = F.cross_entropy(logits, y)
                            loss = loss / self.settings.training.gradient_accumulation_steps

                        if self.scaler is not None:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    if self.settings.optimizer.grad_clip != 0.0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.optimizer.grad_clip)
                    
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Update scheduler if it exists
                    if self.scheduler is not None:
                        self.scheduler.step()

                    # Timing and logging
                    t1 = time.time()
                    dt = t1 - t0
                    t0 = t1
                    
                    metrics = None

                    if self.iter_num % self.settings.training.log_interval == 0 and self.master_process:
                        memory_stats = self.get_memory_usage()
                        memory_metrics = {f"system/{k}": v for k, v in memory_stats.items()}
                        
                        gpu_stats = self.log_gpu_stats()
                        gpu_metrics = {f"system/{k}": v for k, v in gpu_stats.items()}
                        
                        # Garbage collection if memory usage is high
                        if self.settings.system.clear_cache and memory_stats.get("cuda_used_gb", 0) > 0.9 * torch.cuda.get_device_properties(self.device).total_memory:
                            gc.collect()
                            torch.cuda.empty_cache()

                        lossf = loss.item() * self.settings.training.gradient_accumulation_steps
                        
                        # Log training metrics
                        metrics = {
                            "train/iter": self.iter_num,
                            "train/batch_loss": lossf,
                            "train/batch_time_ms": dt * 1000,
                            "optimizer/learning_rate": lr,
                            **memory_metrics,
                            **gpu_metrics
                        }
                        
                        self.log_metrics(metrics)

                    if self.settings.model.use_nViT:
                        self.normalize_matrices()

                    self.iter_num += 1
                    local_iter_num += 1
                    
                    if self.master_process:
                        pbar.update(1)
                        postfix.update({"epoch": current_epoch})
                        if metrics:
                            postfix.update(**{
                                "loss": f"{metrics['train/batch_loss']:.4f}", 
                                "lr": f"{metrics['optimizer/learning_rate']:.4e}", 
                                "time_ms": f"{dt*1000:.1f}"
                            })
                            metrics = None
                        pbar.set_postfix(ordered_dict=postfix)
                    
                    if self.iter_num >= self.settings.training.max_iters:
                        self.mark_training_finished()

                current_epoch += 1

            if self.ddp:
                dist.barrier()
                dist.destroy_process_group()

        except Exception as e:
            self.handle_error(e)
            raise
        finally:
            self.cleanup()

    def get_lr(self, iter_num: int) -> float:
        """Calculate learning rate with warmup and decay"""
        if iter_num < self.settings.optimizer.warmup_iters:
            return self.settings.optimizer.learning_rate * iter_num / self.settings.optimizer.warmup_iters
        if iter_num > self.settings.optimizer.lr_decay_iters:
            return self.settings.optimizer.min_lr
        
        decay_ratio = (iter_num - self.settings.optimizer.warmup_iters) / (
            self.settings.optimizer.lr_decay_iters - self.settings.optimizer.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.settings.optimizer.min_lr + coeff * (
            self.settings.optimizer.learning_rate - self.settings.optimizer.min_lr
        )

    def get_hparams_str(self) -> str:
        """Get hyperparameter string for logging"""
        if not self.settings.model.use_nViT:
            return ""
        
        model_obj = self.get_module(cast(ViT, self.model))
        
        resstr = f"{torch.mean(model_obj.module.sz * (self.settings.model.sz_init_value / self.settings.model.sz_init_scaling)):.5f} " if self.settings.model.use_nViT else "No sz, not using nViT"
        
        for block in model_obj.transformer.h:
            sqk = block.sqk * (block.sqk_init_value/block.sqk_init_scaling)
            attn_alpha = block.attn_alpha * (block.attn_alpha_init_value / block.attn_alpha_init_scaling)
            mlp_alpha = block.mlp_alpha * (block.mlp_alpha_init_value / block.mlp_alpha_init_scaling)
            suv = block.suv * (block.suv_init_value/block.suv_init_scaling)

            resstr += f"{torch.mean(sqk):.5f} "
            resstr += f"{torch.mean(attn_alpha):.5f} "
            resstr += f"{torch.mean(mlp_alpha):.5f} "
            resstr += f"{torch.mean(suv):.5f} "
             
        return resstr

    def write_statistics(self, iter_num: int, lr: float, losses: Dict[str, float]) -> None:
        """Write training statistics to file"""
        stat_fname = Path(self.settings.data.out_dir) / "stat"
        with open(stat_fname, "a" if self.settings.training.init_from == 'resume' else "w") as f:
            # Use the consistent keys
            resstr = f"{iter_num:.6e} {lr:.4e} {losses['train/loss']:.4e} {losses['val/loss']:.4e} "
            resstr += "0.0:.4e " * 9  # Placeholder values
            resstr += self.get_hparams_str() + "\n"
            f.write(resstr)
            f.flush()

    def mark_training_finished(self) -> None:
        """Mark training as finished by writing to a file"""
        self.finished = True
        finished_fname = Path(self.settings.data.out_dir) / "finished"
        with open(finished_fname, "w") as f:
            f.write("1")

    def get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Get dataset-specific transforms"""
        if self.settings.data.dataset.lower() == 'imagenet':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.settings.model.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
            ])
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.settings.model.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
            ])
        elif self.settings.data.dataset.lower() in ['cifar10', 'cifar100']:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(self.settings.model.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            val_transform = transforms.Compose([
                transforms.Resize(self.settings.model.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            raise ValueError(f"Unsupported dataset: {self.settings.data.dataset}")
        
        return train_transform, val_transform

    def setup_scheduler(self) -> None:
        """Setup learning rate scheduler based on settings"""
        if not hasattr(self.settings.optimizer, 'scheduler'):
            return
        
        if self.settings.optimizer.scheduler.type == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.settings.optimizer.lr_decay_iters,
                eta_min=self.settings.optimizer.min_lr
            )
        elif self.settings.optimizer.scheduler.type == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.settings.optimizer.scheduler.factor,
                patience=self.settings.optimizer.scheduler.patience,
                min_lr=self.settings.optimizer.min_lr
            )
        elif self.settings.optimizer.scheduler.type == 'linear':
            scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.settings.optimizer.min_lr / self.settings.optimizer.learning_rate,
                total_iters=self.settings.optimizer.lr_decay_iters
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.settings.optimizer.scheduler.type}")
        
        if self.settings.optimizer.warmup_iters > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.0,
                end_factor=1.0,
                total_iters=self.settings.optimizer.warmup_iters
            )
            self.scheduler = SequentialLR(
                self.optimizer,  # type: ignore
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.settings.optimizer.warmup_iters]
            )
        else:
            self.scheduler = scheduler  # type: ignore

    def log_gpu_stats(self) -> Dict[str, float]:
        """Log multi-GPU training statistics"""
        if not torch.cuda.is_available() or not self.ddp or not self.settings.system.log_gpu_stats:
            return {}
        
        metrics = {}
        for i in range(torch.cuda.device_count()):
            gpu_stats = {
                f"gpu_{i}/memory_used": torch.cuda.memory_allocated(i) / 1e9,
                f"gpu_{i}/memory_cached": torch.cuda.memory_reserved(i) / 1e9,
                f"gpu_{i}/utilization": torch.cuda.utilization(i)
            }
            metrics.update(gpu_stats)
        
        return metrics

    def handle_error(self, error: Exception) -> None:
        """Handle common training errors with helpful messages"""
        if isinstance(error, RuntimeError):
            if "out of memory" in str(error):
                self.logger.error(
                    "GPU OUT OF MEMORY!\n"
                    "Try:\n"
                    "\t1. Reducing batch size\n"
                    "\t2. Reducing model size\n"
                    "\t3. Using gradient accumulation\n"
                    "\t4. Using mixed precision training"
                )
            elif "CUDA error" in str(error):
                self.logger.error(
                    "CUDA ERROR!\n"
                    "Try:\n"
                    "\t1. Checking GPU availability\n"
                    "\t2. Updating CUDA drivers\n"
                    "\t3. Reducing model size"
                )
        elif isinstance(error, ValueError):
            self.logger.error(f"Configuration error: {error}")
        else:
            self.logger.error(f"Unknown error: {error}")

    def handle_signal(self, signum: int, frame: Any) -> None:
        """Handle termination signals gracefully"""
        self.logger.info(f"Received signal {signum}. Performing cleanup...")
        self.cleanup()
        self.save_checkpoint(self.iter_num, {"train/loss": 0.0}, torch.get_rng_state())
        sys.exit(0)

    def get_validation_subset(self, num_samples: Optional[int] = None) -> DataLoader:
        """Create a subset of validation data for quick evaluation"""
        if not self.settings.system.quick_validation:
            return self.val_loader
        
        num_samples = num_samples or self.settings.system.quick_validation_size
        if len(self.val_loader.dataset) <= num_samples:  # type: ignore
            return self.val_loader
        
        indices = torch.randperm(len(self.val_loader.dataset))[:num_samples]  # type: ignore
        subset_dataset = Subset(self.val_loader.dataset, indices)  # type: ignore
        
        return DataLoader(
            subset_dataset,
            batch_size=self.settings.training.batch_size,
            shuffle=False,
            num_workers=self.settings.data.num_workers,
            pin_memory=True if self.settings.system.device == 'cuda' else False
        )

def main():
    trainer = Trainer()
    if trainer.settings.training.eval_only:
        trainer.validate_only()
    else:
        trainer.train()

if __name__ == "__main__":
    main()
