import os
import random
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import omegaconf
import torch
import torch.distributed as dist
import torch.multiprocessing

try:
    import intel_extension_for_pytorch
except ImportError:
    pass

from omegaconf import OmegaConf, listconfig
from torch.nn.parallel import DistributedDataParallel as DDP

from boltz.data.module.training import BoltzTrainingDataModule, DataConfig
from boltz.data.module.utils import transfer_batch_to_device


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rank_zero_only(fn):
    """Decorator that only executes the function on rank 0."""

    def wrapper(*args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapper


def is_rank_zero() -> bool:
    """Check if the current process is rank 0."""
    return not dist.is_initialized() or dist.get_rank() == 0


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    val_metric: float,
    dirpath: str,
    filename: str = "last.ckpt",
) -> None:
    """Save a training checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        The model to save.
    optimizer : torch.optim.Optimizer
        The optimizer state.
    scheduler : Any
        The learning rate scheduler.
    epoch : int
        The current epoch.
    global_step : int
        The global step count.
    val_metric : float
        The validation metric value.
    dirpath : str
        The directory path to save the checkpoint.
    filename : str
        The filename for the checkpoint.

    """
    # Unwrap DDP model if needed
    model_to_save = model.module if isinstance(model, DDP) else model

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metric": val_metric,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Save EMA state if model has it
    if hasattr(model_to_save, "use_ema") and model_to_save.use_ema and model_to_save.ema is not None:
        checkpoint["ema"] = model_to_save.ema.state_dict()

    path = Path(dirpath) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    checkpoint_path: str,
    strict: bool = True,
) -> tuple[int, int, float]:
    """Load a training checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        The model to load into.
    optimizer : torch.optim.Optimizer
        The optimizer to load state into.
    scheduler : Any
        The learning rate scheduler to load state into.
    checkpoint_path : str
        The path to the checkpoint file.
    strict : bool
        Whether to strictly enforce state dict matching.

    Returns
    -------
    tuple[int, int, float]
        The epoch, global step, and validation metric from the checkpoint.

    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load model state
    model_to_load = model.module if isinstance(model, DDP) else model
    model_to_load.load_state_dict(checkpoint["state_dict"], strict=strict)

    # Load optimizer state
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Load EMA state
    if hasattr(model_to_load, "on_load_checkpoint"):
        model_to_load.on_load_checkpoint(checkpoint)

    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    val_metric = checkpoint.get("val_metric", 0.0)

    return epoch, global_step, val_metric


@dataclass
class TrainConfig:
    """Train configuration.

    Attributes
    ----------
    data : DataConfig
        The data configuration.
    model : torch.nn.Module
        The model configuration.
    output : str
        The output directory.
    trainer : Optional[dict]
        The trainer configuration.
    resume : Optional[str]
        The resume checkpoint.
    pretrained : Optional[str]
        The pretrained model.
    wandb : Optional[dict]
        The wandb configuration.
    disable_checkpoint : bool
        Disable checkpoint.
    matmul_precision : Optional[str]
        The matmul precision.
    find_unused_parameters : Optional[bool]
        Find unused parameters.
    save_top_k : Optional[int]
        Save top k checkpoints.
    validation_only : bool
        Run validation only.
    debug : bool
        Debug mode.
    strict_loading : bool
        Fail on mismatched checkpoint weights.
    load_confidence_from_trunk: Optional[bool]
        Load pre-trained confidence weights from trunk.

    """

    data: DataConfig
    model: torch.nn.Module
    output: str
    trainer: Optional[dict] = None
    resume: Optional[str] = None
    pretrained: Optional[str] = None
    wandb: Optional[dict] = None
    disable_checkpoint: bool = False
    matmul_precision: Optional[str] = None
    find_unused_parameters: Optional[bool] = False
    save_top_k: Optional[int] = 1
    validation_only: bool = False
    debug: bool = False
    strict_loading: bool = True
    load_confidence_from_trunk: Optional[bool] = False


def run_validation(
    model: torch.nn.Module,
    data_module: BoltzTrainingDataModule,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Run validation loop.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    data_module : BoltzTrainingDataModule
        The data module.
    device : torch.device
        The device to use.
    use_amp : bool
        Whether to use automatic mixed precision.
    amp_dtype : torch.dtype
        The dtype for AMP autocast.

    Returns
    -------
    dict
        Dictionary of validation metrics.

    """
    # Unwrap DDP for validation hooks
    raw_model = model.module if isinstance(model, DDP) else model

    # Apply EMA weights for validation
    if hasattr(raw_model, "prepare_eval"):
        raw_model.prepare_eval()

    model.eval()
    val_loader = data_module.val_dataloader()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = transfer_batch_to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                raw_model.validation_step(batch, batch_idx)

    # Compute validation metrics
    if hasattr(raw_model, "on_validation_epoch_end"):
        raw_model.on_validation_epoch_end()

    # Collect logged metrics
    metrics = {}
    if hasattr(raw_model, "_logged_metrics"):
        metrics = dict(raw_model._logged_metrics)
        raw_model._logged_metrics.clear()

    # Restore original weights (undo EMA)
    if hasattr(raw_model, "on_validation_end"):
        raw_model.on_validation_end()

    return metrics


def train(raw_config: str, args: list[str]) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run training.

    Parameters
    ----------
    raw_config : str
        The input yaml configuration.
    args : list[str]
        Any command line overrides.

    """
    # Load the configuration
    raw_config = omegaconf.OmegaConf.load(raw_config)

    # Apply input arguments
    args = omegaconf.OmegaConf.from_dotlist(args)
    raw_config = omegaconf.OmegaConf.merge(raw_config, args)

    # Instantiate the task
    cfg = hydra.utils.instantiate(raw_config)
    cfg = TrainConfig(**cfg)

    # Set matmul precision
    if cfg.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)

    # Create trainer dict
    trainer_cfg = cfg.trainer
    if trainer_cfg is None:
        trainer_cfg = {}

    # Flip some arguments in debug mode
    devices = trainer_cfg.get("devices", 1)

    wandb_cfg = cfg.wandb
    if cfg.debug:
        if isinstance(devices, int):
            devices = 1
        elif isinstance(devices, (list, listconfig.ListConfig)):
            devices = [devices[0]]
        trainer_cfg["devices"] = devices
        cfg.data.num_workers = 0
        if wandb_cfg:
            wandb_cfg = None

    # Extract trainer parameters
    accelerator = trainer_cfg.get("accelerator", "gpu")
    precision = trainer_cfg.get("precision", 32)
    gradient_clip_val = trainer_cfg.get("gradient_clip_val", None)
    max_epochs = trainer_cfg.get("max_epochs", -1)
    accumulate_grad_batches = trainer_cfg.get("accumulate_grad_batches", 1)

    # Determine device
    if accelerator in ("gpu", "auto", "cuda"):
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu", 0)
        else:
            device = torch.device("cpu")
    elif accelerator == "xpu":
        device = torch.device("xpu", 0)
    else:
        device = torch.device("cpu")

    # Determine AMP settings
    use_amp = False
    amp_dtype = torch.float32
    if precision == "bf16-mixed" or precision == "bf16":
        use_amp = True
        amp_dtype = torch.bfloat16
    elif precision == "16-mixed" or precision == 16:
        use_amp = True
        amp_dtype = torch.float16

    # Create objects
    data_config = DataConfig(**cfg.data)
    data_module = BoltzTrainingDataModule(data_config)
    model_module = cfg.model

    if cfg.pretrained and not cfg.resume:
        # Load the pretrained weights into the confidence module
        if cfg.load_confidence_from_trunk:
            checkpoint = torch.load(cfg.pretrained, map_location="cpu")

            # Modify parameter names in the state_dict
            new_state_dict = {}
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("structure_module") and not key.startswith(
                    "distogram_module"
                ):
                    new_key = "confidence_module." + key
                    new_state_dict[new_key] = value
            new_state_dict.update(checkpoint["state_dict"])

            # Update the checkpoint with the new state_dict
            checkpoint["state_dict"] = new_state_dict

            # Save the modified checkpoint
            random_string = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=10)
            )
            file_path = os.path.dirname(cfg.pretrained) + "/" + random_string + ".ckpt"
            print(
                f"Saving modified checkpoint to {file_path} created by broadcasting trunk of {cfg.pretrained} to confidence module."
            )
            torch.save(checkpoint, file_path)
        else:
            file_path = cfg.pretrained

        print(f"Loading model from {file_path}")
        model_module = type(model_module).load_from_checkpoint(
            file_path, map_location="cpu", strict=False, **(model_module.hparams)
        )

        if cfg.load_confidence_from_trunk:
            os.remove(file_path)

    # Create wandb logger
    wandb_logger = None
    if wandb_cfg:
        import wandb

        if is_rank_zero():
            wandb_logger = wandb.init(
                name=wandb_cfg["name"],
                group=wandb_cfg["name"],
                dir=cfg.output,
                project=wandb_cfg["project"],
                entity=wandb_cfg["entity"],
            )
            # Save the config to wandb
            config_out = Path(wandb_logger.dir) / "run.yaml"
            with Path.open(config_out, "w") as f:
                OmegaConf.save(raw_config, f)
            wandb_logger.save(str(config_out))

    # Determine number of devices for DDP
    num_devices = 1
    if isinstance(devices, int):
        num_devices = devices
    elif isinstance(devices, (list, listconfig.ListConfig)):
        num_devices = len(devices)

    # Setup DDP if needed
    use_ddp = num_devices > 1
    if use_ddp:
        # TODO: Full DDP setup with torch.distributed.launch or torchrun
        # For now, assume single process multi-GPU is handled externally
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)

    # Move model to device and setup
    model_module.to(device)
    if hasattr(model_module, "setup"):
        model_module.setup("fit")

    # Wrap with DDP if needed
    if use_ddp:
        model_module = DDP(
            model_module,
            device_ids=[device.index],
            find_unused_parameters=cfg.find_unused_parameters,
        )

    raw_model = model_module.module if isinstance(model_module, DDP) else model_module

    # Initialize EMA
    if hasattr(raw_model, "on_train_start"):
        raw_model.on_train_start()

    # Configure optimizers
    optim_result = raw_model.configure_optimizers()
    if isinstance(optim_result, tuple) and len(optim_result) == 2:
        optimizers, schedulers_cfg = optim_result
        optimizer = optimizers[0] if isinstance(optimizers, list) else optimizers
        scheduler_entry = schedulers_cfg[0] if isinstance(schedulers_cfg, list) else schedulers_cfg
        if isinstance(scheduler_entry, dict):
            scheduler = scheduler_entry["scheduler"]
            scheduler_interval = scheduler_entry.get("interval", "epoch")
        else:
            scheduler = scheduler_entry
            scheduler_interval = "epoch"
    else:
        optimizer = optim_result
        scheduler = None
        scheduler_interval = "epoch"

    # GradScaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    # Checkpoint tracking for save_top_k
    dirpath = cfg.output
    best_metrics = []  # list of (metric_value, filepath)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_val_metric = 0.0
    if cfg.resume:
        print(f"Resuming from checkpoint: {cfg.resume}")
        start_epoch, global_step, best_val_metric = load_checkpoint(
            model_module, optimizer, scheduler, cfg.resume, strict=cfg.strict_loading
        )
        print(f"Resumed at epoch {start_epoch}, global step {global_step}")

    # ---- Validation only mode ----
    if cfg.validation_only:
        print("Running validation only...")
        val_metrics = run_validation(
            model_module, data_module, device, use_amp, amp_dtype
        )
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")
        return

    # ---- Training loop ----
    epoch = start_epoch
    max_epoch_count = max_epochs if max_epochs > 0 else 10**9  # effectively infinite

    while epoch < max_epoch_count:
        # Rebuild dataloaders every epoch (as per reload_dataloaders_every_n_epochs=1)
        if hasattr(data_module, "setup"):
            data_module.setup("fit")
        train_loader = data_module.train_dataloader()

        # Restore EMA original weights for training
        if hasattr(raw_model, "on_train_epoch_start"):
            raw_model.on_train_epoch_start()

        model_module.train()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch = transfer_batch_to_device(batch, device)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                loss = raw_model.training_step(batch, batch_idx)

            if loss is None:
                # training_step returned None (skipped batch)
                continue

            # Scale loss for gradient accumulation
            scaled_loss = loss / accumulate_grad_batches
            scaler.scale(scaled_loss).backward()

            # Step optimizer every accumulate_grad_batches steps
            if (batch_idx + 1) % accumulate_grad_batches == 0:
                # Gradient clipping
                if gradient_clip_val is not None and gradient_clip_val > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        raw_model.parameters(), gradient_clip_val
                    )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Step scheduler if interval is "step"
                if scheduler is not None and scheduler_interval == "step":
                    scheduler.step()

                # Update EMA
                if hasattr(raw_model, "on_train_batch_end"):
                    raw_model.on_train_batch_end(None, batch, batch_idx)

                global_step += 1

                # Log training metrics
                if wandb_logger and is_rank_zero():
                    log_dict = {"train/loss": loss.item(), "global_step": global_step}
                    lr = optimizer.param_groups[0]["lr"]
                    log_dict["lr"] = lr

                    # Collect any metrics logged by the model
                    if hasattr(raw_model, "_logged_metrics"):
                        for k, v in raw_model._logged_metrics.items():
                            if k.startswith("train/"):
                                log_dict[k] = v if not torch.is_tensor(v) else v.item()
                        raw_model._logged_metrics.clear()

                    wandb_logger.log(log_dict, step=global_step)

        # Step scheduler if interval is "epoch"
        if scheduler is not None and scheduler_interval == "epoch":
            scheduler.step()

        # ---- Validation ----
        val_metrics = run_validation(
            model_module, data_module, device, use_amp, amp_dtype
        )

        # Get the monitored metric (val/lddt)
        val_lddt = val_metrics.get("val/lddt", 0.0)

        if is_rank_zero():
            print(f"Epoch {epoch} | val/lddt: {val_lddt}")
            for k, v in val_metrics.items():
                print(f"  {k}: {v}")

            # Log validation metrics to wandb
            if wandb_logger:
                wandb_logger.log(
                    {**val_metrics, "epoch": epoch},
                    step=global_step,
                )

            # ---- Checkpointing ----
            if not cfg.disable_checkpoint:
                checkpoint_dir = Path(dirpath) / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save last checkpoint
                save_checkpoint(
                    model_module, optimizer, scheduler,
                    epoch, global_step, val_lddt,
                    str(checkpoint_dir), "last.ckpt",
                )

                # Save top-k checkpoints based on val/lddt (mode=max)
                save_top_k = cfg.save_top_k
                if save_top_k != 0:  # 0 means don't save any best
                    ckpt_name = f"epoch={epoch}-val_lddt={val_lddt:.4f}.ckpt"
                    ckpt_path = str(checkpoint_dir / ckpt_name)

                    if save_top_k == -1:
                        # Save all checkpoints
                        save_checkpoint(
                            model_module, optimizer, scheduler,
                            epoch, global_step, val_lddt,
                            str(checkpoint_dir), ckpt_name,
                        )
                    else:
                        # Save only top k
                        best_metrics.append((val_lddt, ckpt_path))
                        best_metrics.sort(key=lambda x: x[0], reverse=True)

                        # Save current checkpoint
                        save_checkpoint(
                            model_module, optimizer, scheduler,
                            epoch, global_step, val_lddt,
                            str(checkpoint_dir), ckpt_name,
                        )

                        # Remove excess checkpoints
                        while len(best_metrics) > save_top_k:
                            _, path_to_remove = best_metrics.pop()
                            if Path(path_to_remove).exists():
                                os.remove(path_to_remove)

        epoch += 1

    # Cleanup
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()
    if wandb_logger and is_rank_zero():
        wandb_logger.finish()


if __name__ == "__main__":
    arg1 = sys.argv[1]
    arg2 = sys.argv[2:]
    train(arg1, arg2)
