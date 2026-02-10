# Boltz Refactoring Plan: XPU Support & Lightning Removal

## Overview

Two major architectural changes to the Boltz protein structure prediction codebase:

1. **Remove all hardcoded CUDA calls** — make device selection runtime-dynamic (CUDA, XPU, CPU)
2. **Remove PyTorch Lightning dependency** — convert to vanilla PyTorch training/inference loops

---

## Current State Summary

| Metric | Count |
|--------|-------|
| Hardcoded `"cuda"` strings (autocast, device, etc.) | ~37 in src/, ~14 in tests/ |
| Files importing `pytorch_lightning` | 12 |
| `LightningModule` subclasses | 2 (`Boltz1`, `Boltz2`) |
| `LightningDataModule` subclasses | 4 (training v1/v2, inference v1/v2) |
| Lightning Callbacks | 2 (`EMA`, `BoltzWriter`/`BoltzAffinityWriter`) |
| `self.log()` / `self.log_dict()` calls | 58 |
| Lightning Trainer instantiations | 2 (`main.py`, `train.py`) |

---

## Phase 1: Remove Hardcoded CUDA Calls (Device-Agnostic)

### 1.1 — Replace `torch.autocast("cuda", ...)` calls

**Files affected (17 files, ~30 call sites):**

| File | Lines | Current | Target |
|------|-------|---------|--------|
| `model/modules/diffusion.py` | 690, 816 | `torch.autocast("cuda", enabled=False)` | Derive device from input tensor |
| `model/modules/diffusionv2.py` | 513, 603 | Same | Same |
| `model/modules/encodersv2.py` | 312, 481, 544 | Same | Same |
| `model/modules/trunkv2.py` | 311, 462 | `torch.autocast(device_type="cuda", ...)` | Same |
| `model/layers/attention.py` | 119 | Same | Same |
| `model/layers/attentionv2.py` | 99 | Same | Same |
| `model/layers/pairformer.py` | 105 | Same | Same |
| `model/layers/triangular_attention/primitives.py` | 106, 119, 139, 167 | Same | Same |
| `model/layers/confidence_utils.py` | 26 | `torch.amp.autocast("cuda", ...)` | Same |
| `model/loss/bfactor.py` | 24 | Same | Same |
| `model/loss/distogramv2.py` | 27 | Same | Same |
| `model/loss/confidencev2.py` | 98, 149, 362, 523 | Mixed variants | Same |

**Strategy:** Each of these functions receives tensors as input. We'll extract the device type from the first relevant tensor parameter using a helper:

```python
# New utility function in model/modules/utils.py (or similar)
def get_autocast_device_type(tensor_or_device):
    """Get the device type string for torch.autocast from a tensor or device."""
    if isinstance(tensor_or_device, torch.Tensor):
        return tensor_or_device.device.type
    if isinstance(tensor_or_device, torch.device):
        return tensor_or_device.type
    return str(tensor_or_device)
```

Then replace:
```python
# Before
with torch.autocast("cuda", enabled=False):
# After
with torch.autocast(tensor.device.type, enabled=False):
```

### 1.2 — Replace `torch.cuda.empty_cache()` calls

**Files affected:**
- `model/models/boltz2.py` — lines 1025, 1045, 1126
- `model/models/boltz1.py` — lines 633, 687, 1201

**Strategy:** Create a device-aware cache clearing utility:

```python
def empty_device_cache(device_type: str) -> None:
    """Clear device memory cache for the given device type."""
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "xpu":
        torch.xpu.empty_cache()
    # CPU has no cache to clear
```

### 1.3 — Replace hardcoded `device="cuda"` tensor creation

**Files affected:**
- `model/models/boltz2.py` — lines 988, 997 (`torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")`)

**Strategy:** Use `self.device` (which will be passed through after Lightning removal) or derive from model parameters:

```python
# Before
torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")
# After
torch.tensor(0.0, device=self.device)
```

### 1.4 — Replace CUDA capability checks for kernel selection

**Files affected:**
- `model/models/boltz2.py` — lines 363-365
- `model/models/boltz1.py` — lines 267-268

**Strategy:** Generalize the kernel availability check:

```python
# Before
if stage == "predict" and not (
    torch.cuda.is_available()
    and torch.cuda.get_device_properties(torch.device("cuda")).major >= 8.0
):
    self.use_kernels = False

# After
def _check_kernel_support(self, device) -> bool:
    """Check if the device supports custom CUDA kernels."""
    if device.type != "cuda":
        return False
    return torch.cuda.get_device_properties(device).major >= 8
```

### 1.5 — Update CLI accelerator options

**File:** `src/boltz/main.py`

Add `"xpu"` to the CLI accelerator choices (if not already present), and ensure the device type flows through to all components.

### 1.6 — Update test files

**Files:**
- `tests/test_regression.py` — line 27: generalize device selection
- `tests/test_kernels.py` — CUDA-specific tests, add guards/skips for non-CUDA
- `tests/profiling.py` — CUDA profiling calls, add device awareness

---

## Phase 2: Remove PyTorch Lightning Dependency

This is the larger change. Lightning provides: model lifecycle management, training/validation/prediction loops, distributed strategy, logging, checkpointing, callbacks, and data module abstractions.

### 2.1 — Convert `LightningModule` → `nn.Module`

**Files:** `model/models/boltz1.py`, `model/models/boltz2.py`

**Changes required per model class:**

| Lightning Feature | Current Usage | Vanilla PyTorch Replacement |
|---|---|---|
| `LightningModule` base class | `class Boltz2(LightningModule)` | `class Boltz2(nn.Module)` |
| `self.save_hyperparameters()` | Stores all `__init__` args | Store manually in `self.hparams = dict(...)` or use a dataclass |
| `self.log()` / `self.log_dict()` | 58 calls for metrics | Create a `MetricsLogger` utility class that wraps wandb/stdout |
| `self.device` property | Auto-provided by Lightning | Track explicitly via `self._device` property or derive from parameters |
| `self.trainer.global_step` | Used in EMA callback | Pass step count explicitly |
| `training_step()` | Called by Trainer | Called by our custom training loop |
| `validation_step()` | Called by Trainer | Called by our custom validation loop |
| `predict_step()` | Called by Trainer | Called by our custom inference loop |
| `on_validation_epoch_end()` | Called by Trainer | Called by our custom validation loop |
| `configure_optimizers()` | Returns optimizer + scheduler | Extract to standalone function |
| `setup(stage)` | Called by Trainer | Call manually before train/predict |
| `load_from_checkpoint()` | Lightning checkpoint loading | `torch.load()` + `model.load_state_dict()` with compatibility wrapper |

**Key design decisions:**

- `training_step`, `validation_step`, `predict_step` become regular methods (keep the same names for minimal diff)
- `configure_optimizers()` becomes a standalone function or method that returns `(optimizer, scheduler)`
- A `device` property will be added that introspects model parameters
- `self.log()` calls will be replaced with a simple metrics accumulator that can be flushed to wandb/stdout

### 2.2 — Convert `LightningDataModule` → Plain DataLoader factories

**Files:**
- `data/module/training.py` (`BoltzTrainingDataModule`)
- `data/module/trainingv2.py` (`BoltzTrainingDataModule`)
- `data/module/inference.py` (`BoltzInferenceDataModule`)
- `data/module/inferencev2.py` (`Boltz2InferenceDataModule`)

**Changes:**
- Remove `pl.LightningDataModule` base class
- Keep `train_dataloader()`, `val_dataloader()`, `predict_dataloader()` methods as-is (they return `DataLoader` objects)
- Move `transfer_batch_to_device()` logic into the training/inference loop (it handles nested dict/list tensor transfers)
- `setup()` becomes a regular initialization method

### 2.3 — Convert EMA Callback → Standalone EMA class

**File:** `model/optim/ema.py`

Currently `EMA(Callback)` hooks into Lightning's callback system. Convert to a standalone class:

```python
class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model, decay=0.999, ...):
        ...

    def update(self, model, step):
        """Call after each training step."""
        ...

    def apply(self, model):
        """Swap in EMA weights for evaluation."""
        ...

    def restore(self, model):
        """Restore original weights after evaluation."""
        ...

    def state_dict(self) / load_state_dict(self):
        """For checkpointing."""
        ...
```

### 2.4 — Convert Prediction Writers → Standalone writers

**File:** `data/write/writer.py`

`BoltzWriter(BasePredictionWriter)` and `BoltzAffinityWriter(BasePredictionWriter)` — remove Lightning base class, make them plain classes with a `write_prediction(prediction, batch)` method. The `write_interval="batch"` behavior gets handled by calling the writer directly in the inference loop.

### 2.5 — Create Vanilla Training Loop

**New file:** `src/boltz/training/trainer.py` (or modify `scripts/train/train.py`)

Replace `pl.Trainer.fit()` with a custom training loop that handles:

```python
class BoltzTrainer:
    """Vanilla PyTorch training loop for Boltz models."""

    def __init__(self, model, optimizer, scheduler, ema=None,
                 device='cuda', distributed=False, precision='fp32',
                 checkpoint_dir=None, logger=None):
        ...

    def fit(self, train_dataloader, val_dataloader=None,
            max_epochs=-1, resume_from=None):
        """Main training loop."""
        # - Epoch iteration
        # - Batch iteration with gradient accumulation
        # - Mixed precision via torch.amp.GradScaler + autocast
        # - Distributed training via torch.nn.parallel.DistributedDataParallel
        # - Periodic validation
        # - Checkpointing (model, optimizer, scheduler, EMA, step)
        # - Logging to wandb
        # - Gradient clipping
        ...

    def validate(self, val_dataloader):
        """Validation loop."""
        ...

    def predict(self, predict_dataloader, writer=None):
        """Inference loop."""
        ...
```

**Critical features to preserve:**
- Mixed-precision training (bf16-mixed for Boltz2, fp32 for Boltz1)
- DDP multi-GPU support via `torch.nn.parallel.DistributedDataParallel`
- Model checkpointing with "save top-k by val/lddt" logic
- EMA weight swapping during validation
- Reload dataloaders every N epochs
- Gradient norm logging
- OOM recovery with cache clearing

### 2.6 — Create Vanilla Inference Loop

**Modify:** `src/boltz/main.py`

Replace `Trainer.predict()` with:

```python
def run_inference(model, dataloader, writer, device, precision='fp32'):
    """Run inference and write predictions."""
    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = transfer_batch_to_device(batch, device)
            with torch.autocast(device.type, dtype=precision_dtype):
                prediction = model.predict_step(batch, batch_idx)
            writer.write_prediction(prediction, batch)
```

For multi-GPU inference, use `torch.nn.parallel.DistributedDataParallel` with `torch.distributed` setup.

### 2.7 — Checkpoint Compatibility Layer

**New utility:** `src/boltz/model/checkpoint.py`

Lightning checkpoints have a specific format (`state_dict`, `hyper_parameters`, etc.). We need a loader that can read both old Lightning checkpoints and new vanilla checkpoints:

```python
def load_model_checkpoint(model_cls, checkpoint_path, device='cpu', **override_args):
    """Load a model from either a Lightning or vanilla checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Lightning format
    if 'state_dict' in ckpt and 'hyper_parameters' in ckpt:
        hparams = ckpt['hyper_parameters']
        hparams.update(override_args)
        model = model_cls(**hparams)
        model.load_state_dict(ckpt['state_dict'])
    # Vanilla format
    else:
        model = model_cls(**ckpt.get('hparams', {}))
        model.load_state_dict(ckpt['model_state_dict'])

    return model
```

### 2.8 — Update `pyproject.toml`

Remove `pytorch-lightning==2.5.0` from dependencies. May also remove `fairscale` if only used via Lightning (needs verification — it's used for `checkpoint_wrapper`).

### 2.9 — Update training configs

**Files:** `scripts/train/configs/*.yaml`

Update the trainer section to use our new trainer parameters instead of Lightning Trainer kwargs. The model instantiation via Hydra should still work since the model classes keep the same `__init__` signatures.

### 2.10 — Update test files

**Files:**
- `tests/model/layers/test_triangle_attention.py` — replace `pytorch_lightning.seed_everything` with `torch.manual_seed` + `random.seed` + `numpy.random.seed`
- `tests/model/layers/test_outer_product_mean.py` — same
- `tests/test_regression.py` — update model loading to use new checkpoint loader

---

## Phase 3: Agent Deployment Strategy

### Agent Assignments

| Agent | Responsibility | Files |
|-------|---------------|-------|
| **Python Pro #1** | Phase 1 — All CUDA → device-agnostic changes across model layers, modules, and losses | `model/modules/*.py`, `model/layers/*.py`, `model/loss/*.py` |
| **Python Pro #2** | Phase 2.1-2.3 — Convert model classes from LightningModule to nn.Module, convert EMA | `model/models/boltz1.py`, `model/models/boltz2.py`, `model/optim/ema.py` |
| **Refactoring Specialist** | Phase 2.4-2.7 — Create training/inference loops, convert writers, checkpoint compat | `main.py`, `train.py`, `data/write/writer.py`, `data/module/*.py`, new trainer |
| **Code Reviewer** | Review all changes for correctness, device edge cases, distributed training parity | All modified files |

### Execution Order

```
Phase 1 (CUDA removal) ──────────────────────────────┐
                                                       ├──→ Phase 3 (Code Review)
Phase 2 (Lightning removal) ──────────────────────────┘

Within Phase 1: All files can be done in parallel (no dependencies)

Within Phase 2:
  2.1 (Models) ──┐
  2.2 (Data)  ───┼──→ 2.5/2.6 (Training/Inference loops) ──→ 2.7-2.10 (Cleanup)
  2.3 (EMA)   ───┤
  2.4 (Writer) ──┘
```

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Lightning checkpoint incompatibility | Compatibility layer reads both formats (Phase 2.7) |
| Distributed training regression | Preserve DDP setup exactly; test with >1 GPU |
| Mixed-precision breakage | Device-aware autocast with same dtype settings |
| Metric logging gaps | MetricsLogger wrapper preserves all 58 log calls |
| EMA weight loading from old checkpoints | Checkpoint compat layer handles `ema` key |
| `save_hyperparameters` removal | Explicit hparams dict preserves checkpoint reloading |
| `transfer_batch_to_device` nested tensor handling | Port the exact logic into training/inference loops |

---

## Files Changed Summary

| Category | Files Modified | Files Created |
|----------|---------------|---------------|
| Model layers/modules/losses | 13 | 0 |
| Model classes | 2 | 0 |
| Data modules | 4 | 0 |
| Callbacks/writers | 2 | 0 |
| Entry points | 2 | 0 |
| New infrastructure | 0 | 2 (`trainer.py`, `checkpoint.py`) |
| Config | 4 | 0 |
| Tests | 4 | 0 |
| **Total** | **31** | **2** |
