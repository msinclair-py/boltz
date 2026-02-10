# --------------------------------------------------------------------------------------
# Modified from Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# Modified from : https://github.com/BioinfoMachineLearning/bio-diffusion/blob/main/src/utils/__init__.py
# --------------------------------------------------------------------------------------

from typing import Any, Optional

import torch
from torch import nn


class EMA:
    """Implements Exponential Moving Averaging (EMA).

    When training a model, this class maintains moving averages
    of the trained parameters. When evaluating, we use the moving
    averages copy of the trained parameters. When saving, we save
    an additional set of parameters with the prefix `ema`.

    Adapted from:
    https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/ema.py
    https://github.com/BioinfoMachineLearning/bio-diffusion/blob/main/src/utils/__init__.py

    """

    def __init__(
        self,
        decay: float = 0.999,
        apply_ema_every_n_steps: int = 1,
        start_step: int = 0,
        eval_with_ema: bool = True,
        warm_start: bool = True,
    ) -> None:
        """Initialize EMA.

        Parameters
        ----------
        decay: float
            The exponential decay, has to be between 0-1.
        apply_ema_every_n_steps: int, optional (default=1)
            Apply EMA every n global steps.
        start_step: int, optional (default=0)
            Start applying EMA from ``start_step`` global step onwards.
        eval_with_ema: bool, optional (default=True)
            Validate the EMA weights instead of the original weights.
            Note this means that when saving the model, the
            validation metrics are calculated with the EMA weights.
        warm_start: bool, optional (default=True)
            Use warm start for EMA decay.

        """
        if not (0 <= decay <= 1):
            msg = "EMA decay value must be between 0 and 1"
            raise ValueError(msg)

        self._ema_weights: Optional[dict[str, torch.Tensor]] = None
        self._cur_step: Optional[int] = None
        self._weights_buffer: Optional[dict[str, torch.Tensor]] = None
        self.apply_ema_every_n_steps = apply_ema_every_n_steps
        self.start_step = start_step
        self.eval_with_ema = eval_with_ema
        self.decay = decay
        self.warm_start = warm_start

    @property
    def ema_initialized(self) -> bool:
        """Check if EMA weights have been initialized.

        Returns
        -------
        bool
            Whether the EMA weights have been initialized.

        """
        return self._ema_weights is not None

    def state_dict(self) -> dict[str, Any]:
        """Return the current state of the EMA.

        Returns
        -------
        dict[str, Any]
            The current state of the EMA.

        """
        return {
            "cur_step": self._cur_step,
            "ema_weights": self._ema_weights,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state of the EMA.

        Parameters
        ----------
        state_dict: dict[str, Any]
            The state of the EMA to load.

        """
        self._cur_step = state_dict["cur_step"]
        self._ema_weights = state_dict["ema_weights"]

    def should_apply_ema(self, step: int) -> bool:
        """Check if EMA should be applied at the current step.

        Parameters
        ----------
        step: int
            The current global step.

        Returns
        -------
        bool
            True if EMA should be applied, False otherwise.

        """
        return (
            step != self._cur_step
            and step >= self.start_step
            and step % self.apply_ema_every_n_steps == 0
        )

    def _apply_ema(self, model: nn.Module) -> None:
        """Apply EMA update to the model weights.

        Parameters
        ----------
        model: nn.Module
            The model instance.

        """
        decay = self.decay
        if self.warm_start:
            decay = min(decay, (1 + self._cur_step) / (10 + self._cur_step))

        for k, orig_weight in model.state_dict().items():
            ema_weight = self._ema_weights[k]
            if (
                ema_weight.data.dtype != torch.long  # noqa: PLR1714
                and orig_weight.data.dtype != torch.long  # skip non-trainable weights
            ):
                diff = ema_weight.data - orig_weight.data
                diff.mul_(1.0 - decay)
                ema_weight.sub_(diff)

    def update(self, model: nn.Module, step: int) -> None:
        """Update EMA weights. Call after each training step.

        Parameters
        ----------
        model: nn.Module
            The model instance.
        step: int
            The current global step.

        """
        if not self.ema_initialized:
            self._ema_weights = {
                k: p.detach().clone() for k, p in model.state_dict().items()
            }
            device = next(model.parameters()).device
            self._ema_weights = {
                k: p.to(device) for k, p in self._ema_weights.items()
            }

        if self.should_apply_ema(step):
            self._cur_step = step
            self._apply_ema(model)

    def apply(self, model: nn.Module) -> None:
        """Swap in EMA weights for evaluation. Call before validation/prediction.

        Parameters
        ----------
        model: nn.Module
            The model instance.

        """
        if self.ema_initialized and self.eval_with_ema:
            self._weights_buffer = {
                k: p.detach().clone().to("cpu")
                for k, p in model.state_dict().items()
            }
            model.load_state_dict(self._ema_weights, strict=False)

    def restore(self, model: nn.Module) -> None:
        """Restore original weights after evaluation. Call after validation/prediction.

        Parameters
        ----------
        model: nn.Module
            The model instance.

        """
        if self.ema_initialized and self.eval_with_ema:
            model.load_state_dict(self._weights_buffer, strict=False)
            del self._weights_buffer

    def on_train_start(self, model: nn.Module) -> None:
        """Initialize EMA weights. Call at training start.

        Parameters
        ----------
        model: nn.Module
            The model instance.

        """
        if not self.ema_initialized:
            self._ema_weights = {
                k: p.detach().clone() for k, p in model.state_dict().items()
            }

        device = next(model.parameters()).device
        self._ema_weights = {
            k: p.to(device) for k, p in self._ema_weights.items()
        }
