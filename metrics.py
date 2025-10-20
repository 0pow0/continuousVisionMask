import torch
import sys 
import os
import math
from typing import Any, Callable, Optional
from torch.distributions import Distribution
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path

class Insertion():
    def __init__(self, output_transform: Optional[Callable[[Any], torch.Tensor]] = None) -> None:
        self.output_transform = output_transform

    def __call__(self, model, x, attr, file_name, fraction):
        actions, auc_values = self._run_curve(model, x, attr, fraction)
        self._persist_results(file_name, actions, auc_values)
        return auc_values

    def _run_curve(self, model, x, attr, fraction):
        if fraction < 0 or fraction > 1:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")

        if attr.shape != x.shape:
            raise ValueError(
                f"Attribution tensor shape {attr.shape} must match input shape {x.shape}"
            )

        B = attr.shape[0]
        mask = self._initialize_mask(x)

        flat_attr = attr.reshape(B, -1)
        flat_mask = mask.reshape(B, -1)
        total_features = flat_attr.size(1)
        N = max(0, math.floor(total_features * fraction))

        actions = x.new_zeros((B, N + 1))

        sorted_indices = torch.argsort(flat_attr, descending=True, dim=1)

        masked_x = mask * x
        out = model(masked_x)
        actions[:, 0] = self._process_output(out, B, actions.dtype, actions.device)

        batch_idx = torch.arange(B, device=attr.device)
        for step in range(N):
            idx = sorted_indices[:, step]
            self._apply_step(flat_mask, batch_idx, idx)
            masked_x = mask * x
            out = model(masked_x)
            actions[batch_idx, step + 1] = self._process_output(
                out, B, actions.dtype, actions.device
            )

        auc_values = self._compute_auc(actions)
        return actions.detach(), auc_values

    def _initialize_mask(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def _apply_step(
        self, flat_mask: torch.Tensor, batch_idx: torch.Tensor, indices: torch.Tensor
    ) -> None:
        flat_mask[batch_idx, indices] = 1.0

    def _compute_auc(self, actions: torch.Tensor) -> list[float]:
        steps = actions.size(1)
        if steps <= 1:
            return [0.0] * actions.size(0)

        trapezoid = (actions[:, 1:] + actions[:, :-1]) * 0.5
        auc = trapezoid.sum(dim=1) / (steps - 1)
        return auc.detach().cpu().tolist()

    def _persist_results(
        self, file_name: Optional[str], actions: torch.Tensor, auc_values: list[float]
    ) -> None:
        if not file_name:
            return

        path = Path(file_name)
        if path.suffix == "":
            path = path.with_suffix('.pt')
        os.makedirs(path.parent, exist_ok=True)
        torch.save(
            {
                'actions': actions.cpu(),
                'auc': torch.tensor(auc_values, dtype=torch.float32),
            },
            path,
        )

    def plot(self, actions, file_name, auc):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        x = [ (k / len(actions)) for k in range(len(actions)) ]
        fig, ax = plt.subplots()
        ax.plot(x, actions, color='#376795')
        ax.fill_between(x, actions, color='skyblue', alpha=0.3)
        ax.set_xlabel('Fractions of Pixels')
        ax.set_ylabel(r'$P(a)$')
        ax.text(0.85, 0.9, 'AUC={:.4f}'.format(auc.item()), fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
        plt.savefig(file_name, bbox_inches='tight')
        plt.close(fig)

    def _process_output(self, output: Any, batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.output_transform is not None:
            value = self.output_transform(output)
        else:
            value = self._default_output_transform(output)

        if not torch.is_tensor(value):
            value = torch.as_tensor(value)

        if value.ndim == 0:
            value = value.unsqueeze(0)

        if value.shape[0] != batch:
            raise ValueError(f"Expected output batch size {batch}, got {value.shape[0]}")

        if value.ndim > 1:
            value = value.reshape(batch, -1).mean(dim=1)

        return value.detach().to(device=device, dtype=dtype)

    def _default_output_transform(self, output: Any) -> torch.Tensor:
        if isinstance(output, (list, tuple)):
            output = output[0]
        elif isinstance(output, dict):
            try:
                output = next(iter(output.values()))
            except StopIteration as exc:
                raise ValueError('Model output dictionary is empty') from exc

        if isinstance(output, Distribution):
            output = output.mean

        if torch.is_tensor(output):
            return output

        raise TypeError(
            "Unable to convert model output to tensor. Provide an explicit 'output_transform'."
        )


class Deletion(Insertion):
    def _initialize_mask(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    def _apply_step(
        self, flat_mask: torch.Tensor, batch_idx: torch.Tensor, indices: torch.Tensor
    ) -> None:
        flat_mask[batch_idx, indices] = 0.0
