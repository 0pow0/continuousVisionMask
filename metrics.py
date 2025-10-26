import torch
import sys 
import os
import math
from typing import Any, Callable, Optional, Sequence
from torch.distributions import Distribution
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path

class Insertion():
    def __init__(self, output_transform: Optional[Callable[[Any], torch.Tensor]] = None) -> None:
        self.output_transform = output_transform
        self._records: dict[int, dict[str, Any]] = {}
        self._plot_queue: list[tuple[int, torch.Tensor, float]] = []

    def __call__(
        self,
        model,
        x,
        attr,
        fraction,
        sample_ids: Optional[Sequence[int] | torch.Tensor] = None,
    ):
        actions, auc_values = self._run_curve(model, x, attr, fraction)
        ids = self._prepare_ids(sample_ids, actions.size(0))
        self._accumulate_results(actions, auc_values, ids)
        self._queue_plots(actions, auc_values, ids)
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

        # print(f"{attr.shape} {x.shape=}")
        # print(f"{x[0]=}")
        # print(f"{attr[0]=}")
        sorted_indices = torch.argsort(flat_attr, descending=True, dim=1)
        # print(f"{sorted_indices[0]=}")
        # print(f"{attr[0][sorted_indices[0]]=}")
        # print(f"{x[0][sorted_indices[0]]=}")

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

    def flush(self, output_dir: str, prefix: str):
        if not output_dir:
            raise ValueError("output_dir must be provided when flushing metrics.")
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        results_path = output_root / f"{prefix}_results.pt"
        torch.save(self._records, results_path)

        for sample_id, actions, auc in self._plot_queue:
            plot_path = output_root / f"{prefix}_{sample_id}.png"
            self._plot(actions, str(plot_path), auc)

        self._records.clear()
        self._plot_queue.clear()

    def _prepare_ids(
        self,
        sample_ids: Optional[Sequence[int] | torch.Tensor],
        batch_size: int,
    ) -> list[int] | None:
        if sample_ids is None:
            return None
        if torch.is_tensor(sample_ids):
            ids_list = sample_ids.detach().cpu().view(-1).tolist()
        else:
            ids_list = [int(s) for s in sample_ids]
        if len(ids_list) != batch_size:
            raise ValueError(
                f"sample_ids length ({len(ids_list)}) must match batch size ({batch_size})."
            )
        return ids_list

    def _accumulate_results(
        self,
        actions: torch.Tensor,
        auc_values: list[float],
        sample_ids: Optional[list[int]],
    ) -> None:
        if sample_ids is None:
            return
        record = self._records
        actions_cpu = actions.detach().cpu()
        for idx, sample_id in enumerate(sample_ids):
            record[int(sample_id)] = {
                'actions': actions_cpu[idx],
                'auc': float(auc_values[idx]),
            }

    def _queue_plots(
        self,
        actions: torch.Tensor,
        auc_values: list[float],
        sample_ids: Optional[list[int]],
    ):
        if sample_ids is None:
            return
        for idx, sample_id in enumerate(sample_ids):
            self._plot_queue.append(
                (int(sample_id), actions[idx].detach().cpu(), float(auc_values[idx]))
            )

    def _plot(self, actions, file_name, auc):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        x = [ (k / len(actions)) for k in range(len(actions)) ]
        fig, ax = plt.subplots()
        ax.plot(x, actions, color='#376795')
        ax.fill_between(x, actions, color='skyblue', alpha=0.3)
        ax.set_xlabel('Fractions of Pixels')
        ax.set_ylabel(r'$P(a)$')
        ax.text(0.85, 0.9, 'AUC={:.4f}'.format(auc), fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
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
