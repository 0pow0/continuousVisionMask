import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from torch.distributions import Distribution

try:
    from quantus.metrics.complexity.sparseness import (
        Sparseness as QuantusSparseness,
    )
    from quantus.metrics.robustness.local_lipschitz_estimate import (
        LocalLipschitzEstimate as QuantusLocalLipschitzEstimate,
    )
except ImportError:  # pragma: no cover - optional dependency.
    QuantusSparseness = None
    QuantusLocalLipschitzEstimate = None

class Insertion():
    def __init__(
        self,
        output_transform: Optional[Callable[[Any], torch.Tensor]] = None,
        enable_plots: bool = True,
    ) -> None:
        self.output_transform = output_transform
        self.enable_plots = enable_plots
        self._records: dict[int, dict[str, Any]] = {}
        self._plot_order: list[int] = []

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
        self._accumulate_results(actions, attr, ids)
        self._queue_plots(ids)
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

    def flush(self, output_dir: str, prefix: str) -> dict[int, dict[str, Any]]:
        if not output_dir:
            raise ValueError("output_dir must be provided when flushing metrics.")
        if not self._records:
            return {}
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        normalized_records = self._normalize_records()
        results_path = output_root / f"{prefix}_results.pt"
        torch.save(normalized_records, results_path)

        if self.enable_plots:
            for sample_id in self._plot_order:
                record = normalized_records.get(sample_id)
                if not record:
                    continue
                plot_path = output_root / f"{prefix}_{sample_id}.png"
                self._plot(record["actions"], str(plot_path), record["auc"])

            self._plot_order.clear()
        self._records.clear()
        return normalized_records

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
        attr: torch.Tensor,
        sample_ids: Optional[list[int]],
    ) -> None:
        if sample_ids is None:
            return
        actions_cpu = actions.detach().cpu()
        attr_cpu = attr.detach().cpu()
        for idx, sample_id in enumerate(sample_ids):
            self._records[int(sample_id)] = {
                "actions": actions_cpu[idx],
                "attr": attr_cpu[idx],
            }

    def _queue_plots(
        self,
        sample_ids: Optional[list[int]],
    ):
        if sample_ids is None or not self.enable_plots:
            return
        for sample_id in sample_ids:
            self._plot_order.append(int(sample_id))

    def _normalize_records(self) -> dict[int, dict[str, Any]]:
        if not self._records:
            return {}
        ordered_ids = sorted(self._records.keys())
        stacked = torch.stack([self._records[_id]["actions"] for _id in ordered_ids])
        min_vals = stacked.min(dim=1, keepdim=True).values
        max_vals = stacked.max(dim=1, keepdim=True).values
        denom = (max_vals - min_vals).clamp_min(1e-8)
        normalized = (stacked - min_vals) / denom
        auc_values = self._compute_auc(normalized)
        normalized_records: dict[int, dict[str, Any]] = {}
        for idx, sample_id in enumerate(ordered_ids):
            normalized_records[sample_id] = {
                "actions": normalized[idx].detach().cpu(),
                "auc": float(auc_values[idx]),
                "attr": self._records[sample_id]["attr"],
            }
        return normalized_records

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


class QuantusAttributionMetrics:
    """
    Wrapper around Quantus' Sparseness and Local Lipschitz Estimate metrics.

    This helper expects pre-computed attributions and takes care of converting tensors
    to numpy arrays before delegating the actual scoring to Quantus. Local Lipschitz
    requires an `explain_func` compatible with Quantus' API (signature like
    `func(model, inputs, targets, **kwargs)`), which will be used to regenerate
    attributions for perturbed inputs.
    """

    def __init__(
        self,
        *,
        sparseness_kwargs: Optional[dict[str, Any]] = None,
        lipschitz_kwargs: Optional[dict[str, Any]] = None,
        device: Optional[str] = None,
        channel_first: Optional[bool] = None,
    ) -> None:
        if QuantusSparseness is None or QuantusLocalLipschitzEstimate is None:
            raise ImportError(
                "Quantus must be installed to use QuantusAttributionMetrics. "
                "Install it with `pip install quantus`."
            )
        self.sparseness_metric = QuantusSparseness(**(sparseness_kwargs or {}))
        self.lipschitz_metric = QuantusLocalLipschitzEstimate(
            **(lipschitz_kwargs or {})
        )
        self.device = device
        self.channel_first = channel_first
        self._records: dict[int, dict[str, float]] = {}

    def __call__(
        self,
        model: Any,
        x: torch.Tensor | np.ndarray,
        attr: torch.Tensor | np.ndarray,
        *,
        targets: Optional[torch.Tensor | np.ndarray] = None,
        sample_ids: Optional[Sequence[int] | torch.Tensor] = None,
        explain_func: Optional[Callable[..., np.ndarray]] = None,
        explain_func_kwargs: Optional[dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        channel_first: Optional[bool] = None,
        model_predict_kwargs: Optional[dict[str, Any]] = None,
    ) -> dict[str, list[float]]:
        if explain_func is None:
            raise ValueError(
                "explain_func must be provided so LocalLipschitzEstimate can "
                "re-compute attributions for perturbed inputs."
            )

        x_np = self._to_numpy(x)
        attr_np = self._to_numpy(attr)
        if x_np.shape != attr_np.shape:
            raise ValueError(
                f"Input and attribution tensors must have identical shapes, got "
                f"{x_np.shape} and {attr_np.shape}."
            )

        n_samples = x_np.shape[0]
        metric_batch_size = batch_size or n_samples
        y_np = self._prepare_targets(targets, n_samples)
        ids = self._prepare_ids(sample_ids, n_samples)
        resolved_channel_first = (
            channel_first if channel_first is not None else self.channel_first
        )
        resolved_device = self._resolve_device(x)
        explain_kwargs = dict(explain_func_kwargs or {})

        sparseness_scores = self.sparseness_metric(
            model=None,
            x_batch=x_np,
            y_batch=y_np,
            a_batch=attr_np,
            s_batch=None,
            channel_first=resolved_channel_first,
            explain_func=None,
            explain_func_kwargs=None,
            model_predict_kwargs=None,
            softmax=None,
            device=resolved_device,
            batch_size=metric_batch_size,
        )

        lipschitz_scores = self.lipschitz_metric(
            model=model,
            x_batch=x_np,
            y_batch=y_np,
            a_batch=attr_np,
            s_batch=None,
            channel_first=resolved_channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            softmax=None,
            device=resolved_device,
            batch_size=metric_batch_size,
        )

        sparseness_list = self._to_float_list(sparseness_scores)
        lipschitz_list = self._to_float_list(lipschitz_scores)
        self._accumulate(ids, sparseness_list, lipschitz_list)

        return {
            "sparseness": sparseness_list,
            "local_lipschitz": lipschitz_list,
        }

    def summary(self) -> dict[str, float]:
        """
        Compute simple aggregates (nan-aware) over accumulated metric values.
        """
        if not self._records:
            return {}
        sparseness_vals = np.array(
            [item["sparseness"] for item in self._records.values()], dtype=float
        )
        lipschitz_vals = np.array(
            [item["local_lipschitz"] for item in self._records.values()], dtype=float
        )
        return {
            "sparseness_mean": float(np.nanmean(sparseness_vals)),
            "sparseness_std": float(np.nanstd(sparseness_vals)),
            "local_lipschitz_mean": float(np.nanmean(lipschitz_vals)),
            "local_lipschitz_std": float(np.nanstd(lipschitz_vals)),
        }

    def flush(
        self,
        output_path: Optional[str | os.PathLike[Any]] = None,
    ) -> dict[int, dict[str, float]]:
        """
        Persist accumulated per-sample records to disk (optional) and reset state.
        """
        if not self._records:
            return {}

        ordered = {
            sample_id: {
                "sparseness": float(values["sparseness"]),
                "local_lipschitz": float(values["local_lipschitz"]),
            }
            for sample_id, values in sorted(self._records.items())
        }
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ordered, output_path)
        self._records.clear()
        return ordered

    def _to_numpy(self, value: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _prepare_targets(
        self,
        targets: Optional[torch.Tensor | np.ndarray],
        expected_size: int,
    ) -> np.ndarray:
        if targets is None:
            return np.zeros(expected_size, dtype=np.int64)

        targets_np = self._to_numpy(targets)
        if targets_np.shape[0] != expected_size:
            raise ValueError(
                f"Targets length ({targets_np.shape[0]}) does not match batch size "
                f"({expected_size})."
            )
        return targets_np

    def _prepare_ids(
        self,
        sample_ids: Optional[Sequence[int] | torch.Tensor],
        batch_size: int,
    ) -> Optional[list[int]]:
        if sample_ids is None:
            return None
        if torch.is_tensor(sample_ids):
            ids_list = sample_ids.detach().cpu().view(-1).tolist()
        else:
            ids_list = [int(idx) for idx in sample_ids]
        if len(ids_list) != batch_size:
            raise ValueError(
                f"sample_ids length ({len(ids_list)}) must match batch size ({batch_size})."
            )
        return ids_list

    def _accumulate(
        self,
        sample_ids: Optional[list[int]],
        sparseness: list[float],
        lipschitz: list[float],
    ) -> None:
        if sample_ids is None:
            return
        for idx, sample_id in enumerate(sample_ids):
            self._records[int(sample_id)] = {
                "sparseness": float(sparseness[idx]),
                "local_lipschitz": float(lipschitz[idx]),
            }

    def _to_float_list(self, values: Any) -> list[float]:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr.tolist()

    def _resolve_device(self, x: torch.Tensor | np.ndarray) -> Optional[str]:
        if self.device is not None:
            return self.device
        if isinstance(x, torch.Tensor):
            return "gpu" if x.is_cuda else "cpu"
        return None
