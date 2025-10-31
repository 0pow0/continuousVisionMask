#!/usr/bin/env python3
"""
Heatmap plotting utilities.

This module creates composite figures that combine a main heatmap, a shared
0-1 colorbar on the right, and two small heat strips across the top with
descriptive labels. The strips give a quick glance at auxiliary signals
while keeping everything in the same color scale as the main plot.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency during static analysis.
    torch = None  # type: ignore[assignment]


FEATURE_GROUPS: tuple[tuple[str, int], ...] = (
    ("acc", 3),
    ("vel", 3),
    ("gyro", 3),
    ("mag", 3),
    ("goal", 16),
    ("hazard", 16),
    ("vase", 16),
)

BUTTON_FEATURE_GROUPS: tuple[tuple[str, int], ...] = (
    ("acc", 3),
    ("vel", 3),
    ("gyro", 3),
    ("mag", 3),
    ("button", 16),
    ("goal", 16),
    ("gremlin", 16),
    ("hazard", 16),
)

PUSH_FEATURE_GROUPS: tuple[tuple[str, int], ...] = (
    ("acc", 3),
    ("vel", 3),
    ("gyro", 3),
    ("mag", 3),
    ("goal", 16),
    ("hazard", 16),
    ("pillars", 16),
    ("push_box", 16),
)


def infer_feature_groups(*sources: Path | str) -> tuple[tuple[str, int], ...]:
    """
    Choose feature groups based on known environment hints embedded in paths/labels.

    When any source string contains ``button1`` (case-insensitive) we switch to the
    button-specific groups; otherwise the default PointGoal groups are used.
    """
    tokens = " ".join(str(src) for src in sources).lower()
    if "button1" in tokens:
        return BUTTON_FEATURE_GROUPS
    elif "push1" in tokens:
        return PUSH_FEATURE_GROUPS
    return FEATURE_GROUPS


def _groups_from_record(record: dict[str, Any] | None, base: tuple[tuple[str, int], ...]) -> tuple[tuple[str, int], ...]:
    if not record:
        return base
    for key in ("env", "env_name", "environment", "env_id", "task"):
        value = record.get(key)
        if isinstance(value, str) and "button1" in value.lower():
            return BUTTON_FEATURE_GROUPS
        elif isinstance(value, str) and "push1" in value.lower():
            return PUSH_FEATURE_GROUPS
    return base


def feature_index_to_label(index: int, groups: Sequence[tuple[str, int]] = FEATURE_GROUPS) -> str:
    """
    Map a flat feature ``index`` to a semantic label (e.g., ``goal[2]``).

    The lookup walks ``groups`` in order, subtracting each group's length until the
    corresponding segment is found. If the index exceeds all configured groups a
    fallback ``feat[<index>]`` label is returned to avoid crashing when feature
    counts drift.
    """
    if index < 0:
        raise ValueError("index must be non-negative")

    offset = 0
    for name, length in groups:
        upper = offset + length
        if index < upper:
            rel = index - offset
            return f"{name}[{rel}]"
        offset = upper
    return f"feat[{index}]"


def select_top_features(
    attributions: Sequence[float] | np.ndarray,
    *,
    top_k: int = 5,
    groups: Sequence[tuple[str, int]] = FEATURE_GROUPS,
    normalize: bool = True,
    skip_top: int = 0,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """
    Reduce a flat attribution vector to the ``top_k`` strongest absolute scores.
    Set ``skip_top`` to ignore the highest ranked features before selecting.

    Returns normalized scores (0-1), semantic labels, the original indices, and
    the raw absolute scores in that order.
    """
    attr = np.asarray(attributions, dtype=float).reshape(-1)
    if attr.size == 0:
        raise ValueError("attributions must contain at least one element.")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if skip_top < 0:
        raise ValueError("skip_top must be >= 0")
    if skip_top >= attr.size:
        raise ValueError(
            f"skip_top={skip_top} exceeds number of features ({attr.size})."
        )
    available = attr.size - skip_top
    if available < top_k:
        raise ValueError(
            f"Requested top_k={top_k} after skip_top={skip_top} leaves only {available} features."
        )

    sorted_idx = np.argsort(np.abs(attr))[::-1]
    tail_idx = sorted_idx[skip_top:]
    window = tail_idx[:top_k]
    raw_scores = np.abs(attr[window])
    if normalize:
        tail_scores = np.abs(attr[tail_idx])
        peak = tail_scores.max(initial=0.0)
        norm_scores = raw_scores / peak if peak > 0 else raw_scores.copy()
    else:
        norm_scores = raw_scores.copy()

    labels = [feature_index_to_label(int(idx), groups) for idx in window]
    return norm_scores, labels, window.astype(int), raw_scores


def _to_numpy_array(data) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    if torch is not None and torch.is_tensor(data):
        return data.detach().cpu().numpy()
    if isinstance(data, (list, tuple)):
        return np.asarray(data)
    return np.asarray(data)


def _sanitize_sample_id(sample_id) -> str:
    text = str(sample_id)
    return re.sub(r"[^0-9A-Za-z_.-]", "_", text)


def _extract_feature_vector(attr) -> np.ndarray:
    if isinstance(attr, dict):
        for key in ("vector", "features", "feature", "state", "values"):
            if key in attr:
                arr = _to_numpy_array(attr[key])
                if arr.ndim >= 1:
                    return arr.reshape(-1)
    arr = _to_numpy_array(attr)
    if arr.ndim == 0:
        raise ValueError("Attribution array must have at least one dimension.")
    return arr.reshape(-1)


def _iter_sample_ids(sample_ids) -> list[Any]:
    """
    Flatten ``sample_ids`` into a Python list of scalar identifiers.
    Handles tensors, numpy arrays, and nested sequences.
    """
    if torch is not None and torch.is_tensor(sample_ids):
        data = sample_ids.detach().cpu()
        if data.ndim == 0:
            return [data.item()]
        return data.reshape(-1).tolist()
    if isinstance(sample_ids, np.ndarray):
        if sample_ids.ndim == 0:
            return [sample_ids.item()]
        return sample_ids.reshape(-1).tolist()
    if isinstance(sample_ids, (list, tuple)):
        flattened: list[Any] = []
        for item in sample_ids:
            flattened.extend(_iter_sample_ids(item))
        return flattened
    return [sample_ids]


def _iter_observations(observations) -> list[np.ndarray]:
    """
    Flatten ``observations`` into a list of 1D numpy arrays.
    Any higher dimensional tensors/arrays are split along the first axis.
    """
    if torch is not None and torch.is_tensor(observations):
        data = observations.detach().cpu().numpy()
        if data.ndim == 0:
            raise ValueError("Observation tensor must have at least one dimension.")
        if data.ndim == 1:
            return [np.array(data, dtype=float).reshape(-1)]
        return [np.array(chunk, dtype=float).reshape(-1) for chunk in data]

    if isinstance(observations, np.ndarray):
        if observations.ndim == 0:
            raise ValueError("Observation tensor must have at least one dimension.")
        if observations.ndim == 1:
            return [np.array(observations, dtype=float).reshape(-1)]
        if observations.dtype == object:
            vectors: list[np.ndarray] = []
            for item in observations.tolist():
                vectors.extend(_iter_observations(item))
            return vectors
        return [np.array(chunk, dtype=float).reshape(-1) for chunk in observations]

    if isinstance(observations, (list, tuple)):
        vectors: list[np.ndarray] = []
        for item in observations:
            vectors.extend(_iter_observations(item))
        return vectors

    data = _to_numpy_array(observations)
    if data.ndim == 0:
        raise ValueError("Observation tensor must have at least one dimension.")
    return [data.reshape(-1)]


def _register_state_entry(mapping: dict[Any, np.ndarray], sample_id: Any, obs_vector: np.ndarray) -> None:
    """
    Store ``obs_vector`` under multiple keys derived from ``sample_id`` for robust lookup.
    """
    if torch is not None and torch.is_tensor(sample_id):
        if sample_id.ndim == 0:
            sample_id = sample_id.item()
        else:
            sample_id = sample_id.detach().cpu().tolist()
    if isinstance(sample_id, np.ndarray):
        if sample_id.ndim == 0:
            sample_id = sample_id.item()
        else:
            sample_id = sample_id.reshape(-1).tolist()
    if isinstance(sample_id, (list, tuple)):
        for entry in sample_id:
            _register_state_entry(mapping, entry, obs_vector)
        return

    keys: set[Any] = set()
    keys.add(sample_id)
    try:
        string_key = str(sample_id)
    except Exception:
        string_key = None
    else:
        keys.add(string_key)

    if isinstance(sample_id, str):
        try:
            numeric = int(sample_id)
        except ValueError:
            numeric = None
        if numeric is not None:
            keys.add(numeric)
            keys.add(str(numeric))
    elif isinstance(sample_id, (int, np.integer)):
        numeric = int(sample_id)
        keys.add(numeric)
        keys.add(str(numeric))

    for key in keys:
        mapping[key] = obs_vector


def _load_state_vectors(state_path: Path | None) -> dict[Any, np.ndarray]:
    """
    Load per-sample state vectors from ``state_path``.

    The checkpoint is expected to either be:

    * A list of dicts each containing ``sample_id`` and ``obs`` keys.
    * A dict with ``sample_id`` and ``obs`` entries (arrays/lists/tensors).
    * A dict keyed by sample id where each value contains an ``obs`` array.
    """
    if state_path is None:
        return {}
    if torch is None:
        raise ImportError("torch is required to load state files saved with torch.save().")

    payload = torch.load(state_path, map_location="cpu")
    state_map: dict[Any, np.ndarray] = {}

    def add_pair(sample_id: Any, obs_value: Any) -> None:
        vector = _to_numpy_array(obs_value).reshape(-1)
        _register_state_entry(state_map, sample_id, vector)

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "obs" in item:
                sample_id = item.get("sample_id")
                if sample_id is None:
                    continue
                add_pair(sample_id, item["obs"])
        return state_map

    if isinstance(payload, dict):
        if "sample_id" in payload and "obs" in payload:
            sample_ids = _iter_sample_ids(payload["sample_id"])
            observations = _iter_observations(payload["obs"])
            if len(sample_ids) != len(observations):
                raise ValueError(
                    f"State file {state_path} contains {len(sample_ids)} sample_id entries "
                    f"but {len(observations)} observation rows."
                )
            for sid, obs_vector in zip(sample_ids, observations):
                add_pair(sid, obs_vector)
            return state_map
        for key, value in payload.items():
            if isinstance(value, dict) and "obs" in value:
                add_pair(value.get("sample_id", key), value["obs"])
            elif not isinstance(value, dict):
                add_pair(key, value)
        return state_map

    raise TypeError(f"Unsupported state file format: {type(payload)!r}")


def _lookup_state_vector(state_map: dict[Any, np.ndarray], sample_id: Any) -> np.ndarray | None:
    """
    Resolve ``sample_id`` to its corresponding state vector if available.
    """
    if not state_map:
        return None
    candidates = []
    if torch is not None and torch.is_tensor(sample_id):
        if sample_id.ndim == 0:
            sample_id = sample_id.item()
        else:
            for entry in sample_id.reshape(-1).tolist():
                result = _lookup_state_vector(state_map, entry)
                if result is not None:
                    return result
            return None
    if isinstance(sample_id, np.ndarray):
        if sample_id.ndim == 0:
            sample_id = sample_id.item()
        else:
            for entry in sample_id.reshape(-1).tolist():
                result = _lookup_state_vector(state_map, entry)
                if result is not None:
                    return result
            return None
    if isinstance(sample_id, (list, tuple)):
        for entry in sample_id:
            result = _lookup_state_vector(state_map, entry)
            if result is not None:
                return result
        return None

    candidates.append(sample_id)
    try:
        sample_str = str(sample_id)
    except Exception:
        sample_str = None
    else:
        candidates.append(sample_str)

    if isinstance(sample_id, str):
        try:
            sample_int = int(sample_id)
        except ValueError:
            sample_int = None
        if sample_int is not None:
            candidates.extend([sample_int, str(sample_int)])
    elif isinstance(sample_id, (int, np.integer)):
        sample_int = int(sample_id)
        candidates.extend([sample_int, str(sample_int)])

    for candidate in candidates:
        if candidate in state_map:
            return state_map[candidate]
    return None


def _prepare_heatmap(attr, background: np.ndarray | None) -> tuple[np.ndarray, bool]:
    candidate = None
    if isinstance(attr, dict):
        for key in ("heatmap", "spatial", "mask", "image", "map"):
            if key in attr:
                candidate = attr[key]
                break
    else:
        candidate = attr
    if candidate is None or isinstance(candidate, dict):
        if background is not None:
            height, width = background.shape[:2]
            return np.zeros((height, width), dtype=float), False
        return np.zeros((1, 1), dtype=float), False

    arr = _to_numpy_array(candidate)
    if arr.ndim >= 2:
        heat = np.array(arr, dtype=float)
        if heat.ndim > 2:
            spatial = heat.reshape((-1, heat.shape[-2], heat.shape[-1]))
            heat = spatial.mean(axis=0)
        heat = np.squeeze(heat)
        if heat.ndim != 2:
            if background is None:
                return np.zeros((1, 1), dtype=float), False
            height, width = background.shape[:2]
            return np.zeros((height, width), dtype=float), False
        h_min = np.nanmin(heat)
        h_max = np.nanmax(heat)
        if np.isfinite(h_min) and np.isfinite(h_max) and h_max > h_min:
            heat = (heat - h_min) / (h_max - h_min)
        else:
            heat = np.zeros_like(heat, dtype=float)
        heat = np.clip(heat, 0.0, 1.0)
        return heat, True

    if background is not None:
        height, width = background.shape[:2]
        return np.zeros((height, width), dtype=float), False
    return np.zeros((1, 1), dtype=float), False


def _index_frame_images(root: Path) -> tuple[dict[str, Path], dict[int, Path]]:
    by_name: dict[str, Path] = {}
    by_numeric: dict[int, Path] = {}
    if not root.exists():
        raise FileNotFoundError(f"Image root not found: {root}")
    candidates = list(root.glob("*.png"))
    candidates += list(root.glob("*/*.png"))
    for path in candidates:
        if not path.is_file():
            continue
        by_name[path.name] = path
        match = re.search(r"frame_id(\d+)", path.name)
        if match:
            value = match.group(1)
            try:
                by_numeric[int(value)] = path
            except ValueError:
                continue
    return by_name, by_numeric


def _resolve_frame_path(
    sample_id,
    name_index: dict[str, Path],
    numeric_index: dict[int, Path],
) -> Path | None:
    sample_str = str(sample_id)
    sample_digits = re.findall(r"\d+", sample_str)
    candidates = []
    if sample_str.endswith(".png"):
        candidates.append(sample_str)
    candidates.append(f"{sample_str}.png")
    if not sample_str.startswith("frame_id"):
        candidates.append(f"frame_id{sample_str}.png")

    for digits in sample_digits:
        candidates.append(f"frame_id{digits}.png")
        try:
            value = int(digits)
        except ValueError:
            continue
        if value in numeric_index:
            return numeric_index[value]
        candidates.append(f"frame_id{value:010d}.png")

    for candidate in candidates:
        if candidate in name_index:
            return name_index[candidate]
    return None


def render_results_file(
    results_path: Path,
    image_root: Path,
    output_dir: Path,
    *,
    top_k: int = 5,
    skip_top: int = 5,
    cmap: str = "jet",
    heat_alpha: float = 0.6,
    overlay_heatmap: bool | None = None,
    limit: int | None = None,
    state_path: Path | None = None,
) -> dict[str, list]:
    if torch is None:
        raise ImportError("torch is required to load attribution files saved with torch.save().")
    results = torch.load(results_path, map_location="cpu")
    if not isinstance(results, dict):
        raise ValueError("Expected results file to contain a dict keyed by sample id.")

    feature_groups = infer_feature_groups(results_path, image_root)
    name_index, numeric_index = _index_frame_images(image_root)
    state_map = _load_state_vectors(state_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    missing_images: list = []
    skipped: list = []
    missing_states: list = []

    ordered_items = sorted(results.items(), key=lambda item: str(item[0]))
    for sample_id, record in ordered_items:
        attr = record.get("attr") if isinstance(record, dict) else None
        if attr is None:
            skipped.append((sample_id, "missing attr"))
            continue
        try:
            feature_vec = _extract_feature_vector(attr)
        except ValueError as exc:
            skipped.append((sample_id, f"feature error: {exc}"))
            continue
        required = top_k + max(skip_top, 0)
        if feature_vec.size < required:
            skipped.append((sample_id, f"attr length {feature_vec.size} < required {required}"))
            continue

        frame_path = _resolve_frame_path(sample_id, name_index, numeric_index)
        if frame_path is None:
            missing_images.append(sample_id)
            continue

        background = plt.imread(frame_path)
        heatmap, inferred_overlay = _prepare_heatmap(attr, background)
        use_overlay = inferred_overlay if overlay_heatmap is None else overlay_heatmap

        groups = _groups_from_record(record if isinstance(record, dict) else None, feature_groups)
        feature_scores, feature_labels, feature_indices, _ = select_top_features(
            feature_vec,
            top_k=top_k,
            skip_top=skip_top,
            groups=groups,
        )

        safe_id = _sanitize_sample_id(sample_id)
        save_path = output_dir / f"frame_{safe_id}.png"
        print(f"[sample {sample_id}] -> {save_path.name}")

        feature_values = None
        max_index = int(feature_indices.max()) if feature_indices.size else -1
        if state_map:
            state_vector = _lookup_state_vector(state_map, sample_id)
            if state_vector is None:
                missing_states.append((sample_id, "state missing"))
            elif max_index >= state_vector.size:
                missing_states.append(
                    (
                        sample_id,
                        f"state length {state_vector.size} insufficient for feature index {max_index}",
                    )
                )
            else:
                feature_values = state_vector[feature_indices]

        plot_heat_with_strips(
            heatmap,
            feature_scores=feature_scores,
            feature_labels=feature_labels,
            feature_values=feature_values,
            show=False,
            save_path=str(save_path),
            background=background,
            cmap=cmap,
            heat_alpha=heat_alpha,
            overlay_heatmap=use_overlay,
        )
        processed += 1
        if limit is not None and processed >= limit:
            break

    return {
        "processed": processed,
        "missing_images": missing_images,
        "skipped": skipped,
        "missing_states": missing_states,
    }


def render_single_sample(
    results_path: Path,
    image_root: Path,
    sample_id,
    *,
    top_k: int = 5,
    skip_top: int = 5,
    cmap: str = "jet",
    heat_alpha: float = 0.6,
    overlay_heatmap: bool | None = None,
    show_colorbar: bool = True,
    show: bool = True,
    output_path: Path | str | None = None,
    state_path: Path | None = None,
) -> plt.Figure:
    """
    Render a single attribution figure for ``sample_id`` from ``results_path``.

    Parameters mirror :func:`render_results_file` so the figure matches the batch
    renderer's appearance. When ``state_path`` is provided the top-k feature
    state values are shown beneath their labels.
    """
    if torch is None:
        raise ImportError("torch is required to load attribution files saved with torch.save().")
    results = torch.load(results_path, map_location="cpu")
    if not isinstance(results, dict):
        raise ValueError("Expected results file to contain a dict keyed by sample id.")

    resolved_key = sample_id
    if sample_id not in results:
        match = None
        target = str(sample_id)
        for key, value in results.items():
            if str(key) == target:
                match = (key, value)
                break
        if match is None:
            raise KeyError(f"Sample {sample_id!r} not found in results file.")
        resolved_key, record = match
    else:
        record = results[sample_id]

    attr = record.get("attr") if isinstance(record, dict) else None
    if attr is None:
        raise ValueError(f"Sample {resolved_key!r} is missing 'attr' data.")

    feature_vec = _extract_feature_vector(attr)
    required = top_k + max(skip_top, 0)
    if feature_vec.size < required:
        raise ValueError(
            f"attr length {feature_vec.size} < required {required} for sample {resolved_key!r}"
        )

    name_index, numeric_index = _index_frame_images(image_root)
    frame_path = _resolve_frame_path(resolved_key, name_index, numeric_index)
    if frame_path is None:
        raise FileNotFoundError(f"Could not find frame image for sample {resolved_key!r}.")

    background = plt.imread(frame_path)
    heatmap, inferred_overlay = _prepare_heatmap(attr, background)
    use_overlay = inferred_overlay if overlay_heatmap is None else overlay_heatmap

    feature_groups = infer_feature_groups(results_path, image_root)
    groups = _groups_from_record(record if isinstance(record, dict) else None, feature_groups)
    feature_scores, feature_labels, feature_indices, _ = select_top_features(
        feature_vec,
        top_k=top_k,
        skip_top=skip_top,
        groups=groups,
    )

    feature_values = None
    if state_path is not None:
        state_map = _load_state_vectors(state_path)
        state_vector = _lookup_state_vector(state_map, resolved_key)
        if state_vector is None:
            raise KeyError(f"State vector for sample {resolved_key!r} not found in {state_path}.")
        max_index = int(feature_indices.max()) if feature_indices.size else -1
        if max_index >= state_vector.size:
            raise ValueError(
                f"State vector for sample {resolved_key!r} has length {state_vector.size}, "
                f"but feature index {max_index} is required."
            )
        feature_values = state_vector[feature_indices]

    save_path_str = None
    if output_path is not None:
        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path_str = str(save_path)
        print(f"[sample {resolved_key}] -> {Path(save_path_str).name}")
    elif not show:
        raise ValueError("output_path must be provided when show=False.")

    figure = plot_heat_with_strips(
        heatmap,
        feature_scores=feature_scores,
        feature_labels=feature_labels,
        show=show,
        feature_values=feature_values,
        save_path=save_path_str,
        background=background,
        cmap=cmap,
        heat_alpha=heat_alpha,
        overlay_heatmap=use_overlay,
        show_colorbar=show_colorbar,
    )
    return figure


def plot_heat_with_strips(
    heatmap: np.ndarray,
    feature_scores: Sequence[float] | np.ndarray,
    feature_labels: Sequence[str],
    *,
    feature_values: Sequence[float] | np.ndarray | None = None,
    cmap: str = "jet",
    show: bool = True,
    save_path: str | None = None,
    background: np.ndarray | None = None,
    heat_alpha: float = 0.6,
    overlay_heatmap: bool | None = None,
    show_colorbar: bool = True,
) -> plt.Figure:
    """
    Plot ``heatmap`` (optionally over an image) with top feature labels and a heat strip.

    Parameters
    ----------
    heatmap:
        2D array of values in [0, 1] (values outside will be clipped).
    feature_scores:
        1D sequence of normalized importances (0-1) for the features to highlight on top.
    feature_labels:
        Sequence of strings labelling each feature. Must match ``feature_scores`` length.
    feature_values:
        Optional raw feature values aligned with ``feature_labels``. When provided,
        the values are displayed between the label row and the heat strip.
    cmap:
        Matplotlib colormap name. Defaults to ``'jet'`` which mirrors ``cv2.COLORMAP_JET``.
    show:
        When ``True`` (default) the figure is shown via ``plt.show()``.
    save_path:
        Optional path. When provided, the figure is saved after rendering.
    background:
        Optional RGB/RGBA image mapped beneath the heatmap. Must match ``heatmap`` spatial shape.
    heat_alpha:
        Alpha applied when overlaying the heatmap on top of ``background``.
    overlay_heatmap:
        When ``True`` the main heatmap draws even if ``background`` is present.
        Defaults to ``False`` when a background is supplied and ``True`` otherwise.
    show_colorbar:
        When ``True`` (default) renders the shared colorbar to the right of the plot.
        Disable to keep the layout identical without drawing the colorbar.
    """
    data = np.array(heatmap, dtype=float)
    if data.ndim != 2:
        raise ValueError("heatmap must be a 2D array.")

    # Clip to [0, 1] to stay within the color scale.
    data = np.clip(data, 0.0, 1.0)

    if background is not None:
        bg = np.asarray(background)
        if bg.shape[:2] != data.shape:
            raise ValueError("background spatial dimensions must match heatmap shape.")
    else:
        bg = None

    if overlay_heatmap is None:
        overlay_heatmap = bg is None

    feature_scores_arr = np.clip(np.asarray(feature_scores, dtype=float), 0.0, 1.0)
    if feature_scores_arr.ndim != 1:
        raise ValueError("feature_scores must be 1D.")
    if len(feature_labels) != feature_scores_arr.size:
        raise ValueError("feature_labels length must match feature_scores length.")
    if feature_scores_arr.size == 0:
        raise ValueError("At least one feature score is required.")

    values_arr = None
    if feature_values is not None:
        values_arr = np.asarray(feature_values, dtype=float).reshape(-1)
        if values_arr.size != feature_scores_arr.size:
            raise ValueError("feature_values length must match feature_scores length.")

    num_features = feature_scores_arr.size
    scale = min(0.8, max(0.45, 4.5 / max(1, num_features)))
    label_height = 1.0 * scale
    strip_height = 0.9 * scale
    fontsize = 9.5
    if values_arr is not None:
        value_height = 0.75 * scale
        height_ratios = [label_height, value_height, strip_height, 0.2, 12]
        nrows = 5
        row_labels = 0
        row_values = 1
        row_strip = 2
        row_spacer = 3
        row_main = 4
    else:
        height_ratios = [label_height, strip_height, 0.2, 12]
        nrows = 4
        row_labels = 0
        row_values = None
        row_strip = 1
        row_spacer = 2
        row_main = 3

    fig = plt.figure(figsize=(6, 6), constrained_layout=False)
    if show_colorbar:
        grid_ncols = 2
        width_ratios = [20, 1]
    else:
        grid_ncols = 1
        width_ratios = [20]
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=grid_ncols,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.05,
        wspace=0.15,
    )

    # Feature label axis.
    ax_labels = fig.add_subplot(gs[row_labels, 0])
    ax_labels.set_axis_off()
    ax_labels.set_xlim(0, feature_scores_arr.size)
    ax_labels.set_ylim(0, 1)
    for idx, label in enumerate(feature_labels):
        ax_labels.text(
            idx + 0.5,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
        )

    if values_arr is not None and row_values is not None:
        ax_values = fig.add_subplot(gs[row_values, 0])
        ax_values.set_axis_off()
        ax_values.set_xlim(0, feature_scores_arr.size)
        ax_values.set_ylim(0, 1)
        for idx, value in enumerate(values_arr):
            formatted = f"{value:.3f}".rstrip("0").rstrip(".")
            if not formatted or formatted == "-":
                formatted = "0"
            if formatted == "-0":
                formatted = "0"
            ax_values.text(
                idx + 0.5,
                0.5,
                formatted,
                ha="center",
                va="center",
                fontsize=fontsize,
            )

    # Heat strip axis.
    ax_strip = fig.add_subplot(gs[row_strip, 0])
    ax_strip.imshow(
        feature_scores_arr[None, :],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])

    # Spacer axis to keep separation.
    ax_spacer = fig.add_subplot(gs[row_spacer, 0])
    ax_spacer.set_axis_off()

    # Main heatmap axis.
    ax_main = fig.add_subplot(gs[row_main, 0])
    if bg is not None:
        ax_main.imshow(bg, aspect="auto")

    if overlay_heatmap:
        im = ax_main.imshow(
            data,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            alpha=heat_alpha if bg is not None else 1.0,
        )
    else:
        im = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=cmap)
        im.set_array([])
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    # Shared colorbar spanning the full height.
    if show_colorbar:
        cax = fig.add_subplot(gs[:, -1])
        fig.colorbar(im, cax=cax, ticks=np.linspace(0, 1, 6))
        cax.set_ylabel("Intensity", rotation=90)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def demo(seed: int = 7) -> None:
    """Quick demo that combines a recorded frame with synthetic heat strips."""
    image_path = Path(
        "/home/rzuo02/work/Safe-Policy-Optimization/jobs/runs/"
        "single_agent_exp/SafetyPointGoal1-v0/ppo/"
        "seed-000-2025-10-27-11-56-06/renders/epoch_00499/env_0000/"
        "frame_id0009999500.png"
    )
    if not image_path.exists():
        raise FileNotFoundError(
            f"Demo image not found: {image_path}\n"
            "Update the path or provide a different sample before running demo()."
        )

    background = plt.imread(image_path)
    height, width = background.shape[:2]

    rng = np.random.default_rng(seed)
    heatmap = rng.random((height, width))

    total_features = sum(length for _, length in FEATURE_GROUPS)
    feature_attributions = rng.normal(size=total_features)
    skip = 5
    (
        feature_scores,
        feature_labels,
        feature_indices,
        raw_scores,
    ) = select_top_features(feature_attributions, top_k=5, skip_top=skip)

    for rank, (label, idx, raw) in enumerate(
        zip(feature_labels, feature_indices, raw_scores), start=skip + 1
    ):
        print(f"#{rank:>2} {label:<12} (idx {idx:>2}) -> |attr|={raw:.4f}")

    plot_heat_with_strips(
        heatmap,
        feature_scores=feature_scores,
        feature_labels=feature_labels,
        show=True,
        background=background,
        overlay_heatmap=False,
        save_path="foobar.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render attribution plots for PointGoal samples.")
    parser.add_argument("--results", type=Path, help="Path to results .pt file.")
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Directory containing frame subfolders with frame_id*.png files.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        help="Optional path to a .pt file containing state vectors with 'sample_id' and 'obs' entries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where rendered figures will be written.",
    )
    parser.add_argument(
        "--sample-id",
        help="Render only the specified sample id instead of the full results file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Optional destination path when rendering a single sample.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top features to display.")
    parser.add_argument(
        "--skip-top",
        type=int,
        default=5,
        help="Skip this many highest-ranked features before selecting top-k (default: 5).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on processed samples.")
    parser.add_argument("--cmap", default="jet", help="Matplotlib colormap for heatmaps.")
    parser.add_argument(
        "--heat-alpha",
        type=float,
        default=0.6,
        help="Alpha used when overlaying heatmap on the background image.",
    )
    parser.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Disable the colorbar when rendering a single sample.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the figure when rendering a single sample.",
    )
    parser.add_argument(
        "--overlay-heatmap",
        choices=("auto", "on", "off"),
        default="auto",
        help="Control heatmap overlay behaviour (auto uses heuristics).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the synthetic demo instead of processing a results file.",
    )
    cli_args = parser.parse_args()

    if cli_args.demo and cli_args.results:
        parser.error("--demo cannot be combined with --results.")
    if cli_args.output_path and not cli_args.sample_id:
        parser.error("--output-path requires --sample-id.")
    if cli_args.no_colorbar and not cli_args.sample_id:
        parser.error("--no-colorbar is only valid with --sample-id.")
    if cli_args.no_show and not cli_args.sample_id:
        parser.error("--no-show is only valid with --sample-id.")

    if cli_args.overlay_heatmap == "auto":
        overlay_choice: bool | None = None
    elif cli_args.overlay_heatmap == "on":
        overlay_choice = True
    else:
        overlay_choice = False

    if cli_args.demo:
        demo()
    elif cli_args.sample_id is not None:
        if not cli_args.results or not cli_args.image_root:
            parser.error("--results and --image-root are required when --sample-id is set.")
        render_single_sample(
            results_path=cli_args.results,
            image_root=cli_args.image_root,
            sample_id=cli_args.sample_id,
            state_path=cli_args.state,
            top_k=cli_args.top_k,
            skip_top=cli_args.skip_top,
            cmap=cli_args.cmap,
            heat_alpha=cli_args.heat_alpha,
            overlay_heatmap=overlay_choice,
            show_colorbar=not cli_args.no_colorbar,
            show=not cli_args.no_show,
            output_path=cli_args.output_path,
        )
    else:
        if not cli_args.results or not cli_args.image_root:
            parser.error("--results and --image-root are required unless --demo is set.")
        if not cli_args.output_dir:
            parser.error("--output-dir is required for batch rendering.")
        summary = render_results_file(
            results_path=cli_args.results,
            image_root=cli_args.image_root,
            output_dir=cli_args.output_dir,
            state_path=cli_args.state,
            top_k=cli_args.top_k,
            skip_top=cli_args.skip_top,
            cmap=cli_args.cmap,
            heat_alpha=cli_args.heat_alpha,
            overlay_heatmap=overlay_choice,
            limit=cli_args.limit,
        )
        print(f"Processed {summary['processed']} sample(s).")
        if summary["missing_images"]:
            print(f"Missing images for {len(summary['missing_images'])} sample(s): {summary['missing_images']}")
        if summary["skipped"]:
            print(f"Skipped {len(summary['skipped'])} sample(s): {summary['skipped']}")
        if summary["missing_states"]:
            print(
                f"Missing states for {len(summary['missing_states'])} sample(s): {summary['missing_states']}"
            )
