"""SegmentationAPI — clean interface to the Cardiac-Prism model.

Encapsulates config loading, model loading (lazy), preprocessing,
inference, and coordinate mapping for the workstation backend.

Coordinate convention
---------------------
All public methods accept points in **original image space** as (x, y)
(horizontal, vertical) and perform the (y, x) flip + scale + pad mapping
internally.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def load_checkpoint_state(model: torch.nn.Module, state: Mapping) -> Dict:
    """Load inference weights, explicitly handling known legacy omissions.

    Pass a freshly constructed model so legacy omissions retain its seeded
    initialization. Older checkpoints lack the entire sparse click encoder and projection.
    Keep their deterministic initialization only when a complete module is
    absent. Missing learned weights, partial modules and unknown keys are
    errors; they must not be silently accepted via ``strict=False``.
    """
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint 'model' must be a mapping of tensor weights.")

    normalized = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("Checkpoint model entries must have string keys and tensor values.")
        clean_key = key.removeprefix("module.")
        if clean_key in normalized:
            raise ValueError(f"Duplicate checkpoint key after removing DDP prefix: {clean_key}")
        normalized[clean_key] = value

    expected = model.state_dict()
    missing = set(expected) - set(normalized)
    unexpected = set(normalized) - set(expected)
    allowed_missing = set()
    legacy_modules = []
    initialized_prompt_keys = []
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] not in {"point_embed_encoder", "click_feature_projection"}:
            continue
        module_keys = {f"{name}.{key}" for key in module.state_dict()}
        absent = missing & module_keys
        if absent and absent != module_keys:
            raise RuntimeError(f"Incomplete click module in checkpoint: {name}; missing {sorted(absent)}")
        if absent:
            legacy_modules.append(name)
            initialized_prompt_keys.extend(sorted(absent))
            allowed_missing.update(absent)

    # These counters were added after the trained weights and do not affect
    # predictions. Only actual registered statistics buffers may be omitted.
    statistics = {
        name for name, _ in model.named_buffers()
        if name.endswith((".gate_sum", ".gate_count"))
    }
    initialized_statistic_keys = sorted(missing & statistics)
    allowed_missing.update(initialized_statistic_keys)
    invalid_missing = missing - allowed_missing
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"Incompatible checkpoint: missing weights {sorted(invalid_missing)}; "
            f"unexpected keys {sorted(unexpected)}"
        )
    incompatible_shapes = [
        key for key in normalized
        if normalized[key].shape != expected[key].shape
    ]
    if incompatible_shapes:
        raise RuntimeError(f"Checkpoint tensor shape mismatch: {incompatible_shapes}")

    # Fill only the omissions validated above, then require a strict load.
    model.load_state_dict({**expected, **normalized}, strict=True)
    if legacy_modules:
        log.warning(
            "Legacy checkpoint has no weights for %s; using fixed initialization "
            "for these click modules. Reproducibility is restored; segmentation "
            "accuracy still needs validation with representative data.",
            ", ".join(legacy_modules),
        )
    return {
        "legacy_prompt_modules": legacy_modules,
        "initialized_prompt_keys": sorted(initialized_prompt_keys),
        "initialized_statistic_keys": initialized_statistic_keys,
    }


class SegmentationAPI:
    """Wraps the Cardiac-Prism model for interactive slice segmentation.

    Parameters
    ----------
    config_path:
        Path to ``eval_config.yaml``.  Can be overridden by the environment
        variable ``CARDIAC_PRISM_CONFIG``.
    """

    def __init__(self, config_path: str | Path, device: str | None = None) -> None:
        from server.environment import apply_legacy_environment

        apply_legacy_environment()
        self.config_path = Path(os.environ.get("CARDIAC_PRISM_CONFIG", str(config_path)))
        self.requested_device = device or os.environ.get("CARDIAC_PRISM_DEVICE", "auto")
        self._model: Optional[object] = None
        self._cfg: Optional[object] = None
        self._preprocessor: Optional[object] = None
        self._checkpoint_load_report: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_config(self):
        """Load and compose OmegaConf config (Hydra-like merge without Hydra)."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        cfg = OmegaConf.load(str(self.config_path))
        cfg_dir = self.config_path.parent

        if "model" not in cfg:
            model_base = cfg_dir / "model" / "base.yaml"
            if model_base.exists():
                cfg = OmegaConf.merge(OmegaConf.load(str(model_base)), cfg)

        if "data" not in cfg:
            data_base = cfg_dir / "data" / "base.yaml"
            if data_base.exists():
                cfg = OmegaConf.merge(cfg, OmegaConf.load(str(data_base)))

        datasets_yaml = cfg_dir / "data" / "datasets.yaml"
        if datasets_yaml.exists():
            if "data" in cfg:
                cfg.data = OmegaConf.merge(OmegaConf.load(str(datasets_yaml)), cfg.get("data"))
            else:
                cfg.data = OmegaConf.load(str(datasets_yaml))

        # Normalise model structure: ensure cfg.model.model and cfg.model.setting
        if "model" in cfg:
            if "setting" in cfg and "setting" not in cfg.model:
                cfg.model.setting = cfg.setting
                del cfg["setting"]
            if "model" not in cfg.model:
                inner = {k: v for k, v in cfg.model.items() if k != "setting"}
                setting = cfg.model.get("setting")
                cfg.model = OmegaConf.create({"model": inner})
                if setting is not None:
                    cfg.model.setting = setting

        return cfg

    def _ensure_loaded(self) -> None:
        """Lazy-load model and config on first call."""
        if self._model is not None:
            return

        from ..inference.inference import InferenceCore
        from ..inference.resize_transform import SegmentationPreprocessor

        cfg = self._load_config()

        resume_path = Path(os.environ.get("CARDIAC_PRISM_CHECKPOINT", cfg.testing.resume))
        if not resume_path.is_absolute() and 'CARDIAC_PRISM_CHECKPOINT' not in os.environ:
            resume_path = (self.config_path.parent / resume_path).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        requested_device = self.requested_device
        device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                              if requested_device == "auto" else requested_device)
        log.info("Loading model from %s on %s", resume_path, device)

        model = InferenceCore(cfg).to(device)
        # Load storage on CPU, then copy into the already placed model. This
        # avoids a second GPU copy and Torch 1.13's UUID/NVML device-count bug.
        ckpt = torch.load(str(resume_path), map_location='cpu')
        report = load_checkpoint_state(model, ckpt["model"])
        report["checkpoint_path"] = str(resume_path)
        report["click_prompt_seed"] = model.sem_seg_head.predictor.click_prompt_seed
        # Serialized metadata key shared with existing checkpoints.
        report["model_profile"] = ckpt.get("kprism_ui", {})
        model.eval()

        self._cfg = cfg
        self._model = model
        self._checkpoint_load_report = report
        self._preprocessor = SegmentationPreprocessor(
            long_size=cfg.model.setting.long_side_size
        )
        log.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_slice(
        self,
        image: np.ndarray,
        points_pos: List[Tuple[float, float]],
        points_neg: List[Tuple[float, float]],
        q_index: Optional[int],
        *,
        prev_mask: Optional[torch.Tensor] = None,
        click_index: int = 0,
        prepared_image: Optional[np.ndarray] = None,
    ) -> Dict:
        """Run segmentation on a single 2-D slice.

        Parameters
        ----------
        image:
            Float32 array of shape ``(H, W)`` or ``(H, W, 3)``.  Values can be
            in any range; they are normalised internally.
        points_pos:
            Positive clicks as ``[(x, y), ...]`` in **original image pixels**.
        points_neg:
            Negative clicks as ``[(x, y), ...]`` in **original image pixels**.
        q_index:
            Organ q-index (16=RV, 17=MYO, 18=LV) or ``None`` for clicks-only.
        prev_mask:
            Padded-space sigmoid tensor ``(1, long_size, long_size)`` from the
            previous inference run.  ``None`` on first run.
        click_index:
            Number of refinement iterations already completed.  0 on first run.
        prepared_image:
            Optional float32 256x256 slice from cardiac_model.prepare_frame for
            the cardiac fine-tuned checkpoint. Supply this for full-frame
            preprocessing, or for a known already prepared original MM image.
            Omit it for a standalone raw slice. Existing priors must use the
            same checkpoint profile and 512x512 model grid.

        Returns
        -------
        dict with keys:
            ``"mask"``              – ``np.ndarray (H, W)`` bool in original coords
            ``"seg_result_padded"`` – ``Tensor (1, long_H, long_W)`` for next iter
            ``"click_index"``       – updated click_index (caller should store this)
        """
        self._ensure_loaded()

        if self._checkpoint_load_report.get("model_profile", {}).get("preprocessing") == "mm_slice_p0_p98_256_v1":
            from server import cardiac_model
            if q_index not in (None, 16, 17, 18):
                raise ValueError("The cardiac profile supports RV/MYO/LV queries 16/17/18.")
            raw = np.asarray(image, dtype=np.float32)
            if raw.ndim == 3:
                raw = raw[..., 0]
            # Full-frame callers supply the already prepared 256-grid slice;
            # standalone slice callers normalize the supplied slice only.
            prepared = cardiac_model.prepare_frame(raw[None])[0] if prepared_image is None else prepared_image
            plane = cardiac_model.CardiacPlane(prepared, tuple(raw.shape))
            points = [{"x": x, "y": y, "positive": positive}
                      for coordinates, positive in ((points_pos, True), (points_neg, False)) for x, y in coordinates]
            mode = "click" if q_index is None else "organ_click" if points else "organ"
            result = cardiac_model.predict(self, plane, points, 1 if q_index is None else q_index-15,
                                            mode, None if prev_mask is None else prev_mask.detach().cpu().numpy(), click_index)
            return {"mask": result["mask"], "probability": result["probability"],
                    "seg_result_padded": torch.from_numpy(result["prior"]), "click_index": result["iteration"]}

        sp = self._preprocessor
        device = self._model.device

        # Normalise image to float32 [0,1], shape (3, H, W)
        img_np = image.astype(np.float32)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)  # (H, W, 3)
        if img_np.max() > 1.0:
            img_np = img_np / img_np.max()
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()  # (3, H, W)

        orig_h, orig_w = img_tensor.shape[1], img_tensor.shape[2]
        mask_zero = torch.zeros((orig_h, orig_w), dtype=torch.long)
        padded_image, _, size_info, pad_info, scale_factor = sp.resize_and_pad(
            img_tensor, mask_zero
        )
        pad_top, _, pad_left, _ = pad_info

        # Map (x, y) original → (row, col) padded, i.e. (y*scale+pad_top, x*scale+pad_left)
        def _map(pts: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
            return [
                (int(round(y * scale_factor + pad_top)), int(round(x * scale_factor + pad_left)))
                for x, y in pts
            ]

        pos_pts = _map(points_pos)
        neg_pts = _map(points_neg)

        has_clicks = len(points_pos) + len(points_neg) > 0
        has_qindex = q_index is not None

        batched_inputs = [{
            "image": padded_image.to(device),
            "size_info": size_info,
            "pad_info": pad_info,
            "scale_factor": scale_factor,
            "q_index": q_index,
            "user_points": {"pos": pos_pts, "neg": neg_pts},
            "prev_mask": prev_mask,
            "click_index": click_index,
        }]

        # click_mode is derived inside iter_inference from has_qindex/has_clicks,
        # but we also set cfg so the value is consistent for any logging.
        if has_qindex and not has_clicks:
            self._cfg.testing.testing_click_mode = ["1"]
        elif has_qindex and has_clicks:
            self._cfg.testing.testing_click_mode = ["2"]
        else:
            self._cfg.testing.testing_click_mode = ["3"]

        result_dict, _, extra_dict = self._model.iter_inference(batched_inputs)

        last_iter = max(result_dict.keys())
        restored = result_dict[last_iter][0]  # (1, orig_H, orig_W)
        mask_np = (restored[0].numpy() > 0.5).astype(bool)

        seg_result_padded = extra_dict.get("padded_results", {}).get(0, [None])[0]

        return {
            "mask": mask_np,
            "probability": restored[0].numpy().copy(),
            "seg_result_padded": seg_result_padded,
            "click_index": click_index + 1,
        }

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def checkpoint_load_report(self) -> Optional[Dict]:
        """Describe any explicitly initialized legacy checkpoint omissions."""
        return self._checkpoint_load_report

    def reload(self, config_path: Optional[str | Path] = None) -> None:
        """Force a fresh model load (e.g. after config path change)."""
        if config_path is not None:
            self.config_path = Path(config_path)
        self._model = None
        self._cfg = None
        self._preprocessor = None
        self._checkpoint_load_report = None
