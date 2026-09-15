from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; set training.device: cpu for a CPU run")
    return torch.device(requested)


def write_json(path: str | Path, value: Any) -> None:
    def json_safe(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): json_safe(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [json_safe(child) for child in item]
        if isinstance(item, np.generic):
            return json_safe(item.item())
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
