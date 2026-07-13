import base64
import io
import os
import threading
import traceback
from typing import Any

import torch

from inference import _pipeline
from sampling import sample


PIPELINES: dict[str, tuple[Any, Any, Any]] = {}
PIPELINE_LOCK = threading.Lock()
VALID_CHECKPOINTS = {"oss_raw", "oss_turbo"}


def _checkpoint_env(checkpoint: str) -> str:
    return "OSS_RAW" if checkpoint == "oss_raw" else "OSS_TURBO"


def _checkpoint_status(checkpoint: str) -> dict[str, Any]:
    env_name = _checkpoint_env(checkpoint)
    path = os.environ.get(env_name)
    return {
        "env": env_name,
        "path": path,
        "configured": bool(path),
        "exists": os.path.exists(path) if path else False,
    }


def _bool_input(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int_input(value: Any, name: str, default: int, minimum: int | None = None) -> int:
    if value is None:
        result = default
    else:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _float_input(
    value: Any, name: str, default: float | None, minimum: float | None = None
) -> float | None:
    if value is None:
        result = default
    else:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc
    if result is not None and minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _checkpoint_defaults(checkpoint: str) -> tuple[int, float, float | None]:
    if checkpoint == "oss_turbo":
        return 8, 0.0, 1.15
    return 28, 4.5, None


def _get_pipeline(checkpoint: str) -> tuple[Any, Any, Any]:
    with PIPELINE_LOCK:
        if checkpoint not in PIPELINES:
            PIPELINES[checkpoint] = _pipeline(checkpoint=checkpoint)
        return PIPELINES[checkpoint]


def _encode_png(image, output_format: str) -> dict[str, Any]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    if output_format == "base64":
        data = encoded
    elif output_format == "data_uri":
        data = f"data:image/png;base64,{encoded}"
    else:
        raise ValueError("output_format must be 'data_uri' or 'base64'")
    return {
        "format": "png",
        "width": image.width,
        "height": image.height,
        "data": data,
    }


def handler(job):
    job_input = job.get("input") or {}
    if _bool_input(job_input.get("health_check")):
        return {
            "ok": True,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "loaded_checkpoints": sorted(PIPELINES),
            "checkpoints": {
                checkpoint: _checkpoint_status(checkpoint)
                for checkpoint in sorted(VALID_CHECKPOINTS)
            },
        }

    prompt = job_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "input.prompt is required"}

    checkpoint = job_input.get("checkpoint") or os.environ.get(
        "K2_CHECKPOINT", "oss_turbo"
    )
    if checkpoint not in VALID_CHECKPOINTS:
        return {"error": f"checkpoint must be one of {sorted(VALID_CHECKPOINTS)}"}

    default_steps, default_cfg, default_mu = _checkpoint_defaults(checkpoint)
    max_images = _int_input(os.environ.get("K2_MAX_IMAGES"), "K2_MAX_IMAGES", 1, 1)
    num_images = _int_input(job_input.get("num_images"), "num_images", 1, 1)
    if num_images > max_images:
        return {"error": f"num_images must be <= {max_images}"}

    width = _int_input(job_input.get("width"), "width", 1024, 16)
    height = _int_input(job_input.get("height"), "height", 1024, 16)
    steps = _int_input(job_input.get("steps"), "steps", default_steps, 1)
    seed = _int_input(job_input.get("seed"), "seed", 0)
    cfg = _float_input(job_input.get("cfg"), "cfg", default_cfg, 0.0)
    y1 = _float_input(job_input.get("y1"), "y1", 0.5)
    y2 = _float_input(job_input.get("y2"), "y2", 1.15)
    mu = _float_input(job_input.get("mu"), "mu", default_mu)
    output_format = job_input.get("output_format", "data_uri")

    negative_prompt = job_input.get("negative_prompt", "")
    negative_prompts = [negative_prompt] * num_images

    try:
        dit, ae, encoder = _get_pipeline(checkpoint)
        images = sample(
            dit,
            ae,
            encoder,
            [prompt] * num_images,
            negative_prompts=negative_prompts,
            width=width,
            height=height,
            steps=steps,
            guidance=cfg,
            seed=seed,
            y1=y1,
            y2=y2,
            mu=mu,
        )
        return {
            "images": [_encode_png(image, output_format) for image in images],
            "metadata": {
                "checkpoint": checkpoint,
                "prompt": prompt,
                "width": images[0].width if images else width,
                "height": images[0].height if images else height,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "mu": mu,
            },
        }
    except Exception as exc:
        traceback.print_exc()
        return {"error": str(exc)}
