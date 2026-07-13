import sys
from pathlib import Path

import runpod


PROJECT_ROOT = Path("/app")
if not (PROJECT_ROOT / "worker.py").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from worker import handler as worker_handler  # noqa: E402


def handler(job):
    return worker_handler(job)


# RunPod scanner marker: runpod.serverless.start()
runpod.serverless.start({"handler": handler})
