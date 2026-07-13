import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_ENDPOINT_ID = "i2pgt8jfdmk2sn"


def post_json(url, api_key, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request = Request(url, data=body, headers=headers, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url, api_key, timeout):
    headers = {"Authorization": f"Bearer {api_key}"}
    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def decode_image_data(data):
    if data.startswith("data:image/png;base64,"):
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


def save_images(result, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    images = result.get("output", result).get("images", [])
    saved = []
    for index, image in enumerate(images):
        image_bytes = decode_image_data(image["data"])
        path = output_dir / f"runpod_k2_{index}.png"
        path.write_bytes(image_bytes)
        saved.append(path)
    return saved


def main():
    env_api_key = os.environ.get("RUNPOD_API_KEY")
    parser = argparse.ArgumentParser(description="Test a deployed RunPod Krea 2 worker.")
    parser.add_argument(
        "--endpoint-id",
        default=os.environ.get("RUNPOD_ENDPOINT_ID", DEFAULT_ENDPOINT_ID),
        help=f"RunPod endpoint ID (default: {DEFAULT_ENDPOINT_ID}).",
    )
    parser.add_argument(
        "--api-key",
        default=env_api_key,
        help="RunPod API key. Defaults to $RUNPOD_API_KEY.",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Check worker and checkpoint status without generating an image.",
    )
    parser.add_argument("--prompt", default="a fox walking in the snow")
    parser.add_argument("--checkpoint", default="oss_turbo", choices=["oss_raw", "oss_turbo"])
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=Path("runpod_outputs"))
    parser.add_argument(
        "--async",
        dest="async_job",
        action="store_true",
        help="Submit with /run and poll /status instead of using /runsync.",
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("set RUNPOD_API_KEY or pass --api-key")
    if env_api_key:
        print(f"RUNPOD_API_KEY loaded from environment (ends with {env_api_key[-4:]})")
    else:
        print("RUNPOD_API_KEY is not set; using the --api-key argument")

    base_url = f"https://api.runpod.ai/v2/{args.endpoint_id}"
    if args.health_check:
        try:
            result = post_json(
                f"{base_url}/runsync",
                args.api_key,
                {"input": {"health_check": True}},
                timeout=args.timeout,
            )
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"request failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2))
        return 0

    request_input = {
        "prompt": args.prompt,
        "checkpoint": args.checkpoint,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "num_images": args.num_images,
        "output_format": "base64",
    }
    if args.steps is not None:
        request_input["steps"] = args.steps
    if args.cfg is not None:
        request_input["cfg"] = args.cfg

    try:
        if args.async_job:
            submitted = post_json(
                f"{base_url}/run",
                args.api_key,
                {"input": request_input},
                timeout=60,
            )
            job_id = submitted["id"]
            deadline = time.time() + args.timeout
            while time.time() < deadline:
                status = get_json(
                    f"{base_url}/status/{job_id}",
                    args.api_key,
                    timeout=60,
                )
                if status.get("status") in {"COMPLETED", "FAILED", "CANCELLED"}:
                    result = status
                    break
                print(f"status={status.get('status')}; waiting...", flush=True)
                time.sleep(5)
            else:
                raise TimeoutError(f"job {job_id} did not finish within {args.timeout}s")
        else:
            result = post_json(
                f"{base_url}/runsync",
                args.api_key,
                {"input": request_input},
                timeout=args.timeout,
            )
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1

    output = result.get("output", result)
    if result.get("status") in {"FAILED", "CANCELLED", "TIMED_OUT"}:
        print(f"job ended with status: {result['status']}", file=sys.stderr)
        return 1
    if output.get("error"):
        print(f"worker error: {output['error']}", file=sys.stderr)
        return 1

    saved = save_images(result, args.output_dir)
    if not saved:
        print("no images were returned by the worker", file=sys.stderr)
        return 1
    print(json.dumps(result.get("metadata", output.get("metadata", {})), indent=2))
    for path in saved:
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
