import runpod

from worker import handler as worker_handler


def handler(job):
    return worker_handler(job)


# RunPod scanner marker: runpod.serverless.start()
runpod.serverless.start({"handler": handler})
