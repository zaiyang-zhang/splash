#!/usr/bin/env python3

"""End-to-end video job on an isolated server (make test-video-real).

Serves the installed language model with the installed video model, then:
a chat request; POST /v1/videos for the 3 s golden prompt with a preview;
while it runs, generation must get 503 video_job_running and /ready and
/health stay 200; the job's progress must advance; the preview mp4 is
downloadable before the video; the video mp4 at the end; generation
answers again after it.

--golden NPZ injects saved hidden states instead of the text encoder
(SPLASH_VIDEO_HIDDEN_STATES) and requires the final latents to equal
--reference byte for byte. Without it the engine's own hidden states run,
and the result is compared with --reference: latent relative RMS and the
PSNR of the frames against --reference-frames. A memory sampler records
the host's used memory and each process's footprint across the job.
The report goes to --output (JSON).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.tests.smoke_real import (  # noqa: E402
    RealServer,
    SmokeFailure,
    available_port,
    require,
)

GOLDEN_PROMPT = (
    "A golden retriever runs along a beach at sunset, waves crashing, camera tracking from the side. "
    "Sound of waves and the dog's footsteps on wet sand."
)


def call(port, method, path, body=None, timeout=120):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    payload = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    try:
        connection.request(method, path, payload, headers)
        response = connection.getresponse()
        raw = response.read()
        content_type = response.getheader("Content-Type", "")
    finally:
        connection.close()
    if content_type.startswith("application/json"):
        return response.status, json.loads(raw)
    return response.status, raw


class MemorySampler:
    """The host's used memory (active + wired + compressed pages) and the
    resident size of this checkout's engine and video processes, every
    interval seconds."""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _host_used(self):
        output = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = int(re.search(r"page size of (\d+)", output).group(1))
        pages = {
            key: int(value)
            for key, value in re.findall(r'"?([A-Za-z -]+)"?:\s+(\d+)\.', output)
        }
        used = (
            pages.get("Pages active", 0)
            + pages.get("Pages wired down", 0)
            + pages.get("Pages occupied by compressor", 0)
        )
        return used * page

    def _footprints(self):
        output = subprocess.run(
            ["ps", "-axo", "pid=,rss=,command="], capture_output=True, text=True
        ).stdout
        processes = {}
        for line in output.splitlines():
            pid, rss, command = line.strip().split(None, 2)
            words = command.split()
            # This checkout's engine and video processes, by subcommand.
            if (
                words
                and words[0].endswith(str(ROOT / "build/splash"))
                and len(words) > 1
            ):
                processes[words[1]] = processes.get(words[1], 0) + int(rss) * 1024
        return processes

    def _run(self):
        while not self.stop.is_set():
            try:
                self.samples.append(
                    {
                        "t": time.time(),
                        "host_used": self._host_used(),
                        "rss": self._footprints(),
                    }
                )
            except (OSError, ValueError, AttributeError):
                pass
            self.stop.wait(self.interval)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join()
        peak = max(self.samples, key=lambda sample: sample["host_used"], default=None)
        rss = {}
        for sample in self.samples:
            for name, value in sample["rss"].items():
                rss[name] = max(rss.get(name, 0), value)
        return {
            "host_used_peak_bytes": peak["host_used"] if peak else None,
            "host_used_start_bytes": self.samples[0]["host_used"]
            if self.samples
            else None,
            "process_rss_peak_bytes": rss,
            "samples": len(self.samples),
        }


class VideoServer(RealServer):
    """smoke_real's isolated server, serving the video model as well."""

    def __init__(self, arguments, environment):
        package = arguments.package.resolve()
        self.port = available_port()
        self.log = tempfile.NamedTemporaryFile(
            mode="w+", prefix="splash-video-real-", suffix=".log"
        )
        command = [
            sys.executable,
            str(ROOT / "server/server.py"),
            str(package / "target"),
            str(package / "draft"),
            "--tokenizer",
            str(package / "tokenizer"),
            "--model",
            arguments.model,
            "--binary",
            str(arguments.binary.resolve()),
            "--port",
            str(self.port),
            "--video-root",
            str(arguments.video_root.resolve()),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, **environment},
            stdout=self.log,
            stderr=subprocess.STDOUT,
            text=True,
        )


def chat(port, model):
    return call(
        port,
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Say hello in three words."}],
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/splash")
    parser.add_argument("--golden", type=Path, help="hidden states to inject (npz)")
    parser.add_argument(
        "--reference", type=Path, required=True, help="golden final.npz"
    )
    parser.add_argument("--reference-frames", type=Path, help="golden frames.npy")
    parser.add_argument("--seconds", default="3")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    directory = Path(tempfile.mkdtemp(prefix="splash-video-real-", dir=ROOT / "build"))
    environment = {"SPLASH_VIDEO_DIR": str(directory), "SPLASH_VIDEO_KEEP": "1"}
    if arguments.golden:
        environment["SPLASH_VIDEO_HIDDEN_STATES"] = str(arguments.golden)
    server = VideoServer(arguments, environment)
    report = {"golden": bool(arguments.golden), "seconds": arguments.seconds}
    sampler = MemorySampler()
    try:
        started = time.monotonic()
        server.wait_ready(900)
        report["startup_seconds"] = round(time.monotonic() - started, 1)
        port = server.port
        status, models = call(port, "GET", "/v1/models")
        require(
            status == 200
            and any(item.get("output_modalities") for item in models["data"]),
            "the video model is not listed",
        )
        status, reply = chat(port, arguments.model)
        require(status == 200, f"chat before the job failed: {reply}")
        sampler.start()
        status, job = call(
            port,
            "POST",
            "/v1/videos",
            {
                "prompt": GOLDEN_PROMPT,
                "seconds": arguments.seconds,
                "preview": True,
                "seed": 0,
            },
        )
        require(
            status == 200 and job["status"] in ("queued", "in_progress"),
            f"submit failed: {job}",
        )
        job_started = time.monotonic()
        status, refused = chat(port, arguments.model)
        require(
            status == 503 and refused["error"]["code"] == "video_job_running",
            f"generation during the job was not refused: {status} {refused}",
        )
        require(call(port, "GET", "/ready")[0] == 200, "/ready failed during the job")
        require(call(port, "GET", "/health")[0] == 200, "/health failed during the job")
        progress, preview_at = [], None
        while True:
            status, current = call(port, "GET", f"/v1/videos/{job['id']}")
            require(status == 200, f"job lookup failed: {current}")
            elapsed = round(time.monotonic() - job_started, 1)
            if not progress or progress[-1][1:] != [
                current["progress"],
                current["phase"],
            ]:
                progress.append([elapsed, current["progress"], current["phase"]])
            if current["preview"] and preview_at is None:
                code, preview = call(
                    port, "GET", f"/v1/videos/{job['id']}/content?variant=preview"
                )
                require(
                    code == 200 and preview[4:8] == b"ftyp", "the preview is not an mp4"
                )
                preview_at = elapsed
            if current["status"] in ("completed", "failed"):
                break
            time.sleep(2)
        report["job_seconds"] = round(time.monotonic() - job_started, 1)
        report["memory"] = sampler.finish()
        report["job"] = current
        report["progress"] = progress
        report["preview_seconds"] = preview_at
        require(current["status"] == "completed", f"the job failed: {current['error']}")
        values = [entry[1] for entry in progress]
        require(
            values == sorted(values) and len(set(values)) > 5,
            "progress did not advance in steps",
        )
        code, mp4 = call(port, "GET", f"/v1/videos/{job['id']}/content")
        require(code == 200 and mp4[4:8] == b"ftyp", "the video is not an mp4")
        report["mp4_bytes"] = len(mp4)
        (directory / "video.mp4").write_bytes(mp4)
        status, reply = chat(port, arguments.model)
        require(status == 200, f"chat after the job failed: {reply}")

        job_directory = next(directory.glob(f"*/{job['id']}"))
        final = job_directory / "final.npz"
        if arguments.golden:
            same = final.read_bytes() == arguments.reference.read_bytes()
            report["final_identical"] = same
            require(same, "the final latents differ from the golden run")
        reference = np.load(arguments.reference)
        got = np.load(final)
        for name in ("video", "audio"):
            difference = got[name].astype(np.float64) - reference[name]
            report[f"{name}_latent_rel_rms"] = float(
                np.sqrt(
                    (difference**2).sum()
                    / (reference[name].astype(np.float64) ** 2).sum()
                )
            )
        if arguments.reference_frames:
            frames = np.load(arguments.reference_frames, mmap_mode="r")
            produced = np.fromfile(job_directory / "frames.rgb", np.uint8).reshape(
                frames.shape
            )
            values = [psnr(produced[i], frames[i]) for i in range(frames.shape[0])]
            report["frames_identical"] = bool(np.array_equal(produced, frames))
            report["frame_psnr_mean"] = float(
                np.mean([v for v in values if np.isfinite(v)] or [np.inf])
            )
            report["frame_psnr_min"] = float(min(values))
        report["result"] = "pass"
    except SmokeFailure as failure:
        report["result"] = f"fail: {failure}"
        report["server_log"] = server.tail()
    finally:
        if sampler.thread.is_alive():
            report.setdefault("memory", sampler.finish())
        server.close()
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("server_log", "job")},
            default=str,
        )
    )
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
