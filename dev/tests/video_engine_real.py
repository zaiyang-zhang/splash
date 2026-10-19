#!/usr/bin/env python3

"""The vendored video engine against its references (make test-video-real).

Runs `splash video-native` on a request's saved prepared inputs, the 3 s
golden request of the h3 engine's own suite, with the installed weights:
resident, then streamed from SSD with 8 block slots. Both runs' final
latents, the W8 video VAE frames and the TAEH3 preview frames must equal the
references byte for byte (runtime/video/VENDORED), as h3's smoke, VAE and
preview checks require. A cancelled run must exit at a step boundary with
its own status. Each run's wall time and peak footprint go to --output.

  dev/tests/video_engine_real.py --weights VIDEO_ROOT/weights \\
      --references DIR --output build/release/video-engine.json

DIR holds prep/ (layout.npz, text_ref.npy, ada_blocks.npy, ada_final.npy,
sched.npz, noise.npz), final.npz, frames.npy and preview_frames.npy.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CANCELLED = 75


def run(binary, metallib, weights, prep, out, *options):
    command = [
        str(binary),
        "video-native",
        str(weights),
        str(metallib),
        str(prep),
        str(out),
        *options,
    ]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True)
    events = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    if result.returncode:
        raise AssertionError(
            f"video-native failed ({result.returncode}): {events[-1:]}\n{result.stderr[-2000:]}"
        )
    return {
        "seconds": round(time.monotonic() - started, 1),
        "peak_footprint_bytes": max(e.get("peak_footprint_bytes", 0) for e in events),
        "phases": {
            e["phase"]: e["seconds"]
            for e in events
            if e.get("event") == "phase" and e.get("state") == "completed"
        },
    }


def identical(path, reference):
    if reference.suffix == ".npy":
        expected = np.load(reference, mmap_mode="r")
        produced = np.fromfile(path, np.uint8)
        return produced.size == expected.size and np.array_equal(
            produced, np.asarray(expected).reshape(-1)
        )
    return path.read_bytes() == reference.read_bytes()


def cancelled_run(binary, metallib, weights, prep, out):
    """SIGTERM during denoising: the process reports the step it stopped
    after and exits with the cancelled status."""
    process = subprocess.Popen(
        [str(binary), "video-native", str(weights), str(metallib), str(prep), str(out),
         "--phases", "denoise"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )  # fmt: skip
    events = []
    for line in process.stdout:
        events.append(json.loads(line))
        if events[-1].get("event") == "step":
            process.send_signal(signal.SIGTERM)
    status = process.wait()
    last = events[-1] if events else {}
    if status != CANCELLED or last.get("event") != "cancelled":
        raise AssertionError(
            f"cancel did not stop at a step boundary: status {status}, {last}"
        )
    return {"status": status, "stopped_after_step": last.get("step")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/splash")
    parser.add_argument("--metallib", type=Path, default=ROOT / "build/splash.metallib")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    references = arguments.references
    prep = references / "prep"
    report = {}
    with tempfile.TemporaryDirectory(
        prefix="splash-video-engine-", dir=ROOT / "build"
    ) as directory:
        for label, options in (
            ("resident", ["--phases", "denoise,preview,decode", "--preview-rgb"]),
            ("streamed", ["--phases", "denoise", "--stream", "--stream-slots", "8"]),
        ):
            out = Path(directory) / label
            report[label] = run(
                arguments.binary,
                arguments.metallib,
                arguments.weights,
                prep,
                out,
                *options,
            )
            checks = {"final.npz": references / "final.npz"}
            if label == "resident":
                checks |= {
                    "frames.rgb": references / "frames.npy",
                    "preview.rgb": references / "preview_frames.npy",
                }
            for name, reference in checks.items():
                same = identical(out / name, reference)
                report[label][f"{name} identical"] = same
                if not same:
                    raise AssertionError(f"{label}: {name} differs from {reference}")
        report["cancel"] = cancelled_run(
            arguments.binary,
            arguments.metallib,
            arguments.weights,
            prep,
            Path(directory) / "cancel",
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))
    print("video engine references: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
