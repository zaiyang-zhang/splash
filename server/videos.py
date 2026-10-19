"""Video generation: OpenAI-shaped /v1/videos jobs over a video model served
beside the language model (MiniMax-H3, runtime/video).

A job runs its phases in order on one worker thread:

- text: the engine's text encoder turns the prompt into hidden states
  (HiddenStatesRequest), while the language model is still resident;
- the engine hands the GPU over (Exclusive acquire): it drops its cached
  prefixes, KV backing and states and ends its weights' residency;
- `splash video-native`, a process of its own whose exit releases every
  allocation: prep (the DiT inputs: layout, RoPE, adaLN tables, refined
  text, schedule, noise), denoise, where each step advances the progress,
  preview (TAEH3, when asked), decode, audio and mux into an mp4;
- the engine takes the GPU back (Exclusive release).

The GPU is exclusive: from the moment a job is admitted until the last one
finishes, generation requests get 503 video_job_running, and the job starts
once the requests in flight have finished. Health and readiness stay up.
Completed videos are kept in a byte-budgeted store, oldest first evicted, and
DELETE cancels a running job at its next step boundary.
"""

from __future__ import annotations

import json
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

if __package__:
    from .errors import APIError
else:
    from errors import APIError


# OpenAI's `seconds` is a string; MiniMax-H3 renders 3, 10 or 15 s clips at
# 24 fps, 17 n + 5 frames.
CLIP_FRAMES = {"3": 73, "10": 243, "15": 362}
DEFAULT_SECONDS = "3"
VIDEO_SIZE = "1344x768"
VARIANTS = ("turbo", "default")
MAX_PROMPT_CHARACTERS = 32 * 1024
# The hidden states the DiT conditions on (hidden_states[50] of Qwen3-VL-32B).
TEXT_ENCODER_LAYER = 50
# Bytes of finished videos kept for download; the oldest finished job goes
# first. Queued and running jobs are never evicted.
VIDEO_STORE_BUDGET_BYTES = 4 << 30
MAX_QUEUED_JOBS = 4
MAX_LISTED_JOBS = 100
# The share of a job's progress each phase reports when it ends.
PHASE_PROGRESS = {
    "text": 2,
    "prep": 5,
    "denoise": 85,
    "preview": 86,
    "decode": 94,
    "audio": 97,
    "mux": 100,
}
# A cancelled job's process stops at its next step boundary; after this it is
# killed.
CANCEL_GRACE_SECONDS = 120.0
VIDEO_NATIVE_CANCELLED = 75


class VideoCancelled(Exception):
    pass


class VideoFailed(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class VideoRequest:
    prompt: str
    seconds: str = DEFAULT_SECONDS
    variant: str = "turbo"
    seed: int = 0
    preview: bool = False
    audio: bool = True

    @property
    def frames(self):
        return CLIP_FRAMES[self.seconds]


@dataclass
class VideoJob:
    id: str
    model: str
    request: VideoRequest
    directory: Path
    created_at: int = field(default_factory=lambda: int(time.time()))
    status: str = "queued"
    progress: int = 0
    phase: str | None = None
    completed_at: int | None = None
    error: dict | None = None
    # Wall seconds and the peak footprint of the process each phase ran in.
    phases: dict = field(default_factory=dict)
    preview_ready: bool = False
    cancelled: threading.Event = field(default_factory=threading.Event)
    deleted: bool = False

    @property
    def finished(self):
        return self.status in ("completed", "failed")

    def stored_bytes(self):
        total = 0
        for name in ("video.mp4", "preview.mp4"):
            try:
                total += (self.directory / name).stat().st_size
            except OSError:
                pass
        return total

    def public(self):
        """The video object, as OpenAI's API names its fields, with Splash's
        phase, request options and per-phase timings beside them."""
        return {
            "id": self.id,
            "object": "video",
            "model": self.model,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "expires_at": None,
            "size": VIDEO_SIZE,
            "seconds": self.request.seconds,
            "quality": "standard",
            "remixed_from_video_id": None,
            "error": self.error,
            "phase": self.phase,
            "variant": self.request.variant,
            "seed": self.request.seed,
            "preview": self.preview_ready,
            "phases": self.phases,
        }


def _bool(body, name, default):
    value = body.get(name, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise APIError(400, f"{name} must be a boolean")
    return value


def parse_video_request(body, models):
    """A POST /v1/videos body (JSON) as a VideoRequest; `models` are the
    names the video model answers to."""
    if not isinstance(body, dict):
        raise APIError(400, "request body must be an object")
    known = {
        "prompt",
        "model",
        "seconds",
        "size",
        "variant",
        "seed",
        "preview",
        "audio",
    }
    if "input_reference" in body:
        raise APIError(400, "input_reference is not supported", "unsupported_parameter")
    unknown = sorted(set(body) - known)
    if unknown:
        raise APIError(
            400, f"unsupported parameter: {unknown[0]}", "unsupported_parameter"
        )
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise APIError(400, "prompt must be a non-empty string")
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        raise APIError(400, f"prompt is longer than {MAX_PROMPT_CHARACTERS} characters")
    model = body.get("model")
    if model is not None and model not in models:
        raise APIError(404, f"video model {model!r} is not served", "model_not_found")
    seconds = body.get("seconds", DEFAULT_SECONDS)
    if isinstance(seconds, int) and not isinstance(seconds, bool):
        seconds = str(seconds)
    if seconds not in CLIP_FRAMES:
        raise APIError(400, "seconds must be one of " + ", ".join(CLIP_FRAMES))
    size = body.get("size", VIDEO_SIZE)
    if size != VIDEO_SIZE:
        raise APIError(400, f"size must be {VIDEO_SIZE}")
    variant = body.get("variant", "turbo")
    if variant not in VARIANTS:
        raise APIError(400, "variant must be turbo or default")
    seed = body.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        raise APIError(400, "seed must be an integer in [0, 2^31)")
    return VideoRequest(
        prompt,
        seconds,
        variant,
        seed,
        _bool(body, "preview", False),
        _bool(body, "audio", True),
    )


class VideoService:
    """Admits jobs, runs them one at a time on the pipeline, and stores the
    results within a byte budget."""

    def __init__(
        self,
        pipeline,
        model,
        *,
        model_names=(),
        root=None,
        budget_bytes=VIDEO_STORE_BUDGET_BYTES,
        max_queued=MAX_QUEUED_JOBS,
    ):
        self.pipeline = pipeline
        self.model = model
        self.model_names = tuple(dict.fromkeys([model, *model_names]))
        self.budget_bytes = budget_bytes
        self.max_queued = max_queued
        parent = Path(root) if root else None
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="splash-videos-", dir=parent))
        self.jobs = OrderedDict()
        self.queue = []
        self.running = None
        self.lock = threading.Condition()
        self.closing = False
        self.evictions = 0
        self.worker = threading.Thread(
            target=self._run, name="splash-video-jobs", daemon=True
        )
        self.worker.start()

    # -- HTTP surface ------------------------------------------------------

    def holds_gpu(self):
        """Whether a job is queued or running: generation is refused."""
        with self.lock:
            return self.running is not None or bool(self.queue)

    def submit(self, body):
        request = parse_video_request(body, self.model_names)
        with self.lock:
            if self.closing:
                raise APIError(503, "server is shutting down", "server_shutdown")
            if len(self.queue) >= self.max_queued:
                raise APIError(
                    503,
                    f"{self.max_queued} video jobs are already queued",
                    "video_queue_full",
                )
            job_id = "video_" + secrets.token_hex(16)
            directory = self.root / job_id
            directory.mkdir()
            job = VideoJob(job_id, self.model, request, directory)
            self.jobs[job_id] = job
            self.queue.append(job)
            self.lock.notify_all()
            return job.public()

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise APIError(404, "video not found", "not_found_error")
            return job.public()

    def list(self, query):
        try:
            limit = int(query.get("limit", 20))
        except (TypeError, ValueError):
            raise APIError(400, "limit must be an integer") from None
        if not 1 <= limit <= MAX_LISTED_JOBS:
            raise APIError(400, f"limit must be in [1, {MAX_LISTED_JOBS}]")
        order = query.get("order", "desc")
        if order not in ("asc", "desc"):
            raise APIError(400, "order must be asc or desc")
        with self.lock:
            jobs = list(self.jobs.values())
        if order == "desc":
            jobs.reverse()
        after = query.get("after")
        if after is not None:
            ids = [job.id for job in jobs]
            if after not in ids:
                raise APIError(404, "video not found", "not_found_error")
            jobs = jobs[ids.index(after) + 1 :]
        page = jobs[:limit]
        data = [job.public() for job in page]
        return {
            "object": "list",
            "data": data,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
            "has_more": len(jobs) > limit,
        }

    def content(self, job_id, variant):
        """The mp4 of a completed job, or its preview once the preview is
        ready."""
        if variant not in ("video", "preview"):
            raise APIError(
                400, "variant must be video or preview", "unsupported_parameter"
            )
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise APIError(404, "video not found", "not_found_error")
            ready = (
                job.preview_ready if variant == "preview" else job.status == "completed"
            )
            path = job.directory / (
                "preview.mp4" if variant == "preview" else "video.mp4"
            )
            if not ready:
                raise APIError(
                    409 if not job.finished else 404,
                    f"the {variant} of {job_id} is not available"
                    + ("" if job.finished else f" yet (status {job.status})"),
                    "video_not_ready",
                )
            # Read under the lock: a concurrent delete or eviction removes
            # the directory.
            return path.read_bytes()

    def delete(self, job_id):
        with self.lock:
            job = self.jobs.pop(job_id, None)
            if job is None:
                raise APIError(404, "video not found", "not_found_error")
            job.deleted = True
            if job in self.queue:
                self.queue.remove(job)
            running = job is self.running
            if running:
                # The worker removes its directory when the job stops.
                job.cancelled.set()
                self.pipeline.cancel(job)
            else:
                shutil.rmtree(job.directory, ignore_errors=True)
            self.lock.notify_all()
        return {"id": job_id, "object": "video.deleted", "deleted": True}

    def stats(self):
        with self.lock:
            finished = [job for job in self.jobs.values() if job.finished]
            return {
                "model": self.model,
                "queued": len(self.queue),
                "running": self.running.id if self.running else None,
                "stored": len(finished),
                "stored_bytes": sum(job.stored_bytes() for job in finished),
                "budget_bytes": self.budget_bytes,
                "evictions": self.evictions,
            }

    def close(self):
        with self.lock:
            self.closing = True
            for job in self.queue:
                job.cancelled.set()
            self.queue.clear()
            if self.running is not None:
                self.running.cancelled.set()
                self.pipeline.cancel(self.running)
            self.lock.notify_all()
        self.worker.join(timeout=CANCEL_GRACE_SECONDS + 10)
        shutil.rmtree(self.root, ignore_errors=True)

    # -- worker ------------------------------------------------------------

    def _run(self):
        while True:
            with self.lock:
                while not self.queue and not self.closing:
                    self.lock.wait()
                if self.closing:
                    return
                job = self.queue.pop(0)
                self.running = job
                job.status = "in_progress"
            try:
                self._execute(job)
            finally:
                with self.lock:
                    self.running = None
                    if job.deleted:
                        shutil.rmtree(job.directory, ignore_errors=True)
                    self._evict()
                    self.lock.notify_all()

    def _report(self, job, *, phase=None, progress=None, timing=None):
        with self.lock:
            if phase is not None:
                job.phase = phase
            if progress is not None:
                job.progress = max(job.progress, min(100, int(progress)))
            if timing is not None:
                name, values = timing
                job.phases[name] = values

    def _preview_ready(self, job):
        with self.lock:
            job.preview_ready = True

    def _execute(self, job):
        started = time.monotonic()
        try:
            self.pipeline.run(job, self)
        except VideoCancelled:
            self._finish(
                job,
                "failed",
                {"code": "cancelled", "message": "the video job was cancelled"},
            )
        except VideoFailed as error:
            self._finish(job, "failed", {"code": error.code, "message": error.message})
        except Exception as error:
            self._finish(job, "failed", {"code": "video_failed", "message": str(error)})
        else:
            self._finish(job, "completed", None)
        with self.lock:
            job.phases["total"] = {"seconds": round(time.monotonic() - started, 3)}

    def _finish(self, job, status, error):
        with self.lock:
            job.status = status
            job.error = error
            job.phase = None
            job.completed_at = int(time.time())
            if status == "completed":
                job.progress = 100

    def _evict(self):
        finished = [job for job in self.jobs.values() if job.finished]
        total = sum(job.stored_bytes() for job in finished)
        for job in finished:
            if total <= self.budget_bytes:
                break
            total -= job.stored_bytes()
            del self.jobs[job.id]
            shutil.rmtree(job.directory, ignore_errors=True)
            self.evictions += 1


@dataclass(frozen=True)
class VideoInstallation:
    """An h3-video installation (install/videos.py): its record and paths."""

    root: Path
    model: str
    layer: int
    variants: dict
    weights: Path
    tokenizer: Path

    @classmethod
    def read(cls, root):
        root = Path(root)
        record = json.loads((root / "model.json").read_text())
        if record.get("version") != 1 or record.get("kind") != "h3-video":
            raise ValueError(f"{root} is not an h3-video installation")
        return cls(
            root,
            record["model"],
            int(record["text_encoder"]["layer"]),
            {variant["name"]: variant for variant in record["variants"]},
            root / "weights",
            root / "tokenizer",
        )


class H3Pipeline:
    """The phases of a MiniMax-H3 job: the engine's text encoder and memory
    handoff through the native runtime, then `splash video-native`, which
    prepares the request, denoises, decodes the video and the audio and
    writes the mp4 (runtime/video/VideoJob.hpp)."""

    def __init__(
        self,
        installation,
        runtime,
        backend,
        *,
        binary,
        metallib,
        hidden_states_override=None,
        keep_intermediates=False,
    ):
        from transformers import AutoTokenizer

        self.installation = installation
        self.runtime = runtime
        self.backend = backend
        self.binary = str(binary)
        self.metallib = str(metallib)
        # A saved prompt embedding (npz with `hidden`) used instead of the
        # text encoder, for the byte-identity test against the golden run.
        self.hidden_states_override = hidden_states_override
        # Keep the prompt embedding, the prepared inputs, the final latents
        # and the raw frames beside the video (the identity tests read them).
        self.keep_intermediates = keep_intermediates
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(installation.tokenizer), local_files_only=True
        )
        self.processes = {}
        self.process_lock = threading.Lock()

    # -- control -------------------------------------------------------------

    def cancel(self, job):
        with self.process_lock:
            process = self.processes.get(job.id)
        if process is not None and process.poll() is None:
            try:
                process.send_signal(signal.SIGTERM)
            except OSError:
                pass

    def _check(self, job):
        if job.cancelled.is_set():
            raise VideoCancelled()

    # -- phases --------------------------------------------------------------

    def run(self, job, service):
        self._wait_for_generation(job)
        self._check(job)
        directory = job.directory
        handoff = None
        try:
            started = time.monotonic()
            service._report(job, phase="text")
            try:
                prompt = self._text(job, directory)
            except VideoFailed as error:
                # A Mac that cannot hold the encoder beside the language
                # model pages the language model out first.
                if error.code != "insufficient_memory":
                    raise
                handoff = self._acquire(job, service)
                prompt = self._text(job, directory)
            service._report(
                job,
                progress=PHASE_PROGRESS["text"],
                timing=(
                    "text",
                    {"seconds": round(time.monotonic() - started, 3), **prompt},
                ),
            )
            self._check(job)
            if handoff is None:
                handoff = self._acquire(job, service)
            self._generate(job, service, directory)
        finally:
            if handoff is not None:
                restored = self.runtime.exclusive(False)
                service._report(
                    job,
                    timing=(
                        "restore",
                        {
                            "engine_resident_bytes": restored.resident_bytes,
                            "engine_footprint_bytes": restored.footprint_bytes,
                        },
                    ),
                )
            if not self.keep_intermediates:
                for name in ("final.npz", "prompt.npz"):
                    (directory / name).unlink(missing_ok=True)

    def _acquire(self, job, service):
        """The engine's memory handed to the job (Exclusive acquire)."""
        handoff = self.runtime.exclusive(True)
        service._report(
            job,
            timing=(
                "handoff",
                {
                    "released_bytes": handoff.released_bytes,
                    "engine_resident_bytes": handoff.resident_bytes,
                    "engine_footprint_bytes": handoff.footprint_bytes,
                },
            ),
        )
        return handoff

    def _wait_for_generation(self, job):
        # New generation is refused already; the requests in flight finish.
        while not self.backend.idle():
            if job.cancelled.wait(0.2):
                raise VideoCancelled()

    def _text(self, job, directory):
        import numpy as np

        if self.hidden_states_override:
            saved = np.load(self.hidden_states_override)
            hidden = np.ascontiguousarray(saved["hidden"], np.float32)
            ids = saved["ids"].astype(np.int32)
            wall = 0.0
        else:
            ids = np.array(
                self.tokenizer(job.request.prompt, add_special_tokens=False)[
                    "input_ids"
                ],
                np.int32,
            )
            if len(ids) < 2:
                raise VideoFailed(
                    "invalid_prompt", "the prompt needs at least two tokens"
                )
            try:
                event = self.runtime.hidden_states(
                    tuple(int(token) for token in ids), self.installation.layer
                )
            except Exception as error:
                code = getattr(error, "code", b"")
                if code == b"insufficient_memory":
                    raise VideoFailed("insufficient_memory", str(error)) from None
                raise VideoFailed("text_encoder_failed", str(error)) from None
            hidden = np.frombuffer(event.values, np.float32).reshape(
                event.rows, event.width
            )
            wall = event.wall_micros / 1e6
        # Every row of a text-only prompt is text (tag 1).
        np.savez(
            directory / "prompt.npz",
            hidden=hidden,
            tags=np.ones(len(ids), np.int64),
            ids=ids,
        )
        return {"tokens": int(len(ids)), "encoder_seconds": round(wall, 3)}

    def _generate(self, job, service, directory):
        request = job.request
        variant = self.installation.variants[request.variant]
        outputs = ["video"] + (["audio"] if request.audio else [])
        outputs += ["preview"] if request.preview else []
        command = [
            self.binary,
            "video-native",
            str(self.installation.weights),
            self.metallib,
            str(directory / "prompt.npz"),
            str(directory),
            "--variant",
            request.variant,
            "--frames",
            str(request.frames),
            "--seed",
            str(request.seed),
            "--outputs",
            ",".join(outputs),
        ]
        # The DiT streams its block weights from SSD when the installation
        # has the stream image: 15 s fits in ~15 GB instead of ~32 GB, at
        # the same forward time.
        if variant.get("stream"):
            command.append("--stream")
        if self.keep_intermediates:
            command += ["--save-prep", "--frames-rgb"]
        service._report(job, phase="prep")
        with (directory / "stderr.log").open("a") as log:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=log, text=True
            )
        with self.process_lock:
            self.processes[job.id] = process
        if job.cancelled.is_set():
            self.cancel(job)
        peak = 0
        failure = None
        for line in process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            peak = max(peak, int(event.get("peak_footprint_bytes", 0)))
            kind = event.get("event")
            if kind == "step":
                low, high = PHASE_PROGRESS["prep"], PHASE_PROGRESS["denoise"]
                service._report(
                    job, progress=low + (high - low) * event["step"] / event["steps"]
                )
            elif kind == "phase" and event.get("state") == "started":
                service._report(job, phase=event["phase"])
            elif kind == "phase" and event.get("state") == "completed":
                name = event["phase"]
                service._report(
                    job,
                    progress=PHASE_PROGRESS.get(name),
                    timing=(
                        name,
                        {
                            "seconds": event.get("seconds"),
                            "peak_footprint_bytes": event.get("peak_footprint_bytes"),
                        },
                    ),
                )
                if name == "preview":
                    service._preview_ready(job)
            elif kind == "error":
                failure = event.get("message", "")
        process.stdout.close()
        status = process.wait()
        with self.process_lock:
            self.processes.pop(job.id, None)
        if job.cancelled.is_set():
            raise VideoCancelled()
        if failure is not None or status:
            raise VideoFailed(
                "video_engine_failed",
                failure
                or f"video-native exited with status {status}: {self._log_tail(job)}",
            )
        service._report(job, timing=("video_native", {"peak_footprint_bytes": peak}))

    @staticmethod
    def _log_tail(job):
        try:
            lines = (
                (job.directory / "stderr.log").read_text(errors="replace").splitlines()
            )
        except OSError:
            return ""
        return " | ".join(lines[-3:])
