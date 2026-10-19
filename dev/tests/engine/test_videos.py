import json
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dev.tests.test_server import FakeRuntime, Harness
from server import videos

MODEL = "MiniMaxAI/MiniMax-H3"
MP4 = b"\x00\x00\x00\x18ftypmp42" + bytes(32)


class StubPipeline:
    """The phases of a job without a GPU: it reports progress, waits for the
    test to release it, and writes the preview and the video."""

    def __init__(self, hold=True):
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self.started = threading.Event()
        self.cancelled = []

    def cancel(self, job):
        self.cancelled.append(job.id)
        self.release.set()

    def run(self, job, service):
        service._report(job, phase="text", progress=videos.PHASE_PROGRESS["text"])
        self.started.set()
        self.release.wait()
        if job.cancelled.is_set():
            raise videos.VideoCancelled()
        (job.directory / "preview.mp4").write_bytes(b"preview")
        service._preview_ready(job)
        for step in range(1, 9):
            service._report(job, phase="denoise", progress=5 + 75 * step / 8)
        (job.directory / "video.mp4").write_bytes(MP4)
        service._report(
            job, phase="mux", progress=100, timing=("mux", {"seconds": 0.1})
        )


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


class VideoRequestTests(unittest.TestCase):
    def test_defaults_and_openai_field_types(self):
        request = videos.parse_video_request({"prompt": "a dog"}, (MODEL,))
        self.assertEqual(
            (request.seconds, request.frames, request.variant), ("3", 73, "turbo")
        )
        request = videos.parse_video_request(
            {"prompt": "a dog", "seconds": 15, "model": MODEL, "size": "1344x768"},
            (MODEL,),
        )
        self.assertEqual((request.seconds, request.frames), ("15", 362))

    def test_rejections_name_the_field(self):
        cases = (
            ({}, 400, "prompt"),
            ({"prompt": " "}, 400, "prompt"),
            ({"prompt": "x", "seconds": "4"}, 400, "seconds"),
            ({"prompt": "x", "size": "1280x720"}, 400, "size"),
            ({"prompt": "x", "model": "sora-2"}, 404, "video model"),
            ({"prompt": "x", "input_reference": "a.png"}, 400, "input_reference"),
            ({"prompt": "x", "variant": "fast"}, 400, "variant"),
            ({"prompt": "x", "seed": -1}, 400, "seed"),
            ({"prompt": "x", "preview": "yes"}, 400, "preview"),
            ({"prompt": "x", "quality": "high"}, 400, "quality"),
        )
        for body, status, field in cases:
            with self.subTest(body=body):
                with self.assertRaises(videos.APIError) as raised:
                    videos.parse_video_request(body, (MODEL,))
                self.assertEqual(raised.exception.status, status)
                self.assertIn(field, raised.exception.message)


class VideoServerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.harness = Harness(FakeRuntime())
        self.services = []

    def tearDown(self):
        for service in self.services:
            service.pipeline.release.set()
            service.close()
        self.harness.close()
        self.directory.cleanup()

    def serve(self, pipeline, **options):
        service = videos.VideoService(
            pipeline, MODEL, root=self.directory.name, **options
        )
        self.services.append(service)
        self.harness.app.videos = service
        return service

    def json(self, method, path, body=None):
        status, content_type, payload = self.harness.request(method, path, body)
        return (
            status,
            content_type,
            json.loads(payload) if content_type == "application/json" else payload,
        )

    def test_job_lifecycle_holds_the_gpu_and_serves_the_mp4(self):
        pipeline = StubPipeline()
        service = self.serve(pipeline)
        status, _, video = self.json(
            "POST", "/v1/videos", {"prompt": "a dog", "preview": True}
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            {
                key: video[key]
                for key in ("object", "model", "seconds", "size", "quality")
            },
            {
                "object": "video",
                "model": MODEL,
                "seconds": "3",
                "size": "1344x768",
                "quality": "standard",
            },
        )
        self.assertIn(video["status"], ("queued", "in_progress"))
        self.assertTrue(video["id"].startswith("video_"))
        pipeline.started.wait(5)
        _, _, running = self.json("GET", f"/v1/videos/{video['id']}")
        self.assertEqual((running["status"], running["phase"]), ("in_progress", "text"))
        self.assertEqual(running["progress"], videos.PHASE_PROGRESS["text"])

        # Generation is refused while the job holds the GPU; health stays up.
        status, _, error = self.json(
            "POST",
            "/v1/chat/completions",
            {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual((status, error["error"]["code"]), (503, "video_job_running"))
        self.assertEqual(self.json("GET", "/ready")[0], 200)
        self.assertEqual(self.json("GET", "/health")[0], 200)
        status, _, error = self.json("GET", f"/v1/videos/{video['id']}/content")
        self.assertEqual((status, error["error"]["code"]), (409, "video_not_ready"))

        pipeline.release.set()
        wait_for(lambda: service.get(video["id"])["status"] == "completed")
        self.assertFalse(service.holds_gpu())
        _, _, done = self.json("GET", f"/v1/videos/{video['id']}")
        self.assertEqual(
            (done["progress"], done["preview"], done["error"]), (100, True, None)
        )
        self.assertIsInstance(done["completed_at"], int)
        status, content_type, mp4 = self.harness.request(
            "GET", f"/v1/videos/{video['id']}/content"
        )
        self.assertEqual((status, content_type, mp4), (200, "video/mp4", MP4))
        status, _, preview = self.harness.request(
            "GET", f"/v1/videos/{video['id']}/content?variant=preview"
        )
        self.assertEqual((status, preview), (200, b"preview"))
        self.assertEqual(
            self.json("GET", f"/v1/videos/{video['id']}/content?variant=thumbnail")[0],
            400,
        )
        _, _, listed = self.json("GET", "/v1/videos?limit=1")
        self.assertEqual(
            (
                listed["object"],
                [item["id"] for item in listed["data"]],
                listed["has_more"],
            ),
            ("list", [video["id"]], False),
        )
        _, _, models = self.json("GET", "/v1/models")
        entry = next(item for item in models["data"] if item["id"] == MODEL)
        self.assertEqual(entry["output_modalities"], ["video", "audio"])
        _, _, status_json = self.json("GET", "/status")
        self.assertEqual(status_json["videos"]["stored"], 1)

        status, _, deleted = self.json("DELETE", f"/v1/videos/{video['id']}")
        self.assertEqual(
            (status, deleted),
            (200, {"id": video["id"], "object": "video.deleted", "deleted": True}),
        )
        self.assertEqual(self.json("GET", f"/v1/videos/{video['id']}")[0], 404)
        self.assertFalse(any(Path(service.root).iterdir()))

    def test_delete_cancels_a_running_job_and_frees_the_gpu(self):
        pipeline = StubPipeline()
        service = self.serve(pipeline)
        _, _, video = self.json("POST", "/v1/videos", {"prompt": "a dog"})
        pipeline.started.wait(5)
        status, _, _ = self.json("DELETE", f"/v1/videos/{video['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(pipeline.cancelled, [video["id"]])
        wait_for(lambda: not service.holds_gpu())
        wait_for(lambda: not any(Path(service.root).iterdir()))
        self.assertEqual(self.json("GET", f"/v1/videos/{video['id']}")[0], 404)

    def test_queue_bound_and_store_budget(self):
        pipeline = StubPipeline()
        # One finished job's video and preview fit the budget; two do not.
        service = self.serve(pipeline, max_queued=1, budget_bytes=len(MP4) + 16)
        first = self.json("POST", "/v1/videos", {"prompt": "one"})[2]
        pipeline.started.wait(5)
        second = self.json("POST", "/v1/videos", {"prompt": "two"})[2]
        status, _, error = self.json("POST", "/v1/videos", {"prompt": "three"})
        self.assertEqual((status, error["error"]["code"]), (503, "video_queue_full"))
        pipeline.release.set()
        wait_for(
            lambda: (
                service.stats()["queued"] == 0 and service.stats()["running"] is None
            )
        )
        # Two finished videos exceed the budget: the older one is evicted.
        self.assertEqual(self.json("GET", f"/v1/videos/{first['id']}")[0], 404)
        self.assertEqual(
            self.json("GET", f"/v1/videos/{second['id']}")[2]["status"], "completed"
        )
        self.assertEqual(service.stats()["evictions"], 1)

    def test_without_a_video_model_the_routes_are_absent(self):
        self.assertEqual(self.json("GET", "/v1/videos")[0], 404)
        self.assertEqual(self.json("POST", "/v1/videos", {"prompt": "x"})[0], 404)


FAKE_VIDEO_NATIVE = """#!{python}
import json, sys, time
out = sys.argv[5]
phases = sys.argv[sys.argv.index("--phases") + 1].split(",")
def emit(**event):
    event.update(footprint_bytes=1, peak_footprint_bytes=5 << 30)
    print(json.dumps(event), flush=True)
emit(event="ready", frames=73, width=1344, height=768, text_rows=32)
emit(event="phase", phase="denoise", state="started")
for step in range(1, 9):
    emit(event="step", phase="denoise", step=step, steps=8, wall_ms=1.0, device_mb=1.0)
emit(event="phase", phase="denoise", state="completed", seconds=0.1, steps=8)
if "preview" in phases:
    open(out + "/preview.yuv", "wb").write(b"\\0" * 16)
    emit(event="phase", phase="preview", state="completed", seconds=0.1, frames=73)
if "{fail}" == "yes":
    emit(event="error", message="the video VAE produced non-finite values")
    sys.exit(1)
open(out + "/frames.rgb", "wb").write(b"\\0" * 16)
emit(event="phase", phase="decode", state="completed", seconds=0.1, frames=73)
emit(event="done")
"""


class H3PipelineTests(unittest.TestCase):
    """The GPU phases' events as the pipeline reads them, with a stand-in
    for `splash video-native`."""

    def pipeline(self, directory, fail=False):
        binary = Path(directory) / "video-native"
        binary.write_text(
            FAKE_VIDEO_NATIVE.format(
                python=sys.executable, fail="yes" if fail else "no"
            )
        )
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
        installation = SimpleNamespace(
            weights=Path(directory),
            tokenizer=Path(directory),
            layer=50,
            variants={"turbo": {"name": "turbo", "steps": 8, "stream": True}},
        )
        with mock.patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=None
        ):
            pipeline = videos.H3Pipeline(
                installation, None, None, binary=sys.executable, metallib="unused"
            )
        pipeline.binary = str(binary)
        return pipeline

    def run_generate(self, fail=False, preview=True):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.pipeline(directory, fail)
            pipeline._encode_preview = lambda job, out: (
                out / "preview.mp4"
            ).write_bytes(b"p")
            job = videos.VideoJob(
                "video_1",
                MODEL,
                videos.VideoRequest("x", preview=preview),
                Path(directory),
            )
            reports = []
            service = SimpleNamespace(
                _report=lambda job, **report: reports.append(report),
                _preview_ready=lambda job: reports.append({"preview": True}),
            )
            try:
                pipeline._generate(job, service, Path(directory))
                return reports, None
            except videos.VideoFailed as error:
                return reports, error

    def test_steps_advance_progress_and_the_preview_is_published_early(self):
        reports, error = self.run_generate()
        self.assertIsNone(error)
        progress = [r["progress"] for r in reports if "progress" in r]
        self.assertEqual(progress, sorted(progress))
        self.assertEqual(progress[-1], videos.PHASE_PROGRESS["decode"])
        names = [r["timing"][0] for r in reports if "timing" in r]
        self.assertEqual(names, ["denoise", "preview", "decode", "video_native"])
        self.assertIn({"preview": True}, reports)
        native = next(
            r["timing"][1]
            for r in reports
            if r.get("timing", ("",))[0] == "video_native"
        )
        self.assertEqual(native["peak_footprint_bytes"], 5 << 30)

    def test_an_engine_error_fails_the_job_with_its_message(self):
        _, error = self.run_generate(fail=True)
        self.assertEqual(error.code, "video_engine_failed")
        self.assertIn("non-finite", error.message)


if __name__ == "__main__":
    unittest.main()
