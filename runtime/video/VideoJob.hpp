#pragma once

// One video job's GPU phases, in a process of their own (`splash video-native`):
// the request's DiT inputs (prep), the denoising loop, the TAEH3 preview, the
// video and audio VAE decodes and the mp4 mux of the vendored MiniMax-H3
// engine (runtime/video/VENDORED), driven through its C ABI (video/abi/h3.h).
// The server's job runner (server/videos.py) starts it once the language
// engine has handed over its memory, reads its events and stops it with
// SIGTERM to cancel: the process exits at the next step boundary, and its
// exit releases every Metal allocation at once.
//
//   splash video-native WEIGHTS_ROOT METALLIB INPUT OUT_DIR [options]
//
// WEIGHTS_ROOT is an h3-video installation's weights/: dit-turbo/,
// dit-default/, vae/, audio/, preview/, fl2va/ and vdn/. INPUT is the prompt's
// hidden states (an npz holding `hidden`, f32 [L][5120]), which the engine
// prepares itself, or a saved prep directory (layout.npz, text_ref.npy,
// ada_blocks.npy, ada_final.npy, sched.npz, noise.npz). OUT_DIR receives
// final.npz (the final latents), preview.mp4 and video.mp4.
// Options: --variant turbo|default, --frames N (17 n + 5; 73 for 3 s),
// --seed N, --steps N, --chunk N, --radius N, --no-qkfix,
// --qkfix-blocks 45,48,49, --stream, --stream-slots N, --vae-precision w8|f16,
// --outputs latents,preview,video,audio (the default: video and audio), and
// for the tests --save-prep (OUT_DIR/prep), --frames-rgb and --preview-rgb
// (the raw uint8 frames, frames.rgb and preview.rgb).
//
// Events are JSON lines on stdout, each with the process's footprint and its
// peak: "ready", "phase" (started/completed, with seconds), "step"
// (denoising progress), then "done"; an error exits 1 after an "error"
// event, cancellation exits kVideoCancelledExit after a "cancelled" event.
// The engine's own log lines (H3_LOG) go to stderr.

namespace splash::video {

inline constexpr int kVideoCancelledExit = 75;

// argv[1] is "video-native". Returns the process exit status.
[[nodiscard]] int runVideoJob(int argc, char **argv);

} // namespace splash::video
