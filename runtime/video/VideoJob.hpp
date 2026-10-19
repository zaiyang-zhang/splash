#pragma once

// One video job's GPU phases, in a process of their own (`splash video-native`):
// the DiT denoising loop, the TAEH3 preview and the video VAE decode of the
// vendored MiniMax-H3 engine (runtime/video/VENDORED), driven through its C ABI
// (video/abi/h3.h). The server's job runner (server/videos.py) starts it once
// the language engine has paged out its memory, reads its events and stops it
// with SIGTERM to cancel: the process exits at the next step boundary and its
// exit releases every Metal allocation at once.
//
//   splash video-native WEIGHTS_ROOT METALLIB PREP_DIR OUT_DIR [options]
//
// WEIGHTS_ROOT holds dit-turbo/, dit-default/, vae/ and preview/ (an h3-video
// installation's weights/); PREP_DIR the request's prepared inputs
// (layout.npz, text_ref.npy, ada_blocks.npy, ada_final.npy, sched.npz,
// noise.npz). OUT_DIR receives final.npz (the final latents), preview.yuv
// (yuv420p frames) and frames.rgb (uint8 RGB frames), as the phases run.
// Options: --variant turbo|default, --chunk N, --radius N, --no-qkfix,
// --qkfix-blocks 45,48,49, --stream, --stream-slots N, --vae-precision w8|f16,
// --phases denoise,preview,decode (any subset, in this order), --steps N, and
// for the tests --preview-rgb (also preview.rgb, the preview's RGB frames).
//
// Events are JSON lines on stdout, each with the process's footprint and its
// peak: "ready", "phase" (started/completed), "step" (denoising progress),
// then "done"; an error exits 1 after an "error" event, cancellation exits
// kVideoCancelledExit after a "cancelled" event. The engine's own log lines
// (H3_LOG) go to stderr.

namespace splash::video {

inline constexpr int kVideoCancelledExit = 75;

// argv[1] is "video-native". Returns the process exit status.
[[nodiscard]] int runVideoJob(int argc, char **argv);

} // namespace splash::video
