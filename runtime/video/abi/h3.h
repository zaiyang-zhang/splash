/*
 * h3: MiniMax-H3 on-device engine (M5 Pro, macOS 27) -- C ABI of build/libh3.dylib.
 *
 * One engine per request. The frontend (python -m h3) prepares the request inputs (text encoder, layout, RoPE,
 * adaLN tables, schedule, noise) and hands them over with the request options; the engine owns the device, the
 * weights, the activations and every kernel choice. Phases:
 *
 *   h3_create            device, kernel library, request geometry
 *   h3_dit_load          DiT weights (resident or SSD-streamed), key-range table, activations; prepared layout
 *   h3_denoise           all steps: forward + Euler update, per-step callback      (or h3_dit_forward: one forward)
 *   h3_dit_release       frees the DiT memory (weights, slots, activations)
 *   h3_decode            video VAE: final latents -> uint8 frames
 *   h3_preview           TAEH3 preview decoder: final latents -> uint8 frames (+ yuv420p)
 *   h3_destroy
 *
 * Every call returns H3_OK or an error code; h3_error() has the message. Inputs are borrowed for the duration of the
 * call (h3_dit_load: until h3_dit_release). The only environment variable the engine reads is H3_LOG (0 quiet,
 * 1 phase summaries (default), 2 memory plan and details).
 */
#ifndef H3_H3_H
#define H3_H3_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3_ABI_VERSION 1
#define H3_MAX_QKFIX_BLOCKS 8

typedef enum h3_status {
  H3_OK = 0,
  H3_ERROR_ARGUMENT = 1, /* request or inputs inconsistent */
  H3_ERROR_IO = 2,       /* file missing, unreadable or malformed (weights, stream image) */
  H3_ERROR_METAL = 3,    /* device, kernel library, pipeline or allocation failure */
  H3_ERROR_GPU = 4,      /* a command buffer failed */
  H3_ERROR_STATE = 5,    /* phase called out of order */
} h3_status;

typedef enum h3_variant { H3_VARIANT_TURBO = 0, H3_VARIANT_DEFAULT = 1 } h3_variant;
typedef enum h3_vae_precision { H3_VAE_W8 = 0, H3_VAE_F16 = 1 } h3_vae_precision;

typedef struct h3_engine h3_engine;

/* Where the engine finds its files. */
typedef struct h3_config {
  const char *metallib; /* build/metal/h3.metallib */
  const char *weights;  /* weights root: dit-turbo/, dit-default/, vae/, preview/ */
} h3_config;

/* User-facing request options. */
typedef struct h3_request {
  h3_variant variant;                         /* turbo: 8 steps, default: 50 (from the schedule) */
  int32_t frames;                             /* 17 n + 5 at 24 fps: 3 s = 73, 10 s = 243, 15 s = 362 */
  int32_t width, height;                      /* 1344 x 768 */
  int32_t window_chunk, window_radius;        /* window softmax: 5 / 1; anchor frames first and last */
  int32_t qkfix;                              /* fp16 Q / K in the QKFIX blocks */
  int32_t qkfix_blocks[H3_MAX_QKFIX_BLOCKS];  /* 45, 48, 49 */
  int32_t n_qkfix_blocks;
  int32_t stream;                             /* SSD weight streaming */
  int32_t stream_slots;                       /* block slots in the ring (8) */
  const char *stream_image;                   /* NULL: <weights>/dit-<variant>/stream.bin */
  h3_vae_precision vae_precision;
} h3_request;

/* Shapes that follow from the request (no engine needed). */
typedef struct h3_geometry {
  int32_t frames, width, height;
  int32_t latent_frames, latent_height, latent_width;
  int32_t video_rows, audio_rows; /* DiT latent rows: [video_rows][96], [audio_rows][32] */
  int32_t decode_frames;          /* frames of h3_decode */
  int32_t preview_frames;         /* frames of h3_preview */
} h3_geometry;

/* Prepared layout of the packed DiT sequence [text L | audio | video] (python -m h3 prep). */
typedef struct h3_dit_inputs {
  int32_t text_rows;     /* L */
  const uint8_t *tags;   /* [M] modality: 0 video, 1 text, 2 audio */
  const uint16_t *rope;  /* bf16 [M][96]: cos 48 | sin 48 */
  const uint16_t *text;  /* bf16 [L][5376] refined text */
} h3_dit_inputs;

/* Denoising schedule (rectified-flow Euler, eta 0, one forward per step). */
typedef struct h3_schedule {
  int32_t steps;
  const uint16_t *ada_blocks;               /* bf16 [steps][50][3][6][5376] */
  const uint16_t *ada_final;                /* bf16 [steps][3][2][5376] */
  const float *video_t, *audio_t;           /* [steps] */
  const float *video_sigmas, *audio_sigmas; /* [steps + 1] */
} h3_schedule;

typedef struct h3_forward_stats {
  double wall_ms, embed_ms, blocks_span_ms, blocks_gpu_ms, final_ms;
  double linear_ms, attention_ms, other_ms, branch_ms; /* GPU time by step class */
  double gpu_busy_ms, gpu_idle_ms, gpu_gap_max_ms;
  int32_t gpu_gaps, command_buffers;
  double stream_wait_ms, stream_wait_max_ms, stream_read_gb, stream_read_s, stream_read_max_ms;
  int32_t stream_waits, stream_blocks;
  double device_mb;
} h3_forward_stats;

typedef struct h3_dit_info {
  int32_t rows, padded_rows, text_rows, audio_rows, video_rows;
  int32_t window_items, chunked_groups, partial_slots;
  double density;           /* allowed (q, k) pairs / rows^2 */
  int32_t stream_slots;     /* 0: weights resident */
  double weights_mb;        /* resident weights + stream slots */
  double activations_mb;    /* MemoryPlan buffers */
  double device_mb;
} h3_dit_info;

typedef struct h3_decode_stats {
  double wall_s, gpu_s, host_copy_s, upload_s, gpu_span_s;
  int64_t nonfinite;
  double device_mb;
} h3_decode_stats;

typedef struct h3_preview_stats {
  double wall_s, gpu_s, host_copy_s;
  int64_t nonfinite;
  int32_t chunks, frames;
  double device_mb;
} h3_preview_stats;

/* Called after every denoising step with the updated latents ([video_rows][96], [audio_rows][32]). */
typedef void (*h3_step_callback)(void *user, int32_t step, const h3_forward_stats *stats, const float *video,
                                 const float *audio);

h3_request h3_default_request(void);
h3_status h3_geometry_of(const h3_request *request, h3_geometry *out);
/* Key-range table of the window softmax for this request and text length: int32 words; returns the word count
 * (call with capacity 0 to size the buffer), or -1 on a bad request. */
int64_t h3_window_table(const h3_request *request, int32_t text_rows, int32_t *words, int64_t capacity);

h3_status h3_create(const h3_config *config, const h3_request *request, h3_engine **out);
void h3_destroy(h3_engine *engine);
const char *h3_error(const h3_engine *engine);

h3_status h3_dit_load(h3_engine *engine, const h3_dit_inputs *inputs);
h3_status h3_dit_info_of(const h3_engine *engine, h3_dit_info *out);
h3_status h3_dit_forward(h3_engine *engine, const uint16_t *ada_blocks, const uint16_t *ada_final,
                         const float *video_in, const float *audio_in, float *video_out, float *audio_out,
                         h3_forward_stats *stats);
h3_status h3_denoise(h3_engine *engine, const h3_schedule *schedule, const float *video_noise,
                     const float *audio_noise, float *video_out, float *audio_out, h3_step_callback callback,
                     void *user);
void h3_dit_release(h3_engine *engine);

h3_status h3_decode(h3_engine *engine, const float *video_latents, uint8_t *frames, h3_decode_stats *stats);
h3_status h3_preview(h3_engine *engine, const float *video_latents, uint8_t *frames, uint8_t *yuv420,
                     h3_preview_stats *stats);

#ifdef __cplusplus
}
#endif

#endif /* H3_H3_H */
