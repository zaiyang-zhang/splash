#pragma once

// Parameter layouts of the text encoder's attention kernels
// (kernels/prefill/encoder.metal, model/QwenEncoder.cpp).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The kernels run one simdgroup per head row, four dimensions per lane.
#define SPLASH_ENCODER_HEAD_DIMENSION 128

// Q/K RMS norm and rotary embedding of the fused QKV rows.
struct EncoderRopeParams {
  uint32_t rows;
  uint32_t query_heads;
  uint32_t kv_heads;
  float norm_epsilon;
};

// Causal attention of every prompt row.
struct EncoderAttentionParams {
  uint32_t rows;
  uint32_t query_heads;
  uint32_t kv_heads;
  float scale;
};

static_assert(sizeof(EncoderRopeParams) == 16,
              "encoder rotary parameters are 16 bytes on both sides");
static_assert(sizeof(EncoderAttentionParams) == 16,
              "encoder attention parameters are 16 bytes on both sides");
