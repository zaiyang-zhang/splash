#pragma once

#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::ops {

// The text encoder's attention (model/QwenEncoder.hpp,
// kernels/prefill/encoder.metal): Qwen3-VL's Q/K RMS norms and rotary
// embedding of the fused QKV rows, and causal GQA attention over one prompt.
// Heads have SPLASH_ENCODER_HEAD_DIMENSION dimensions.
struct EncoderAttentionShape final {
  uint32_t rows = 0;
  uint32_t queryHeads = 0;
  uint32_t kvHeads = 0;
};

class EncoderAttention final {
public:
  // qkv [rows][queries | keys | values] -> head-major queries
  // [query head][row][d] and keys and values [kv head][row][d]; queries and
  // keys RMS normalized with their weights and rotated by the row's
  // position at inverseFrequencies[d / 2].
  static void addNormRope(metal::CommandGraph &graph, metal::MetalBuffer qkv,
                          metal::MetalBuffer queryNorm, metal::MetalBuffer keyNorm,
                          metal::MetalBuffer inverseFrequencies, metal::MetalBuffer queries,
                          metal::MetalBuffer keys, metal::MetalBuffer values,
                          EncoderAttentionShape shape, float normEpsilon);
  // output [rows][query head][d]: causal softmax(q k^T / sqrt(d)) v.
  static void add(metal::CommandGraph &graph, metal::MetalBuffer queries, metal::MetalBuffer keys,
                  metal::MetalBuffer values, metal::MetalBuffer output, EncoderAttentionShape shape);
};

} // namespace splash::ops
