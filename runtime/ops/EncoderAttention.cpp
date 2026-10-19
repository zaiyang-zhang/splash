#include "ops/EncoderAttention.hpp"

#include "metal/abi/Encoder.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace splash::ops {
namespace {

void requireShape(const EncoderAttentionShape &shape) {
  if (!shape.rows || !shape.queryHeads || !shape.kvHeads || shape.queryHeads % shape.kvHeads)
    throw std::invalid_argument("invalid text encoder attention shape");
}

// One simdgroup per (row, head).
constexpr metal::DispatchSize kSimdgroup{32, 1, 1};

} // namespace

void EncoderAttention::addNormRope(metal::CommandGraph &graph, metal::MetalBuffer qkv,
                                   metal::MetalBuffer queryNorm, metal::MetalBuffer keyNorm,
                                   metal::MetalBuffer inverseFrequencies, metal::MetalBuffer queries,
                                   metal::MetalBuffer keys, metal::MetalBuffer values,
                                   EncoderAttentionShape shape, float normEpsilon) {
  requireShape(shape);
  const uint32_t heads = shape.queryHeads + 2 * shape.kvHeads;
  const uint64_t headBytes = uint64_t{shape.rows} * SPLASH_ENCODER_HEAD_DIMENSION * 2;
  if (qkv.sizeBytes() < headBytes * heads || queries.sizeBytes() < headBytes * shape.queryHeads ||
      keys.sizeBytes() < headBytes * shape.kvHeads || values.sizeBytes() < headBytes * shape.kvHeads ||
      inverseFrequencies.sizeBytes() < SPLASH_ENCODER_HEAD_DIMENSION / 2 * sizeof(float))
    throw std::invalid_argument("text encoder rotary buffers are below the shape");
  graph.add("encoder_qk_norm_rope",
            {std::move(qkv), std::move(queryNorm), std::move(keyNorm), std::move(inverseFrequencies),
             std::move(queries), std::move(keys), std::move(values)},
            EncoderRopeParams{shape.rows, shape.queryHeads, shape.kvHeads, normEpsilon},
            {shape.rows, heads, 1}, kSimdgroup);
}

void EncoderAttention::add(metal::CommandGraph &graph, metal::MetalBuffer queries, metal::MetalBuffer keys,
                           metal::MetalBuffer values, metal::MetalBuffer output, EncoderAttentionShape shape) {
  requireShape(shape);
  const uint64_t headBytes = uint64_t{shape.rows} * SPLASH_ENCODER_HEAD_DIMENSION * 2;
  if (queries.sizeBytes() < headBytes * shape.queryHeads || output.sizeBytes() < headBytes * shape.queryHeads ||
      keys.sizeBytes() < headBytes * shape.kvHeads || values.sizeBytes() < headBytes * shape.kvHeads)
    throw std::invalid_argument("text encoder attention buffers are below the shape");
  graph.add("encoder_attention", {std::move(queries), std::move(keys), std::move(values), std::move(output)},
            EncoderAttentionParams{shape.rows, shape.queryHeads, shape.kvHeads,
                                   1.0F / std::sqrt(float(SPLASH_ENCODER_HEAD_DIMENSION))},
            {shape.rows, shape.queryHeads, 1}, kSimdgroup);
}

} // namespace splash::ops
