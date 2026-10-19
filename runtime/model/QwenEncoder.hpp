#pragma once

// The text encoder of a video model (MiniMax-H3): Qwen3-VL-32B's language
// model run as an encoder. H3 conditions on hidden_states[50], the
// unnormalized residual stream after decoder layer 50 of 64, of every prompt
// token, so only the token table and the first 50 layers are prepared and
// loaded; the later layers, the final norm and the head are never read.
//
// The source is the MLX affine 4-bit, group-64 checkpoint
// (mlx-community/Qwen3-VL-32B-Instruct-4bit). Preparation reorders each
// projection into the packed 256-row tiles without requantization, as for an
// MLX target (AffinePreparation), and the layers run the affine prefill
// kernels of the targets: RMS norms with Q4 input sums, fused QKV, gate, up
// with the SiLU gate and residual projections. Only the attention differs:
// Qwen3-VL attends with 64 query and 8 KV heads of 128 dimensions, Q/K RMS
// norms and full rotary embedding, which kernels/prefill/encoder.metal runs
// causally over the prompt in one pass. A text prompt's M-RoPE positions are
// the same on every axis, so the rotation is the plain 1-D one.

#include "metal/MetalBackend.hpp"
#include "model/AffinePreparation.hpp"
#include "model/Model.hpp"
#include "model/PreparedFiles.hpp"
#include "model/WeightStore.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

struct QwenEncoderLayout final {
  static constexpr std::string_view layerMagic = "MDTE0001";

  // Decoder layers of the checkpoint, and those evaluated: the encoder's
  // output is hidden_states[layers].
  uint32_t sourceLayers = 0;
  uint32_t layers = 0;
  uint32_t hiddenSize = 0;
  uint32_t vocabularySize = 0;
  uint32_t intermediateSize = 0;
  uint32_t queryHeads = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;
  float rotaryTheta = 0.0F;
  float normEpsilon = 0.0F;
  // One prefill pass: the affine prefill kernels' row bound.
  uint32_t maximumTokens = ExecutionLimits::prefillTokenBudget;

  [[nodiscard]] constexpr uint32_t attentionWidth() const noexcept { return queryHeads * headDimension; }
  [[nodiscard]] constexpr uint32_t kvWidth() const noexcept { return kvHeads * headDimension; }
  // The fused projection's rows: queries, keys, values.
  [[nodiscard]] constexpr uint32_t qkvWidth() const noexcept { return attentionWidth() + 2 * kvWidth(); }
  bool operator==(const QwenEncoderLayout &) const = default;
};

// Qwen3-VL-32B evaluated through `layers` decoder layers (H3: 50).
[[nodiscard]] QwenEncoderLayout qwen3VlEncoderLayout(uint32_t layers);

// The prepared images of an MLX Qwen3-VL checkpoint: encoder/layer-N.bin for
// the evaluated layers, then encoder/embedding.bin. Planned once; each image
// is written when it is first prepared or opened.
class QwenEncoderLoader final {
public:
  QwenEncoderLoader(metal::MetalBackend &backend, const std::filesystem::path &directory,
                    const QwenEncoderLayout &layout, PreparationCheck admitConversion = {});
  ~QwenEncoderLoader();
  [[nodiscard]] const QwenEncoderLayout &layout() const noexcept;
  // Every image's cache identity and size, layers first.
  [[nodiscard]] std::span<const PreparedWeight> weights() const noexcept;
  // Writes every missing image and maps none.
  void prepare();
  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile embedding();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] uint64_t preparedEncoderBytes(const QwenEncoderLayout &layout);
// The planned image of encoder layer `layer`: its sections at their offsets.
[[nodiscard]] affine::Image encoderLayerImage(const QwenEncoderLayout &layout, uint32_t layer);

struct QwenEncoderLayerWeights final {
  ops::NormWeights inputNorm;
  ops::Projection qkvProjection;
  ops::NormWeights queryNorm;
  ops::NormWeights keyNorm;
  ops::Projection outputProjection;
  ops::NormWeights postAttentionNorm;
  ops::Projection gateProjection;
  ops::Projection upProjection;
  ops::Projection downProjection;
};

struct QwenEncoderWeights final {
  QwenEncoderLayout layout;
  std::vector<QwenEncoderLayerWeights> layers;
  ops::EmbeddingWeights tokenEmbedding;
  std::vector<WeightFileRecord> files;
};

// Maps every prepared image, writing the missing ones first.
[[nodiscard]] QwenEncoderWeights loadQwenEncoderWeights(QwenEncoderLoader &loader);

// Timings of one encode, for the server's job record.
struct EncoderTiming final {
  uint32_t tokens = 0;
  double wallMilliseconds = 0.0;
  double gpuMilliseconds = 0.0;
};

class QwenEncoder final {
public:
  QwenEncoder(metal::MetalBackend &backend, QwenEncoderWeights weights);

  [[nodiscard]] const QwenEncoderLayout &layout() const noexcept { return weights_.layout; }
  // hidden_states[layout().layers] of every token, row-major f32
  // [tokens][hiddenSize] holding bf16 values. Activations are allocated for
  // this prompt through `admit` and released on return.
  [[nodiscard]] std::vector<float> encode(std::span<const uint32_t> tokens,
                                          const metal::AllocationAdmission &admit,
                                          EncoderTiming *timing = nullptr) const;

private:
  metal::MetalBackend &backend_;
  QwenEncoderWeights weights_;
  ops::Linear linear_;
  metal::MetalBuffer inverseFrequencies_;
};

} // namespace splash::model
