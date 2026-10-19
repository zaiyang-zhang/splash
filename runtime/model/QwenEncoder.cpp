#include "model/QwenEncoder.hpp"

#include "metal/CommandGraph.hpp"
#include "metal/abi/Encoder.h"
#include "model/AffinePlan.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightLayout.hpp"
#include "ops/Embedding.hpp"
#include "ops/EncoderAttention.hpp"

#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace splash::model {
namespace {

using affine::copy;
using affine::image;
using affine::Image;
using affine::projection;

// The rows the affine prefill kernels tile: a projection's input and output
// buffers hold whole 32-row tiles.
constexpr uint32_t kPrefillTileRows = 32;
constexpr std::string_view kSourcePrefix = "language_model.model.";

void validateConfiguration(const SafetensorsCheckpoint &source, const QwenEncoderLayout &layout) {
  const std::pair<const char *, double> fields[] = {
      {"num_hidden_layers", layout.sourceLayers}, {"hidden_size", layout.hiddenSize},
      {"vocab_size", layout.vocabularySize}, {"intermediate_size", layout.intermediateSize},
      {"num_attention_heads", layout.queryHeads}, {"num_key_value_heads", layout.kvHeads},
      {"head_dim", layout.headDimension}, {"rope_theta", layout.rotaryTheta},
      {"attention_bias", 0}, {"tie_word_embeddings", 0}};
  for (const auto &[key, value] : fields) source.requireConfigNumber(key, value);
  // The norms run with the layout's float epsilon, 1e-6 as the checkpoint
  // states it in double.
  source.requireConfigNumber("rms_norm_eps", 1e-6);
  if (layout.normEpsilon != 1e-6F) throw WeightStoreError("text encoder norm epsilon is not 1e-6");
  source.requireConfigString("model_type", "qwen3_vl_text");
  source.requireConfigString("hidden_act", "silu");
  source.requireConfigString("rope_scaling.rope_type", "default");
}

Image layerImage(const QwenEncoderLayout &layout, uint32_t layer) {
  Image result = image("layer-" + std::to_string(layer) + ".bin", QwenEncoderLayout::layerMagic, layer, 0);
  const std::string prefix = std::string(kSourcePrefix) + "layers." + std::to_string(layer) + ".";
  const std::string attention = prefix + "self_attn.";
  copy(result, prefix + "input_layernorm.weight", {layout.hiddenSize});
  projection(result, {{attention + "q_proj", layout.attentionWidth()}, {attention + "k_proj", layout.kvWidth()},
                      {attention + "v_proj", layout.kvWidth()}},
             layout.qkvWidth(), layout.hiddenSize);
  copy(result, attention + "q_norm.weight", {layout.headDimension});
  copy(result, attention + "k_norm.weight", {layout.headDimension});
  projection(result, {{attention + "o_proj", layout.hiddenSize}}, layout.hiddenSize, layout.attentionWidth());
  copy(result, prefix + "post_attention_layernorm.weight", {layout.hiddenSize});
  const std::string mlp = prefix + "mlp.";
  projection(result, {{mlp + "gate_proj", layout.intermediateSize}}, layout.intermediateSize, layout.hiddenSize);
  projection(result, {{mlp + "up_proj", layout.intermediateSize}}, layout.intermediateSize, layout.hiddenSize);
  projection(result, {{mlp + "down_proj", layout.hiddenSize}}, layout.hiddenSize, layout.intermediateSize);
  return result;
}

// The token rows as stored: 4-bit codes, scales and biases.
Image embeddingImage(const QwenEncoderLayout &layout) {
  Image result = image("embedding.bin", kEmbeddingMagic, layout.vocabularySize, layout.hiddenSize);
  const std::string table = std::string(kSourcePrefix) + "embed_tokens";
  result.quantized.emplace_back(table, 4);
  copy(result, table + ".weight", {layout.vocabularySize, layout.hiddenSize / 8}, "U32");
  copy(result, table + ".scales", {layout.vocabularySize, layout.hiddenSize / kQ4GroupElements});
  copy(result, table + ".biases", {layout.vocabularySize, layout.hiddenSize / kQ4GroupElements});
  return result;
}

std::vector<Image> images(const QwenEncoderLayout &layout) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) result.push_back(layerImage(layout, layer));
  result.push_back(embeddingImage(layout));
  return result;
}

void requireLayout(const QwenEncoderLayout &layout) {
  if (!layout.layers || layout.layers >= layout.sourceLayers || !layout.hiddenSize || !layout.vocabularySize ||
      !layout.intermediateSize || !layout.queryHeads || !layout.kvHeads || layout.queryHeads % layout.kvHeads ||
      layout.headDimension != SPLASH_ENCODER_HEAD_DIMENSION || !(layout.rotaryTheta > 0.0F) ||
      !(layout.normEpsilon > 0.0F) || !layout.maximumTokens ||
      layout.maximumTokens > ExecutionLimits::prefillTokenBudget)
    throw WeightStoreError("text encoder layout is inconsistent");
  validateQ4Layout(layout.qkvWidth(), layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth());
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
}

uint32_t roundUp(uint32_t value, uint32_t multiple) { return (value + multiple - 1) / multiple * multiple; }

} // namespace

QwenEncoderLayout qwen3VlEncoderLayout(uint32_t layers) {
  QwenEncoderLayout layout;
  layout.sourceLayers = 64;
  layout.layers = layers;
  layout.hiddenSize = 5120;
  layout.vocabularySize = 151936;
  layout.intermediateSize = 25600;
  layout.queryHeads = 64;
  layout.kvHeads = 8;
  layout.headDimension = 128;
  layout.rotaryTheta = 5000000.0F;
  layout.normEpsilon = 1e-6F;
  return layout;
}

struct QwenEncoderLoader::Impl {
  metal::MetalBackend &backend;
  QwenEncoderLayout layout;
  SafetensorsCheckpoint source;
  std::vector<Image> images; // layers, embedding
  std::vector<PreparedWeight> weights;
  PreparedFiles files;
  Impl(metal::MetalBackend &backend, const std::filesystem::path &directory, const QwenEncoderLayout &layout,
       PreparationCheck admitConversion)
      : backend(backend), layout(layout), source(directory, [&backend] { backend.checkOperation(); }),
        files([&backend] { backend.checkOperation(); }, std::move(admitConversion),
              [this] { source.checkUnchanged(); }) {
    requireLayout(layout);
    validateConfiguration(source, layout);
    images = model::images(layout);
    for (Image &image : images) {
      backend.checkOperation();
      affine::bind(image, source);
      weights.push_back(affine::affineImageWeight(image, "encoder", directory.string()));
    }
  }
  WeightFile open(size_t index) {
    const Image &image = images[index];
    return files.open(backend, weights[index], affine::affineImageWriter(image), image.magic, image.layer,
                      image.type);
  }
};

QwenEncoderLoader::QwenEncoderLoader(metal::MetalBackend &backend, const std::filesystem::path &directory,
                                     const QwenEncoderLayout &layout, PreparationCheck admitConversion)
    : impl_(std::make_unique<Impl>(backend, directory, layout, std::move(admitConversion))) {}
QwenEncoderLoader::~QwenEncoderLoader() = default;
const QwenEncoderLayout &QwenEncoderLoader::layout() const noexcept { return impl_->layout; }
std::span<const PreparedWeight> QwenEncoderLoader::weights() const noexcept { return impl_->weights; }
void QwenEncoderLoader::prepare() {
  PreparedWeights().requireSpace(impl_->weights, [this] { impl_->backend.checkOperation(); });
  for (size_t index = 0; index < impl_->images.size(); ++index)
    static_cast<void>(impl_->files.prepare(impl_->weights[index], affine::affineImageWriter(impl_->images[index])));
}
WeightFile QwenEncoderLoader::layer(uint32_t index) {
  if (index >= impl_->layout.layers) throw WeightStoreError("text encoder layer is out of range");
  return impl_->open(index);
}
WeightFile QwenEncoderLoader::embedding() { return impl_->open(impl_->images.size() - 1); }

uint64_t preparedEncoderBytes(const QwenEncoderLayout &layout) {
  uint64_t bytes = 0;
  for (const Image &image : images(layout)) bytes += image.bytes;
  return bytes;
}

affine::Image encoderLayerImage(const QwenEncoderLayout &layout, uint32_t layer) {
  return layerImage(layout, layer);
}

QwenEncoderWeights loadQwenEncoderWeights(QwenEncoderLoader &loader) {
  QwenEncoderWeights result;
  result.layout = loader.layout();
  const QwenEncoderLayout &l = result.layout;
  loader.prepare();
  result.layers.reserve(l.layers);
  for (uint32_t index = 0; index < l.layers; ++index) {
    WeightFile file = loader.layer(index);
    QwenEncoderLayerWeights &layer = result.layers.emplace_back();
    layer.inputNorm = readNorm(file, l.hiddenSize, false, "input-norm");
    layer.qkvProjection = readAffineProjection(file, l.qkvWidth(), l.hiddenSize, "qkv");
    layer.queryNorm = readNorm(file, l.headDimension, false, "query-norm");
    layer.keyNorm = readNorm(file, l.headDimension, false, "key-norm");
    layer.outputProjection = readAffineProjection(file, l.hiddenSize, l.attentionWidth(), "attention-output");
    layer.postAttentionNorm = readNorm(file, l.hiddenSize, false, "post-attention-norm");
    layer.gateProjection = readAffineProjection(file, l.intermediateSize, l.hiddenSize, "gate");
    layer.upProjection = readAffineProjection(file, l.intermediateSize, l.hiddenSize, "up");
    layer.downProjection = readAffineProjection(file, l.hiddenSize, l.intermediateSize, "down");
    file.finish();
    result.files.push_back(file.record());
  }
  WeightFile file = loader.embedding();
  result.tokenEmbedding = readAffineEmbedding(file, l.vocabularySize, l.hiddenSize, "embedding");
  file.finish();
  result.files.push_back(file.record());
  return result;
}

QwenEncoder::QwenEncoder(metal::MetalBackend &backend, QwenEncoderWeights weights)
    : backend_(backend), weights_(std::move(weights)), linear_(backend.capabilities()) {
  requireLayout(weights_.layout);
  if (weights_.layers.size() != weights_.layout.layers)
    throw std::invalid_argument("text encoder weights do not cover its layers");
  // inv_freq[i] = theta^(-2i / d), as the reference computes it in fp32.
  const uint32_t pairs = weights_.layout.headDimension / 2;
  inverseFrequencies_ = backend_.allocateBuffer(uint64_t{pairs} * sizeof(float), metal::BufferStorage::Shared,
                                                "encoder-inverse-frequencies");
  auto *frequencies = static_cast<float *>(inverseFrequencies_.contents());
  for (uint32_t i = 0; i < pairs; ++i)
    frequencies[i] = 1.0F / std::pow(weights_.layout.rotaryTheta,
                                     float(2 * i) / float(weights_.layout.headDimension));
}

std::vector<float> QwenEncoder::encode(std::span<const uint32_t> tokens, const metal::AllocationAdmission &admit,
                                       EncoderTiming *timing) const {
  const QwenEncoderLayout &l = weights_.layout;
  const auto rows = static_cast<uint32_t>(tokens.size());
  if (!rows || rows > l.maximumTokens)
    throw std::invalid_argument("a text encoder prompt holds 1 to " + std::to_string(l.maximumTokens) +
                                " tokens");
  for (const uint32_t token : tokens)
    if (token >= l.vocabularySize) throw std::invalid_argument("prompt token is outside the vocabulary");
  const auto started = std::chrono::steady_clock::now();
  const uint64_t storage = roundUp(rows, kPrefillTileRows);
  const uint64_t widest = std::max(l.hiddenSize, l.attentionWidth());

  struct Allocation {
    const char *label;
    uint64_t bytes;
  };
  const Allocation plan[] = {
      {"encoder-tokens", uint64_t{rows} * sizeof(uint32_t)},
      {"encoder-hidden-a", storage * l.hiddenSize * 2},
      {"encoder-hidden-b", storage * l.hiddenSize * 2},
      {"encoder-normalized", storage * l.hiddenSize * 2},
      {"encoder-sums", storage * (widest / kQ4GroupElements) * sizeof(float)},
      {"encoder-down-sums", storage * (l.intermediateSize / kQ4GroupElements) * sizeof(float)},
      {"encoder-qkv", storage * l.qkvWidth() * 2},
      {"encoder-queries", uint64_t{rows} * l.attentionWidth() * 2},
      {"encoder-keys", uint64_t{rows} * l.kvWidth() * 2},
      {"encoder-values", uint64_t{rows} * l.kvWidth() * 2},
      {"encoder-attention", storage * l.attentionWidth() * 2},
      {"encoder-attention-output", storage * l.hiddenSize * 2},
      {"encoder-gate", storage * l.intermediateSize * 2},
      {"encoder-intermediate", storage * l.intermediateSize * 2},
  };
  std::vector<metal::MetalBuffer> buffers;
  for (const Allocation &allocation : plan) {
    metal::MetalBuffer buffer;
    const metal::AllocationResult admitted = admit(allocation.bytes, [&] {
      buffer = backend_.allocateBuffer(allocation.bytes, metal::BufferStorage::Shared, allocation.label);
    });
    if (!admitted)
      throw metal::MetalAllocationError(std::string("text encoder activations: ") +
                                            metal::allocationFailureName(admitted.failure),
                                        admitted.failure);
    buffers.push_back(std::move(buffer));
  }
  const auto &[tokenIds, hiddenA, hiddenB, normalized, sums, downSums, qkv, queries, keys, values, attention,
               attentionOutput, gate, intermediate] =
      std::tie(buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5], buffers[6], buffers[7],
               buffers[8], buffers[9], buffers[10], buffers[11], buffers[12], buffers[13]);
  std::memcpy(tokenIds.contents(), tokens.data(), tokens.size_bytes());

  double gpuMilliseconds = 0.0;
  const auto submit = [&](const metal::CommandGraph &graph) {
    gpuMilliseconds += backend_.submitCommand(graph.dispatches()).gpuSeconds * 1000.0;
  };
  {
    metal::CommandGraph graph;
    ops::Embedding::add(graph, tokenIds, weights_.tokenEmbedding, hiddenA, rows);
    submit(graph);
  }
  const ops::EncoderAttentionShape shape{rows, l.queryHeads, l.kvHeads};
  for (uint32_t index = 0; index < l.layers; ++index) {
    const QwenEncoderLayerWeights &layer = weights_.layers[index];
    const metal::MetalBuffer &input = index & 1 ? hiddenB : hiddenA;
    const metal::MetalBuffer &output = index & 1 ? hiddenA : hiddenB;
    metal::CommandGraph graph;
    ops::Normalization::addRmsWithQ4Sums(graph, input, layer.inputNorm, normalized, sums, l.hiddenSize, rows);
    linear_.addPrefill(graph, normalized, layer.qkvProjection, qkv, sums, rows);
    ops::EncoderAttention::addNormRope(graph, qkv, layer.queryNorm.buffer, layer.keyNorm.buffer,
                                       inverseFrequencies_, queries, keys, values, shape, l.normEpsilon);
    ops::EncoderAttention::add(graph, queries, keys, values, attention, shape);
    linear_.addPrefillSums(graph, attention, sums, layer.outputProjection, rows);
    linear_.addPrefillResidual(graph, attention, layer.outputProjection, input, attentionOutput, sums, rows);
    ops::Normalization::addRmsWithQ4Sums(graph, attentionOutput, layer.postAttentionNorm, normalized, sums,
                                         l.hiddenSize, rows);
    linear_.addPrefill(graph, normalized, layer.gateProjection, gate, sums, rows);
    linear_.addPrefillUpWithGate(graph, normalized, layer.upProjection, gate, intermediate, sums, downSums, rows);
    linear_.addPrefillResidual(graph, intermediate, layer.downProjection, attentionOutput, output, downSums, rows);
    submit(graph);
  }
  const metal::MetalBuffer &result = l.layers & 1 ? hiddenB : hiddenA;
  const auto *bf16 = static_cast<const uint16_t *>(result.contents());
  std::vector<float> hidden(size_t{rows} * l.hiddenSize);
  for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = std::bit_cast<float>(uint32_t{bf16[i]} << 16);
  if (timing)
    *timing = {rows, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(),
               gpuMilliseconds};
  return hidden;
}

} // namespace splash::model
