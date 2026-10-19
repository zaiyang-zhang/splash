#pragma once

#include "DFlashDraft.hpp"
#include "Model.hpp"
#include "Qwen3_6Moe.hpp"
#include "Qwen3_8.hpp"
#include "QwenEncoder.hpp"
#include "ops/Vision.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace splash::model {

using TargetLayout = std::variant<Qwen3_8Layout, Qwen3_6MoeLayout>;

// Where a model's weights come from: files already in the packed layout, or
// an MLX or GGUF checkpoint, and the draft's DFlash2 checkpoint, prepared
// into cached files when it loads. The vision tower is None for a model
// installed with --language-only.
enum class TargetSource : uint8_t { Packed, Mlx, Gguf };
enum class DraftSource : uint8_t { Packed, Checkpoint };
enum class VisionSource : uint8_t { Packed, Mlx, Gguf, None };

// Package metadata validated before weight buffers are loaded. The engine
// consumes capabilities; model loading consumes the concrete layouts.
struct ModelDescriptor final {
  std::string name;
  TargetLayout target;
  DFlashDraftLayout draft;
  ops::VisionLayout vision;
  ModelCapabilities capabilities;
  kv::Layout targetKvLayout;
  CompositeStateLayout stateLayout;
  // Container selection belongs to loading; runtime dispatch follows each weight.
  TargetSource targetSource = TargetSource::Packed;
  DraftSource draftSource = DraftSource::Packed;
  VisionSource visionSource = VisionSource::Packed;

  // A model installed with --language-only has no vision tower: it loads no
  // vision weights and serves no image requests.
  [[nodiscard]] bool hasVision() const noexcept {
    return visionSource != VisionSource::None;
  }
  [[nodiscard]] bool valid() const noexcept;
};

// Derives the capabilities and cache layouts the engine consumes from the
// concrete target and draft layouts.
[[nodiscard]] ModelDescriptor makeModelDescriptor(std::string name,
                                                  TargetLayout target,
                                                  DFlashDraftLayout draft,
                                                  ops::VisionLayout vision);
[[nodiscard]] ModelDescriptor
inspectModelPackage(const std::filesystem::path &root);

// A video model, served beside the language model (serve-native --video):
// an installation whose model.json states "kind": "h3-video". MiniMax-H3's
// engine images live under weights/, in the vendored engine's layout
// (runtime/video): per variant dit-<variant>/ (the int8 Hadamard-rotated
// blocks, QKFIX fp16 tensors, globals and io tensors, and the SSD stream
// image), vae/ (the W8 video VAE), preview/ (TAEH3) and vdn/ (the OpenVDN
// stage checkpoints request preparation reads). Its text encoder is an MLX
// Qwen3-VL checkpoint under encoder/, prepared and run like a language
// model's target (QwenEncoder.hpp) up to `encoder.layers`.
struct VideoVariant final {
  std::string name;
  // Denoising steps, one DiT forward each.
  uint32_t steps = 0;
  // The variant's weights can be streamed from SSD (stream.bin).
  bool streamable = false;
};

struct VideoModelDescriptor final {
  std::string name;
  QwenEncoderLayout encoder;
  std::filesystem::path encoderDirectory;
  std::filesystem::path weightsDirectory;
  std::vector<VideoVariant> variants;
};

[[nodiscard]] VideoModelDescriptor inspectVideoModel(const std::filesystem::path &root);

} // namespace splash::model
