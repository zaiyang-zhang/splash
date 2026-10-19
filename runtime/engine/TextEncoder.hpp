#pragma once

// The text encoder of the video model the engine serves beside its language
// model (serve-native --video VIDEO_ROOT): Qwen3-VL-32B truncated to the
// layers MiniMax-H3 conditions on (model/QwenEncoder.hpp), prepared from the
// installation's encoder/ checkpoint. Each request maps the prepared files,
// encodes on the engine's backend and unmaps them again, so the encoder holds
// memory only while it runs; its activations are admitted by the memory
// governor like the engine's own.
//
// The same class backs the development command
//   splash encoder-native ENCODER_DIR METALLIB prepare
//   splash encoder-native ENCODER_DIR METALLIB encode TOKENS_FILE OUT_FILE
// which prepares the images once (the installer runs it) or writes the hidden
// states of a prompt of little-endian uint32 token ids as f32
// [tokens][hidden] (dev/tests/video).

#include "metal/MetalBackend.hpp"
#include "model/QwenEncoder.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace splash::engine {

class TextEncoder final {
public:
  // `layers` decoder layers of the checkpoint at `directory` (H3: 50).
  TextEncoder(metal::MetalBackend &backend, std::filesystem::path directory, uint32_t layers);

  [[nodiscard]] const model::QwenEncoderLayout &layout() const noexcept { return layout_; }
  [[nodiscard]] uint64_t preparedBytes() const noexcept { return model::preparedEncoderBytes(layout_); }
  // Writes every missing prepared image, maps none.
  void prepare(model::PreparationCheck admitConversion = {});
  // hidden_states[layer] of the prompt, f32 [tokens][hiddenSize]. `layer`
  // must be the encoder's. Throws std::invalid_argument for a prompt or layer
  // it cannot encode.
  [[nodiscard]] std::vector<float> encode(std::span<const uint32_t> tokens, uint32_t layer,
                                          const metal::AllocationAdmission &admit,
                                          model::EncoderTiming *timing = nullptr);

private:
  metal::MetalBackend &backend_;
  std::filesystem::path directory_;
  model::QwenEncoderLayout layout_;
};

// argv[1] is "encoder-native". Returns the process exit status.
[[nodiscard]] int runTextEncoderCommand(int argc, char **argv);

} // namespace splash::engine
