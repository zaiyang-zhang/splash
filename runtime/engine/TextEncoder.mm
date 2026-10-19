#include "engine/TextEncoder.hpp"

#include "StderrLine.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iterator>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace splash::engine {

TextEncoder::TextEncoder(metal::MetalBackend &backend, std::filesystem::path directory, uint32_t layers)
    : backend_(backend), directory_(std::move(directory)), layout_(model::qwen3VlEncoderLayout(layers)) {}

void TextEncoder::prepare(model::PreparationCheck admitConversion) {
  model::QwenEncoderLoader loader(backend_, directory_, layout_, std::move(admitConversion));
  loader.prepare();
}

std::vector<float> TextEncoder::encode(std::span<const uint32_t> tokens, uint32_t layer,
                                       const metal::AllocationAdmission &admit, model::EncoderTiming *timing) {
  if (layer != layout_.layers)
    throw std::invalid_argument("the text encoder returns hidden states after layer " +
                                std::to_string(layout_.layers));
  if (tokens.empty() || tokens.size() > layout_.maximumTokens)
    throw std::invalid_argument("a text encoder prompt holds 1 to " + std::to_string(layout_.maximumTokens) +
                                " tokens");
  const auto started = std::chrono::steady_clock::now();
  model::QwenEncoderLoader loader(backend_, directory_, layout_);
  const model::QwenEncoder encoder(backend_, model::loadQwenEncoderWeights(loader));
  std::vector<float> hidden = encoder.encode(tokens, admit, timing);
  if (timing)
    timing->wallMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  return hidden;
}

namespace {

class CommandError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

std::vector<uint32_t> readTokens(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw CommandError("cannot read " + path.string());
  std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (bytes.empty() || bytes.size() % sizeof(uint32_t))
    throw CommandError(path.string() + " must hold little-endian uint32 token ids");
  std::vector<uint32_t> tokens(bytes.size() / sizeof(uint32_t));
  std::memcpy(tokens.data(), bytes.data(), bytes.size());
  return tokens;
}

// The development command admits every allocation: it is the only work on
// the GPU, and the run fails where Metal refuses one.
metal::AllocationResult admitAll(uint64_t, const std::function<void()> &allocate) {
  allocate();
  return true;
}

} // namespace

int runTextEncoderCommand(int argc, char **argv) {
  try {
    if (argc < 5)
      throw CommandError("usage: splash encoder-native ENCODER_DIR METALLIB prepare | encode TOKENS_FILE OUT_FILE "
                         "[--layers N]");
    const std::filesystem::path directory = argv[2];
    metal::MetalBackend backend(argv[3]);
    const std::string_view command = argv[4];
    uint32_t layers = 50;
    int next = command == "encode" ? 7 : 5;
    if (command == "encode" && argc < 7) throw CommandError("encode needs TOKENS_FILE and OUT_FILE");
    for (; next < argc; next += 2) {
      if (std::string_view(argv[next]) != "--layers" || next + 1 >= argc)
        throw CommandError("unexpected argument " + std::string(argv[next]));
      layers = static_cast<uint32_t>(std::stoul(argv[next + 1]));
    }
    TextEncoder encoder(backend, directory, layers);
    const auto started = std::chrono::steady_clock::now();
    if (command == "prepare") {
      encoder.prepare();
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      printf("{\"prepared_bytes\": %llu, \"seconds\": %.3f}\n",
             static_cast<unsigned long long>(encoder.preparedBytes()), seconds);
      return 0;
    }
    if (command != "encode") throw CommandError("unknown encoder command " + std::string(command));
    const std::vector<uint32_t> tokens = readTokens(argv[5]);
    model::EncoderTiming timing;
    const std::vector<float> hidden = encoder.encode(tokens, layers, admitAll, &timing);
    std::ofstream out(argv[6], std::ios::binary);
    out.write(reinterpret_cast<const char *>(hidden.data()), static_cast<std::streamsize>(hidden.size() * 4));
    if (!out) throw CommandError("cannot write " + std::string(argv[6]));
    printf("{\"tokens\": %u, \"hidden_size\": %u, \"wall_ms\": %.3f, \"gpu_ms\": %.3f}\n", timing.tokens,
           encoder.layout().hiddenSize, timing.wallMilliseconds, timing.gpuMilliseconds);
    return 0;
  } catch (const std::exception &error) {
    writeStderrLine(std::string("error: ") + error.what());
    return 1;
  }
}

} // namespace splash::engine
