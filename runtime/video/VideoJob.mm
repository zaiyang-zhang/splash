#include "video/VideoJob.hpp"

#include "video/abi/h3.h"
#include "video/model/Numpy.hpp"

#include <mach/mach.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace splash::video {
namespace {

class JobError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

std::atomic<bool> gCancelled{false};
// The event stream: the process's stdout, which the engine's own log lines
// (H3_LOG, written to stdout) leave for stderr.
FILE *gEvents = stdout;

void requestCancel(int) { gCancelled.store(true, std::memory_order_relaxed); }

// The process's physical footprint, current and lifetime peak, as Activity
// Monitor's Memory column counts it: Metal allocations included.
struct Footprint {
  uint64_t current = 0, peak = 0;
};

Footprint footprint() {
  task_vm_info_data_t info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
    return {};
  return {info.phys_footprint, std::max<uint64_t>(info.phys_footprint, info.ledger_phys_footprint_peak)};
}

std::string quoted(std::string_view text) {
  std::string out = "\"";
  for (const char c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      char escape[8];
      snprintf(escape, sizeof escape, "\\u%04x", c);
      out += escape;
    } else {
      out += c;
    }
  }
  return out + "\"";
}

// One event line: {"event": NAME, FIELDS...} where fields are preformatted
// JSON members.
void emit(std::string_view event, const std::string &fields = {}) {
  const Footprint f = footprint();
  std::string line = "{\"event\": " + quoted(event) + (fields.empty() ? "" : ", " + fields) +
                     ", \"footprint_bytes\": " + std::to_string(f.current) +
                     ", \"peak_footprint_bytes\": " + std::to_string(f.peak) + "}\n";
  fputs(line.c_str(), gEvents);
  fflush(gEvents);
}

std::string member(std::string_view name, const std::string &value) { return quoted(name) + ": " + value; }
std::string number(double value) {
  char text[64];
  snprintf(text, sizeof text, "%.3f", value);
  return text;
}

void phaseStarted(std::string_view phase) {
  emit("phase", member("phase", quoted(phase)) + ", " + member("state", quoted("started")));
}

void phaseCompleted(std::string_view phase, std::chrono::steady_clock::time_point started,
                    const std::string &fields = {}) {
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  emit("phase", member("phase", quoted(phase)) + ", " + member("state", quoted("completed")) + ", " +
                    member("seconds", number(seconds)) + (fields.empty() ? "" : ", " + fields));
}

struct Options {
  std::filesystem::path weights, metallib, input, out;
  h3_request request = h3_default_request();
  // The input is a saved prep directory instead of a prompt embedding.
  bool savedPrep = false;
  bool preview = false, video = true, audio = true;
  uint64_t seed = 0;
  int32_t steps = 0;
  // Development outputs: the raw preview and video frames (preview.rgb,
  // frames.rgb) and the prepared inputs (prep/).
  bool previewRgb = false, framesRgb = false, savePrep = false;
};

int32_t parseCount(std::string_view option, std::string_view value) {
  int32_t result = 0;
  try {
    size_t used = 0;
    result = std::stoi(std::string(value), &used);
    if (used != value.size() || result < 0) throw std::invalid_argument("negative");
  } catch (const std::exception &) {
    throw JobError(std::string(option) + " requires a nonnegative integer");
  }
  return result;
}

Options parse(int argc, char **argv) {
  if (argc < 6)
    throw JobError("usage: splash video-native WEIGHTS_ROOT METALLIB INPUT OUT_DIR [options]");
  Options o;
  o.weights = argv[2];
  o.metallib = argv[3];
  o.input = argv[4];
  o.out = argv[5];
  o.savedPrep = std::filesystem::is_directory(o.input);
  for (int i = 6; i < argc; ++i) {
    const std::string_view option(argv[i]);
    if (option == "--no-qkfix") {
      o.request.qkfix = 0;
      continue;
    }
    if (option == "--stream") {
      o.request.stream = 1;
      continue;
    }
    if (option == "--preview-rgb" || option == "--frames-rgb" || option == "--save-prep") {
      (option == "--preview-rgb" ? o.previewRgb : option == "--frames-rgb" ? o.framesRgb : o.savePrep) = true;
      continue;
    }
    if (i + 1 >= argc) throw JobError(std::string(option) + " requires a value");
    const std::string_view value(argv[++i]);
    if (option == "--variant") {
      if (value != "turbo" && value != "default") throw JobError("--variant requires turbo or default");
      o.request.variant = value == "turbo" ? H3_VARIANT_TURBO : H3_VARIANT_DEFAULT;
    } else if (option == "--frames") {
      o.request.frames = parseCount(option, value);
    } else if (option == "--seed") {
      o.seed = static_cast<uint64_t>(parseCount(option, value));
    } else if (option == "--chunk") {
      o.request.window_chunk = parseCount(option, value);
    } else if (option == "--radius") {
      o.request.window_radius = parseCount(option, value);
    } else if (option == "--qkfix-blocks") {
      o.request.n_qkfix_blocks = 0;
      std::stringstream blocks{std::string(value)};
      for (std::string block; std::getline(blocks, block, ',');) {
        if (block.empty()) continue;
        if (o.request.n_qkfix_blocks == H3_MAX_QKFIX_BLOCKS) throw JobError("too many --qkfix-blocks");
        o.request.qkfix_blocks[o.request.n_qkfix_blocks++] = parseCount(option, block);
      }
    } else if (option == "--stream-slots") {
      o.request.stream_slots = parseCount(option, value);
    } else if (option == "--vae-precision") {
      if (value != "w8" && value != "f16") throw JobError("--vae-precision requires w8 or f16");
      o.request.vae_precision = value == "w8" ? H3_VAE_W8 : H3_VAE_F16;
    } else if (option == "--outputs") {
      o.preview = o.video = o.audio = false;
      std::stringstream outputs{std::string(value)};
      for (std::string output; std::getline(outputs, output, ',');) {
        if (output == "preview") o.preview = true;
        else if (output == "video") o.video = true;
        else if (output == "audio") o.audio = true;
        else if (!output.empty() && output != "latents") throw JobError("--outputs takes latents, preview, video and audio");
      }
    } else if (option == "--steps") {
      o.steps = parseCount(option, value);
    } else {
      throw JobError("unexpected argument " + std::string(option));
    }
  }
  return o;
}

void check(h3::Status status) {
  if (!status.ok()) throw JobError(status.message());
}

// An engine of the C ABI, destroyed on every path.
class Engine final {
public:
  explicit Engine(const Options &options) {
    const std::string metallib = options.metallib.string(), weights = options.weights.string();
    const h3_config config{metallib.c_str(), weights.c_str()};
    call(h3_create(&config, &options.request, &engine_), "create");
  }
  ~Engine() { h3_destroy(engine_); }
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;

  h3_engine *get() const noexcept { return engine_; }
  void call(h3_status status, std::string_view what) const {
    if (status != H3_OK)
      throw JobError(std::string(what) + ": " + (engine_ ? h3_error(engine_) : "invalid request or files"));
  }

private:
  h3_engine *engine_ = nullptr;
};

void writeFile(const std::filesystem::path &path, const void *data, size_t bytes) {
  const std::filesystem::path staging = path.string() + ".partial";
  FILE *file = fopen(staging.c_str(), "wb");
  if (!file) throw JobError("cannot write " + staging.string());
  const bool written = fwrite(data, 1, bytes, file) == bytes;
  if (fclose(file) != 0 || !written) throw JobError("cannot write " + staging.string());
  std::filesystem::rename(staging, path);
}

void cancelPoint(std::string_view phase, int32_t step = -1) {
  if (!gCancelled.load(std::memory_order_relaxed)) return;
  emit("cancelled", member("phase", quoted(phase)) + ", " + member("step", std::to_string(step)));
  // Every command of the step has completed; exiting releases its memory.
  _exit(kVideoCancelledExit);
}

// The DiT's inputs: the engine's own prep (h3_prep) of a prompt embedding,
// or a saved prep directory's arrays.
struct DitInputs {
  h3_dit_inputs dit{};
  h3_schedule schedule{};
  const float *videoNoise = nullptr, *audioNoise = nullptr;
  // A saved prep's arrays, which the pointers above borrow.
  std::map<std::string, h3::model::NpyArray> layout, sched, noise;
  h3::model::NpyArray text, adaBlocks, adaFinal;
  std::vector<float> videoT, audioT, videoSigmas, audioSigmas;
};

int32_t savedFrames(const std::filesystem::path &prep) {
  std::map<std::string, h3::model::NpyArray> layout;
  check(h3::model::readNpz((prep / "layout.npz").string(), layout));
  int64_t frames = 0;
  check(layout.at("frames").toInt64(frames));
  return static_cast<int32_t>(frames);
}

void readSaved(const Options &o, const h3_geometry &g, DitInputs &in) {
  const std::filesystem::path &prep = o.input;
  check(h3::model::readNpz((prep / "layout.npz").string(), in.layout));
  int64_t textRows = 0;
  check(in.layout.at("L").toInt64(textRows));
  const size_t rows = in.layout.at("tags").count();
  const h3::model::NpyArray *tags = nullptr, *rope = nullptr;
  check(h3::model::member(in.layout, "tags", "|u1", 0, tags));
  check(h3::model::member(in.layout, "rope", "<u2", rows * 96, rope));
  check(h3::model::readNpy((prep / "text_ref.npy").string(), in.text));
  if (in.text.dtype != "<u2" || in.text.count() != size_t(textRows) * 5376)
    throw JobError("text_ref.npy: expected bf16 [L][5376]");
  in.dit = {static_cast<int32_t>(textRows), tags->as<uint8_t>(), rope->as<uint16_t>(), in.text.as<uint16_t>()};
  check(h3::model::readNpy((prep / "ada_blocks.npy").string(), in.adaBlocks));
  check(h3::model::readNpy((prep / "ada_final.npy").string(), in.adaFinal));
  check(h3::model::readNpz((prep / "sched.npz").string(), in.sched));
  check(h3::model::readNpz((prep / "noise.npz").string(), in.noise));
  check(in.sched.at("v_t").toFloat32(in.videoT));
  check(in.sched.at("a_t").toFloat32(in.audioT));
  check(in.sched.at("v_sigmas").toFloat32(in.videoSigmas));
  check(in.sched.at("a_sigmas").toFloat32(in.audioSigmas));
  const h3::model::NpyArray *video = nullptr, *audio = nullptr;
  check(h3::model::member(in.noise, "video", "<f4", size_t(g.video_rows) * 96, video));
  check(h3::model::member(in.noise, "audio", "<f4", size_t(g.audio_rows) * 32, audio));
  in.videoNoise = video->as<float>();
  in.audioNoise = audio->as<float>();
  int32_t steps = static_cast<int32_t>(in.videoT.size());
  if (o.steps) steps = std::min(steps, o.steps);
  if (in.adaBlocks.dtype != "<u2" || in.adaBlocks.shape.empty() || in.adaBlocks.shape[0] < steps ||
      in.adaFinal.shape.empty() || in.adaFinal.shape[0] < steps)
    throw JobError("the adaLN tables do not cover the schedule's steps");
  in.schedule = {steps,          in.adaBlocks.as<uint16_t>(), in.adaFinal.as<uint16_t>(),
                 in.videoT.data(), in.audioT.data(),           in.videoSigmas.data(),
                 in.audioSigmas.data()};
}

void prepare(const Options &o, Engine &engine, DitInputs &in) {
  const auto started = std::chrono::steady_clock::now();
  phaseStarted("prep");
  std::map<std::string, h3::model::NpyArray> embedding;
  check(h3::model::readNpz(o.input.string(), embedding));
  if (!embedding.count("hidden")) throw JobError(o.input.string() + ": no member hidden");
  const h3::model::NpyArray &hidden = embedding.at("hidden");
  if (hidden.dtype != "<f4" || hidden.shape.size() != 2 || hidden.shape[1] != 5120)
    throw JobError(o.input.string() + ": hidden states must be f4 [L][5120]");
  const h3_prep_inputs inputs{hidden.as<float>(), static_cast<int32_t>(hidden.shape[0]), o.steps, o.seed};
  h3_prepared p{};
  engine.call(h3_prep(engine.get(), &inputs, &p), "prepare the request");
  if (o.savePrep) engine.call(h3_prep_write(engine.get(), (o.out / "prep").c_str()), "write the prep");
  in.dit = {p.text_rows, p.tags, p.rope, p.text};
  in.schedule = {p.steps, p.ada_blocks, p.ada_final, p.video_t, p.audio_t, p.video_sigmas, p.audio_sigmas};
  in.videoNoise = p.video_noise;
  in.audioNoise = p.audio_noise;
  phaseCompleted("prep", started,
                 member("text_rows", std::to_string(p.text_rows)) + ", " + member("steps", std::to_string(p.steps)));
}

struct Latents {
  std::vector<float> video, audio;
};

Latents denoise(const Options &o, Engine &engine, DitInputs &in, const h3_geometry &g) {
  const auto started = std::chrono::steady_clock::now();
  phaseStarted("denoise");
  engine.call(h3_dit_load(engine.get(), &in.dit), "load the DiT");
  Latents latents{std::vector<float>(size_t(g.video_rows) * 96), std::vector<float>(size_t(g.audio_rows) * 32)};
  struct Progress {
    int32_t steps;
  } progress{in.schedule.steps};
  const h3_step_callback onStep = [](void *user, int32_t step, const h3_forward_stats *stats, const float *,
                                     const float *) {
    const auto *state = static_cast<const Progress *>(user);
    emit("step", member("phase", quoted("denoise")) + ", " + member("step", std::to_string(step + 1)) + ", " +
                     member("steps", std::to_string(state->steps)) + ", " +
                     member("wall_ms", number(stats->wall_ms)) + ", " +
                     member("device_mb", number(stats->device_mb)));
    cancelPoint("denoise", step + 1);
  };
  engine.call(h3_denoise(engine.get(), &in.schedule, in.videoNoise, in.audioNoise, latents.video.data(),
                         latents.audio.data(), onStep, &progress),
              "denoise");
  const auto v = h3::model::NpyArray::of("<f4", {g.video_rows, 96}, latents.video.data(), latents.video.size() * 4);
  const auto a = h3::model::NpyArray::of("<f4", {g.audio_rows, 32}, latents.audio.data(), latents.audio.size() * 4);
  check(h3::model::writeNpz((o.out / "final.npz").string(), {{"video", &v}, {"audio", &a}}));
  h3_dit_release(engine.get());
  if (!o.savedPrep) h3_prep_release(engine.get());
  phaseCompleted("denoise", started, member("steps", std::to_string(in.schedule.steps)));
  return latents;
}

void writeMp4(const std::filesystem::path &path, const std::vector<uint8_t> &frames, int32_t count,
              const h3_geometry &g, const std::vector<float> &wav) {
  const std::filesystem::path staging = path.string() + ".partial.mp4";
  h3_mp4_stats stats{};
  if (h3_write_mp4(staging.c_str(), frames.data(), count, g.width, g.height, wav.empty() ? nullptr : wav.data(),
                   wav.empty() ? 0 : g.audio_samples, &stats) != H3_OK)
    throw JobError("cannot write " + path.string());
  std::filesystem::rename(staging, path);
}

void preview(const Options &o, Engine &engine, const h3_geometry &g, const Latents &latents) {
  const auto started = std::chrono::steady_clock::now();
  phaseStarted("preview");
  std::vector<uint8_t> frames(size_t(g.width) * g.height * 3 * g.preview_frames);
  h3_preview_stats stats{};
  engine.call(h3_preview(engine.get(), latents.video.data(), frames.data(), nullptr, &stats), "preview");
  if (stats.nonfinite) throw JobError("the preview decoder produced non-finite values");
  if (o.previewRgb) writeFile(o.out / "preview.rgb", frames.data(), frames.size());
  writeMp4(o.out / "preview.mp4", frames, g.preview_frames, g, {});
  phaseCompleted("preview", started, member("frames", std::to_string(g.preview_frames)));
}

std::vector<uint8_t> decode(const Options &o, Engine &engine, const h3_geometry &g, const Latents &latents) {
  const auto started = std::chrono::steady_clock::now();
  phaseStarted("decode");
  std::vector<uint8_t> frames(size_t(g.width) * g.height * 3 * g.decode_frames);
  h3_decode_stats stats{};
  engine.call(h3_decode(engine.get(), latents.video.data(), frames.data(), &stats), "decode");
  if (stats.nonfinite) throw JobError("the video VAE produced non-finite values");
  if (o.framesRgb) writeFile(o.out / "frames.rgb", frames.data(), frames.size());
  phaseCompleted("decode", started,
                 member("frames", std::to_string(g.decode_frames)) + ", " + member("device_mb", number(stats.device_mb)));
  return frames;
}

std::vector<float> decodeAudio(Engine &engine, const h3_geometry &g, const Latents &latents) {
  const auto started = std::chrono::steady_clock::now();
  phaseStarted("audio");
  std::vector<float> wav(size_t(2) * g.audio_samples);
  h3_audio_stats stats{};
  engine.call(h3_decode_audio(engine.get(), latents.audio.data(), wav.data(), &stats), "decode the audio");
  if (stats.nonfinite) throw JobError("the audio VAE produced non-finite values");
  phaseCompleted("audio", started, member("samples", std::to_string(stats.samples)));
  return wav;
}

void run(const Options &options) {
  Options o = options;
  if (o.savedPrep) o.request.frames = savedFrames(o.input);
  h3_geometry g{};
  if (h3_geometry_of(&o.request, &g) != H3_OK) throw JobError("the request's clip geometry is invalid");
  std::filesystem::create_directories(o.out);
  emit("ready", member("frames", std::to_string(g.frames)) + ", " + member("width", std::to_string(g.width)) +
                    ", " + member("height", std::to_string(g.height)));
  Engine engine(o);
  Latents latents;
  // Each phase drains its own pool: the Metal objects a phase autoreleases,
  // the DiT's buffers among them, are freed before the next phase allocates.
  @autoreleasepool {
    DitInputs inputs;
    if (o.savedPrep) readSaved(o, g, inputs);
    else prepare(o, engine, inputs);
    cancelPoint("prep");
    latents = denoise(o, engine, inputs, g);
  }
  cancelPoint("denoise");
  if (o.preview) {
    @autoreleasepool {
      preview(o, engine, g, latents);
    }
    cancelPoint("preview");
  }
  std::vector<uint8_t> frames;
  if (o.video || o.framesRgb) {
    @autoreleasepool {
      frames = decode(o, engine, g, latents);
    }
    cancelPoint("decode");
  }
  std::vector<float> wav;
  if (o.audio) {
    @autoreleasepool {
      wav = decodeAudio(engine, g, latents);
    }
    cancelPoint("audio");
  }
  if (o.video) {
    const auto started = std::chrono::steady_clock::now();
    phaseStarted("mux");
    writeMp4(o.out / "video.mp4", frames, g.decode_frames, g, wav);
    phaseCompleted("mux", started);
  }
}

} // namespace

int runVideoJob(int argc, char **argv) {
  const int events = dup(STDOUT_FILENO);
  if (events >= 0 && dup2(STDERR_FILENO, STDOUT_FILENO) >= 0) {
    if (FILE *stream = fdopen(events, "w")) gEvents = stream;
  }
  struct sigaction action {};
  action.sa_handler = requestCancel;
  sigemptyset(&action.sa_mask);
  for (const int number : {SIGTERM, SIGINT}) sigaction(number, &action, nullptr);
  std::signal(SIGPIPE, SIG_IGN);
  @autoreleasepool {
    try {
      run(parse(argc, argv));
      emit("done");
      return 0;
    } catch (const std::exception &error) {
      emit("error", member("message", quoted(error.what())));
      return 1;
    }
  }
}

} // namespace splash::video
