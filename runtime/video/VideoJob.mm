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
  fputs(line.c_str(), stdout);
  fflush(stdout);
}

std::string member(std::string_view name, const std::string &value) { return quoted(name) + ": " + value; }
std::string number(double value) {
  char text[64];
  snprintf(text, sizeof text, "%.3f", value);
  return text;
}

struct Options {
  std::filesystem::path weights, metallib, prep, out;
  h3_request request = h3_default_request();
  bool denoise = true, preview = false, decode = true;
  // Also write the preview's RGB frames, preview.rgb (dev/tests).
  bool previewRgb = false;
  int32_t steps = 0;
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
    throw JobError("usage: splash video-native WEIGHTS_ROOT METALLIB PREP_DIR OUT_DIR [options]");
  Options o;
  o.weights = argv[2];
  o.metallib = argv[3];
  o.prep = argv[4];
  o.out = argv[5];
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
    if (option == "--preview-rgb") {
      o.previewRgb = true;
      continue;
    }
    if (i + 1 >= argc) throw JobError(std::string(option) + " requires a value");
    const std::string_view value(argv[++i]);
    if (option == "--variant") {
      if (value != "turbo" && value != "default") throw JobError("--variant requires turbo or default");
      o.request.variant = value == "turbo" ? H3_VARIANT_TURBO : H3_VARIANT_DEFAULT;
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
    } else if (option == "--phases") {
      o.denoise = o.preview = o.decode = false;
      std::stringstream phases{std::string(value)};
      for (std::string phase; std::getline(phases, phase, ',');) {
        if (phase == "denoise") o.denoise = true;
        else if (phase == "preview") o.preview = true;
        else if (phase == "decode") o.decode = true;
        else throw JobError("--phases takes denoise, preview and decode");
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

struct Prepared {
  std::map<std::string, h3::model::NpyArray> layout, sched, noise;
  h3::model::NpyArray text, adaBlocks, adaFinal;
  int32_t frames = 0, textRows = 0;
};

Prepared readLayout(const std::filesystem::path &prep) {
  Prepared p;
  check(h3::model::readNpz((prep / "layout.npz").string(), p.layout));
  int64_t frames = 0, text = 0;
  check(p.layout.at("frames").toInt64(frames));
  check(p.layout.at("L").toInt64(text));
  p.frames = static_cast<int32_t>(frames);
  p.textRows = static_cast<int32_t>(text);
  return p;
}

void denoise(const Options &o, Engine &engine, Prepared &p, const h3_geometry &g) {
  const auto started = std::chrono::steady_clock::now();
  emit("phase", member("phase", quoted("denoise")) + ", " + member("state", quoted("started")));
  const size_t rows = p.layout.at("tags").count();
  const h3::model::NpyArray *tags = nullptr, *rope = nullptr;
  check(h3::model::member(p.layout, "tags", "|u1", 0, tags));
  check(h3::model::member(p.layout, "rope", "<u2", rows * 96, rope));
  check(h3::model::readNpy((o.prep / "text_ref.npy").string(), p.text));
  if (p.text.dtype != "<u2" || p.text.count() != size_t(p.textRows) * 5376)
    throw JobError("text_ref.npy: expected bf16 [L][5376]");
  const h3_dit_inputs inputs{p.textRows, tags->as<uint8_t>(), rope->as<uint16_t>(), p.text.as<uint16_t>()};
  engine.call(h3_dit_load(engine.get(), &inputs), "load the DiT");
  check(h3::model::readNpy((o.prep / "ada_blocks.npy").string(), p.adaBlocks));
  check(h3::model::readNpy((o.prep / "ada_final.npy").string(), p.adaFinal));
  check(h3::model::readNpz((o.prep / "sched.npz").string(), p.sched));
  check(h3::model::readNpz((o.prep / "noise.npz").string(), p.noise));
  std::vector<float> videoT, audioT, videoSigmas, audioSigmas;
  check(p.sched.at("v_t").toFloat32(videoT));
  check(p.sched.at("a_t").toFloat32(audioT));
  check(p.sched.at("v_sigmas").toFloat32(videoSigmas));
  check(p.sched.at("a_sigmas").toFloat32(audioSigmas));
  int32_t steps = static_cast<int32_t>(videoT.size());
  if (o.steps) steps = std::min(steps, o.steps);
  const h3::model::NpyArray *videoNoise = nullptr, *audioNoise = nullptr;
  check(h3::model::member(p.noise, "video", "<f4", size_t(g.video_rows) * 96, videoNoise));
  check(h3::model::member(p.noise, "audio", "<f4", size_t(g.audio_rows) * 32, audioNoise));
  if (p.adaBlocks.dtype != "<u2" || p.adaBlocks.shape.empty() || p.adaBlocks.shape[0] < steps ||
      p.adaFinal.shape.empty() || p.adaFinal.shape[0] < steps)
    throw JobError("the adaLN tables do not cover the schedule's steps");
  const h3_schedule schedule{steps,          p.adaBlocks.as<uint16_t>(), p.adaFinal.as<uint16_t>(),
                             videoT.data(),  audioT.data(),              videoSigmas.data(),
                             audioSigmas.data()};
  std::vector<float> video(videoNoise->count()), audio(audioNoise->count());
  struct Progress {
    int32_t steps;
  } progress{steps};
  const h3_step_callback onStep = [](void *user, int32_t step, const h3_forward_stats *stats, const float *,
                                     const float *) {
    const auto *state = static_cast<const Progress *>(user);
    emit("step", member("phase", quoted("denoise")) + ", " + member("step", std::to_string(step + 1)) + ", " +
                     member("steps", std::to_string(state->steps)) + ", " +
                     member("wall_ms", number(stats->wall_ms)) + ", " +
                     member("device_mb", number(stats->device_mb)));
    cancelPoint("denoise", step + 1);
  };
  engine.call(h3_denoise(engine.get(), &schedule, videoNoise->as<float>(), audioNoise->as<float>(), video.data(),
                         audio.data(), onStep, &progress),
              "denoise");
  const auto v = h3::model::NpyArray::of("<f4", {g.video_rows, 96}, video.data(), video.size() * 4);
  const auto a = h3::model::NpyArray::of("<f4", {g.audio_rows, 32}, audio.data(), audio.size() * 4);
  check(h3::model::writeNpz((o.out / "final.npz").string(), {{"video", &v}, {"audio", &a}}));
  h3_dit_release(engine.get());
  p.adaBlocks = {};
  p.adaFinal = {};
  p.text = {};
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  emit("phase", member("phase", quoted("denoise")) + ", " + member("state", quoted("completed")) + ", " +
                    member("seconds", number(seconds)) + ", " + member("steps", std::to_string(steps)));
}

std::vector<float> finalVideoLatents(const Options &o, const h3_geometry &g) {
  std::map<std::string, h3::model::NpyArray> latents;
  check(h3::model::readNpz((o.out / "final.npz").string(), latents));
  const h3::model::NpyArray *video = nullptr;
  check(h3::model::member(latents, "video", "<f4", size_t(g.video_rows) * 96, video));
  return {video->as<float>(), video->as<float>() + video->count()};
}

void preview(const Options &o, Engine &engine, const h3_geometry &g, const std::vector<float> &latents) {
  const auto started = std::chrono::steady_clock::now();
  emit("phase", member("phase", quoted("preview")) + ", " + member("state", quoted("started")));
  const size_t frameBytes = size_t(g.width) * g.height * 3;
  std::vector<uint8_t> frames(frameBytes * g.preview_frames), yuv(frameBytes / 2 * g.preview_frames);
  h3_preview_stats stats{};
  engine.call(h3_preview(engine.get(), latents.data(), frames.data(), yuv.data(), &stats), "preview");
  if (stats.nonfinite) throw JobError("the preview decoder produced non-finite values");
  writeFile(o.out / "preview.yuv", yuv.data(), yuv.size());
  if (o.previewRgb) writeFile(o.out / "preview.rgb", frames.data(), frames.size());
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  emit("phase", member("phase", quoted("preview")) + ", " + member("state", quoted("completed")) + ", " +
                    member("seconds", number(seconds)) + ", " +
                    member("frames", std::to_string(g.preview_frames)));
}

void decode(const Options &o, Engine &engine, const h3_geometry &g, const std::vector<float> &latents) {
  const auto started = std::chrono::steady_clock::now();
  emit("phase", member("phase", quoted("decode")) + ", " + member("state", quoted("started")));
  std::vector<uint8_t> frames(size_t(g.width) * g.height * 3 * g.decode_frames);
  h3_decode_stats stats{};
  engine.call(h3_decode(engine.get(), latents.data(), frames.data(), &stats), "decode");
  if (stats.nonfinite) throw JobError("the video VAE produced non-finite values");
  writeFile(o.out / "frames.rgb", frames.data(), frames.size());
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  emit("phase", member("phase", quoted("decode")) + ", " + member("state", quoted("completed")) + ", " +
                    member("seconds", number(seconds)) + ", " +
                    member("frames", std::to_string(g.decode_frames)) + ", " +
                    member("device_mb", number(stats.device_mb)));
}

void run(const Options &options) {
  Options o = options;
  Prepared p = readLayout(o.prep);
  o.request.frames = p.frames;
  h3_geometry g{};
  if (h3_geometry_of(&o.request, &g) != H3_OK) throw JobError("the request's clip geometry is invalid");
  std::filesystem::create_directories(o.out);
  emit("ready", member("frames", std::to_string(g.frames)) + ", " + member("width", std::to_string(g.width)) +
                    ", " + member("height", std::to_string(g.height)) + ", " +
                    member("text_rows", std::to_string(p.textRows)));
  Engine engine(o);
  // Each phase drains its own pool: the Metal objects a phase autoreleases,
  // the DiT's buffers among them, are freed before the next phase allocates.
  if (o.denoise) {
    @autoreleasepool {
      denoise(o, engine, p, g);
    }
  }
  cancelPoint("denoise");
  if (!o.preview && !o.decode) return;
  const std::vector<float> latents = finalVideoLatents(o, g);
  if (o.preview) {
    @autoreleasepool {
      preview(o, engine, g, latents);
    }
  }
  cancelPoint("preview");
  if (o.decode) {
    @autoreleasepool {
      decode(o, engine, g, latents);
    }
  }
}

} // namespace

int runVideoJob(int argc, char **argv) {
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
