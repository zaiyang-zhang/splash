"""The audio track of a video job, while the vendored engine has no audio
VAE: runs in the h3 frontend's environment (server/videos.py, H3Pipeline),
whose BigVGAN audio VAE (MLX) decodes the final audio latents.

  video_audio.py FINAL_NPZ FRAMES OUT_WAV
"""

import sys

import numpy as np
from h3.decode import decode_audio
from minimax_h3_mlx.media import save_wav
from minimax_h3_mlx.packing import audio_latent_num_frames


def main():
    final, frames, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    audio = np.load(final)["audio"]
    wav, rate, _ = decode_audio(audio, audio_latent_num_frames(frames), 0.0)
    save_wav(out, wav, rate)


if __name__ == "__main__":
    main()
