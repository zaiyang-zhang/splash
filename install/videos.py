#!/usr/bin/env python3

"""Install the video model `splash serve --video-model` names.

A video installation is an assembly, like a language model's (assembly.py),
of kind h3-video: models/.video/assemblies/<SHA-256 of its record>, which the
selection link models/.video/OWNER/REPO points to. It holds:

  model.json   the record: the sources at their commits, the license
               acknowledgement, the text encoder layer, the variants and the
               converted images' identity (ModelDescriptor.mm
               inspectVideoModel, server/videos.py VideoInstallation)
  encoder/     the text encoder's MLX checkpoint (config.json and shards),
               which the engine prepares and runs like a target
               (model/QwenEncoder.hpp); `prepare` prepares it once
  tokenizer/   MiniMax-H3's tokenizer (FL2VA/tokenizer), which the server
               encodes prompts with
  weights/     the engine images (dit-turbo, dit-default, vae, preview) and
               the OpenVDN stage checkpoints request preparation reads (vdn)

The weights are downloaded under their publishers' licenses, which the user
acknowledges once per license text: MiniMax H3 and its derivatives (the
VDN-H3 weights) under the MiniMax H3 Community License Agreement, which limits
use to its Applicable Territory. The engine images are converted once by the
vendored engine's converters (install/video) into the prepared-weight cache,
video/<key>, keyed by the converters' sources and the source commits;
--converted adopts images an h3 checkout already converted instead.

  install/videos.py --video-model MiniMaxAI/MiniMax-H3 [--accept-license] prepare
  install/videos.py --video-model MiniMaxAI/MiniMax-H3 verify [--full]
  install/videos.py --video-model MiniMaxAI/MiniMax-H3 link
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from . import hub, models, paths
else:
    import hub
    import models
    import paths

CONVERTERS = paths.ROOT / "install" / "video"
VIDEO = ".video"
KIND = "h3-video"
# The engine images an installation links under weights/.
IMAGE_DIRECTORIES = ("vae", "preview")


@dataclass(frozen=True)
class Source:
    repo: str
    # A commit, or None for the default branch.
    revision: str | None
    # The repository files the installation reads (fnmatch patterns).
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class Variant:
    name: str
    steps: int
    # The OpenVDN stage checkpoint the variant's DiT is converted from.
    stage: str
    # The stage's turbo adapter applies (8-step distilled).
    turbo: bool


@dataclass(frozen=True)
class VideoFamily:
    name: str
    sources: dict
    text_encoder_layer: int
    variants: tuple[Variant, ...]
    license_name: str
    license_url: str


STAGES = ("stage-b-step-2000", "stage-dmd-step-250")
FAMILIES = {
    "MiniMaxAI/MiniMax-H3": VideoFamily(
        "MiniMax-H3",
        {
            "h3": Source(
                "MiniMaxAI/MiniMax-H3",
                None,
                (
                    "LICENSE",
                    "FL2VA/*.json",
                    "FL2VA/transformer/*",
                    "FL2VA/video_vae/*",
                    "FL2VA/audio_vae/*",
                    "FL2VA/tokenizer/*",
                ),
            ),
            "vdn": Source(
                "OpenVDN/vdn-minimax-h3",
                None,
                tuple(
                    f"{stage}/{part}"
                    for stage in STAGES
                    for part in (
                        "adapters/*",
                        "linear_branch/*",
                        "metadata.json",
                        "model_spec.json",
                    )
                )
                + ("NOTICE", "LICENSE"),
            ),
            "taeh3": Source(
                "madebyollin/taehv",
                "62f7591f59dfbb4c3c02b7a621d180a9eeaba26c",
                ("safetensors/taeh3.safetensors", "LICENSE"),
            ),
            "text_encoder": Source(
                "mlx-community/Qwen3-VL-32B-Instruct-4bit",
                None,
                ("config.json", "*.safetensors"),
            ),
        },
        50,
        (
            Variant("turbo", 8, "stage-dmd-step-250", True),
            Variant("default", 50, "stage-b-step-2000", False),
        ),
        "MiniMax H3 Community License Agreement",
        "https://huggingface.co/MiniMaxAI/MiniMax-H3",
    )
}

LICENSE_NOTICE = """\
{model} is distributed under the {license} ({url}).
Its terms limit every use of MiniMax H3 and its derivatives, the VDN-H3
weights and this installation's converted images included, to the license's
Applicable Territory, and add an Acceptable Use Policy. The license's own
words on the territory:

{territory}

The VDN-H3 weights (OpenVDN/vdn-minimax-h3) are a derivative of MiniMax H3
under the same agreement; OpenVDN's code is Apache-2.0 (its NOTICE is in
THIRD_PARTY_NOTICES). The TAEH3 preview weights (madebyollin/taehv) are MIT.
The text encoder (Qwen3-VL-32B) keeps its own license.

Read the full agreement at {url} (LICENSE in the repository) before
installing. By continuing you confirm you accept it and that you will use
MiniMax H3 only within its Applicable Territory."""


class VideoError(models.ModelError):
    pass


def family_of(model):
    family = FAMILIES.get(model)
    if family is None:
        raise VideoError(
            f"{model} is not a supported video model; supported: {', '.join(FAMILIES)}"
        )
    return family


def selection_link(models_root: Path, model: str) -> Path:
    models.validate_repo_id(model)
    return models_root / VIDEO / model


def weight_cache() -> Path:
    """The prepared-weight cache the engine uses (PreparedWeights.cpp)."""
    configured = os.environ.get("SPLASH_WEIGHT_CACHE")
    return (
        Path(configured)
        if configured
        else Path.home() / "Library" / "Caches" / "Splash" / "weights"
    )


# -- license -------------------------------------------------------------------


def territory_excerpt(license_text: str) -> str:
    """The license's sentences that define its territory."""
    markers = (
        "expressly limited",
        "Territory” means",
        "Territories” means",
        "outside the Applicable",
    )
    lines = [
        line.strip()
        for line in license_text.splitlines()
        if any(marker in line for marker in markers) and len(line) < 600
    ]
    return "\n".join(f"  {line}" for line in lines) or "  (see the agreement)"


def acknowledge(
    models_root: Path, model: str, family: VideoFamily, license_path: Path, accept: bool
):
    """The user's acceptance of this license text, asked once per text and
    recorded in models/.video/licenses.json."""
    text = license_path.read_text(errors="replace")
    digest = hashlib.sha256(text.encode()).hexdigest()
    record_path = models_root / VIDEO / "licenses.json"
    try:
        accepted = json.loads(record_path.read_text())
    except (OSError, ValueError):
        accepted = {}
    previous = accepted.get(model)
    if isinstance(previous, dict) and previous.get("sha256") == digest:
        return previous
    print(
        LICENSE_NOTICE.format(
            model=model,
            license=family.license_name,
            url=family.license_url,
            territory=territory_excerpt(text),
        ),
        flush=True,
    )
    if not accept:
        if not sys.stdin.isatty():
            raise VideoError(
                f"{model} needs its license accepted: rerun with --accept-video-license "
                "(splash serve) or --accept-license (install/videos.py)"
            )
        answer = input("Type 'accept' to accept the license and continue: ")
        if answer.strip().lower() != "accept":
            raise VideoError("the license was not accepted; nothing was installed")
    entry = {
        "license": family.license_name,
        "sha256": digest,
        "accepted_at": datetime.datetime.now(datetime.UTC).isoformat(
            timespec="seconds"
        ),
    }
    accepted[model] = entry
    record_path.parent.mkdir(parents=True, exist_ok=True)
    staging = record_path.with_suffix(".tmp")
    staging.write_text(json.dumps(accepted, indent=2, sort_keys=True) + "\n")
    os.replace(staging, record_path)
    return entry


# -- sources -------------------------------------------------------------------


def resolve(family: VideoFamily, *, need):
    """The sources at their commits, with the files `need` names downloaded:
    name -> (Repository, {repository file: local path})."""
    resolved = {}
    for name, source in family.sources.items():
        if name not in need:
            continue
        repo = hub.Repository.resolve(source.repo, source.revision)
        if repo.unreachable_reason:
            print(
                f"{source.repo}: {repo.unreachable_reason}; using the cached snapshot",
                flush=True,
            )
        files = sorted(
            file
            for file in repo.files
            if any(fnmatch.fnmatch(file, p) for p in source.patterns)
        )
        if not files:
            raise VideoError(
                f"{source.repo}@{repo.revision} has none of the files it needs"
            )
        resolved[name] = (repo, repo.download(files))
    return resolved


# -- conversion ----------------------------------------------------------------


def converter_identity() -> str:
    """The converters' sources, by content: an edit converts again."""
    digest = hashlib.sha256(b"splash-video-converters-v1\0")
    for path in sorted(CONVERTERS.rglob("*.py")):
        digest.update(path.relative_to(CONVERTERS).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def conversion_key(family: VideoFamily, commits: dict) -> str:
    record = {
        "converters": converter_identity(),
        "sources": {name: commits[name] for name in ("h3", "vdn", "taeh3")},
        "variants": [variant.name for variant in family.variants],
    }
    return hashlib.sha256(models.json_bytes(record)).hexdigest()


def run_converter(python, arguments, environment):
    command = [str(python), *arguments]
    print(
        "Converting ·",
        " ".join(Path(a).name if "/" in a else a for a in arguments[:3]),
        flush=True,
    )
    if subprocess.run(command, cwd=CONVERTERS, env=environment).returncode:
        raise VideoError(f"conversion failed: {' '.join(command)}")


def snapshot_root(repo) -> Path:
    """Where a source's files are: its local directory or Hub snapshot."""
    return repo.directory or hub.snapshot(repo.name, repo.revision)


def convert(family: VideoFamily, sources, key: str, python) -> Path:
    """The engine images, converted once into the weight cache's video/<key>
    and published atomically; an existing entry is reused."""
    published = weight_cache() / "video" / key
    if (published / "complete.json").is_file():
        return published
    published.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=published.parent))
    try:
        vdn = staging / "vdn"
        vdn.symlink_to(snapshot_root(sources["vdn"][0]))
        environment = dict(
            os.environ, H3_WEIGHTS_DIR=str(staging), PYTHONPATH=str(CONVERTERS)
        )
        for variant in family.variants:
            out = staging / f"dit-{variant.name}"
            stage = vdn / variant.stage
            turbo = (
                ["--turbo", str(stage / "adapters" / "turbo")] if variant.turbo else []
            )
            for step in ("blocks", "qkfix"):
                run_converter(
                    python,
                    [
                        "convert_dit.py",
                        step,
                        "--stage",
                        str(stage),
                        *turbo,
                        "--out",
                        str(out),
                    ],
                    environment,
                )
            for step in ("io", "stream"):
                run_converter(
                    python, ["convert_dit.py", step, "--out", str(out)], environment
                )
        fl2va = next(
            path
            for file, path in sources["h3"][1].items()
            if file.startswith("FL2VA/video_vae/")
        ).parent
        run_converter(
            python,
            ["convert_vae.py", "--src", str(fl2va), "--out", str(staging / "vae"),
             "--tag", "w8", "--i8", "qkv,o,1,2"],
            environment,
        )  # fmt: skip
        taeh3 = sources["taeh3"][1]["safetensors/taeh3.safetensors"]
        run_converter(
            python,
            ["convert_preview.py", str(taeh3), str(staging / "preview")],
            environment,
        )
        vdn.unlink()
        (staging / "complete.json").write_bytes(models.json_bytes({"key": key}))
        os.replace(staging, published)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return published


def adopt(converted: Path, family: VideoFamily) -> Path:
    """Images an h3 checkout converted (its weights/ directory), checked for
    every directory the engine reads."""
    converted = converted.resolve()
    for name in [f"dit-{v.name}" for v in family.variants] + list(IMAGE_DIRECTORIES):
        manifest = "manifest_w8.json" if name == "vae" else "manifest.json"
        if not (converted / name / manifest).is_file():
            raise VideoError(
                f"{converted / name} holds no converted images ({manifest})"
            )
    return converted


# -- assembly ------------------------------------------------------------------


def build(models_root: Path, record: dict, links: dict) -> Path:
    """The assembly of record and links (assembly path -> target), published
    atomically under models/.video/assemblies/<record digest>."""
    data = models.json_bytes(record)
    root = models_root / VIDEO / "assemblies"
    published = root / hashlib.sha256(data).hexdigest()
    if (published / "model.json").is_file():
        return published
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=models.ENTRY_STAGING, dir=root))
    try:
        for name, target in links.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)
        (staging / "model.json").write_bytes(data)
        os.replace(staging, published)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return published


def verify(link: Path, *, full=False) -> dict:
    """The installation's record, once every link it serves from resolves;
    --full also checks the converted images' manifests parse."""
    record = models.read_json(link / "model.json")
    if record.get("version") != 1 or record.get("kind") != KIND:
        raise VideoError(f"{link} is not an {KIND} installation")
    for directory in ("encoder", "tokenizer", "weights"):
        for path in (
            (link / directory).rglob("*") if (link / directory).is_dir() else ()
        ):
            if path.is_symlink() and not path.exists():
                raise VideoError(f"broken link in the installation: {path}")
    for name in ("encoder/config.json", "tokenizer/tokenizer.json"):
        if not (link / name).is_file():
            raise VideoError(f"the installation is missing {name}")
    for variant in record["variants"]:
        if not (
            link / "weights" / f"dit-{variant['name']}" / "manifest.json"
        ).is_file():
            raise VideoError(f"the installation is missing dit-{variant['name']}")
    if full:
        for manifest in (link / "weights").glob("*/manifest*.json"):
            json.loads(manifest.read_text())
    return record


def prepare(args):
    models_root = args.models.resolve()
    family = family_of(args.video_model)
    link = selection_link(models_root, args.video_model)
    with models.installation_lock(models_root):
        sources = resolve(
            family,
            need={"h3", "text_encoder"}
            | ({"vdn", "taeh3"} if not args.converted else set()),
        )
        h3_repo, h3_files = sources["h3"]
        license_entry = acknowledge(
            models_root,
            args.video_model,
            family,
            h3_files["LICENSE"],
            args.accept_license,
        )
        commits = {name: repo.revision for name, (repo, _) in sources.items()}
        if args.converted:
            weights = adopt(args.converted, family)
            converted = {"from": str(weights)}
        else:
            key = conversion_key(family, commits)
            weights = convert(
                family, sources, key, args.converter_python or sys.executable
            )
            converted = {"key": key}
        encoder_repo, encoder_files = sources["text_encoder"]
        links = {
            f"encoder/{Path(name).name}": path for name, path in encoder_files.items()
        }
        links.update(
            {
                f"tokenizer/{Path(name).name}": path
                for name, path in h3_files.items()
                if name.startswith("FL2VA/tokenizer/")
            }
        )
        names = [f"dit-{variant.name}" for variant in family.variants]
        names += list(IMAGE_DIRECTORIES) + ["vdn"]
        links.update({f"weights/{name}": weights / name for name in names})
        if not args.converted:
            links["weights/vdn"] = snapshot_root(sources["vdn"][0])
        record = {
            "version": 1,
            "kind": KIND,
            "model": args.video_model,
            "sources": {
                name: {"repo": repo.name, "revision": repo.revision}
                for name, (repo, _) in sorted(sources.items())
            },
            "license": license_entry,
            "text_encoder": {
                "model": encoder_repo.name,
                "layer": family.text_encoder_layer,
            },
            "variants": [
                {
                    "name": v.name,
                    "steps": v.steps,
                    "stream": (weights / f"dit-{v.name}" / "stream.bin").is_file(),
                }
                for v in family.variants
            ],
            "converted": converted,
        }
        published = build(models_root, record, links)
        for repo, files in sources.values():
            if repo.directory is None and files:
                path = next(iter(files.values()))
                snapshot = hub.snapshot_of(path)
                if snapshot is not None:
                    hub.pin(snapshot, repo.name, link)
        models.link_selection(link, published)
    # The text encoder is prepared once, like a target, before the first
    # serve maps it (engine/TextEncoder.hpp).
    command = [
        str(paths.BINARY),
        "encoder-native",
        str(link / "encoder"),
        str(paths.BINARY.with_name("splash.metallib")),
        "prepare",
        "--layers",
        str(family.text_encoder_layer),
    ]
    print("Preparing · the text encoder", flush=True)
    if subprocess.run(command).returncode:
        raise VideoError("text encoder preparation failed")
    verify(link)
    print(f"Splash video model {args.video_model} is installed at {link}.", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Install a Splash video model")
    parser.add_argument("--models", type=Path, default=paths.MODELS)
    parser.add_argument(
        "--video-model", required=True, help="Hugging Face repository ID"
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="accept the model's license without the interactive prompt",
    )
    parser.add_argument(
        "--converted",
        type=Path,
        help="adopt the engine images an h3 checkout converted (its weights/)",
    )
    parser.add_argument(
        "--converter-python",
        type=Path,
        help="interpreter for the converters (the VAE's needs MLX)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("verify").add_argument("--full", action="store_true")
    commands.add_parser("link", help="print the selection link")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        family_of(args.video_model)
        link = selection_link(args.models.resolve(), args.video_model)
        if args.command == "link":
            print(link)
            return 0
        signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))
        if args.command == "prepare":
            prepare(args)
        else:
            verify(link, full=args.full)
            print(f"Splash video model {args.video_model} preflight passed.")
    except (models.ModelError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
