"""publish_hf.py — package the checkpoint as a Hugging Face repo (Phase 6).

Stages the trained checkpoint plus the model card and inference code into a
self-contained folder laid out like a HF repo, and optionally pushes it.

    python publish_hf.py                       # just stage into hf_repo/
    python publish_hf.py --push --repo you/Posaidon   # stage + upload

Pushing needs `huggingface_hub` and a token (run `huggingface-cli login`, or set
HF_TOKEN). The model uses a custom architecture/tokenizer, so the repo ships the
loader (minigpt_mlx.py) and runs via `python minigpt_mlx.py --generate`.
"""
import argparse
import os
import shutil


def get_args():
    p = argparse.ArgumentParser(description="package the checkpoint for Hugging Face")
    p.add_argument("--out_dir", default="out", help="checkpoint directory to publish")
    p.add_argument("--stage_dir", default="hf_repo", help="where to assemble the repo")
    p.add_argument("--repo", default=None, help="HF repo id, e.g. you/Posaidon")
    p.add_argument("--push", action="store_true", help="upload --stage_dir to --repo")
    return p.parse_args()


def stage(out_dir, stage_dir):
    # (source path, name in the repo). The model card becomes the repo README.
    files = [
        ("MODEL_CARD.md", "README.md"),
        (os.path.join(out_dir, "config.json"), "config.json"),
        (os.path.join(out_dir, "model.safetensors"), "model.safetensors"),
        (os.path.join(out_dir, "tokenizer.json"), "tokenizer.json"),
        ("minigpt_mlx.py", "minigpt_mlx.py"),
        ("build_kafka_corpus.py", "build_kafka_corpus.py"),
    ]
    os.makedirs(stage_dir, exist_ok=True)
    for src, dst in files:
        if not os.path.exists(src):
            raise FileNotFoundError(f"missing {src} — train first or check --out_dir")
        shutil.copyfile(src, os.path.join(stage_dir, dst))
        print(f"  staged {dst:20s} ({os.path.getsize(src)/1e6:.2f} MB)")
    return [dst for _, dst in files]


def main():
    args = get_args()
    print(f"staging {args.out_dir} -> {args.stage_dir}/")
    stage(args.out_dir, args.stage_dir)

    if not args.push:
        print(f"\ndone. inspect {args.stage_dir}/, then: "
              f"python publish_hf.py --push --repo you/Posaidon")
        return

    if not args.repo:
        raise SystemExit("--push needs --repo you/Posaidon")
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=args.stage_dir, repo_id=args.repo, repo_type="model")
    print(f"\npushed to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
