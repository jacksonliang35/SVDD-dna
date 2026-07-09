#!/usr/bin/env python3
"""
W&B-free decode script for SVDD-dna.

Put this file in the repository root and run it instead of decode.py.
It intentionally does not import, initialize, log to, or finish a W&B run.

For compatibility with the current Enformer.py, this script still expects local
checkpoint files in the artifact-style paths used by the repo, unless you pass
--diffusion_ckpt_path / --reward_ckpt_path. If custom paths are passed, the
script creates local symlinks at the paths Enformer.py expects.
"""

import argparse
import os
import random
import shutil
import sys
import types
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from Enformer import BaseModel, BaseModelMultiSep, ConvHead, EnformerTrunk, TimedEnformerTrunk  # noqa: E402


DNA_DIFFUSION_EXPECTED = Path("artifacts/DNA_Diffusion:v0/last.ckpt")
DNA_REWARD_EXPECTED = Path("artifacts/DNA_evaluation:v0/model.ckpt")
RNA_DIFFUSION_EXPECTED = Path("artifacts/RNA_Diffusion:v0/best.ckpt")
RNA_REWARD_EXPECTED = Path("artifacts/RNA_evaluation:v0/model.ckpt")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def expected_paths_for_task(task: str) -> Tuple[Path, Path]:
    task_l = task.lower()
    if task_l in {"rna", "rna_old", "rna_saluki"}:
        return RNA_DIFFUSION_EXPECTED, RNA_REWARD_EXPECTED
    return DNA_DIFFUSION_EXPECTED, DNA_REWARD_EXPECTED


def prepare_expected_file(src: Path, expected: Path, force_link: bool = False) -> Path:
    """
    Enformer.py has hard-coded checkpoint paths. This function makes sure those
    expected paths exist, optionally by symlinking a user-provided local file.
    """
    src = Path(src)
    expected = Path(expected)

    if src == expected:
        if not expected.exists():
            raise FileNotFoundError(
                f"Missing required local checkpoint: {expected}\n"
                "Either place the file there or pass a local checkpoint with "
                "--diffusion_ckpt_path / --reward_ckpt_path."
            )
        return expected

    if not src.exists():
        raise FileNotFoundError(f"Checkpoint source does not exist: {src}")

    expected.parent.mkdir(parents=True, exist_ok=True)

    if expected.exists() or expected.is_symlink():
        if not force_link:
            print(f"Using existing expected checkpoint: {expected}")
            return expected
        if expected.is_dir() and not expected.is_symlink():
            raise IsADirectoryError(f"Will not replace directory: {expected}")
        expected.unlink()

    try:
        os.symlink(src.resolve(), expected)
        print(f"Linked {expected} -> {src.resolve()}")
    except OSError:
        shutil.copy2(src, expected)
        print(f"Copied {src} -> {expected}")

    return expected


def patch_grelu_checkpoint_if_needed(path: Path) -> None:
    """
    Fix older gReLU checkpoints that store data_params inside hyper_parameters
    rather than as a top-level checkpoint key.
    """
    ckpt = torch_load(path)
    if not isinstance(ckpt, dict):
        return

    hp = ckpt.get("hyper_parameters", {})
    if not isinstance(hp, dict):
        hp = {}

    changed = False

    if "data_params" not in ckpt and "data_params" in hp:
        ckpt["data_params"] = hp["data_params"]
        changed = True

    if "performance" not in ckpt:
        ckpt["performance"] = hp.get("performance", {})
        changed = True

    if changed:
        backup = Path(str(path) + ".bak")
        if not backup.exists() and not path.is_symlink():
            shutil.copy2(path, backup)
            print(f"Backup saved to {backup}")
        torch.save(ckpt, path)
        print(f"Patched gReLU metadata in {path}")


def build_model(args, device="cuda"):
    if args.model == "enformer":
        common_trunk = EnformerTrunk(
            n_conv=7,
            channels=1536,
            n_transformers=11,
            n_heads=8,
            key_len=64,
            attn_dropout=0.05,
            pos_dropout=0.01,
            ff_dropout=0.4,
            crop_len=0,
        )
        reg_head = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func="avg")
        return BaseModel(
            embedding=common_trunk,
            head=reg_head,
            cdq=args.cdq,
            batch_size=args.batch_size,
            val_batch_num=1,
            task=args.task,
            n_tasks=args.n_task,
            saluki_body=args.saluki_body,
            device=torch.device(device)
        )

    if args.model == "multienformer":
        common_trunk = EnformerTrunk(
            n_conv=7,
            channels=1536,
            n_transformers=11,
            n_heads=8,
            key_len=64,
            attn_dropout=0.05,
            pos_dropout=0.01,
            ff_dropout=0.4,
            crop_len=0,
        )
        reg_head = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func="avg")
        return BaseModelMultiSep(
            embedding=common_trunk,
            head=reg_head,
            cdq=args.cdq,
            batch_size=args.batch_size,
            val_batch_num=args.val_batch_num,
            device=torch.device(device)
        )

    if args.model == "timedenformer":
        common_trunk = TimedEnformerTrunk(
            n_conv=7,
            channels=1536,
            n_transformers=11,
            n_heads=8,
            key_len=64,
            attn_dropout=0.05,
            pos_dropout=0.01,
            ff_dropout=0.4,
            crop_len=0,
        )
        reg_head = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func="avg")
        return BaseModel(
            embedding=common_trunk,
            head=reg_head,
            cdq=args.cdq,
            batch_size=args.batch_size,
            val_batch_num=args.val_batch_num,
            timed=True,
            task=args.task,
            n_tasks=args.n_task,
            saluki_body=args.saluki_body,
            device=torch.device(device)
        )

    raise NotImplementedError(f"Unknown model: {args.model}")


def load_model_state(model, checkpoint_path: Path, strict: bool = True) -> None:
    checkpoint = torch_load(checkpoint_path)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        state = checkpoint
    model.load_state_dict(state, strict=strict)


def reward_predict(model, reward_model, token_batch: torch.Tensor, n_tasks: int) -> torch.Tensor:
    onehot_samples = model.transform_samples(token_batch)
    task_l = getattr(model, "task", "").lower()

    if task_l == "rna_saluki":
        pred = reward_model(model.transform_samples_saluki(token_batch).float()).detach().squeeze(2)
    elif n_tasks == 1:
        pred = reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0]
    else:
        pred = reward_model(onehot_samples.float().transpose(1, 2)).detach()

    return pred.reshape(-1), onehot_samples


@torch.no_grad()
def controlled_decode_local(model, gen_batch_num: int, sample_M: int, n_tasks: int, variant: str = "MC", show_progress: bool = True):
    """
    Controlled decoding with tqdm.

    variant="MC" uses the learned soft value model:
        ref_model.controlled_sample(model.embedding, model.head, ...)

    variant="PM" uses the posterior-mean/Tweedie path:
        ref_model.controlled_sample_tweedie(model.reward_model, ...)

    For PM there is no separate learned value-function score, so the returned
    value_func_preds mirrors the reward-model predictions for generated samples.
    """
    variant = variant.upper()
    if variant not in {"MC", "PM"}:
        raise ValueError(f"variant must be 'MC' or 'PM', got: {variant!r}")

    if not hasattr(model, "ref_model"):
        raise AttributeError("The constructed model does not have ref_model; cannot decode.")
    if not hasattr(model, "reward_model"):
        raise AttributeError("The constructed model does not have reward_model; cannot score samples.")

    reward_model = model.reward_model
    model.eval()
    model.ref_model.eval()
    reward_model.eval()

    samples = []
    value_func_preds = []
    reward_model_preds = []

    task_l = getattr(model, "task", "").lower()
    tweedie_task = "rna_saluki" if task_l == "rna_saluki" else "dna"
    guided_desc = f"{variant} guided generation"

    for _ in tqdm(range(gen_batch_num), desc=guided_desc, unit="batch", dynamic_ncols=True, disable=not show_progress,):
        if variant == "MC":
            batch_samples = model.ref_model.controlled_sample(
                model.embedding,
                model.head,
                eval_sp_size=model.NUM_SAMPLES_PER_BATCH,
                sample_M=sample_M,
            )
        else:  # variant == "PM"
            batch_samples = model.ref_model.controlled_sample_tweedie(
                reward_model,
                eval_sp_size=model.NUM_SAMPLES_PER_BATCH,
                sample_M=sample_M,
                task=tweedie_task,
            )

        samples.append(batch_samples.detach())

        pred, onehot_samples = reward_predict(model, reward_model, batch_samples, n_tasks)

        if variant == "MC":
            value = model.head(
                model.embedding(onehot_samples.float())
            ).squeeze(2).detach().reshape(-1)
        else:
            # PM uses reward feedback directly, not a separately trained value head.
            # Keep the saved .npz schema unchanged.
            value = pred.detach().reshape(-1)

        value_func_preds.append(value)
        reward_model_preds.append(pred)

    print(f"{variant} guided sampling done.")

    baseline_preds = []
    all_preds = []

    for i in tqdm(
        range(gen_batch_num * sample_M),
        desc="Baseline generation",
        unit="batch",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        batch = model.ref_model.decode_sample(eval_sp_size=model.NUM_SAMPLES_PER_BATCH)
        pred, _ = reward_predict(model, reward_model, batch, n_tasks)

        if i < gen_batch_num:
            baseline_preds.append(pred)
        all_preds.append(pred)

    print("Baseline sampling done.")

    all_values = torch.cat(all_preds)
    k = int(len(all_values) / sample_M)
    top_k_values, _ = torch.topk(all_values, k)

    return (
        samples,
        torch.cat(value_func_preds),
        torch.cat(reward_model_preds),
        top_k_values,
        torch.cat(baseline_preds),
    )


def tensor_list_to_numpy(samples):
    if not samples:
        return np.array([])
    return torch.cat([x.detach().cpu() for x in samples], dim=0).numpy()


def run(args) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This repo's Enformer/BaseModel path calls .cuda() internally. "
            "Run on a CUDA GPU or edit Enformer.py to support CPU."
        )

    set_seed(args.seed)

    expected_diffusion, expected_reward = expected_paths_for_task(args.task)
    diffusion_ckpt = Path(args.diffusion_ckpt_path) if args.diffusion_ckpt_path else expected_diffusion
    reward_ckpt = Path(args.reward_ckpt_path) if args.reward_ckpt_path else expected_reward

    prepare_expected_file(diffusion_ckpt, expected_diffusion, force_link=args.force_artifact_link)
    prepare_expected_file(reward_ckpt, expected_reward, force_link=args.force_artifact_link)

    if args.patch_grelu_ckpt:
        patch_grelu_checkpoint_if_needed(expected_reward)

    print("loading model")
    device = args.device if hasattr(args, "device") else "cuda"
    model = build_model(args, device)

    if args.pre_model_path is not None:
        print("loading pretrained value model:", args.pre_model_path)
        load_model_state(model, Path(args.pre_model_path), strict=True)

    if args.load_checkpoint_path is not None:
        print("loading stored value model:", args.load_checkpoint_path)
        load_model_state(model, Path(args.load_checkpoint_path), strict=True)

    print("total params:", sum(p.numel() for p in model.parameters()))

    model.to(device)
    print("Using device:", device)

    model.eval()

    gen_samples, value_func_preds, reward_model_preds, selected_baseline_preds, baseline_preds = controlled_decode_local(
        model=model,
        gen_batch_num=args.val_batch_num,
        sample_M=args.sample_M,
        n_tasks=args.n_task,
        variant=args.variant,
        show_progress=not args.no_tqdm,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.output_name or f"{args.task}-{args.reward_name}-{args.sampler}-{args.variant}.npz"
    if not out_name.endswith(".npz"):
        out_name += ".npz"
    out_path = out_dir / out_name

    arrays: Dict[str, np.ndarray] = {
        "decoding": reward_model_preds.detach().cpu().numpy(),
        "baseline": baseline_preds.detach().cpu().numpy(),
        "value_func": value_func_preds.detach().cpu().numpy(),
        "selected_baseline": selected_baseline_preds.detach().cpu().numpy(),
    }
    if args.save_samples:
        arrays["sample_tokens"] = tensor_list_to_numpy(gen_samples)

    np.savez(out_path, **arrays)
    print(f"saved: {out_path}")
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SVDD-dna decode.py replacement without W&B")

    parser.add_argument("--task", type=str, default="DNA")
    parser.add_argument("--model", type=str, default="enformer", choices=["enformer", "multienformer", "timedenformer"])
    parser.add_argument("--n_task", type=int, default=1)
    parser.add_argument("--saluki_body", type=int, default=0)
    parser.add_argument("--cdq", action="store_true", default=False)

    parser.add_argument("--sampler", type=str, default="svdd")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_M", type=int, default=5)
    parser.add_argument("--val_batch_num", type=int, default=1)
    parser.add_argument("--variant", type=str, default="MC", choices=["MC", "PM"])
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--pre_model_path", type=str, default=None)
    parser.add_argument("--load_checkpoint_path", type=str, default=None)
    parser.add_argument("--diffusion_ckpt_path", type=str, default=None)
    parser.add_argument("--reward_ckpt_path", type=str, default=None)

    parser.add_argument("--reward_name", type=str, default="HepG2")
    parser.add_argument("--out_dir", type=str, default="./log")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--save_samples", action="store_true", default=False)

    parser.add_argument("--patch_grelu_ckpt", action="store_true", default=False)
    parser.add_argument("--force_artifact_link", action="store_true", default=False)
    parser.add_argument("--no_tqdm", action="store_true", default=False)

    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
