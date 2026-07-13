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
DNA_VALUE_EXPECTED = Path("artifacts/DNA_value:v0/human_enhancer_diffusion_enformer_7_11_1536_16_ep10_it3500.pt")

RNA_DIFFUSION_EXPECTED = Path("artifacts/RNA_Diffusion:v0/best.ckpt")
RNA_REWARD_EXPECTED = Path("artifacts/RNA_evaluation:v0/model.ckpt")
RNA_MRL_VALUE_EXPECTED = Path("artifacts/RNA_MRL_value:v0/rna_MRL_diffusion_convgru_6_64_512_ep10_it2800.pt")
RNA_STABILITY_VALUE_EXPECTED = Path("artifacts/RNA_Stability_value:v0/rna_saluki_diffusion_enformer_7_11_1536_16_ep10_it3200.pt")


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


def _strip_checkpoint_wrappers(key: str) -> str:
    """Remove common wrappers added by DataParallel, compile, or Lightning."""
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _extract_checkpoint_state(checkpoint_path: Path):
    checkpoint = torch_load(checkpoint_path)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint at {checkpoint_path} must contain a dictionary.")

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint at {checkpoint_path} does not contain a valid state dictionary.")

    return {_strip_checkpoint_wrappers(key): value for key, value in state.items()}


def _clean_nested_submodule_state(state, submodule_name: str):
    """Normalize a separately saved embedding/head state dictionary."""
    prefix = f"{submodule_name}."
    cleaned = {}

    for raw_key, value in state.items():
        key = _strip_checkpoint_wrappers(raw_key)
        if key.startswith(prefix):
            key = key[len(prefix):]
        cleaned[key] = value

    return cleaned


def load_value_model_state(model, checkpoint_path: Path) -> None:
    """Load only embedding and head, preserving ref_model and reward_model."""
    checkpoint = torch_load(checkpoint_path)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint at {checkpoint_path} must contain a dictionary.")

    # Support a smaller value-only checkpoint format.
    if isinstance(checkpoint.get("embedding_state_dict"), dict) and isinstance(checkpoint.get("head_state_dict"), dict):
        embedding_state = _clean_nested_submodule_state(checkpoint["embedding_state_dict"], "embedding")
        head_state = _clean_nested_submodule_state(checkpoint["head_state_dict"], "head")
        ignored_roots = []
    else:
        if "model_state_dict" in checkpoint:
            full_state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            full_state = checkpoint["state_dict"]
        else:
            full_state = checkpoint

        if not isinstance(full_state, dict):
            raise TypeError(f"Checkpoint at {checkpoint_path} does not contain a valid state dictionary.")

        full_state = {_strip_checkpoint_wrappers(key): value for key, value in full_state.items()}
        embedding_state = {key[len("embedding."):]: value for key, value in full_state.items() if key.startswith("embedding.")}
        head_state = {key[len("head."):]: value for key, value in full_state.items() if key.startswith("head.")}
        ignored_roots = sorted({key.split(".", 1)[0] for key in full_state if not key.startswith(("embedding.", "head."))})

    if not embedding_state:
        raise KeyError(
            f"No embedding.* weights were found in {checkpoint_path}. "
            "Check that this is a value-model checkpoint."
        )

    if not head_state:
        raise KeyError(
            f"No head.* weights were found in {checkpoint_path}. "
            "Check that this is a value-model checkpoint."
        )

    try:
        model.embedding.load_state_dict(embedding_state, strict=True)
        model.head.load_state_dict(head_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"The value-network architecture does not match {checkpoint_path}. "
            "Check --model and the Enformer architecture used to create the checkpoint."
        ) from exc

    embedding_tensors = len(embedding_state)
    head_tensors = len(head_state)
    embedding_parameters = sum(value.numel() for value in embedding_state.values() if torch.is_tensor(value))
    head_parameters = sum(value.numel() for value in head_state.values() if torch.is_tensor(value))

    print(
        f"Loaded value network from {checkpoint_path}: "
        f"{embedding_tensors} embedding tensors ({embedding_parameters:,} values), "
        f"{head_tensors} head tensors ({head_parameters:,} values)."
    )

    if ignored_roots:
        print("Ignored non-value checkpoint components:", ", ".join(ignored_roots))

def sampler_uses_value_model(args) -> bool:
    sampler = str(args.sampler).strip().lower()
    variant = str(args.variant).strip().upper()
    return variant == "MC" and sampler in {"svdd", "svdd_max", "smc"}


def resolve_value_checkpoint(args):
    pre_model_path = getattr(args, "pre_model_path", None)
    load_checkpoint_path = getattr(args, "load_checkpoint_path", None)

    if pre_model_path is not None and load_checkpoint_path is not None:
        print("Both --pre_model_path and --load_checkpoint_path were provided; --load_checkpoint_path will be used.")

    selected_path = load_checkpoint_path or pre_model_path
    if selected_path is not None:
        return Path(selected_path)

    if not sampler_uses_value_model(args):
        return None

    task = str(args.task).strip().lower()

    if task == "dna":
        default_path = DNA_VALUE_EXPECTED
    elif task == "rna":
        default_path = RNA_MRL_VALUE_EXPECTED
    elif task == "rna_saluki":
        default_path = RNA_STABILITY_VALUE_EXPECTED
    else:
        raise ValueError(
            f"No default value-network checkpoint is defined for task={args.task!r}. "
            "Provide --load_checkpoint_path explicitly."
        )

    print(f"No value checkpoint specified; using default for task={task!r}: {default_path}")
    return default_path

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
def pretrained_decode_local(model, gen_batch_num: int, show_progress: bool = True):
    """Sample from the pretrained diffusion model without SVDD or SMC guidance."""
    if gen_batch_num < 1:
        raise ValueError(f"gen_batch_num must be positive, got: {gen_batch_num}")
    if not hasattr(model, "ref_model"):
        raise AttributeError("The constructed model does not have ref_model; cannot decode.")
    if not hasattr(model.ref_model, "decode_sample"):
        raise AttributeError("ref_model does not have decode_sample.")
    if not hasattr(model, "NUM_SAMPLES_PER_BATCH"):
        raise AttributeError("The constructed model does not define NUM_SAMPLES_PER_BATCH.")

    model.eval()
    model.ref_model.eval()
    samples = []

    for _ in tqdm(range(gen_batch_num), desc="Pretrained generation", unit="batch", dynamic_ncols=True, disable=not show_progress):
        batch_samples = model.ref_model.decode_sample(eval_sp_size=model.NUM_SAMPLES_PER_BATCH)
        samples.append(batch_samples.detach())

    print("Pretrained sampling done.")
    return samples

@torch.no_grad()
def controlled_decode_local(model, gen_batch_num: int, sample_M: int, n_tasks: int, variant: str = "MC", method: str = "max", alpha: float = 1.0, sample_baseline: bool = False, show_progress: bool = True):
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
    # value_func_preds = []
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
                selection_method=method,
                selection_alpha=alpha,
            )
        else:  # variant == "PM"
            batch_samples = model.ref_model.controlled_sample_tweedie(
                reward_model,
                eval_sp_size=model.NUM_SAMPLES_PER_BATCH,
                sample_M=sample_M,
                options="True",
                task=tweedie_task,
                selection_method=method,
                selection_alpha=alpha,
            )

        samples.append(batch_samples.detach())

        pred, _ = reward_predict(model, reward_model, batch_samples, n_tasks)

        # if variant == "MC":
        #     value = model.head(
        #         model.embedding(onehot_samples.float())
        #     ).squeeze(2).detach().reshape(-1)
        # else:
        #     # PM uses reward feedback directly, not a separately trained value head.
        #     # Keep the saved .npz schema unchanged.
        #     value = pred.detach().reshape(-1)

        # value_func_preds.append(value)

        reward_model_preds.append(pred)

    print(f"{variant} guided sampling done.")

    if sample_baseline:
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
            # torch.cat(value_func_preds),
            torch.cat(reward_model_preds),
            top_k_values,
            torch.cat(baseline_preds),
        )
    
    else:
        # return samples, torch.cat(value_func_preds), torch.cat(reward_model_preds)
        return samples, torch.cat(reward_model_preds)

@torch.no_grad()
def controlled_decode_smc_local(model, gen_batch_num: int, n_tasks: int, alpha: float, variant: str = "MC", 
        num_steps=None, ess_resample_ratio: float = 0.5, resampling_method: str = "ess", x0_mode: str = "soft", 
        reward_channel: int = 0, mc_timed=None, sample_baseline: bool = False, show_progress: bool = True
    ):
    """Run one global SMC population per generated batch and preserve the existing decode return schema."""
    variant = variant.upper()

    if variant not in {"MC", "PM"}:
        raise ValueError(f"variant must be 'MC' or 'PM', got: {variant!r}")
    if gen_batch_num < 1:
        raise ValueError(f"gen_batch_num must be positive, got: {gen_batch_num}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got: {alpha}")
    if not hasattr(model, "ref_model"):
        raise AttributeError("The constructed model does not have ref_model; cannot decode.")
    if not hasattr(model, "reward_model"):
        raise AttributeError("The constructed model does not have reward_model; cannot score samples.")
    if not hasattr(model.ref_model, "controlled_sample_smc"):
        raise AttributeError("ref_model does not have controlled_sample_smc; add it to diffusion_gosai.py first.")
    if variant == "MC" and not hasattr(model.ref_model, "_mc_value_from_tokens"):
        raise AttributeError("ref_model does not have _mc_value_from_tokens; add the MC helper to diffusion_gosai.py first.")

    reward_model = model.reward_model
    mc_timed = bool(getattr(model, "timed", False)) if mc_timed is None else bool(mc_timed)
    task_l = getattr(model, "task", "").lower()
    smc_task = "rna_saluki" if task_l == "rna_saluki" else "dna"
    smc_steps = int(num_steps) if num_steps is not None else int(model.ref_model.config.sampling.steps)

    model.eval()
    model.ref_model.eval()
    reward_model.eval()

    samples = []
    # value_func_preds = []
    reward_model_preds = []

    for _ in tqdm(range(gen_batch_num), desc=f"SMC-{variant} guided generation", unit="population", dynamic_ncols=True, disable=not show_progress):
        batch_samples = model.ref_model.controlled_sample_smc(
            reward_model=reward_model,
            alpha=alpha,
            num_steps=num_steps,
            eval_sp_size=model.NUM_SAMPLES_PER_BATCH,
            ess_resample_ratio=ess_resample_ratio,
            resampling_method=resampling_method,
            variant=variant,
            pre_scorer_embedding=model.embedding if variant == "MC" else None,
            pre_scorer_head=model.head if variant == "MC" else None,
            mc_timed=mc_timed,
            task=smc_task,
            reward_channel=reward_channel,
            x0_mode=x0_mode,
        )

        samples.append(batch_samples.detach())
        pred, _ = reward_predict(model, reward_model, batch_samples, n_tasks)

        # if variant == "MC":
        #     final_mc_index = smc_steps - 1 if mc_timed else None
        #     value = model.ref_model._mc_value_from_tokens(
        #         batch_samples,
        #         model.embedding,
        #         model.head,
        #         time_index=final_mc_index,
        #         timed=mc_timed,
        #         reward_channel=reward_channel,
        #     ).detach().reshape(-1)
        # else:
        #     value = pred.detach().reshape(-1)

        # value_func_preds.append(value)
        reward_model_preds.append(pred)

    print(f"SMC-{variant} guided sampling done.")

    if sample_baseline:
        baseline_preds = []
        all_preds = []

        for i in tqdm(range(gen_batch_num), desc="Baseline generation", unit="batch", dynamic_ncols=True, disable=not show_progress):
            batch = model.ref_model.decode_sample(eval_sp_size=model.NUM_SAMPLES_PER_BATCH)
            pred, _ = reward_predict(model, reward_model, batch, n_tasks)

            if i < gen_batch_num:
                baseline_preds.append(pred)

            all_preds.append(pred)

        print("Baseline sampling done.")

        all_values = torch.cat(all_preds)

        # return samples, torch.cat(value_func_preds), torch.cat(reward_model_preds), top_k_values, torch.cat(baseline_preds)
        return samples, torch.cat(reward_model_preds), all_values, torch.cat(baseline_preds)
    
    else:
        # return samples, torch.cat(value_func_preds), torch.cat(reward_model_preds)
        return samples, torch.cat(reward_model_preds)

def tensor_list_to_numpy(samples):
    if not samples:
        return np.array([])
    return torch.cat([x.detach().cpu() for x in samples], dim=0).numpy()

def initialize_decode_model(args, device):
    # BaseModel compares task names against lowercase strings.
    args.task = str(args.task).strip().lower()

    expected_diffusion, expected_reward = expected_paths_for_task(args.task)
    diffusion_path_arg = getattr(args, "diffusion_ckpt_path", None)
    reward_path_arg = getattr(args, "reward_ckpt_path", None)
    diffusion_ckpt = Path(diffusion_path_arg) if diffusion_path_arg else expected_diffusion
    reward_ckpt = Path(reward_path_arg) if reward_path_arg else expected_reward

    # These calls do not load the models. They only make the files available
    # at the hard-coded paths that BaseModel.__init__ will use.
    prepare_expected_file(
        diffusion_ckpt,
        expected_diffusion,
        force_link=getattr(args, "force_artifact_link", False),
    )
    prepare_expected_file(
        reward_ckpt,
        expected_reward,
        force_link=getattr(args, "force_artifact_link", False),
    )

    if getattr(args, "patch_grelu_ckpt", False):
        patch_grelu_checkpoint_if_needed(expected_reward)

    print("Constructing BaseModel; diffusion and reward models load during initialization.")
    model = build_model(args, device)

    value_checkpoint = resolve_value_checkpoint(args)

    if sampler_uses_value_model(args):
        if value_checkpoint is None:
            raise ValueError(
                f"sampler={args.sampler!r} with variant='MC' requires a trained "
                "value model. Provide --load_checkpoint_path or --pre_model_path."
            )

        if not value_checkpoint.exists():
            raise FileNotFoundError(f"Value checkpoint does not exist: {value_checkpoint}")

        print("Loading value network only:", value_checkpoint)
        load_value_model_state(model, value_checkpoint)
    
    elif value_checkpoint is not None:
        print(
            f"Ignoring value checkpoint {value_checkpoint} because "
            f"sampler={args.sampler!r}, variant={args.variant!r} does not use the MC value network."
        )

    model.to(device)
    model.eval()
    model.ref_model.eval()
    model.reward_model.eval()

    print("Using device:", device)
    print("Total parameters:", sum(parameter.numel() for parameter in model.parameters()))

    return model

def run(args) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This repo's Enformer/BaseModel path calls .cuda() internally. "
            "Run on a CUDA GPU or edit Enformer.py to support CPU."
        )

    set_seed(args.seed)

    # expected_diffusion, expected_reward = expected_paths_for_task(args.task)
    # diffusion_ckpt = Path(args.diffusion_ckpt_path) if args.diffusion_ckpt_path else expected_diffusion
    # reward_ckpt = Path(args.reward_ckpt_path) if args.reward_ckpt_path else expected_reward
    # value_ckpt = Path(args.value_ckpt_path) if args.value_ckpt_path else DNA_VALUE_EXPECTED

    # prepare_expected_file(diffusion_ckpt, expected_diffusion, force_link=args.force_artifact_link)
    # prepare_expected_file(reward_ckpt, expected_reward, force_link=args.force_artifact_link)
    # prepare_expected_file(value_ckpt, DNA_VALUE_EXPECTED, force_link=args.force_artifact_link)

    # if args.patch_grelu_ckpt:
    #     patch_grelu_checkpoint_if_needed(expected_reward)

    # print("loading base models")
    # device = args.device if hasattr(args, "device") else "cuda"
    # model = build_model(args, device)

    # if args.variant == "MC" and args.sampler in {"svdd", "svdd_max", "smc"}:
    #     print("loading value model:", value_ckpt)
    #     load_model_state(model, value_ckpt, strict=True)

    # # if args.pre_model_path is not None:
    # #     print("loading pretrained value model:", args.pre_model_path)
    # #     load_model_state(model, Path(args.pre_model_path), strict=True)

    # # if args.load_checkpoint_path is not None:
    # #     print("loading stored value model:", args.load_checkpoint_path)
    # #     load_model_state(model, Path(args.load_checkpoint_path), strict=True)


    # print("total params:", sum(p.numel() for p in model.parameters()))

    # model.to(device)
    # print("Using device:", device)

    # model.eval()

    device = getattr(args, "device", "cuda")
    model = initialize_decode_model(args, device)

    if args.sampler == "pretrained":

        print("Using pretrained diffusion sampler")

        gen_samples = pretrained_decode_local(
            model=model,
            gen_batch_num=args.val_batch_num,
            show_progress=not args.no_tqdm,
        )

        # This is post-hoc evaluation only. The reward model does not
        # influence generation.
        model.reward_model.eval()

        with torch.no_grad():
            reward_model_preds = torch.cat([
                reward_predict(model, model.reward_model, batch_samples, args.n_task)[0]
                for batch_samples in gen_samples
            ])

    elif args.sampler == "svdd_max":

        print(f"Using {args.sampler} sampler with {args.variant} variant")

        gen_samples, reward_model_preds = controlled_decode_local(
            model=model,
            gen_batch_num=args.val_batch_num,
            sample_M=args.sample_M,
            n_tasks=args.n_task,
            variant=args.variant,
            method="max",
            alpha=1.0,      # Does not matter for max selection, but still needs to be passed
            show_progress=not args.no_tqdm,
        )
    elif args.sampler == "svdd":

        print(f"Using {args.sampler} sampler with alpha={args.alpha} and {args.variant} variant")

        gen_samples, reward_model_preds = controlled_decode_local(
            model=model,
            gen_batch_num=args.val_batch_num,
            sample_M=args.sample_M,
            n_tasks=args.n_task,
            variant=args.variant,
            method="resample",
            alpha=args.alpha,
            show_progress=not args.no_tqdm,
        )
    elif args.sampler == "smc":

        print(f"Using {args.sampler} sampler with alpha={args.alpha} and {args.variant} variant")

        gen_samples, reward_model_preds = controlled_decode_smc_local(
            model=model,
            gen_batch_num=args.val_batch_num,
            n_tasks=args.n_task,
            alpha=args.alpha,
            variant=args.variant,
            ess_resample_ratio=.5,
            resampling_method="ess",
            x0_mode=args.x0_mode,
            show_progress=not args.no_tqdm,
        )
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.sampler == "pretrained":
        out_name = args.output_name or f"{args.task}-{args.sampler}-S{args.seed}.npz"
    elif args.sampler == "smc":
        out_name = args.output_name or f"{args.task}-{args.sampler}-{args.variant}-M{args.batch_size}-S{args.seed}.npz"
    else:   # svdd or svdd_max
        out_name = args.output_name or f"{args.task}-{args.sampler}-{args.variant}-M{args.sample_M}-S{args.seed}.npz"
    if not out_name.endswith(".npz"):
        out_name += ".npz"
    out_path = out_dir / out_name

    arrays: Dict[str, np.ndarray] = {
        "reward": reward_model_preds.detach().cpu().numpy(),
        # "baseline": baseline_preds.detach().cpu().numpy(),
        # "value_func": value_func_preds.detach().cpu().numpy(),
        # "selected_baseline": selected_baseline_preds.detach().cpu().numpy(),
    }
    if args.save_samples:
        arrays["sample"] = tensor_list_to_numpy(gen_samples)
    
    print("Reward:", arrays["reward"])
    print("Reward mean:", np.mean(arrays["reward"]))
    print("Reward std:", np.std(arrays["reward"]))
    print("Reward lower 20% quantile mean:", np.mean(arrays["reward"][arrays["reward"] <= np.quantile(arrays["reward"], 0.2)]))

    if not args.no_save:
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

    parser.add_argument("--sampler", type=str, default="svdd", choices=["pretrained", "svdd", "svdd_max", "smc"])
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_M", type=int, default=20)
    parser.add_argument("--val_batch_num", type=int, default=1)
    parser.add_argument("--variant", type=str, default="PM", choices=["MC", "PM"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--x0_mode", type=str, default="soft", choices=["soft", "hard"])
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--value_ckpt_path", type=str, default=None)
    # parser.add_argument("--load_checkpoint_path", type=str, default=None)
    parser.add_argument("--diffusion_ckpt_path", type=str, default=None)
    parser.add_argument("--reward_ckpt_path", type=str, default=None)

    parser.add_argument("--reward_name", type=str, default="HepG2")
    parser.add_argument("--out_dir", type=str, default="./output_dna")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--save_samples", action="store_true", default=False)

    parser.add_argument("--patch_grelu_ckpt", action="store_true", default=False)
    parser.add_argument("--force_artifact_link", action="store_true", default=False)
    # parser.add_argument("--sample_baseline", action="store_true", default=False)
    parser.add_argument("--no_tqdm", action="store_true", default=False)
    parser.add_argument("--no_save", action="store_true", default=False)

    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
