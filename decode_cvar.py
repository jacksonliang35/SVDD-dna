#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from decode_new import (
    DNA_REWARD_CHANNELS,
    initialize_decode_model,
    normalize_reward_name,
    resolve_oracle_reward_channel,
    reward_predict,
    set_seed,
    tensor_list_to_numpy,
)


@torch.no_grad()
def controlled_decode_cvar_local(model, gen_batch_num: int, sample_M: int, n_tasks: int, alpha: float, 
    cvar_beta: float, cvar_eta: float, cvar_lambda: float = 1.0, variant: str = "MC", reward_channel: int = 0, 
    show_progress: bool = True
):
    variant = variant.upper()
    if variant not in {"MC", "PM"}:
        raise ValueError(f"variant must be 'MC' or 'PM', got: {variant!r}")

    reward_model = model.reward_model
    task_l = getattr(model, "task", "").lower()
    tweedie_task = "rna_saluki" if task_l == "rna_saluki" else "dna"

    model.eval()
    model.ref_model.eval()
    reward_model.eval()

    samples = []
    reward_model_preds = []

    for _ in tqdm(range(gen_batch_num), desc=f"CVaR-{variant} guided generation", unit="batch", dynamic_ncols=True, disable=not show_progress):
        if variant == "MC":
            batch_samples = model.ref_model.controlled_sample(
                model.embedding,
                model.head,
                eval_sp_size=model.NUM_SAMPLES_PER_BATCH,
                sample_M=sample_M,
                selection_method="resample",
                selection_alpha=alpha,
                cvar_beta=cvar_beta,
                cvar_eta=cvar_eta,
                cvar_lambda=cvar_lambda,
                reward_channel=reward_channel,
                show_progress=show_progress
            )
        else:
            batch_samples = model.ref_model.controlled_sample_tweedie(
                reward_model,
                eval_sp_size=model.NUM_SAMPLES_PER_BATCH,
                sample_M=sample_M,
                options="True",
                task=tweedie_task,
                selection_method="resample",
                selection_alpha=alpha,
                cvar_beta=cvar_beta,
                cvar_eta=cvar_eta,
                cvar_lambda=cvar_lambda,
                reward_channel=reward_channel,
                show_progress=show_progress
            )

        samples.append(batch_samples.detach())
        pred, _ = reward_predict(model, reward_model, batch_samples, n_tasks, reward_channel=reward_channel)
        reward_model_preds.append(pred)

    print(f"CVaR-{variant} guided sampling done.")
    return samples, torch.cat(reward_model_preds)


@torch.no_grad()
def controlled_decode_smc_cvar_local(model, gen_batch_num: int, n_tasks: int, alpha: float, cvar_beta: float, cvar_eta: float, cvar_lambda: float = 1.0, variant: str = "MC", x0_mode: str = "soft", reward_channel: int = 0, show_progress: bool = True):
    variant = variant.upper()
    if variant not in {"MC", "PM"}:
        raise ValueError(f"variant must be 'MC' or 'PM', got: {variant!r}")

    reward_model = model.reward_model
    mc_timed = bool(getattr(model, "timed", False))
    task_l = getattr(model, "task", "").lower()
    smc_task = "rna_saluki" if task_l == "rna_saluki" else "dna"

    model.eval()
    model.ref_model.eval()
    reward_model.eval()

    samples = []
    reward_model_preds = []

    for _ in tqdm(range(gen_batch_num), desc=f"CVaR-SMC-{variant} guided generation", unit="population", dynamic_ncols=True, disable=not show_progress):
        batch_samples = model.ref_model.controlled_sample_smc(
            alpha=alpha,
            eval_sp_size=model.NUM_SAMPLES_PER_BATCH,
            cvar_beta=cvar_beta,
            cvar_eta=cvar_eta,
            cvar_lambda=cvar_lambda,
            ess_resample_ratio=.5,
            resampling_method="ess",
            variant=variant,
            reward_model=reward_model,
            pre_scorer_embedding=model.embedding if variant == "MC" else None,
            pre_scorer_head=model.head if variant == "MC" else None,
            mc_timed=mc_timed,
            task=smc_task,
            reward_channel=reward_channel,
            x0_mode=x0_mode,
            show_progress=show_progress
        )

        samples.append(batch_samples.detach())
        pred, _ = reward_predict(model, reward_model, batch_samples, n_tasks, reward_channel=reward_channel)
        reward_model_preds.append(pred)

    print(f"CVaR-SMC-{variant} guided sampling done.")
    return samples, torch.cat(reward_model_preds)


def run(args) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("This repo's Enformer/BaseModel path requires CUDA.")

    set_seed(args.seed)
    reward_channel = resolve_oracle_reward_channel(args.task, args.reward_name)
    cvar_eta = args.cvar_eta if args.cvar_eta is not None else -args.cvar_eta_in_reward
    cvar_lambda = args.cvar_lambda
    print(f"Reward objective: {args.reward_name}", f"Reward channel: {reward_channel}")
    model = initialize_decode_model(args, args.device)

    for cvar_lambda in [0.1,0.2,0.4,0.6,0.8,1.0]:
        for cvar_eta in [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0]:
    
            print(f"CVaR beta={args.cvar_beta}, eta={cvar_eta}, lambda={args.cvar_lambda}; cost=-reward")

            if args.sampler == "svdd":
                gen_samples, reward_model_preds = controlled_decode_cvar_local(
                    model=model,
                    gen_batch_num=args.val_batch_num,
                    sample_M=args.sample_M,
                    n_tasks=args.n_task,
                    alpha=args.alpha,
                    cvar_beta=args.cvar_beta,
                    cvar_eta=cvar_eta,
                    cvar_lambda=cvar_lambda,
                    variant=args.variant,
                    reward_channel=reward_channel,
                    show_progress=not args.no_tqdm,
                )
            elif args.sampler == "smc":
                gen_samples, reward_model_preds = controlled_decode_smc_cvar_local(
                    model=model,
                    gen_batch_num=args.val_batch_num,
                    n_tasks=args.n_task,
                    alpha=args.alpha,
                    cvar_beta=args.cvar_beta,
                    cvar_eta=cvar_eta,
                    cvar_lambda=cvar_lambda,
                    variant=args.variant,
                    x0_mode=args.x0_mode,
                    reward_channel=reward_channel,
                    show_progress=not args.no_tqdm,
                )
            else:
                raise ValueError(f"Unknown sampler: {args.sampler}")
            
            arrays = {"reward": reward_model_preds.detach().cpu().numpy()}
            if args.save_samples:
                arrays["sample"] = tensor_list_to_numpy(gen_samples)
            
            # print("Reward:", arrays["reward"])
            print("Reward mean:", np.mean(arrays["reward"]))
            print("Reward std:", np.std(arrays["reward"]))
            print(f"Reward lower {(1-args.cvar_beta) * 100}% quantile:", np.quantile(arrays["reward"], (1-args.cvar_beta)))
            print(f"Reward lower {(1-args.cvar_beta) * 100}% quantile mean:", np.mean(arrays["reward"][arrays["reward"] <= np.quantile(arrays["reward"], (1-args.cvar_beta))]))
            print("------------------------------------------------------------------------")

    return None

    if str(args.task).lower() == "dna":
        reward_tag = normalize_reward_name(args.reward_name)
    elif str(args.task).lower() == "rna":
        reward_tag = "mrl"
    else:
        reward_tag = "stability"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = args.sample_M if args.sampler == "svdd" else args.batch_size
    out_name = args.output_name or f"{args.task}-{reward_tag}-cvar-{args.sampler}-{args.variant}-M{budget}-S{args.seed}.npz"
    if not out_name.endswith(".npz"):
        out_name += ".npz"
    out_path = out_dir / out_name

    if not args.no_save:
        np.savez(out_path, **arrays)
        print(f"saved: {out_path}")
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CVaR-weighted SVDD/SMC decoding")
    parser.add_argument("--task", type=str, default="DNA")
    parser.add_argument("--model", type=str, default="enformer", choices=["enformer", "multienformer", "timedenformer"])
    parser.add_argument("--n_task", type=int, default=1)
    parser.add_argument("--saluki_body", type=int, default=0)
    parser.add_argument("--cdq", action="store_true", default=False)
    parser.add_argument("--sampler", type=str, default="svdd", choices=["svdd", "smc"])

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cvar_beta", type=float, default=.8)
    eta_group = parser.add_mutually_exclusive_group(required=True)
    eta_group.add_argument("--cvar_eta", type=float, default=None, help="CVaR threshold in cost space.")
    eta_group.add_argument("--cvar_eta_in_reward", type=float, default=None, help="CVaR threshold in reward space; converted internally using cost = -reward.")
    parser.add_argument("--cvar_lambda", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sample_M", type=int, default=20)
    parser.add_argument("--val_batch_num", type=int, default=1)
    parser.add_argument("--variant", type=str, default="PM", choices=["MC", "PM"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--x0_mode", type=str, default="soft", choices=["soft", "hard"])
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--value_ckpt_path", type=str, default=None)
    parser.add_argument("--diffusion_ckpt_path", type=str, default=None)
    parser.add_argument("--reward_ckpt_path", type=str, default=None)
    parser.add_argument("--reward_name", type=str, default="hepg2", choices=tuple(DNA_REWARD_CHANNELS.keys()))

    parser.add_argument("--out_dir", type=str, default="./output_cvar")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--save_samples", action="store_true", default=False)
    parser.add_argument("--patch_grelu_ckpt", action="store_true", default=False)
    parser.add_argument("--force_artifact_link", action="store_true", default=False)
    parser.add_argument("--no_tqdm", action="store_true", default=False)
    parser.add_argument("--no_save", action="store_true", default=False)

    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())