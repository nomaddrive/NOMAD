#!/usr/bin/env python3
"""
Train a goal prediction model from the dataset produced by collect_dataset.py.

Inputs saved by collect_dataset.py:
  - agent_states: (B, A, 3) = (x, y, yaw)
  - roadgraph:    (B, R, F_rg) = [x, y, seg_len, seg_w, seg_h, orientation, type_one_hot(21)]
  - goal_positions: (B, A, 2) = (gx, gy)

This script trains either:
  - GoalPredictor: deterministic regressor (MSE)
  - GoalPredictorMDN: multimodal Mixture Density Network (NLL)
"""

from __future__ import annotations

import argparse
import json
import math
import os
# import nni
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import (
    GoalPredictor,
    GoalPredictorMDN,
    masked_mse,
    mdn_nll_2d,
)

# Optional Weights & Biases logging
try:
    import wandb  # type: ignore
except Exception:
    wandb = None  # type: ignore


class WorldDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor], indices: torch.Tensor):
        self.agent_states = data["agent_states"]  # (B, A, 3)
        self.agent_mask = data["agent_mask"]  # (B, A)
        self.cont_agent_mask = data["cont_agent_mask"]  # (B, A)
        self.roadgraph = data["roadgraph"]  # (B, R, F)
        self.goal_positions = data["goal_positions"]  # (B, A, 2)
        self.roadgraph_mask = data["roadgraph_mask"]  # (B, R)
        self.indices = indices

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        return (
            self.agent_states[idx],  # (A,3)
            self.agent_mask[idx],  # (A,)
            self.cont_agent_mask[idx],  # (A,)
            self.roadgraph[idx],     # (R,F)
            self.roadgraph_mask[idx],  # (R,)
            self.goal_positions[idx] # (A,2)
        )


def compute_norm_stats(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean/std over batch dimension (dim=0) while keeping remaining dims."""
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(eps)
    return mean, std


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


def ade(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # pred/target: (B,A,2)
    diff = pred - target
    l2 = torch.linalg.norm(diff, dim=-1)  # (B,A)
    if mask is not None:
        l2 = l2 * mask.float()
        denom = mask.float().sum().clamp_min(1.0)
        return (l2.sum() / denom)
    return l2.mean()


@torch.no_grad()
def mdn_minade(
    mu: torch.Tensor,            # (B,A,K,2)
    log_sigma: torch.Tensor,     # (B,A,K,2)
    rho: torch.Tensor,           # (B,A,K)
    logits: torch.Tensor,        # (B,A,K)
    goals: torch.Tensor,         # (B,A,2) in original (unnormalized) space desired for metric
    mean_goal: torch.Tensor,     # (A,2)
    std_goal: torch.Tensor,      # (A,2)
    mask: torch.Tensor | None = None,  # (B,A)
    n_samples: int = 32,
) -> torch.Tensor:
    """Compute minADE by sampling from a 2D MDN.

    Returns: scalar tensor (mean of per-agent min distances, masked if provided).
    """
    B, A, K, _ = mu.shape
    device = mu.device

    # Component probabilities
    probs = F.softmax(logits, dim=-1)  # (B,A,K)

    # Sample component indices: shape (S,B, A) then permute -> (B,A,S)
    comp_idx = torch.distributions.Categorical(probs=probs).sample((n_samples,)).permute(1, 2, 0)

    # Prepare gathering indices to select component params
    # Expand tensors to (B,A,S,K,*) and gather over K dim=3
    S = n_samples
    idx_k = comp_idx.unsqueeze(-1).unsqueeze(-1)  # (B,A,S,1,1)

    mu_exp = mu.unsqueeze(2).expand(B, A, S, K, 2)
    log_sigma_exp = log_sigma.unsqueeze(2).expand(B, A, S, K, 2)
    rho_exp = rho.unsqueeze(2).expand(B, A, S, K)

    mu_sel = torch.gather(mu_exp, 3, idx_k.expand(B, A, S, 1, 2)).squeeze(3)          # (B,A,S,2)
    log_sigma_sel = torch.gather(log_sigma_exp, 3, idx_k.expand(B, A, S, 1, 2)).squeeze(3)  # (B,A,S,2)
    rho_sel = torch.gather(rho_exp, 3, comp_idx.unsqueeze(-1)).squeeze(-1)            # (B,A,S)

    sigma_sel = torch.exp(log_sigma_sel)  # (B,A,S,2)

    # Sample from correlated 2D Gaussian
    z1 = torch.randn(B, A, S, device=device)
    z2 = torch.randn(B, A, S, device=device)
    # x = mu_x + sigma_x * z1
    x = mu_sel[..., 0] + sigma_sel[..., 0] * z1
    # y = mu_y + sigma_y * (rho*z1 + sqrt(1-rho^2)*z2)
    eps = 1e-6
    y = mu_sel[..., 1] + sigma_sel[..., 1] * (
        rho_sel * z1 + torch.sqrt(torch.clamp(1 - rho_sel ** 2, min=eps)) * z2
    )

    samples_n = torch.stack([x, y], dim=-1)  # (B,A,S,2) normalized space

    # Denormalize to original space for metric
    mg = mean_goal.view(1, A, 1, 2).to(device)
    sg = std_goal.view(1, A, 1, 2).to(device)
    samples = samples_n * sg + mg  # (B,A,S,2)

    # Distances to targets
    goals_exp = goals.unsqueeze(2)  # (B,A,1,2)
    dists = torch.linalg.norm(samples - goals_exp, dim=-1)  # (B,A,S)
    min_dists = dists.min(dim=-1).values  # (B,A)

    if mask is not None:
        w = mask.float()
        denom = w.sum().clamp_min(1.0)
        return (min_dists * w).sum() / denom
    else:
        return min_dists.mean()


def infer_agent_mask(agent_states: torch.Tensor, tol: float = 0.0) -> torch.Tensor:
    """Heuristic mask for valid agents if dataset contains padding.
    Marks agents valid if any of (x,y,yaw) has non-zero magnitude beyond tol.
    agent_states: (B,A,3).
    Returns: (B,A) bool mask.
    """
    mag = agent_states.abs().sum(dim=-1)
    return mag > tol


def infer_road_mask(roadgraph: torch.Tensor, use_types: bool = True) -> torch.Tensor:
    """Infer valid road points (B,R) from padded roadgraph features.

    By convention, roadgraph feature layout is:
    [x, y, seg_len, seg_w, seg_h, orientation, type_one_hot(21)]

    Padded rows typically have all zeros, especially in the one-hot tail.
    Prefer using the one-hot tail to determine validity if available; else
    fall back to any-nonzero across features.
    """
    if roadgraph.size(-1) >= 7 and use_types:
        type_feats = roadgraph[..., 6:]
        mask = (type_feats.abs().sum(dim=-1) > 0)
    else:
        mask = (roadgraph.abs().sum(dim=-1) > 0)
    return mask


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str,
    mean_state: torch.Tensor,
    std_state: torch.Tensor,
    mean_goal: torch.Tensor,
    std_goal: torch.Tensor,
    grad_clip: float | None = 1.0,
    mdn_minade_samples: int = 32,
    epoch_idx: int | None = None,
    scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts] = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_ade = 0.0
    n_batches = 0

    for batch_i, (agent_states, agent_mask, cont_agent_mask, roadgraph, roadgraph_mask, goals) in enumerate(loader):
        # Shapes per-batch: (B,A,3), (B,A), (B,R,F), (B,A,2)
        B = agent_states.shape[0]
        agent_states = agent_states.to(device)
        agent_mask = agent_mask.to(device)
        cont_agent_mask = cont_agent_mask.to(device)
        roadgraph = roadgraph.to(device)
        roadgraph_mask = roadgraph_mask.to(device)
        goals = goals.to(device)

        road_mask = roadgraph_mask                       # (B,R) a alias

        # Normalize inputs/targets
        agent_states_n = normalize(agent_states, mean_state, std_state)
        goals_n = normalize(goals, mean_goal, std_goal)

        optimizer.zero_grad(set_to_none=True)

        if loss_type == "det":
            pred_n = model(agent_states_n, roadgraph, agent_mask=agent_mask, road_mask=road_mask)  # [B,A,2]
            loss = masked_mse(pred_n, goals_n, cont_agent_mask)
            # For ADE metric, compare in original scale
            pred = denormalize(pred_n, mean_goal, std_goal)
            batch_ade = ade(pred, goals, cont_agent_mask)
        else:  # mdn
            mu, log_sigma, rho, logits = model(agent_states_n, roadgraph, agent_mask=agent_mask, road_mask=road_mask)
            loss = mdn_nll_2d(goals_n, mu, log_sigma, rho, logits, mask=cont_agent_mask)
            # minADE via sampling from GMM
            with torch.no_grad():
                batch_ade = mdn_minade(
                    mu, log_sigma, rho, logits,
                    goals=goals,
                    mean_goal=mean_goal,
                    std_goal=std_goal,
                    mask=cont_agent_mask,
                    n_samples=mdn_minade_samples,
                )

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        # Step cosine warm restarts scheduler per batch with fractional epoch progress
        if scheduler is not None and epoch_idx is not None:
            progress = epoch_idx + (batch_i + 1) / max(1, len(loader))
            scheduler.step(progress)

        total_loss += float(loss.detach().cpu())
        total_ade += float(batch_ade.detach().cpu())
        n_batches += 1

    return total_loss / max(1, n_batches), total_ade / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_type: str,
    mean_state: torch.Tensor,
    std_state: torch.Tensor,
    mean_goal: torch.Tensor,
    std_goal: torch.Tensor,
    mdn_minade_samples: int = 32,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    n_batches = 0

    for agent_states, agent_mask, cont_agent_mask, roadgraph, roadgraph_mask, goals in loader:
        agent_states = agent_states.to(device)
        agent_mask = agent_mask.to(device)
        cont_agent_mask = cont_agent_mask.to(device)
        roadgraph = roadgraph.to(device)
        roadgraph_mask = roadgraph_mask.to(device)
        goals = goals.to(device)

        road_mask = roadgraph_mask
        agent_states_n = normalize(agent_states, mean_state, std_state)
        goals_n = normalize(goals, mean_goal, std_goal)

        if loss_type == "det":
            pred_n = model(agent_states_n, roadgraph, agent_mask=agent_mask, road_mask=road_mask)
            loss = masked_mse(pred_n, goals_n, cont_agent_mask)
            pred = denormalize(pred_n, mean_goal, std_goal)
            batch_ade = ade(pred, goals, cont_agent_mask)
        else:
            mu, log_sigma, rho, logits = model(agent_states_n, roadgraph, agent_mask=agent_mask, road_mask=road_mask)
            loss = mdn_nll_2d(goals_n, mu, log_sigma, rho, logits, mask=cont_agent_mask)
            batch_ade = mdn_minade(
                mu, log_sigma, rho, logits,
                goals=goals,
                mean_goal=mean_goal,
                std_goal=std_goal,
                mask=cont_agent_mask,
                n_samples=mdn_minade_samples,
            )

        total_loss += float(loss.detach().cpu())
        total_ade += float(batch_ade.detach().cpu())
        n_batches += 1

    return total_loss / max(1, n_batches), total_ade / max(1, n_batches)


def main():
    p = argparse.ArgumentParser(description="Train goal prediction model")
    p.add_argument("--dataset", type=str, required=True, help="Path to .pt dataset saved by collect_dataset.py")
    p.add_argument("--out_dir", type=str, default="runs/goal_pred/allus", help="Output directory for checkpoints")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"]) 
    p.add_argument("--model", type=str, default="mdn", choices=["det", "mdn"], help="Deterministic or MDN")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--road_feat_dim", type=int, default=27)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--enc_layers", type=int, default=2)
    p.add_argument("--dec_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_components", type=int, default=5, help="MDN components (for model=mdn)")
    p.add_argument("--mdn_minade_samples", type=int, default=32, help="Number of samples per agent for minADE when model=mdn")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.005, help="Weight decay for AdamW")
    # lr scheduler options
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine_wr", "cosine", "lambda"], help="Learning rate scheduler")
    p.add_argument("--t0", type=int, default=10, help="T_0 for CosineAnnealingWarmRestarts (in epochs)")
    p.add_argument("--t_mult", type=int, default=1, help="T_mult for CosineAnnealingWarmRestarts")
    p.add_argument("--eta_min", type=float, default=0.0, help="Minimum lr for cosine annealing")
    # wandb options
    p.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="gpudrive-goal", help="W&B project name")
    p.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team/user)")
    p.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name")
    p.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="Optional list of W&B tags")
    p.add_argument("--wandb_mode", type=str, default=None, choices=[None, "online", "offline", "disabled"], help="W&B mode override")
    p.add_argument("--wandb_watch", action="store_true", help="Enable wandb.watch for gradients/parameters")
    p.add_argument("--wandb_log_artefact", type=bool, default=False, help="Log model checkpoints as W&B artefacts")
    args = p.parse_args()

    # NNI
    # nni_configs = nni.get_next_parameter()
    # args.batch_size = nni_configs["batch_size"]
    # args.lr = nni_configs["lr"]
    # args.scheduler = nni_configs["scheduler"]
    # args.weight_decay = nni_configs["weight_decay"]
    # args.num_components = nni_configs["num_components"]

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    data = torch.load(args.dataset, map_location="cpu")
    agent_states = data["agent_states"].float()  # (B,A,3)
    roadgraph = data["roadgraph"].float()        # (B,R,F)
    goals = data["goal_positions"].float()       # (B,A,2)
    B = agent_states.shape[0]

    # Train/val split by worlds
    perm = torch.randperm(B)
    n_val = max(1, int(B * args.val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = WorldDataset(data, train_idx)
    val_ds = WorldDataset(data, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Normalization stats computed on train split
    mean_state, std_state = compute_norm_stats(agent_states[train_idx])  # shapes (A,3)
    mean_goal, std_goal = compute_norm_stats(goals[train_idx])           # shapes (A,2)
    # Broadcast-friendly for batching
    mean_state = mean_state.to(device)
    std_state = std_state.to(device)
    mean_goal = mean_goal.to(device)
    std_goal = std_goal.to(device)

    # Model
    if args.model == "det":
        model = GoalPredictor(
            road_feat_dim=args.road_feat_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            enc_layers=args.enc_layers,
            dec_layers=args.dec_layers,
            dropout=args.dropout,
        ).to(device)
        loss_type = "det"
    else:
        model = GoalPredictorMDN(
            road_feat_dim=args.road_feat_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            enc_layers=args.enc_layers,
            dec_layers=args.dec_layers,
            dropout=args.dropout,
            num_components=args.num_components,
        ).to(device)
        loss_type = "mdn"

    # Number of parameters in model
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params} parameters.")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine_wr":
        # We'll step per-batch with fractional epochs; T_0 is interpreted in epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, T_0=args.t0, T_mult=args.t_mult, eta_min=args.eta_min
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=args.epochs, eta_min=args.eta_min
        )
    elif args.scheduler == "lambda":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=lambda epoch: 0.95 ** epoch 
        )
    out_dir = Path(args.out_dir)
    out_dir = out_dir / args.model / f"bs{args.batch_size}_scheduler{args.scheduler}_lr{args.lr}_wd{args.weight_decay}_numcomp{args.num_components}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # W&B setup
    run = None
    if args.use_wandb:
        if wandb is None:
            print("[wandb] wandb not installed; disable logging or `pip install wandb`.")
        else:
            config = {
                **vars(args),
                "dataset": str(args.dataset),
                "train_size": int(train_idx.numel()),
                "val_size": int(val_idx.numel()),
                "road_feat_dim": int(args.road_feat_dim),
            }
            # Count parameters
            config["param_count_total"] = int(sum(p.numel() for p in model.parameters()))
            config["param_count_trainable"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            try:
                run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_run_name,
                    tags=args.wandb_tags,
                    config=config,
                    mode=args.wandb_mode,
                )
                if args.wandb_watch:
                    wandb.watch(model, log="gradients", log_freq=100)
            except TypeError:
                # Older wandb versions may not support mode=; fall back to env var
                if args.wandb_mode is not None:
                    os.environ["WANDB_MODE"] = str(args.wandb_mode)
                run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_run_name,
                    tags=args.wandb_tags,
                    config=config,
                )

    best_val_ade = math.inf

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ade = train_one_epoch(
            model, train_loader, optim, device, loss_type,
            mean_state, std_state, mean_goal, std_goal,
            grad_clip=args.grad_clip, mdn_minade_samples=args.mdn_minade_samples,
            epoch_idx=epoch - 1, scheduler=scheduler
        )
        val_loss, val_ade = evaluate(
            model, val_loader, device, loss_type,
            mean_state, std_state, mean_goal, std_goal,
            mdn_minade_samples=args.mdn_minade_samples,
        )

        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} minADE {train_ade:.3f} | val_loss {val_loss:.4f} minADE {val_ade:.3f}")

        if run is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/minADE": train_ade,
                "val/loss": val_loss,
                "val/minADE": val_ade,
                "lr": optim.param_groups[0]["lr"],
            }, step=epoch)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": (scheduler.state_dict() if scheduler is not None else None),
            "args": vars(args),
            "norm": {
                "mean_state": mean_state.detach().cpu(),
                "std_state": std_state.detach().cpu(),
                "mean_goal": mean_goal.detach().cpu(),
                "std_goal": std_goal.detach().cpu(),
            },
            "metrics": {
                "train_loss": train_loss,
                "train_ade": train_ade,
                "val_loss": val_loss,
                "val_ade": val_ade,
            },
        }
        # Save rolling last and epoch-tagged checkpoints
        torch.save(ckpt, out_dir / f"model_{epoch}.pt")
        torch.save(ckpt, out_dir / "last.pt")
        if run is not None and args.wandb_log_artefact:
            try:
                art_last = wandb.Artifact("model-last", type="model")
                art_last.add_file(str(out_dir / "last.pt"))
                run.log_artifact(art_last)
            except Exception:
                # Fallback to simple file sync
                wandb.save(str(out_dir / "last.pt"), base_path=str(out_dir))
        if val_ade < best_val_ade:
            best_val_ade = val_ade
            torch.save(ckpt, out_dir / "best.pt")
            if run is not None and args.wandb_log_artefact:
                try:
                    art_best = wandb.Artifact("model-best", type="model")
                    art_best.add_file(str(out_dir / "best.pt"))
                    run.log_artifact(art_best)
                    wandb.summary["best_val_ADE"] = best_val_ade
                except Exception:
                    wandb.save(str(out_dir / "best.pt"), base_path=str(out_dir))

    print(f"Best val ADE: {best_val_ade:.3f}")
    if run is not None:
        wandb.summary["best_val_ADE"] = best_val_ade
        run.finish()

    # nni.report_final_result(best_val_ade)


if __name__ == "__main__":
    main()
