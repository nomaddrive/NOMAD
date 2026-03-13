"""Extract expert states and actions from Waymo Open Dataset."""
import torch
import os
import numpy as np
import imageio
import logging
import argparse
import pufferlib
import pufferlib.pytorch
import yaml
import wandb
import random
from collections import defaultdict
from tqdm import tqdm
from box import Box
from collect_multi_discrete_demo import Experience
from evaluate_bc import evaluate

from gpudrive.env.config import EnvConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.networks.actor_critic import NeuralNet


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return pufferlib.namespace(**config)


def get_model_parameters(policy):
    """Helper function to count the number of trainable parameters."""
    params = filter(lambda p: p.requires_grad, policy.parameters())
    return sum([np.prod(p.size()) for p in params])


def init_wandb(config, name, resume=True):
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        group=config.wandb.group,
        mode=config.wandb.mode,
        tags=config.wandb.tags,
        config={
            "all_config": dict(config),
            "environment": dict(config.environment),
            "train": dict(config.train),
            "vec": dict(config.vec),
        },
        name=name,
        save_code=True,
        resume=False,
    )

    return wandb


def make_agent(env, config):
    """Create a policy based on the environment."""

    if config.train.continue_training:
        print("Loading checkpoint...")
        # Load checkpoint
        saved_cpt = torch.load(
            f=config.train.model_cpt,
            map_location=config.train.device,
            weights_only=False,
        )

        single_action_space = env.single_action_space
        if hasattr(single_action_space, "nvec"):
            action_dims = list(map(int, single_action_space.nvec))
            policy = NeuralNet(
                input_dim=config.train.network.input_dim,
                action_dims=action_dims,
                hidden_dim=config.train.network.hidden_dim,
                dropout=config.train.network.dropout,
                config=config.environment,
            )
        else:
            policy = NeuralNet(
                input_dim=config.train.network.input_dim,
                action_dim=single_action_space.n,
                hidden_dim=config.train.network.hidden_dim,
                dropout=config.train.network.dropout,
                config=config.environment,
            )
        # Detect multi-discrete vs single-discrete in checkpoint
        # ckpt_action_dims = saved_cpt.get("action_dims", None)
        # if ckpt_action_dims is not None and ckpt_action_dims:
        #     policy = NeuralNet(
        #         input_dim=saved_cpt["model_arch"]["input_dim"],
        #         action_dims=ckpt_action_dims,
        #         hidden_dim=saved_cpt["model_arch"]["hidden_dim"],
        #         config=config.environment,
        #     )
        # else:
        #     policy = NeuralNet(
        #         input_dim=saved_cpt["model_arch"]["input_dim"],
        #         action_dim=saved_cpt["action_dim"],
        #         hidden_dim=saved_cpt["model_arch"]["hidden_dim"],
        #         config=config.environment,
        #     )

        # Load the model parameters
        policy.load_state_dict(saved_cpt["parameters"])

        return policy

    else:
        # Start from scratch
        # Detect multi-discrete action space (Gym MultiDiscrete has attribute nvec)
        single_action_space = env.single_action_space
        if hasattr(single_action_space, "nvec"):
            action_dims = list(map(int, single_action_space.nvec))
            return NeuralNet(
                input_dim=config.train.network.input_dim,
                action_dims=action_dims,
                hidden_dim=config.train.network.hidden_dim,
                dropout=config.train.network.dropout,
                config=config.environment,
            )
        else:
            return NeuralNet(
                input_dim=config.train.network.input_dim,
                action_dim=single_action_space.n,
                hidden_dim=config.train.network.hidden_dim,
                dropout=config.train.network.dropout,
                config=config.environment,
            )


def train(config, env):
    policy = make_agent(env=env, config=config).to(
        config.train.device
    )
    policy.train()
    
    config.train.network.num_parameters = get_model_parameters(policy)

    if wandb.run is None:
        config.wandb = init_wandb(config, config.train.exp_id)
        config.train.__dict__.update(dict(config.wandb.config.train))
    else:
        config.wandb = wandb

    experience = Experience.load(path=config.experience.experience_path)
    test_experience = Experience.load(path=config.experience.test_experience_path)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs)

    best_val_loss = float('inf')

    # Create a progress bar for the overall training
    epoch_pbar = tqdm(
        range(config.train.epochs),
        desc="Training Progress",
        unit="epoch",
        position=0,
        leave=True
    )
    
    for epoch in epoch_pbar:
        logging.info(f"Epoch {epoch}/{config.train.epochs}")
        indices = torch.randperm(experience.ptr)

        # Drop the last incomplete batch
        num_samples = (experience.ptr // config.train.batch_size) * config.train.batch_size

        # Create a progress bar for the epoch
        pbar = tqdm(
            range(0, num_samples, config.train.batch_size),
            desc=f"Training (epoch {epoch + 1})",
            unit="batch",
            position=1,
            leave=False
        )

        for start in pbar:
            end = start + config.train.batch_size
            batch_indices = indices[start:end]

            obs_batch = experience.obs[batch_indices].to(config.train.device)
            actions_batch = experience.actions[batch_indices].to(config.train.device)

            # Forward pass
            if config.train.action_type == "multi_discrete":
                logits_list = policy.get_logits(obs_batch)
                # Compute loss for multi-discrete actions
                loss = 0
                for i, logits in enumerate(logits_list):
                    loss += torch.nn.functional.cross_entropy(
                        logits,
                        actions_batch[:, i],
                    )
            elif config.train.action_type == "discrete":
                logits = policy.get_logits(obs_batch)

                # Compute loss
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    actions_batch,
                )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

            config.wandb.log({
                "loss": loss.item(),
                "epoch": epoch,
                "batch_size": end - start,
                "learning_rate": optimizer.param_groups[0]["lr"],
            })

        logging.info(f"Epoch {epoch}/{config.train.epochs} completed.")

        # Validation loop
        policy.eval()
        val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            val_indices = torch.arange(test_experience.ptr)
            for start in range(0, test_experience.ptr, config.train.batch_size):
                end = min(start + config.train.batch_size, test_experience.ptr)
                batch_indices = val_indices[start:end]
                
                obs_batch = test_experience.obs[batch_indices].to(config.train.device)
                actions_batch = test_experience.actions[batch_indices].to(config.train.device)
                
                if config.train.action_type == "multi_discrete":
                    logits_list = policy.get_logits(obs_batch)
                    batch_loss = 0
                    for i, logits in enumerate(logits_list):
                        batch_loss += torch.nn.functional.cross_entropy(
                            logits,
                            actions_batch[:, i],
                        )
                elif config.train.action_type == "discrete":
                    logits = policy.get_logits(obs_batch)
                    batch_loss = torch.nn.functional.cross_entropy(
                        logits,
                        actions_batch,
                    )
                
                val_loss += batch_loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            config.wandb.run.summary["best_val_loss"] = best_val_loss

        config.wandb.log({"val_loss": avg_val_loss, "epoch": epoch})
        policy.train()

        lr_scheduler.step()
        
        # Update the epoch progress bar with current loss
        epoch_pbar.set_postfix({"Last Loss": f"{loss.item():.4f}" if 'loss' in locals() else "N/A", "Val Loss": f"{avg_val_loss:.4f}"})

        if (epoch+1) % config.train.checkpoint_interval == 0 or (epoch+1) == config.train.epochs or epoch+1==1:
            path = os.path.join(config.train.checkpoint_path, config.train.exp_id)
            if not os.path.exists(path):
                os.makedirs(path)
            model_name = f"model_{config.train.exp_id}_{epoch+1:06d}.pt"
            model_path = os.path.join(path, model_name)
            action_dim_value = getattr(policy, "action_dim", None)
            action_dims_value = getattr(policy, "action_dims", None)

            state = {
                "parameters": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "update": epoch+1,
                "model_name": model_name,
                "model_class": policy.__class__.__name__,
                "model_arch": config.train.network,
                # Keep legacy key but if single head was replaced with multi, store list
                "action_dim": action_dim_value if action_dim_value is not None else action_dims_value,
                "action_dims": action_dims_value,
                "exp_id": config.train.exp_id,
                "num_params": config.train.network["num_parameters"],
            }
            torch.save(state, model_path)


def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "best_val_loss"},
    "parameters": {
        "learning_rate": {"max": 0.001, "min": 0.00001},
        "batch_size": {"values": [64, 128, 256, 512]},
        "weight_decay": {"max": 0.1, "min": 0.001},
    },
}


if __name__ == "__main__":

    parse = argparse.ArgumentParser(
        description="Generate expert actions and observations from Waymo Open Dataset."
    )
    parse.add_argument("--config", "-c", default="baselines/bc/config/bc_base_torch.yaml", type=str, help="Path to the configuration file.")
    parse.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    parse.add_argument("--sweep_id", type=str, default=None, help="Sweep ID to join")
    args = parse.parse_args()
    # Load default configs
    config = load_config(args.config)

    if args.sweep or args.sweep_id:
        def sweep_train():
            wandb.init()
            # Reload config to ensure clean state for each run
            local_config = load_config(args.config)
            
            # Update config with sweep parameters
            if 'learning_rate' in wandb.config:
                local_config.train.learning_rate = wandb.config.learning_rate
            if 'batch_size' in wandb.config:
                local_config.train.batch_size = wandb.config.batch_size
            if 'weight_decay' in wandb.config:
                local_config.train.weight_decay = wandb.config.weight_decay
                
            seed_everything(
                seed=local_config.seed,
                torch_deterministic=local_config.train.torch_deterministic,
            )

            # Make dataloader
            data_loader = SceneDataLoader(
                **local_config.data_loader
            )

            env_config = EnvConfig(
                **local_config.environment
            )

            # Make environment
            env = GPUDriveTorchEnv(
                config=env_config,
                data_loader=data_loader,
                max_cont_agents=local_config.environment.max_controlled_agents,
                device=local_config.train.device,
                action_type=local_config.train.action_type,
            )

            train(local_config, env)

            env.close()
            del env
            
        if args.sweep_id:
            wandb.agent(args.sweep_id, function=sweep_train, project=config.wandb.project, entity=config.wandb.entity)
        else:
            sweep_id = wandb.sweep(sweep=sweep_configuration, project=config.wandb.project, entity=config.wandb.entity)
            print(f"Sweep ID: {sweep_id}")
            wandb.agent(sweep_id, function=sweep_train, count=10)

    else:
        seed_everything(
            seed=config.seed,
            torch_deterministic=config.train.torch_deterministic,
        )

        # Make dataloader
        data_loader = SceneDataLoader(
            **config.data_loader
        )

        env_config = EnvConfig(
            **config.environment
        )

        # Make environment
        # You should NOT use PufferGPUDrive, which will reset the environment after each episode.
        # Instead, you should use GPUDriveTorchEnv directly and implement your own resampling logic
        env = GPUDriveTorchEnv(
            config=env_config,
            data_loader=data_loader,
            max_cont_agents=config.environment.max_controlled_agents,
            device=config.train.device,
            action_type=config.train.action_type,
        )

        train(config, env)

        env.close()
        del env
