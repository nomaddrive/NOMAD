import copy
from typing import List, Union, Sequence, Optional
import torch
from torch import nn
from torch.distributions.utils import logits_to_probs
import pufferlib.models
from gpudrive.env import constants
from huggingface_hub import PyTorchModelHubMixin
from box import Box

import madrona_gpudrive

TOP_K_ROAD_POINTS = madrona_gpudrive.kMaxAgentMapObservationsCount


def log_prob(logits, value):
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)


def entropy(logits):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)


def sample_logits(
    logits: Union[torch.Tensor, Sequence[torch.Tensor]],
    action: Optional[torch.Tensor] = None,
    deterministic: bool = False,
):
    """
    Sample from (multi)discrete policy heads.

    Args:
        logits: A single tensor [B, A] or a sequence of tensors, one per action head.
        action: Optional provided actions. Shape [B] for single head, [B, H] for multi-head.
        deterministic: If True, pick argmax per head instead of sampling.

    Returns:
        actions: Tensor [B] (single head) or [B, H] (multi-head) of sampled actions.
        logprob: Tensor [B] summed over heads.
        entropy: Tensor [B] summed over heads.
    """

    # Normalize input to a list for unified processing
    if isinstance(logits, torch.Tensor):
        logits_list: List[torch.Tensor] = [logits]
    else:
        logits_list = list(logits)

    batch_size = logits_list[0].shape[0]

    # Numerical stable log-softmax form for log probs
    norm_logits = [l - l.logsumexp(dim=-1, keepdim=True) for l in logits_list]

    # Determine / validate actions
    if action is None:
        sampled: List[torch.Tensor] = []
        for l in (norm_logits if deterministic else logits_list):
            if deterministic:
                sampled.append(l.argmax(dim=-1))
            else:
                sampled.append(
                    torch.multinomial(logits_to_probs(l), 1).squeeze(-1)
                )
        actions = torch.stack(sampled, dim=-1)  # [B, H]
    else:
        # Provided action
        if action.dim() == 1:
            # Could be single head or flattened multi-head (unsupported); assume single
            actions = action.view(batch_size, 1)
        else:
            actions = action.view(batch_size, -1)
        if actions.shape[1] != len(logits_list):
            raise ValueError(
                f"Provided action has {actions.shape[1]} dims, expected {len(logits_list)}"
            )

    # Compute log probabilities per head and entropy
    per_head_logprob: List[torch.Tensor] = []
    per_head_entropy: List[torch.Tensor] = []
    for l_norm, a in zip(norm_logits, actions.unbind(dim=-1)):
        a_long = a.long().unsqueeze(-1)
        lp = l_norm.gather(-1, a_long).squeeze(-1)
        per_head_logprob.append(lp)
        # entropy: - sum p log p ; p = exp(log p)
        p = l_norm.exp()
        per_head_entropy.append(-(p * l_norm).sum(-1))

    logprob = torch.stack(per_head_logprob, dim=-1).sum(-1)
    ent = torch.stack(per_head_entropy, dim=-1).sum(-1)

    # Squeeze actions for single-head case for backward compat
    if len(logits_list) == 1:
        actions_out = actions.squeeze(-1)  # [B]
    else:
        actions_out = actions  # [B, H]

    return actions_out, logprob, ent


class NeuralNet(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/Emerge-Lab/gpudrive",
    docs_url="https://arxiv.org/abs/2502.14706",
    tags=["ffn"],
):
    def __init__(
        self,
        action_dim: int = 91,  # legacy single head (7 * 13)
        action_dims: Optional[Sequence[int]] = None,  # new multi-discrete
        input_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.00,
        act_func: str = "tanh",
        max_controlled_agents: int = 64,
        obs_dim: int = 2984,  # Size of the flattened observation vector (hardcoded)
        config=None,  # Optional config
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Multi-discrete handling (backward compatible)
        if action_dims is None:
            self.action_dims = [int(action_dim)]
        else:
            self.action_dims = [int(a) for a in action_dims]
        self.num_action_heads = len(self.action_dims)
        self.action_dim = self.action_dims[0] if self.num_action_heads == 1 else None
        self.max_controlled_agents = max_controlled_agents
        self.max_observable_agents = max_controlled_agents - 1
        self.obs_dim = obs_dim
        self.num_modes = 3  # Ego, partner, road graph
        self.dropout = dropout
        self.act_func = nn.Tanh() if act_func == "tanh" else nn.GELU()

        # Indices for unpacking the observation
        self.ego_state_idx = constants.EGO_FEAT_DIM
        self.partner_obs_idx = (
            constants.PARTNER_FEAT_DIM * self.max_controlled_agents
        )
        if config is not None:
            self.config = Box(config)
            if "reward_type" in self.config:
                if self.config.reward_type == "reward_conditioned":
                    # Agents know their "type", consisting of three weights
                    # that determine the reward (collision, goal, off-road)
                    self.ego_state_idx += 3
                    self.partner_obs_idx += 3

            self.vbd_in_obs = self.config.vbd_in_obs

        # Calculate the VBD predictions size: 91 timesteps * 5 features = 455
        self.vbd_size = 91 * 5

        self.ego_embed = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(self.ego_state_idx, input_dim)
            ),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
        )

        self.partner_embed = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(constants.PARTNER_FEAT_DIM, input_dim)
            ),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
        )

        self.road_map_embed = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(constants.ROAD_GRAPH_FEAT_DIM, input_dim)
            ),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
        )

        if self.vbd_in_obs:
            self.vbd_embed = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Linear(self.vbd_size, input_dim)
                ),
                nn.LayerNorm(input_dim),
                self.act_func,
                nn.Dropout(self.dropout),
                pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
            )

        self.shared_embed = nn.Sequential(
            nn.Linear(self.input_dim * self.num_modes, self.hidden_dim),
            nn.Dropout(self.dropout),
        )

        # Actor: one linear head per action dimension (categorical)
        self.actor_heads = nn.ModuleList(
            [
                pufferlib.pytorch.layer_init(
                    nn.Linear(hidden_dim, a_dim), std=0.01
                )
                for a_dim in self.action_dims
            ]
        )
        # Backward compatibility (some external code might reference .actor)
        if self.num_action_heads == 1:
            self.actor = self.actor_heads[0]
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_dim, 1), std=1
        )

    def encode_observations(self, observation):

        if self.vbd_in_obs:
            (
                ego_state,
                road_objects,
                road_graph,
                vbd_predictions,
            ) = self.unpack_obs(observation)
        else:
            ego_state, road_objects, road_graph = self.unpack_obs(observation)

        # Embed the ego state
        ego_embed = self.ego_embed(ego_state)

        if self.vbd_in_obs:
            vbd_embed = self.vbd_embed(vbd_predictions)
            # Concatenate the VBD predictions with the ego state embedding
            ego_embed = torch.cat([ego_embed, vbd_embed], dim=1)

        # Max pool
        partner_embed, _ = self.partner_embed(road_objects).max(dim=1)
        road_map_embed, _ = self.road_map_embed(road_graph).max(dim=1)

        # Concatenate the embeddings
        embed = torch.cat([ego_embed, partner_embed, road_map_embed], dim=1)

        return self.shared_embed(embed)

    def forward(self, obs, action=None, deterministic=False):

        # Encode the observations
        hidden = self.encode_observations(obs)

        # Decode the actions
        value = self.critic(hidden)
        logits_list = [head(hidden) for head in self.actor_heads]

        action, logprob, ent = sample_logits(
            logits_list if self.num_action_heads > 1 else logits_list[0],
            action=action,
            deterministic=deterministic,
        )

        return action, logprob, ent, value
    
    def get_logits(self, obs):
        """
        Get the logits for the given observations without sampling actions.
        This is useful for evaluation or when you need the logits directly.
        """
        hidden = self.encode_observations(obs)
        logits_list = [head(hidden) for head in self.actor_heads]
        if self.num_action_heads == 1:
            return logits_list[0]
        return logits_list

    def unpack_obs(self, obs_flat):
        """
        Unpack the flattened observation into the ego state, visible simulator state.

        Args:
            obs_flat (torch.Tensor): Flattened observation tensor of shape (batch_size, obs_dim).

        Returns:
            tuple: If vbd_in_obs is True, returns (ego_state, road_objects, road_graph, vbd_predictions).
                Otherwise, returns (ego_state, road_objects, road_graph).
        """

        # Unpack modalities
        ego_state = obs_flat[:, : self.ego_state_idx]
        partner_obs = obs_flat[:, self.ego_state_idx : self.partner_obs_idx]

        if self.vbd_in_obs:
            # Extract the VBD predictions (last 455 elements)
            vbd_predictions = obs_flat[:, -self.vbd_size :]

            # The rest (excluding ego_state and partner_obs) is the road graph
            roadgraph_obs = obs_flat[:, self.partner_obs_idx : -self.vbd_size]
        else:
            # Without VBD, all remaining elements are road graph observations
            roadgraph_obs = obs_flat[:, self.partner_obs_idx :]

        road_objects = partner_obs.view(
            -1, self.max_observable_agents, constants.PARTNER_FEAT_DIM
        )
        road_graph = roadgraph_obs.view(
            -1, TOP_K_ROAD_POINTS, constants.ROAD_GRAPH_FEAT_DIM
        )

        if self.vbd_in_obs:
            return ego_state, road_objects, road_graph, vbd_predictions
        else:
            return ego_state, road_objects, road_graph
