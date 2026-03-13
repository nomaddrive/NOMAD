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
        action_dims: Optional[Sequence[int]] = None,  # multi-discrete sizes
        input_dim: int = 64,  # modality embedding dimension (per ego/partner/road)
        hidden_dim: int = 128,  # kept for backward compat (unused in new trunk except if specified)
        dropout: float = 0.01,
        act_func: str = "gelu",
        max_controlled_agents: int = 64,
        obs_dim: int = 2984,
        config=None,
        # Architecture enhancement parameters
        shared_width: int = 512,
        shared_depth: int = 1,  # number of residual blocks at shared width
        projection_width: int = 256,  # width after shared trunk fed to policy/value cores
        policy_residual: bool = True,
        value_residual: bool = True,
        per_head_mlp: bool = False,  # add small MLP before each categorical head
        head_hidden: int = 256,  # width inside per-head MLP if enabled
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.shared_width = shared_width
        self.projection_width = projection_width
        self.per_head_mlp = per_head_mlp
        self.head_hidden = head_hidden

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
        self.act_func = nn.Tanh() if act_func.lower() == "tanh" else nn.GELU()

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
        else:
            self.vbd_in_obs = False

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

        # ----- Enhanced shared trunk -----
        concat_in = self.input_dim * self.num_modes

        class ResidualBlock(nn.Module):
            def __init__(self, width: int, act: nn.Module, dropout: float):
                super().__init__()
                self.lin1 = nn.Linear(width, width)
                self.lin2 = nn.Linear(width, width)
                self.act = act
                self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
                self.ln = nn.LayerNorm(width)

            def forward(self, x):
                h = self.lin1(x)
                h = self.act(h)
                h = self.dropout(h)
                h = self.lin2(h)
                x = x + h
                return self.ln(x)

        residual_blocks = []
        for _ in range(shared_depth):
            residual_blocks.append(ResidualBlock(shared_width, self.act_func, self.dropout))

        self.shared_trunk = nn.Sequential(
            nn.Linear(concat_in, shared_width),
            nn.LayerNorm(shared_width),
            self.act_func,
            nn.Dropout(self.dropout),
            *residual_blocks,
            nn.Linear(shared_width, projection_width),
            nn.LayerNorm(projection_width),
            self.act_func,
        )

        # Separate policy/value residual refinement (optional)
        self.policy_core = ResidualBlock(projection_width, self.act_func, self.dropout) if policy_residual else nn.Identity()
        self.value_core = ResidualBlock(projection_width, self.act_func, self.dropout) if value_residual else nn.Identity()

        # Actor heads: either simple linear or per-head MLP
        actor_heads: List[nn.Module] = []
        for a_dim in self.action_dims:
            if per_head_mlp:
                actor_heads.append(
                    nn.Sequential(
                        pufferlib.pytorch.layer_init(
                            nn.Linear(projection_width, head_hidden)
                        ),
                        self.act_func,
                        nn.Dropout(self.dropout),
                        pufferlib.pytorch.layer_init(
                            nn.Linear(head_hidden, a_dim), std=0.01
                        ),
                    )
                )
            else:
                actor_heads.append(
                    pufferlib.pytorch.layer_init(
                        nn.Linear(projection_width, a_dim), std=0.01
                    )
                )
        self.actor_heads = nn.ModuleList(actor_heads)
        # Backward compatibility (some external code might reference .actor)
        if self.num_action_heads == 1:
            self.actor = self.actor_heads[0]
        self.critic_head = pufferlib.pytorch.layer_init(
            nn.Linear(projection_width, 1), std=1
        )

    def encode_observations(self, observation):
        if self.vbd_in_obs:
            ego_state, road_objects, road_graph, vbd_predictions = self.unpack_obs(
                observation
            )
        else:
            ego_state, road_objects, road_graph = self.unpack_obs(observation)

        ego_embed = self.ego_embed(ego_state)
        if self.vbd_in_obs:
            vbd_embed = self.vbd_embed(vbd_predictions)
            ego_embed = torch.cat([ego_embed, vbd_embed], dim=1)

        partner_embed, _ = self.partner_embed(road_objects).max(dim=1)
        road_map_embed, _ = self.road_map_embed(road_graph).max(dim=1)

        embed = torch.cat([ego_embed, partner_embed, road_map_embed], dim=1)
        return self.shared_trunk(embed)

    def forward(self, obs, action=None, deterministic=False):
        # Encode observations + shared trunk
        hidden = self.encode_observations(obs)
        # Policy/value refinement
        policy_latent = self.policy_core(hidden)
        value_latent = self.value_core(hidden)
        # Critic
        value = self.critic_head(value_latent)
        # Actor logits
        logits_list = [head(policy_latent) for head in self.actor_heads]
        # Sample actions
        action, logprob, ent = sample_logits(
            logits_list if self.num_action_heads > 1 else logits_list[0],
            action=action,
            deterministic=deterministic,
        )
        return action, logprob, ent, value

    def get_distribution(self, obs):
        # Encode observations + shared trunk
        hidden = self.encode_observations(obs)
        # Policy/value refinement
        policy_latent = self.policy_core(hidden)
        value_latent = self.value_core(hidden)
        # Actor logits
        logits_list = [head(policy_latent) for head in self.actor_heads]
        # Softmax
        norm_logits = [l - l.logsumexp(dim=-1, keepdim=True) for l in logits_list]
        return norm_logits

    def get_logits(self, obs):
        """
        Get the logits for the given observations without sampling actions.
        This is useful for evaluation or when you need the logits directly.
        """
        hidden = self.encode_observations(obs)
        policy_latent = self.policy_core(hidden)
        logits_list = [head(policy_latent) for head in self.actor_heads]
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
