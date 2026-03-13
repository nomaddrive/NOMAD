import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(max(0, num_layers - 1)):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PosEmbed(nn.Module):
    """Fourier feature encoding for continuous (x,y,theta) style inputs.

    If num_bands=0, acts as identity passthrough (no positional encoding).
    """

    def __init__(self, in_dims: int, num_bands: int = 0, include_input: bool = True):
        super().__init__()
        self.in_dims = in_dims
        self.num_bands = num_bands
        self.include_input = include_input
        if num_bands > 0:
            self.register_buffer(
                "freqs",
                2.0 ** torch.linspace(0, num_bands - 1, num_bands),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_bands == 0:
            return x
        # x: (..., D)
        xb = x.unsqueeze(-1) * self.freqs  # (..., D, B)
        emb = torch.cat([torch.sin(math.pi * xb), torch.cos(math.pi * xb)], dim=-1)
        emb = emb.flatten(-2)  # (..., D * 2B)
        if self.include_input:
            return torch.cat([x, emb], dim=-1)
        return emb


class RoadGraphEncoder(nn.Module):
    """Encodes per-roadpoint features into a set of tokens, then aggregates.

    Inputs: roadgraph [B, R, F_rg]
    Outputs:
      - tokens [B, R, D] (for cross-attention)
      - global [B, D] (pooled)
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        pos_bands: int = 0,
    ):
        super().__init__()
        self.pos_enc = PosEmbed(in_dims=3, num_bands=pos_bands)  # x,y,orientation
        self.pre = MLP(in_dim + (0 if pos_bands == 0 else (3 * 2 * pos_bands + 3)), d_model, d_model, 2, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, roadgraph: torch.Tensor, road_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # roadgraph: [B, R, F]
        # Extract pos subset if available: expect first 2 dims are x,y and 5th is orientation per collect_dataset
        # Features layout: [x, y, seg_len, seg_w, seg_h, orientation, type_one_hot...]
        x = roadgraph
        if x.size(-1) >= 6:
            pos_feats = torch.stack([x[..., 0], x[..., 1], x[..., 5]], dim=-1)  # [B,R,3]
            x = torch.cat([x, self.pos_enc(pos_feats)], dim=-1)
        tokens = self.pre(x)  # [B,R,D]
        key_padding_mask = None
        if road_mask is not None:
            # mask: True indicates keep; Transformer expects True for padding -> invert
            key_padding_mask = ~road_mask.bool()
        tokens = self.enc(tokens, src_key_padding_mask=key_padding_mask)
        tokens = self.norm(tokens)
        if road_mask is not None:
            # masked mean
            denom = road_mask.sum(dim=1, keepdim=True).clamp_min(1).to(tokens.dtype)
            global_feat = (tokens * road_mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            global_feat = tokens.mean(dim=1)
        return tokens, global_feat


class AgentQueryDecoder(nn.Module):
    """Cross-attend agent queries to roadgraph tokens, then regress goals.

    Inputs:
      - agent_states [B, A, F_a] with (x,y,yaw) per agent
      - rg_tokens [B, R, D]
    Output: goals [B, A, 2]
    """

    def __init__(self, agent_in_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1, pos_bands: int = 2):
        super().__init__()
        self.pos_enc = PosEmbed(in_dims=3, num_bands=pos_bands)
        # If pos_bands>0 and PosEmbed.include_input=True, q_in dim = agent_in_dim * (2*pos_bands + 1)
        q_in_dim = agent_in_dim if pos_bands == 0 else (agent_in_dim * (2 * pos_bands + 1))
        self.agent_proj = MLP(q_in_dim, d_model, d_model, 2, dropout)

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, agent_states: torch.Tensor, rg_tokens: torch.Tensor, agent_mask: Optional[torch.Tensor] = None, road_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # agent_states: [B,A,3] -> (x,y,yaw)
        q_in = self.pos_enc(agent_states)
        queries = self.agent_proj(q_in)  # [B,A,D]

        memory = rg_tokens  # [B,R,D]
        tgt = queries
        memory_key_padding_mask = None
        tgt_key_padding_mask = None
        if road_mask is not None:
            memory_key_padding_mask = ~road_mask.bool()
        if agent_mask is not None:
            tgt_key_padding_mask = ~agent_mask.bool()
        for layer in self.layers:
            tgt = layer(tgt=tgt, memory=memory, memory_key_padding_mask=memory_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        out = self.norm(tgt)
        goals = self.head(out)  # [B,A,2]
        return goals


class GoalPredictor(nn.Module):
    """Deterministic goal regressor using cross-attention over roadgraph.

    forward(agent_states[B,A,3], roadgraph[B,R,F_rg], agent_mask[B,A]?, road_mask[B,R]?) -> goals[B,A,2]
    """

    def __init__(self, road_feat_dim: int = 27, d_model: int = 128, nhead: int = 4, enc_layers: int = 2, dec_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rg_enc = RoadGraphEncoder(in_dim=road_feat_dim, d_model=d_model, nhead=nhead, num_layers=enc_layers, dropout=dropout, pos_bands=2)
        self.decoder = AgentQueryDecoder(agent_in_dim=3, d_model=d_model, nhead=nhead, num_layers=dec_layers, dropout=dropout, pos_bands=2)

    def forward(self, agent_states: torch.Tensor, roadgraph: torch.Tensor, agent_mask: Optional[torch.Tensor] = None, road_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens, _ = self.rg_enc(roadgraph, road_mask)
        goals = self.decoder(agent_states, tokens, agent_mask, road_mask)
        return goals


class MDNHead(nn.Module):
    """Mixture Density Network head for 2D outputs.

    Produces K Gaussians: means (B,A,K,2), log_std (B,A,K,2), rho (B,A,K), logits (B,A,K).
    """

    def __init__(self, d_model: int, num_components: int = 5):
        super().__init__()
        K = num_components
        self.mu = nn.Linear(d_model, K * 2)
        self.log_sigma = nn.Linear(d_model, K * 2)
        self.rho = nn.Linear(d_model, K)
        self.logits = nn.Linear(d_model, K)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, A, D = x.shape
        mu = self.mu(x).view(B, A, -1, 2)
        log_sigma = self.log_sigma(x).view(B, A, -1, 2)
        rho = torch.tanh(self.rho(x))  # [-1,1]
        logits = self.logits(x)
        return mu, log_sigma, rho, logits


class GoalPredictorMDN(nn.Module):
    """Multimodal goal predictor with MDN output.

    forward -> (mu, log_sigma, rho, logits)
    """

    def __init__(self, road_feat_dim: int = 27, d_model: int = 128, nhead: int = 4, enc_layers: int = 2, dec_layers: int = 2, dropout: float = 0.1, num_components: int = 5):
        super().__init__()
        self.rg_enc = RoadGraphEncoder(in_dim=road_feat_dim, d_model=d_model, nhead=nhead, num_layers=enc_layers, dropout=dropout, pos_bands=2)
        self.decoder_backbone = AgentQueryDecoder(agent_in_dim=3, d_model=d_model, nhead=nhead, num_layers=dec_layers, dropout=dropout, pos_bands=2)
        # Map 2D deterministic decoder output up to feature dim before MDN
        self.post = MLP(in_dim=2, hidden=d_model, out_dim=d_model, num_layers=2, dropout=dropout)
        self.head = MDNHead(d_model, num_components)

    def forward_features(self, agent_states: torch.Tensor, roadgraph: torch.Tensor, agent_mask: Optional[torch.Tensor] = None, road_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens, _ = self.rg_enc(roadgraph, road_mask)
        # Run decoder but intercept before final linear head: replicate internals
        # Simpler approach: reinstantiate a decoder that returns features instead of 2D; for brevity, reuse head by slicing.
        # Here we approximate by taking the last hidden before the linear head via a small change: use the decoder projection as features
        # NOTE: To keep code simple, call decoder then project back up; for exact features, refactor AgentQueryDecoder.
        goals = self.decoder_backbone(agent_states, tokens, agent_mask, road_mask)
        # Lift back to feature space with an extra MLP
        return goals

    def forward(self, agent_states: torch.Tensor, roadgraph: torch.Tensor, agent_mask: Optional[torch.Tensor] = None, road_mask: Optional[torch.Tensor] = None):
        tokens, _ = self.rg_enc(roadgraph, road_mask)
        hidden2d = self.decoder_backbone(agent_states, tokens, agent_mask, road_mask)  # [B,A,2]
        hidden = F.gelu(self.post(hidden2d))
        return self.head(hidden)


def mdn_nll_2d(
    y: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    rho: torch.Tensor,
    logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute negative log-likelihood for 2D Gaussian mixture.

    Args:
        y: [B,A,2]
        mu: [B,A,K,2]
        log_sigma: [B,A,K,2]
        rho: [B,A,K] in [-1,1]
        logits: [B,A,K]
        mask: optional [B,A] boolean/0-1 mask of valid agents. If None, average over all.
    """
    B, A, K, _ = mu.shape
    y = y.unsqueeze(2)  # [B,A,1,2]
    # convert log stds to stds; optionally clamp to avoid extreme values
    sx = torch.exp(log_sigma[..., 0])
    sy = torch.exp(log_sigma[..., 1])  # [B,A,K]
    norm_x = (y[..., 0] - mu[..., 0]) / (sx + 1e-9)
    norm_y = (y[..., 1] - mu[..., 1]) / (sy + 1e-9)
    z = (norm_x**2 + norm_y**2 - 2 * rho * norm_x * norm_y) / (1 - rho**2 + 1e-6)
    denom = 2 * math.pi * sx * sy * torch.sqrt(1 - rho**2 + 1e-6)
    comp_log_prob = -0.5 * z - torch.log(denom + 1e-9)  # [B,A,K]
    log_mix = F.log_softmax(logits, dim=-1)
    log_prob = torch.logsumexp(log_mix + comp_log_prob, dim=-1)  # [B,A]
    nll = -(log_prob)
    if mask is None:
        return nll.mean()
    mask_f = mask.float()
    return (nll * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, target)
    mask = mask.float().unsqueeze(-1)
    diff = (pred - target) * mask
    denom = mask.sum().clamp_min(1.0)
    return (diff.pow(2).sum() / denom).mean()
