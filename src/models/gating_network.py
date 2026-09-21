import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    """Query-level gating network.

    Takes CLIP query embedding (512-d) as input and outputs 3 modality weights
    per query. The same weights are applied to all candidates.

    Architecture: 3-layer MLP with LayerNorm, GELU and Dropout.
    Optional: learnable temperature and modality scaling parameters.
    """

    def __init__(self, input_dim=512, hidden_dim=128, num_modalities=3, dropout=0.3,
                 learnable_temp=False, learnable_scale=False, weight_reg=0.0,
                 init_tau_audio=0.5, init_tau_text=1.0, init_tau_visual=1.0,
                 init_scale_audio=0.8, init_scale_text=1.0, init_scale_visual=1.0,
                 constant_weights=False, init_weights=None):
        super().__init__()
        self.num_modalities = num_modalities
        self.learnable_temp = learnable_temp
        self.learnable_scale = learnable_scale
        self.weight_reg = weight_reg
        self.target_weights = None
        self.constant_weights = constant_weights

        if constant_weights:
            # Learn a single global weight vector (same for all queries)
            if init_weights is not None:
                self.logits = nn.Parameter(torch.log(torch.tensor(init_weights, dtype=torch.float32) + 1e-7))
            else:
                self.logits = nn.Parameter(torch.zeros(num_modalities))
            return

        self.layer_norm_in = nn.LayerNorm(input_dim)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dim // 2, num_modalities)

        if learnable_temp:
            self.log_tau_audio = nn.Parameter(torch.log(torch.tensor(init_tau_audio)))
            self.log_tau_text = nn.Parameter(torch.log(torch.tensor(init_tau_text)))
            self.log_tau_visual = nn.Parameter(torch.log(torch.tensor(init_tau_visual)))
        else:
            self.register_buffer('tau_audio', torch.tensor(init_tau_audio))
            self.register_buffer('tau_text', torch.tensor(init_tau_text))
            self.register_buffer('tau_visual', torch.tensor(init_tau_visual))

        if learnable_scale:
            self.log_scale_audio = nn.Parameter(torch.log(torch.tensor(init_scale_audio)))
            self.log_scale_text = nn.Parameter(torch.log(torch.tensor(init_scale_text)))
            self.log_scale_visual = nn.Parameter(torch.log(torch.tensor(init_scale_visual)))
        else:
            self.register_buffer('scale_audio', torch.tensor(init_scale_audio))
            self.register_buffer('scale_text', torch.tensor(init_scale_text))
            self.register_buffer('scale_visual', torch.tensor(init_scale_visual))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def get_temperatures(self):
        if self.learnable_temp:
            return {
                'audio': torch.exp(self.log_tau_audio).clamp(0.05, 5.0),
                'text': torch.exp(self.log_tau_text).clamp(0.05, 5.0),
                'visual': torch.exp(self.log_tau_visual).clamp(0.05, 5.0),
            }
        else:
            return {
                'audio': self.tau_audio,
                'text': self.tau_text,
                'visual': self.tau_visual,
            }

    def get_scales(self):
        if self.learnable_scale:
            return {
                'audio': torch.exp(self.log_scale_audio).clamp(0.1, 2.0),
                'text': torch.exp(self.log_scale_text).clamp(0.1, 2.0),
                'visual': torch.exp(self.log_scale_visual).clamp(0.1, 2.0),
            }
        else:
            return {
                'audio': self.scale_audio,
                'text': self.scale_text,
                'visual': self.scale_visual,
            }

    def forward(self, query_emb):
        """
        Args:
            query_emb: (batch, input_dim) query embedding (e.g., CLIP 512-d)
        Returns:
            weights: (batch, num_modalities) modality weights per query
        """
        if self.constant_weights:
            batch = query_emb.shape[0]
            w = F.softmax(self.logits, dim=-1)
            return w.unsqueeze(0).expand(batch, -1)

        x = query_emb.float() if query_emb.dtype == torch.float16 else query_emb
        x = self.layer_norm_in(x)

        h = F.gelu(self.fc1(x))
        h = self.ln1(h)
        h = self.drop1(h)

        h2 = F.gelu(self.fc2(h))
        h2 = self.ln2(h2)
        h2 = self.drop2(h2)

        out = self.fc3(h2)

        weights = F.softmax(out, dim=-1)
        return weights  # (batch, 3)

    def regularization_loss(self):
        """L2 regularization toward target weights (e.g., InverseRank heuristic).
        For constant-weights mode regularizes the learnable vector directly;
        for MLP mode the caller supplies the forward mean separately."""
        if self.constant_weights:
            current = F.softmax(self.logits, dim=-1)
            if self.target_weights is not None:
                return self.weight_reg * F.mse_loss(current, self.target_weights.to(current.device))
        return 0.0


class GatingNetworkPerCandidate(nn.Module):
    """Per-candidate gating network.

    Takes query embedding + candidate modality similarities and outputs
    per-candidate modality weights.
    """

    def __init__(self, input_dim=512, sim_dim=3, hidden_dim=64, num_modalities=3,
                 min_weight=0.0, dropout=0.2, learnable_temp=False, learnable_scale=False):
        super().__init__()
        self.min_weight = min_weight
        self.num_modalities = num_modalities
        self.learnable_temp = learnable_temp
        self.learnable_scale = learnable_scale

        # Query encoder
        self.query_fc = nn.Linear(input_dim, hidden_dim)
        self.query_ln = nn.LayerNorm(hidden_dim)
        self.query_drop = nn.Dropout(dropout)

        # Similarity encoder
        self.sim_ln = nn.LayerNorm(sim_dim)
        self.sim_fc1 = nn.Linear(sim_dim, hidden_dim)
        self.sim_ln1 = nn.LayerNorm(hidden_dim)
        self.sim_drop1 = nn.Dropout(dropout)
        self.sim_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.sim_ln2 = nn.LayerNorm(hidden_dim // 2)
        self.sim_drop2 = nn.Dropout(dropout)

        # Fusion
        self.fusion_fc = nn.Linear(hidden_dim + hidden_dim // 2, num_modalities)

        if learnable_temp:
            self.log_tau_audio = nn.Parameter(torch.log(torch.tensor(0.5)))
            self.log_tau_text = nn.Parameter(torch.log(torch.tensor(1.0)))
            self.log_tau_visual = nn.Parameter(torch.log(torch.tensor(1.0)))
        else:
            self.register_buffer('tau_audio', torch.tensor(0.5))
            self.register_buffer('tau_text', torch.tensor(1.0))
            self.register_buffer('tau_visual', torch.tensor(1.0))

        if learnable_scale:
            self.log_scale_audio = nn.Parameter(torch.log(torch.tensor(0.8)))
            self.log_scale_text = nn.Parameter(torch.log(torch.tensor(1.0)))
            self.log_scale_visual = nn.Parameter(torch.log(torch.tensor(1.0)))
        else:
            self.register_buffer('scale_audio', torch.tensor(0.8))
            self.register_buffer('scale_text', torch.tensor(1.0))
            self.register_buffer('scale_visual', torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def get_temperatures(self):
        if self.learnable_temp:
            return {
                'audio': torch.exp(self.log_tau_audio).clamp(0.05, 5.0),
                'text': torch.exp(self.log_tau_text).clamp(0.05, 5.0),
                'visual': torch.exp(self.log_tau_visual).clamp(0.05, 5.0),
            }
        else:
            return {'audio': self.tau_audio, 'text': self.tau_text, 'visual': self.tau_visual}

    def get_scales(self):
        if self.learnable_scale:
            return {
                'audio': torch.exp(self.log_scale_audio).clamp(0.1, 2.0),
                'text': torch.exp(self.log_scale_text).clamp(0.1, 2.0),
                'visual': torch.exp(self.log_scale_visual).clamp(0.1, 2.0),
            }
        else:
            return {'audio': self.scale_audio, 'text': self.scale_text, 'visual': self.scale_visual}

    def forward(self, query_emb, sim_v, sim_t, sim_a):
        """
        Args:
            query_emb: (batch, input_dim) query embedding
            sim_v: (batch, n_candidates) visual similarities
            sim_t: (batch, n_candidates) text similarities
            sim_a: (batch, n_candidates) audio similarities
        Returns:
            weights: (batch, n_candidates, num_modalities) per-candidate weights
        """
        batch, n = sim_v.shape
        x = query_emb.float() if query_emb.dtype == torch.float16 else query_emb
        q_feat = self.query_drop(F.gelu(self.query_ln(self.query_fc(x))))  # (batch, hidden)

        x_sim = torch.stack([sim_v, sim_t, sim_a], dim=-1)  # (batch, n, 3)
        x_sim = self.sim_ln(x_sim)
        h = self.sim_drop1(F.gelu(self.sim_ln1(self.sim_fc1(x_sim))))
        h2 = self.sim_drop2(F.gelu(self.sim_ln2(self.sim_fc2(h))))  # (batch, n, hidden//2)

        # Broadcast query features to all candidates and concatenate
        q_feat_expanded = q_feat.unsqueeze(1).expand(-1, n, -1)  # (batch, n, hidden)
        fused = torch.cat([q_feat_expanded, h2], dim=-1)  # (batch, n, hidden + hidden//2)

        out = self.fusion_fc(fused)  # (batch, n, 3)

        weights = F.softmax(out, dim=-1)
        if self.min_weight > 0:
            weights = torch.clamp(weights, min=self.min_weight)
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights

    def regularization_loss(self):
        if self.learnable_temp:
            tau = self.get_temperatures()
            reg = (tau['audio'] - 0.5).pow(2) + (tau['text'] - 1.0).pow(2) + (tau['visual'] - 1.0).pow(2)
            return 0.01 * reg
        return 0.0
