import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv

# ============================================================
# GATDualPhase: Low-Rating Sensitive Attention GAT
# ============================================================
class GATDualPhase(nn.Module):
    def __init__(self, user_dim, movie_dim, edge_dim,
                 hidden_dim=128, heads=2, dropout=0.2,
                 lambda_contrast=0.4, alpha_dis=0.6):
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.alpha_dis = alpha_dis

        # -----------------------------
        # Edge feature projection
        # -----------------------------
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, 64, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # -----------------------------
        # Satisfaction-phase encoder
        # -----------------------------
        self.gat_sat = HeteroConv({
            ('user', 'review', 'movie'): GATv2Conv(
                (user_dim, movie_dim),
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False
            ),
            ('movie', 'rev_by', 'user'): GATv2Conv(
                (movie_dim, user_dim),
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False
            )
        })

        # -----------------------------
        # Dissatisfaction-phase encoder
        # -----------------------------
        self.gat_dis = HeteroConv({
            ('user', 'review', 'movie'): GATv2Conv(
                (user_dim, movie_dim),
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False
            ),
            ('movie', 'rev_by', 'user'): GATv2Conv(
                (movie_dim, user_dim),
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False
            )
        })

        # -----------------------------
        # Fusion + Regression Head
        # -----------------------------
        self.user_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.movie_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, data):
        x_u, x_m = data['user'].x, data['movie'].x
        e_attr = data['user', 'review', 'movie'].edge_attr

        # ---- Edge attributes dimensional alignment ----
        in_dim = self.edge_proj[0].in_features
        if e_attr.size(1) > in_dim:
            e_attr = e_attr[:, :in_dim]
        elif e_attr.size(1) < in_dim:
            pad = torch.zeros(e_attr.size(0), in_dim - e_attr.size(1), device=e_attr.device)
            e_attr = torch.cat([e_attr, pad], dim=1)

        e_emb = self.edge_proj(e_attr)
        edge_index = data['user', 'review', 'movie'].edge_index

        # ---- Dual-phase propagation ----
        x_sat = self.gat_sat(
            {'user': x_u, 'movie': x_m},
            {('user', 'review', 'movie'): edge_index,
             ('movie', 'rev_by', 'user'): edge_index.flip(0)}
        )

        x_dis = self.gat_dis(
            {'user': x_u, 'movie': x_m},
            {('user', 'review', 'movie'): edge_index,
             ('movie', 'rev_by', 'user'): edge_index.flip(0)}
        )

        u_sat, m_sat = x_sat['user'], x_sat['movie']
        u_dis, m_dis = x_dis['user'], x_dis['movie']

        # ---- Feature fusion ----
        u_fused = torch.cat([u_sat, u_dis], dim=-1)
        m_fused = torch.cat([m_sat, m_dis], dim=-1)

        u_final = F.relu(self.user_fc(u_fused))
        m_final = F.relu(self.movie_fc(m_fused))

        # ---- Prediction (per edge) ----
        u_idx, m_idx = edge_index
        h_u = u_final[u_idx]
        h_m = m_final[m_idx]
        h = torch.cat([h_u, h_m], dim=-1)
        y_pred = self.pred_head(h).squeeze()

        return y_pred, (u_sat, u_dis, m_sat, m_dis)
