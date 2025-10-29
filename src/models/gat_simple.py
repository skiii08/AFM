# ============================================================
# Simple GATv2 for Movie Recommendation (AFM-compatible features)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class SimpleGAT(nn.Module):
    """
    GATv2-based model using user/movie node features and edge_attr (36~38D)
    - user.x: (N_u, 221)
    - movie.x: (N_m, 636)
    - edge_attr: (E, 36~38)
    - edge_index: (2, E)
    """

    def __init__(self, user_dim=221, movie_dim=636, edge_dim=38,
                 hidden_dim=128, heads=4, dropout=0.2):
        super().__init__()

        # ---- Edge projection ----
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 32, bias=True)
        )

        # ---- GATv2 layers ----
        self.gat_user2movie = GATv2Conv(
            (user_dim, movie_dim),
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=32,
            dropout=dropout,
            add_self_loops=False,
        )

        self.gat_movie2user = GATv2Conv(
            (movie_dim, user_dim),
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=32,
            dropout=dropout,
            add_self_loops=False,
        )

        # ---- Prediction layer ----
        self.fc_pred = nn.Sequential(
            nn.Linear(hidden_dim * heads * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        et = ('user', 'review', 'movie')
        edge_index = data[et].edge_index
        e_attr = data[et].edge_attr
        e_attr = self.edge_proj(e_attr)

        x_u = data['user'].x
        x_m = data['movie'].x

        # 双方向GAT
        h_m = self.gat_user2movie((x_u, x_m), edge_index, e_attr)
        h_u = self.gat_movie2user((x_m, x_u), edge_index.flip(0), e_attr)

        src, dst = edge_index
        pair_feat = torch.cat([h_u[src], h_m[dst]], dim=1)
        out = self.fc_pred(pair_feat).squeeze(1)
        return out
