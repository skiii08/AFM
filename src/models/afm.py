# src/models/afm.py
# ============================================================
# Attentional Factorization Machine v2
# Edge-MLP + Sentiment Conditioned Attention + LayerNorm
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class AFM(nn.Module):
    """
    AFM v2:
      - Edge特徴をMLPで圧縮 (36 → 16)
      - Sentiment差分(18D)をattentionに注入
      - LayerNormで安定化
    """

    def __init__(self,
                 user_dim: int,
                 movie_dim: int,
                 edge_dim: int,
                 embedding_dim: int = 32,
                 attn_size: int = 128,
                 dropout: float = 0.2):
        super().__init__()

        # --- 埋め込み層 ---
        self.user_embed = nn.Linear(user_dim, embedding_dim, bias=False)
        self.movie_embed = nn.Linear(movie_dim, embedding_dim, bias=False)
        self.norm_u = nn.LayerNorm(embedding_dim)
        self.norm_m = nn.LayerNorm(embedding_dim)

        # --- Edge前処理 (36→32→16) ---
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # --- Sentiment Conditioned Attention ---
        self.attn_input_dim = embedding_dim + 16 + 18
        self.attn_layer = nn.Sequential(
            nn.Linear(self.attn_input_dim, attn_size),
            nn.ReLU(),
            nn.Linear(attn_size, 1, bias=False)
        )

        # --- その他パラメータ ---
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear_u = nn.Linear(user_dim, 1, bias=False)
        self.linear_m = nn.Linear(movie_dim, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.movie_embed.weight)
        for m in self.edge_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        for m in self.attn_layer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.linear_u.weight)
        nn.init.xavier_uniform_(self.linear_m.weight)

    def forward(self, u_feat, m_feat, e_attr):
        """
        u_feat: (B, user_dim)
        m_feat: (B, movie_dim)
        e_attr: (B, 36)
        """

        # --- 埋め込み ---
        v_u = self.norm_u(self.user_embed(u_feat))
        v_m = self.norm_m(self.movie_embed(m_feat))
        interaction = v_u * v_m  # (B, k)

        # --- Edge特徴変換 ---
        e_proj = self.edge_proj(e_attr)

        # --- Sentiment差分 (3.0中心化) ---
        if e_attr.size(1) >= 36:
            sentiment = e_attr[:, 18:36] - 3.0
        else:
            sentiment = e_attr.new_zeros(e_attr.size(0), 18)

        # --- Attention計算 ---
        attn_input = torch.cat([interaction, e_proj, sentiment], dim=-1)
        a = self.attn_layer(attn_input)  # (B, 1)

        # --- 加重交互作用 ---
        weighted = self.dropout(interaction * a)
        inter_sum = weighted.sum(dim=1)  # (B,)

        # --- 線形項 ---
        linear = self.bias + self.linear_u(u_feat).squeeze(1) + self.linear_m(m_feat).squeeze(1)

        # --- 最終出力 ---
        y_hat = linear + inter_sum
        return y_hat
