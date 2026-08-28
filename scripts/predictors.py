# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved

import torch
import random, math
# from ..models.gin_predictor import NasBenchGINPredictorAgent , NasBenchGINPredictorAgent_Reg
# from ..models.gin_predictor_celu import NasBenchGINPredictorAgentCELU
# from gnn_lib.data import Data, Batch
# from ..utils.metric_logger import MetricLogger
# from ..utils.utils_solver import CosineLR, gen_batch_idx, make_agent_optimizer
from torch_geometric.nn import SAGEConv, GCNConv, GINConv, global_mean_pool, BatchNorm, JumpingKnowledge, GATConv, global_add_pool
from torch import nn, optim
import torch.nn.functional as F

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mlp(in_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )

# permformer.py
# -------------------------
# helper: sinusoidal rank embedding
# -------------------------
def sinusoid_rank_embed(rank_idx: torch.Tensor, dim: int) -> torch.Tensor:
    """
    rank_idx: (B, L)  各位置のランク(0..L-1) ※パディング位置は使わない前提でゼロにしてOK
    return: (B, L, dim)
    """
    device = rank_idx.device
    half = dim // 2
    if half == 0:
        return torch.zeros(rank_idx.shape + (dim,), device=device)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device).float() / half)
    angles = rank_idx[..., None].float() * freqs  # (B,L,half)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B,L,2*half)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb

# -------------------------
# helper: ALiBi bias (relative distance bias)
# -------------------------
def build_alibi_avg(L: int, n_heads: int, device: torch.device) -> torch.Tensor:
    """
    Return averaged ALiBi bias for (L, L), usable as additive attn_mask.
    """
    # slopes
    def get_slopes(n):
        m = 1 << (n - 1).bit_length() // 2 * 2  # next lower power of two heuristic
        m = 2 ** int(math.floor(math.log2(n)))   # standard: largest power of 2 <= n
        base = torch.pow(2.0, -torch.arange(0, m, device=device).float() / m)
        if m < n:
            extra = torch.pow(2.0, -torch.arange(1, 2*(n-m)+1, 2, device=device).float() / m)
            base = torch.cat([base, extra], dim=0)
        return base[:n]
    slopes = get_slopes(n_heads)  # (H,)

    pos = torch.arange(L, device=device)
    dist = (pos[None, :] - pos[:, None]).abs().float()  # (L, L)
    # bias per head: -(distance) * slope
    bias_heads = -(dist[None, :, :] * slopes[:, None, None])  # (H, L, L)
    alibi_avg = bias_heads.mean(dim=0)  # (L, L)
    return alibi_avg

# -------------------------
# PermFormer
# -------------------------
class PermFormerModel(nn.Module):
    """
    Permutation-only -> scalar regression.
    - Variable length supported with padding_id (= max_n)
    - Token embedding for "item id"
    - Sinusoidal rank embedding for "position"
    - ALiBi relative bias (averaged across heads, passed as additive attn_mask)
    """
    def __init__(
        self,
        max_n: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.2,
        use_cls: bool = True,
        seed = 0
    ):
        super().__init__()
        self.max_n = max_n
        self.pad_id = max_n                 # 0..max_n-1 are items, max_n is [PAD]
        self.use_cls = use_cls
        self.d_model = d_model

        # +1 for padding token
        self.item_embed = nn.Embedding(num_embeddings=max_n + 1, embedding_dim=d_model, padding_idx=self.pad_id)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,  # ★ これを追加
        )

        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls else None

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        # set_seed(seed)

    def forward(self, perm: torch.Tensor) -> torch.Tensor:
        """
        perm: LongTensor (B, L) with values in {0..max_n-1} and padding = -1 allowed.
            If you do not use padding, simply pass 0..n-1 and L=n.
        returns: (B,) float
        """
        device = perm.device
        B, L = perm.shape

        # map -1 -> pad_id
        if (perm < 0).any():
            perm = perm.clone()
            perm[perm < 0] = self.pad_id

        # embeddings
        tok = self.item_embed(perm)  # (B, L, d)

        # rank embedding only for valid positions; paddings will be zeroed via mask later
        rank_idx = torch.arange(L, device=device)[None, :].expand(B, L)  # (B, L)
        pos = sinusoid_rank_embed(rank_idx, self.d_model)                # (B, L, d)

        x = tok + pos  # (B, L, d)

        # [CLS]
        if self.use_cls:
            cls_tok = self.cls.expand(B, 1, self.d_model)  # (B, 1, d)
            x = torch.cat([cls_tok, x], dim=1)             # (B, L+1, d)
            L_eff = L + 1
        else:
            L_eff = L

        # key padding mask: True = mask(pad)
        pad_mask = (perm == self.pad_id)  # (B, L)
        if self.use_cls:
            # prepend False for the CLS position
            pad_mask = torch.cat(
                [torch.zeros(B, 1, dtype=torch.bool, device=device), pad_mask],
                dim=1
            )  # (B, L_eff)

        pad_mask_float = pad_mask.to(x.dtype)
        pad_mask_float = pad_mask_float.masked_fill(pad_mask, float("-inf"))
        # ALiBi (averaged bias) as additive float mask
        alibi = build_alibi_avg(
            L_eff,
            self.encoder.layers[0].self_attn.num_heads,
            device
        ).to(x.dtype)  # (L_eff, L_eff)

        # encode (float attn mask, bool key_padding_mask)
        x = self.encoder(
            x,
            mask=alibi,                    # (L_eff, L_eff) float additive bias
            src_key_padding_mask=pad_mask_float  # (B, L_eff) bool
        )
        x = self.dropout(x)

        # pooling
        if self.use_cls:
            g = x[:, 0]  # CLS
        else:
            valid = ~pad_mask  # (B, L_eff)
            g = (x * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)

        # head
        y = self.head(g).squeeze(-1)  # (B,)
        return y

class GINModel(nn.Module):
    """
    GIN 回帰モデル
    - in_dim   : 入力ノード特徴次元
    - hidden   : 隠れ次元
    - layers   : GINConv 層数
    - dropout  : ドロップアウト率
    - use_bn   : 各層に BatchNorm を入れるか
    - residual : 残差接続を使うか（形が合う時のみ）
    - train_eps: GIN の ε を学習可能にするか
    """
    def __init__(self, in_dim, hidden=128, layers=3, dropout=0.2,
                use_bn=True, residual=False, train_eps=True, seed=0):
        super().__init__()
        assert layers >= 1

        self.dropout  = dropout
        self.use_bn   = use_bn
        self.residual = residual
        # set_seed(seed)

        # GINConv スタック（最初だけ in_dim→hidden、以降は hidden→hidden）
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(mlp(in_dim, hidden, hidden), train_eps=train_eps))
        for _ in range(layers - 1):
            self.convs.append(GINConv(mlp(hidden, hidden, hidden), train_eps=train_eps))

        if use_bn:
            self.bns = nn.ModuleList([BatchNorm(hidden) for _ in range(layers)])
        else:
            self.bns = None

        # グラフ回帰ヘッド
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, conv in enumerate(self.convs):
            h = conv(x, edge_index)
            if self.use_bn:
                h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h if (self.residual and x.shape == h.shape) else h  # 残差（形一致時）

        g = global_add_pool(x, batch)  # グラフ読み出し（mean）
        out = self.head(g).squeeze(-1)
        return out

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_dim, hidden=128, layers=4, dropout=0.2, seed=0):
        super().__init__()
        self.dropout = dropout
        # set_seed(seed)
        ch = [in_dim] + [hidden] * layers
        self.convs = torch.nn.ModuleList([SAGEConv(ch[i], ch[i+1]) for i in range(layers)])
        self.head  = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden//2, 1)
        )
    def forward(self, data, return_emb = False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        out = self.head(g).squeeze(-1)
        if return_emb:
            return out, g
        return out

class JKGCNModel(nn.Module):
    """
    GCN + Jumping Knowledge で多スケール表現を融合する回帰モデル。
    - jk_mode: 'cat' | 'max' | 'lstm'
      * 'cat'  : 全層を連結（出力次元 = hidden * layers）
      * 'max'  : 次元ごとに層方向 max（出力次元 = hidden）
      * 'lstm' : 層列を BiLSTM で融合（出力次元 = hidden）
    """
    def __init__(self, in_dim, hidden=128, layers=4, dropout=0.2,
                jk_mode='lstm', use_bn=True, seed=0):
        super().__init__()
        assert layers >= 2, "layers は 2 以上を推奨（JK の旨味が出ます）"
        assert jk_mode in ['cat', 'max', 'lstm']

        self.hidden   = hidden
        self.layers   = layers
        self.dropout  = dropout
        self.jk_mode  = jk_mode
        self.use_bn   = use_bn
        # set_seed(seed)

        # GCN 層（最初だけ in_dim→hidden、以降は hidden→hidden）
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))

        # BatchNorm（任意）
        if use_bn:
            self.bns = nn.ModuleList([BatchNorm(hidden) for _ in range(layers)])
        else:
            self.bns = None

        # Jumping Knowledge
        if jk_mode == 'cat':
            jk_out_dim = hidden * layers
            self.jk = JumpingKnowledge(mode='cat')
        elif jk_mode == 'max':
            jk_out_dim = hidden
            self.jk = JumpingKnowledge(mode='max')
        else:  # 'lstm'
            jk_out_dim = hidden
            self.jk = JumpingKnowledge(mode='lstm', channels=hidden, num_layers=layers)

        # 回帰ヘッド
        self.head = nn.Sequential(
            nn.Linear(jk_out_dim, jk_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(jk_out_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        xs = []  # 各層のノード埋め込みを貯める
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        # 層方向（時間方向）に沿って Jumping Knowledge で融合（ノードごと）
        x_jk = self.jk(xs)  # [num_nodes, jk_out_dim]

        # グラフ読み出し（順列不変）：mean pooling
        g = global_mean_pool(x_jk, batch)  # [num_graphs, jk_out_dim]

        out = self.head(g).squeeze(-1)     # [num_graphs]
        return out

class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2, dropout=0.2, seed=0):
        super().__init__()
        # set_seed(seed)
        self.dropout = dropout
        ch = [in_dim] + [hidden] * layers
        self.convs = nn.ModuleList([GCNConv(ch[i], ch[i+1]) for i in range(layers)])
        self.head  = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        out = self.head(g).squeeze(-1)
        return out

class GATModel(torch.nn.Module):
    """
    Graph Attention Network (GAT) によるグラフ回帰モデル
    """
    def __init__(self, in_dim, hidden=128, heads=2, layers=3, dropout=0.2, seed=0):
        super().__init__()
        self.dropout = dropout
        # set_seed(seed)

        ch = [in_dim] + [hidden] * layers
        self.convs = torch.nn.ModuleList()
        self.heads = heads

        # GATConv 層スタック
        for i in range(layers):
            self.convs.append(
                GATConv(
                    in_channels=ch[i],
                    out_channels=hidden,
                    heads=heads,
                    concat=False,        # 各ヘッドを平均（Trueなら結合）
                    dropout=dropout
                )
            )

        # 回帰ヘッド
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, batch)
        out = self.head(g).squeeze(-1)
        return out