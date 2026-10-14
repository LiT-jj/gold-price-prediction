"""
黄金 v2 · 时序 Transformer 模型

架构设计（参考现代大模型）：
  输入: [B, L, D]   L=32 交易日窗口，D=98 维（含时间特征）
  1. Patch Embedding —— 把每个交易日（或连续 2 日合并）线性投影到 d_model
  2. 双通道位置编码
       - 可学习时序位置编码（捕捉窗口内的相对位置）
       - 日历 sin/cos 编码（跨窗口的绝对时间位置）
  3. Transformer Encoder × N
       - Multi-Head Self-Attention（因果掩码，保证不偷看未来）
       - Pre-LN + GELU FFN
       - dropout
  4. 双池化：最后有效 token + mean pooling → 拼接
  5. 双头：分类头（涨/跌）+ 回归头（收益率）
  6. 联合损失：BCE + λ·MSE

★因果掩码★
  Transformer 会因为能看到整个窗口而泄漏未来——窗口内第 t 日不应
  看到 t+1 日。必须用上三角掩码强制因果性。
  这是本项目 v1 教训（严格前视防范）在深度模型上的对应实现。
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

SEED = 20261014
torch.manual_seed(SEED)
np.random.seed(SEED)

TARGETS = ["fwd_1d", "fwd_3d", "fwd_5d", "fwd_10d"]
CN = {"fwd_1d": "1日", "fwd_3d": "3日", "fwd_5d": "1周", "fwd_10d": "2周"}


# ============ 数据集 ============
class SeqDataset:
    """构建 [N, L, D] 的滑动窗口序列。"""

    def __init__(self, df, feat_cols, target, window=32):
        self.window = window
        self.feat_cols = feat_cols
        vals = df[feat_cols].values.astype(np.float32)
        # 用训练期统计量做标准化，避免测试期信息进入归一化
        self.mu = np.nanmean(vals[:int(len(vals) * 0.6)], axis=0)
        self.sd = np.nanstd(vals[:int(len(vals) * 0.6)], axis=0) + 1e-8
        Xn = (vals - self.mu) / self.sd
        Xn = np.clip(np.nan_to_num(Xn, nan=0.0, posinf=5.0, neginf=-5.0), -8, 8)

        self.X = Xn
        self.y = df[target].values.astype(np.float32)
        self.y_bin = (df[target].values > 0).astype(np.float32)
        self.index = df.index
        self.ret = self.y.copy()

    def make(self, start_pos, end_pos):
        """返回 [start+L, end) 区间内所有完整窗口。"""
        L = self.window
        starts = np.arange(start_pos, end_pos - L)
        if len(starts) == 0:
            return (np.zeros((0, L, len(self.feat_cols)), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), pd.DatetimeIndex))
        xi = np.stack([self.X[s:s + L] for s in starts])          # [N,L,D]
        yi = np.array([self.y[s + L - 1] for s in starts])        # 窗口末日之后一日
        yb = np.array([self.y_bin[s + L - 1] for s in starts])
        ri = np.array([self.ret[s + L - 1] for s in starts])
        di = self.index[np.array([s + L - 1 for s in starts])]
        return xi, yi, yb, ri, di

    def make_shuffled(self, start_pos, end_pos, rng):
        """打乱 token 时间维，用于检验模型是否真的用到时序结构。"""
        xi, yi, yb, ri, di = self.make(start_pos, end_pos)
        idx = rng.permutation(xi.shape[1])
        return xi[:, idx, :], yi, yb, ri, di


# ============ 模型 ============
class PositionalEncoding(nn.Module):
    """正弦位置编码 + 可学习日历嵌入（双通道）。"""

    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))          # [1,max_len,d]
        self.cal_proj = nn.Linear(8, d_model)               # 日历 8 维

    def forward(self, x, cal):
        # x: [B,L,d]  cal: [B,L,8]
        return x + self.pe[:, :x.size(1)] + self.cal_proj(cal)


class GoldTransformer(nn.Module):
    def __init__(self, n_feat, n_cal=8, d_model=128, nhead=8,
                 layers=3, dim_ff=256, dropout=0.2, window=32):
        super().__init__()
        self.window = window
        self.input_proj = nn.Linear(n_feat, d_model)
        self.cal_norm = nn.LayerNorm(n_cal)
        self.pos = PositionalEncoding(d_model, max_len=window + 8)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)

        self.norm = nn.LayerNorm(d_model)
        self.head_cls = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.head_reg = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1))

        self.register_buffer("mask_cache", torch.zeros(1, 1, window, window))

    def _causal_mask(self, L, device):
        m = torch.triu(torch.ones(L, L, device=device), diagonal=1)
        return m.bool()

    def forward(self, x, cal):
        B, L, _ = x.shape
        h = self.input_proj(x)                                  # [B,L,d]
        h = self.pos(h, self.cal_norm(cal))
        mask = self._causal_mask(L, x.device)
        h = self.encoder(h, mask=mask)
        h = self.norm(h)
        last = h[:, -1, :]                                      # 最后 token
        mean = h.mean(dim=1)                                   # 均值池化
        z = torch.cat([last, mean], dim=-1)
        return self.head_cls(z).squeeze(-1), self.head_reg(z).squeeze(-1)


class LSTMBaseline(nn.Module):
    """纯循环基线：检验 attention 是否真的必要。"""

    def __init__(self, n_feat, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x, cal=None):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1), torch.zeros(1)


# ============ 训练 ============
def train_model(model, Xtr, ytr_bin, rtr, Xva, yva_bin, rva,
                cal_tr=None, cal_va=None,
                epochs=60, lr=1e-3, batch=128, patience=12, verbose=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_va, best_state, bad = np.inf, None, 0
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            xb = Xtr[b]
            cb = cal_tr[b] if cal_tr is not None else torch.zeros(
                len(b), xb.shape[1], 8)
            logit, reg = model(xb, cb)
            loss = F.binary_cross_entropy_with_logits(logit, ytr_bin[b]) \
                + 0.3 * F.mse_loss(torch.tanh(reg), rtr[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if len(Xva) > 0:
            model.eval()
            with torch.no_grad():
                logit, _ = model(Xva, cal_va)
                p = torch.sigmoid(logit)
                vl = F.binary_cross_entropy(p.clamp(1e-7, 1 - 1e-7), yva_bin)
            if vl.item() < best_va:
                best_va, bad = vl.item(), 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state:
        model.load_state_dict(best_state)
    return best_va


CAL_TR = None
CAL_VA = None


def predict(model, X, cal=None):
    model.eval()
    with torch.no_grad():
        c = cal if cal is not None else torch.zeros(len(X), X.shape[1], 8)
        logit, _ = model(X, c)
        return torch.sigmoid(logit).numpy()


if __name__ == "__main__":
    print("本模块供 v2_experiment.py 调用。请运行 v2_experiment.py")