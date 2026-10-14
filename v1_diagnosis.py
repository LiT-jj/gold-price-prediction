"""
v1 结论的严格诊断汇总（用于报告）
"""
import os
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
SPLIT = "2019-01-01"
TARGETS = ["fwd_1d", "fwd_3d", "fwd_5d", "fwd_10d"]
CN = {"fwd_1d": "1日", "fwd_3d": "3日", "fwd_5d": "1周", "fwd_10d": "2周"}
ALL_T = TARGETS

df = pd.read_csv(os.path.join(BASE, "data", "gold_features.csv"),
                 index_col=0, parse_dates=True)
feats = [c for c in df.columns if c not in ALL_T + ["close"]]

rows = []
for tgt in TARGETS:
    sub = df[[tgt] + feats].dropna()
    y = (sub[tgt] > 0).astype(int)
    X = sub[feats]
    m = np.asarray(sub.index >= SPLIT)
    prob = np.load(os.path.join(OUT, f"pred_{tgt}_logistic.npy"))
    if len(prob) != m.sum():
        continue
    ret = sub.loc[m, tgt].values
    act = y[m].values
    base = max(act.mean(), 1 - act.mean())
    pred = (prob > 0.5).astype(int)
    correct = (pred == act).astype(float)

    sig = np.where(prob > 0.6, 1, np.where(prob < 0.4, -1, 0))
    act_mask = sig != 0
    n_l = int((sig[act_mask] == 1).sum())
    n_s = int((sig[act_mask] == -1).sum())
    expo = (n_l - n_s) / max(1, (n_l + n_s))
    strat = sig[act_mask] * ret[act_mask]
    diff_bp = (strat.mean() - ret[act_mask].mean()) * 1e4

    rows.append({
        "target": tgt, "cn": CN[tgt],
        "acc": correct.mean(), "baseline": base,
        "edge": correct.mean() - base,
        "n_long": n_l, "n_short": n_s,
        "net_long_expo": expo,
        "strategy_minus_bh_bp": diff_bp,
        "bh_bp": ret.mean() * 1e4,
    })

out = pd.DataFrame(rows)
print("=" * 78)
print("v1 诊断汇总：为什么'看起来能赚'其实是 beta 暴露")
print("=" * 78)
print(f"{'周期':<6s}{'准确率':>9s}{'基线':>9s}{'超额':>9s}"
      f"{'多/空笔数':>14s}{'净多头':>9s}{'策略−持有':>12s}")
print("-" * 78)
for _, r in out.iterrows():
    print(f"{r['cn']:<6s}{r['acc']*100:>8.2f}%{r['baseline']*100:>8.2f}%"
          f"{r['edge']*100:>+8.2f}pp"
          f"{str(int(r['n_long']))+'/'+str(int(r['n_short'])):>14s}"
          f"{r['net_long_expo']*100:>+8.1f}%{r['strategy_minus_bh_bp']:>+11.2f}bp")
print("-" * 78)
print("\n结论：四个周期的方向准确率全部低于各自多数类基线；")
print("      '扣成本后净盈利'完全来自净多头敞口 × 黄金单边上涨。")
print("      策略收益减去买入持有基准后，1周/1日均为负或接近零。")

out.to_csv(os.path.join(OUT, "v1_diagnosis.csv"),
           index=False, encoding="utf-8-sig")
json.dump({"rows": out.to_dict("records")},
          open(os.path.join(OUT, "v1_diagnosis.json"), "w",
               encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
print(f"\n已存 {os.path.join(OUT, 'v1_diagnosis.csv')}")