"""
★关键审查：2周期的"超额"到底是择时能力还是幸存者偏差？

疑点：2026 年买入持有 −125.36bp，策略 +22.86bp，超额 +148.21bp。
      这个巨额超额可能完全来自「躲开了一次暴跌」，而非日常的择时微调。
检验：剔除2026（或任一单一年份）后超额是否仍然成立。
      这与彩票项目里的"多重比较"同源——不能靠个别异常年份撑结论。
"""
import os
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
SPLIT = "2019-01-01"
SEED = 20261014
rng = np.random.default_rng(SEED)

df = pd.read_csv(os.path.join(BASE, "data", "gold_features.csv"),
                 index_col=0, parse_dates=True)
feats = [c for c in df.columns
         if c not in ["fwd_1d", "fwd_3d", "fwd_5d", "fwd_10d", "close"]]

tgt = "fwd_10d"
sub = df[[tgt] + feats].dropna()
m = np.asarray(sub.index >= SPLIT)
ret = sub.loc[m, tgt].values
idx = np.asarray(sub.index)[m]
prob = np.load(os.path.join(OUT, f"pred_{tgt}_logistic.npy"))
sig = np.where(prob > 0.6, 1, np.where(prob < 0.4, -1, 0))
act = sig != 0

d = pd.DataFrame({
    "dt": idx[act], "strat": sig[act] * ret[act], "bh": ret[act]})
d["year"] = pd.DatetimeIndex(d["dt"]).year
d["diff"] = d["strat"] - d["bh"]

print("=" * 72)
print("★ 留一法审查：剔除任一年度后，超额是否仍成立")
print("=" * 72)
full = d["diff"].mean() * 1e4
print(f"  全样本超额 {full:+.2f}bp\n")
print(f"{'剔除年份':<10s}{'剩余超额':>12s}{'剩余样本':>10s}  判定")
print("-" * 72)
loo = {}
for yr in sorted(d["year"].unique()):
    sub_d = d[d["year"] != yr]
    diff = sub_d["diff"].mean() * 1e4
    loo[yr] = float(diff)
    ok = "仍成立" if diff > 0 else "★消失"
    print(f"{yr:<10d}{diff:>+11.2f}bp{len(sub_d):>10d}  {ok}")
print("-" * 72)
n_neg = sum(1 for v in loo.values() if v <= 0)
print(f"  剔除后超额转负的年份数: {n_neg}/{len(loo)}")

# ---- 关键：只看剔除 2026 后的表现 ----
print("\n" + "=" * 72)
print("重点：剔除 2026 年（持有 −125bp 的灾难年）后")
print("=" * 72)
d2 = d[d["year"] != 2026]
diff2 = d2["diff"].mean() * 1e4
print(f"  剩余超额 {diff2:+.2f}bp   样本 {len(d2)} 笔")
print(f"  {'结论依赖2026 的极端行情，不稳健' if diff2 <= 5 else '剔除后仍有可观超额'}")

# ---- 进一步：按持有收益分组，看模型是否只在熊市有效 ----
print("\n" + "=" * 72)
print("分组检验：模型在什么市况下有效？")
print("=" * 72)
d["bh_q"] = pd.qcut(d["bh"], 5, labels=["最差20%", "较差", "中性", "较好", "最好20%"])
for q, g in d.groupby("bh_q", observed=True):
    diff = g["diff"].mean() * 1e4
    acc_corr = None
    print(f"  {str(q):<10s} n={len(g):4d}  "
          f"持有{g['bh'].mean()*1e4:+8.2f}bp  策略{g['strat'].mean()*1e4:+8.2f}bp"
          f"  超额{diff:+8.2f}bp")
print("\n  → 若超额主要出现在'最差20%'组，说明模型只学会了避开下跌行情。")

# 统计：超额与持有收益的相关性
corr = np.corrcoef(d["bh"], d["diff"])[0, 1]
print(f"  超额与持有收益的相关系数 = {corr:+.4f}")
print(f"  {'负相关 → 超额来自避险而非择时' if corr < -0.1 else '相关性弱'}")

# ---- 结论 ----
print("\n" + "=" * 72)
print("判定")
print("=" * 72)
verdict = {
    "loo_all_positive": bool(n_neg == 0),
    "excl_2026_bp": float(diff2),
    "corr_with_bh": float(corr),
    "depends_on_extreme": bool(diff2 <= 5),
}
print(f"  留一法全部为正: {verdict['loo_all_positive']}")
print(f"  剔除2026 后超额: {diff2:+.2f}bp")
print(f"  超额与行情相关性: {corr:+.4f}")
if verdict["depends_on_extreme"]:
    print("\n  ★ 结论：超额高度依赖单一极端年份的避险效果，")
    print("    不构成可重复的择时能力。样本外优势不可信。")
else:
    print("\n  ★ 结论：剔除极端年份后仍有超额，超额更可能是真实的择时能力。")

json.dump(verdict, open(os.path.join(OUT, "loo_check.json"), "w",
                         encoding="utf-8"), ensure_ascii=False,
          indent=2, default=float)
print(f"\n已存 {os.path.join(OUT, 'loo_check.json')}")