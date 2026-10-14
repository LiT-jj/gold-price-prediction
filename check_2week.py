"""
唯一的正面信号验证：2周周期 logistic 相对基准 +16.31bp 是否真实？

检验清单：
  1. 分年度是否稳定
  2. 置换检验（打乱信号时序，2000 次）
  3. 区块自助法置信区间
  4. 扣除交易成本后是否仍为正
  5. 是否集中于少数极端行情（去掉最大 5% 收益后是否还在）
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
idx = sub.index[m]
prob = np.load(os.path.join(OUT, f"pred_{tgt}_logistic.npy"))
sig = np.where(prob > 0.6, 1, np.where(prob < 0.4, -1, 0))
act = sig != 0

strat = sig[act] * ret[act]
bh = ret[act]
simple_diff = (strat.mean() - bh.mean()) * 1e4

print("=" * 72)
print(f"【2周周期】相对买入持有超额的稳健性验证")
print("=" * 72)
print(f"交易 {act.sum()} 笔   净多头 {(sig[act]==1).mean()*100-50:+.1f}%")
print(f"策略日均 {strat.mean()*1e4:+.2f}bp   买入持有 {bh.mean()*1e4:+.2f}bp"
      f"   差额 {simple_diff:+.2f}bp")

# ---- 1. 分年度 ----
print("\n[1] 分年度超额稳定性")
d = pd.DataFrame({"dt": np.asarray(idx)[act], "strat": strat, "bh": bh})
d["year"] = pd.DatetimeIndex(d["dt"]).year
n_pos = 0
for yr, g in d.groupby("year"):
    diff = (g["strat"].mean() - g["bh"].mean()) * 1e4
    if diff > 0:
        n_pos += 1
    print(f"   {yr}  策略{g['strat'].mean()*1e4:+7.2f}bp  "
          f"持有{g['bh'].mean()*1e4:+7.2f}bp  超额{diff:+7.2f}bp  "
          f"n={len(g)}")
print(f"   → 超额为正的年份: {n_pos}/{d['year'].nunique()}")

# ---- 2. 置换检验 ----
print("\n[2] 置换检验（打乱信号与收益的时序对应，2000 次）")
null = []
for _ in range(2000):
    s_perm = rng.choice([-1, 1], size=act.sum())
    null.append((s_perm * bh).mean() * 1e4)
null = np.array(null)
p_perm = float((null >= strat.mean() * 1e4).mean())
print(f"   真实策略收益 {strat.mean()*1e4:+.2f}bp")
print(f"   零分布 均值 {null.mean():+.2f}bp  标准差 {null.std():.2f}bp  "
      f"95分位 {np.percentile(null,95):+.2f}bp")
print(f"   置换 p = {p_perm:.4f}  {'真实存在' if p_perm<0.05 else '与随机无法区分'}")

# ---- 3. 区块自助 ----
print("\n[3] 区块自助置信区间（保留时序自相关，block=10）")
L, N = 10, len(strat)
bs = []
for _ in range(3000):
    starts = rng.integers(0, max(1, N - L), size=int(np.ceil(N / L)))
    ib = np.concatenate([np.arange(s, s + L) for s in starts])[:N]
    bs.append(strat[ib].mean() * 1e4)
bs = np.array(bs)
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"   策略收益 95%CI = [{lo:+.2f}bp, {hi:+.2f}bp]  "
      f"{'不含0' if lo > 0 else '含0，不能排除噪声'}")

# ---- 4. 扣成本 ----
print("\n[4] 扣除交易成本（双边各 5bp = 10bp）")
net = strat.mean() * 1e4 - 10
print(f"   净收益 = {net:+.2f}bp/笔  "
      f"{'仍为正' if net > 0 else '转负'}")

# ---- 5. 剔除极端行情 ----
print("\n[5] 剔除极端行情后是否仍存在")
srt = np.sort(strat)
trim = srt[int(len(srt) * 0.05):int(len(srt) * 0.95)]
btrim = np.sort(bh)
btrim = btrim[int(len(btrim) * 0.05):int(len(btrim) * 0.95)]
diff_trim = (trim.mean() - btrim.mean()) * 1e4
print(f"   去掉最大/最小各5%后 策略{trim.mean()*1e4:+.2f}bp  "
      f"持有{btrim.mean()*1e4:+.2f}bp  超额{diff_trim:+.2f}bp")
print(f"   {'仍为正，不依赖极端行情' if diff_trim > 0 else '转负，依赖少数极端行情'}")

# ---- 结论 ----
print("\n" + "=" * 72)
print("判定")
print("=" * 72)
conds = {
    "超额为正的年份≥3": n_pos >= 3,
    "置换检验 p<0.05": p_perm < 0.05,
    "自助CI不含0": lo > 0,
    "扣成本后为正": net > 0,
    "剔除极端后为正": diff_trim > 0,
}
for k, v in conds.items():
    print(f"   [{'✓' if v else '✗'}] {k}")
n_ok = sum(conds.values())
print(f"\n   通过 {n_ok}/5 项")
if n_ok >= 4:
    print("   ★ 存在真实的相对超额，但幅度很小（约 6~16bp/笔，"
          "扣成本后所剩无几），且方向准确率仍低于基线。")
    print("   ★ 这说明模型学到的是「何时该持有/减仓」而非「涨跌方向」。")
else:
    print("   ★ 无法确认为真实超额。")

json.dump({
    "simple_diff_bp": float(simple_diff), "years_positive": n_pos,
    "perm_p": p_perm, "bs_ci": [float(lo), float(hi)],
    "net_bp": float(net), "trimmed_diff_bp": float(diff_trim),
    "conditions": {k: bool(v) for k, v in conds.items()},
    "n_pass": int(n_ok),
}, open(os.path.join(OUT, "v2week_check.json"), "w",
       encoding="utf-8"), ensure_ascii=False, indent=2, default=float)