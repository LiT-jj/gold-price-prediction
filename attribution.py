"""
矛盾排查：准确率低于基线，为何"扣成本后净盈利"为正？

假设：交易收益来自 beta 暴露（黄金整体上涨的漂移），而非方向预测能力。
检验方法：与"无脑长期持有黄金"对比，并做市场中性化处理。
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

df = pd.read_csv(os.path.join(BASE, "data", "gold_features.csv"),
                 index_col=0, parse_dates=True)
feats = [c for c in df.columns if c not in TARGETS + ["close"]]

print("=" * 72)
print("★ 矛盾排查：净盈利的真正来源")
print("=" * 72)

# ---- 1. 测试期黄金本身的漂移 ----
print("\n【1】测试期黄金自身表现（这是关键）")
sub_all = df.dropna(subset=["fwd_1d"])
mte = sub_all.index >= SPLIT
for tgt in TARGETS:
    r = df.loc[mte, tgt].dropna()
    cum = (1 + r).prod() - 1
    print(f"  {CN[tgt]:4s} 测试期累计涨幅 {cum*100:+7.2f}%   "
          f"日均收益 {r.mean()*100:+.4f}%   上涨占比 {(r>0).mean()*100:.1f}%")
print("\n  → 若累计涨幅为正，则'永远持有'本身就能盈利，")
print("    模型可能只是偶然押对了方向，而非预测准确。")

# ---- 2. 策略收益分解：方向贡献 vs beta 贡献 ----
print("\n" + "=" * 72)
print("【2】收益分解：方向判断贡献多少？")
print("=" * 72)
decomp = {}
for tgt in ["fwd_1d", "fwd_5d"]:
    prob = np.load(os.path.join(OUT, f"pred_{tgt}_logistic.npy"))
    sub = df[[tgt] + feats].dropna()
    m = np.asarray(sub.index >= SPLIT)
    ret = sub.loc[m, tgt].values
    idx = sub.index[m]
    dfq = pd.DataFrame({"ret": ret, "prob": prob}, index=idx)

    print(f"\n  ── {CN[tgt]} ──")
    # 买入持有基准（测试期年化）
    bh_bp = dfq["ret"].mean() * 1e4
    # 模型信号
    sig = np.where(prob > 0.6, 1, np.where(prob < 0.4, -1, 0))
    act = sig != 0
    strat_bp = (sig[act] * dfq["ret"].values[act]).mean() * 1e4
    # 关键：策略与基准的相关性（beta）
    active_ret = sig[act] * dfq["ret"].values[act]
    all_dates_ret = np.zeros(len(dfq))
    all_dates_ret[act] = active_ret
    beta = np.polyfit(dfq["ret"].values, all_dates_ret, 1)[0]
    corr = np.corrcoef(dfq["ret"].values, all_dates_ret)[0, 1]
    # 市场中性化后收益 = 残差
    resid = all_dates_ret - beta * dfq["ret"].values
    resid_active = resid[act]
    alpha_bp = resid_active.mean() * 1e4

    decomp[tgt] = {
        "buy_hold_bp": float(bh_bp),
        "strategy_bp": float(strat_bp),
        "beta": float(beta),
        "corr_with_market": float(corr),
        "alpha_bp_after_neutral": float(alpha_bp),
        "n_trades": int(act.sum()),
    }
    print(f"    买入持有基准      {bh_bp:+8.2f} bp/次")
    print(f"    模型策略毛收益    {strat_bp:+8.2f} bp/次")
    print(f"    策略对黄金beta    {beta:8.3f}   （>1=方向正确但承担满仓风险）")
    print(f"    与基准相关系数    {corr:8.3f}")
    print(f"    ★市场中性化后alpha {alpha_bp:+8.2f} bp/次  ← 真正的预测能力")
    if alpha_bp > 0:
        print(f"    → alpha为正，说明存在超出beta的预测能力")
    else:
        print(f"    → alpha为负，说明收益完全来自beta暴露，无预测能力")

# ---- 3. 逐年alpha稳定性 ----
print("\n" + "=" * 72)
print("【3】逐年 alpha 稳定性（真正的预测能力是否稳定）")
print("=" * 72)
for tgt in ["fwd_1d", "fwd_5d"]:
    prob = np.load(os.path.join(OUT, f"pred_{tgt}_logistic.npy"))
    sub = df[[tgt] + feats].dropna()
    m = np.asarray(sub.index >= SPLIT)
    ret = sub.loc[m, tgt].values
    idx = sub.index[m]
    sig = np.where(prob > 0.6, 1, np.where(prob < 0.4, -1, 0))
    act = sig != 0
    tmp = pd.DataFrame({
        "year": idx.year, "ret": ret, "sig": sig, "act": act})
    tmp["strat"] = np.where(act, tmp["sig"] * tmp["ret"], np.nan)
    print(f"\n  ── {CN[tgt]} ──")
    pos_years = 0
    for yr, g in tmp.groupby("year"):
        g = g.dropna(subset=["strat"])
        if len(g) < 20:
            continue
        beta = np.polyfit(g["ret"].values, g["strat"].values, 1)[0]
        alpha = (g["strat"] - beta * g["ret"]).mean() * 1e4
        bh = g["ret"].mean() * 1e4
        st = g["strat"].mean() * 1e4
        mark = "↑" if alpha > 0 else "↓"
        if alpha > 0:
            pos_years += 1
        print(f"    {yr}  买入持有{bh:+8.2f}bp  策略{st:+8.2f}bp  "
              f"beta={beta:5.2f}  alpha={alpha:+7.2f}bp {mark}")
    print(f"    alpha为正的年份: {pos_years}/{len(list(tmp.groupby('year')))}")

# ---- 4. 判定 ----
print("\n" + "=" * 72)
print("【结论】")
print("=" * 72)
d1 = decomp.get("fwd_1d", {})
d5 = decomp.get("fwd_5d", {})
concl = {
    "beta_exposure_is_main_driver": bool(d1.get("beta", 0) > 0.5),
    "day1_alpha": d1.get("alpha_bp_after_neutral"),
    "week1_alpha": d5.get("alpha_bp_after_neutral"),
}
print(f"  1日 策略beta = {d1.get('beta', float('nan')):.3f}")
print(f"  1周 策略beta = {d5.get('beta', float('nan')):.3f}")
print(f"  1日 中性化后alpha = {d1.get('alpha_bp_after_neutral', float('nan')):+.2f} bp")
print(f"  1周 中性化后alpha = {d5.get('alpha_bp_after_neutral', float('nan')):+.2f} bp")
print()
if (d1.get("alpha_bp_after_neutral", 0) <= 0
        and d5.get("alpha_bp_after_neutral", 0) <= 0):
    print("  ★ 判定：交易盈利 = 黄金上涨的beta暴露，与方向预测无关。")
    print("    市场中性化后alpha ≤ 0 → 模型没有可验证的预测能力。")
    print("    2019-2026 测试期黄金单边上涨，任何'偏多'的模型都会显示盈利。")
else:
    print("  ★ 判定：存在部分超出beta的alpha，但需进一步检验稳定性。")

with open(os.path.join(OUT, "attribution.json"), "w", encoding="utf-8") as f:
    json.dump({"decomposition": decomp, "conclusion": concl},
              f, ensure_ascii=False, indent=2, default=float)
print(f"\n已存 {os.path.join(OUT, 'attribution.json')}")