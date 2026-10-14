"""
决定性检验：策略 alpha 是否显著异于零？

前面发现：总体 alpha 为正，但逐年只有 2/8 和 4/8 年为正。
这高度提示"总体 alpha 正"是少数年份暴利拉高的。
用两种零分布判定：
  1. 置换检验（打乱信号与收益的时间对应关系）
  2. 区块自助法（保留时序自相关，Block bootstrap）
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
SEED = 20261014

df = pd.read_csv(os.path.join(BASE, "data", "gold_features.csv"),
                 index_col=0, parse_dates=True)
feats = [c for c in df.columns if c not in TARGETS + ["close"]]


def neutral_alpha(ret, sig, active):
    """市场中性化后的每笔平均 alpha（bp）。"""
    strat = np.where(active, sig * ret, 0.0)
    if active.sum() < 10:
        return np.nan
    beta = np.polyfit(ret, strat, 1)[0]
    resid = strat[active] - beta * ret[active]
    return float(resid.mean() * 1e4)


def main():
    rng = np.random.default_rng(SEED)
    print("=" * 72)
    print("★ 决定性检验：alpha 显著性（置换检验 + 区块自助法）")
    print("=" * 72)

    out = {}
    for tgt in ["fwd_1d", "fwd_3d", "fwd_5d", "fwd_10d"]:
        for mname in ["logistic", "rf", "gb"]:
            p = os.path.join(OUT, f"pred_{tgt}_{mname}.npy")
            if not os.path.exists(p):
                continue
            prob = np.load(p)
            sub = df[[tgt] + feats].dropna()
            m = np.asarray(sub.index >= SPLIT)
            ret = sub.loc[m, tgt].values
            sig = np.where(prob > 0.6, 1, np.where(prob < 0.4, -1, 0))
            active = sig != 0
            n_act = int(active.sum())
            if n_act < 30:
                print(f"{CN[tgt]+'/'+mname:<22s}{n_act:>7d}   "
                      f"交易样本过少，跳过显著性检验")
                continue
            real_alpha = neutral_alpha(ret, sig, active)

            # ---- 置换零分布：随机打乱信号 ----
            n_perm = 2000
            perm = np.empty(n_perm)
            sig_act = sig[active]
            ret_act = ret[active]
            full_ret = ret
            for i in range(n_perm):
                sp = rng.permutation(sig_act)
                tmp = np.zeros_like(full_ret)
                tmp[active] = sp
                perm[i] = neutral_alpha(full_ret, tmp, active)
            perm = perm[np.isfinite(perm)]
            p_perm = float((perm >= real_alpha).mean())

            # ---- 区块自助置信区间 ----
            bs = []
            L = 10
            n = len(ret_act)
            n_blocks = max(1, int(np.ceil(n / L)))
            for _ in range(2000):
                starts = rng.integers(0, max(1, n - L), size=n_blocks)
                idx_b = np.concatenate(
                    [np.arange(s, s + L) for s in starts])[:n]
                bs.append(float((sig_act[idx_b] * ret_act[idx_b]).mean() * 1e4))
            bs = np.array(bs)
            ci_bs = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

            key = f"{tgt}_{mname}"
            out[key] = {
                "n_trades": n_act,
                "real_alpha_bp": real_alpha,
                "perm_p": p_perm,
                "perm_mean": float(perm.mean()),
                "perm_std": float(perm.std()),
                "perm_p95": float(np.percentile(perm, 95)),
                "bs_ci": ci_bs,
                "sig_perm": bool(p_perm < 0.05),
                "sig_bs": bool(ci_bs[0] > 0),
            }

    # 输出汇总
    print(f"\n{'模型':<22s}{'交易数':>7s}{'实际alpha':>11s}"
          f"{'置换p':>9s}{'BS95%下限':>12s}  判定")
    print("-" * 72)
    for tgt in TARGETS:
        for mname in ["logistic", "rf", "gb"]:
            k = f"{tgt}_{mname}"
            if k not in out:
                continue
            r = out[k]
            sig = r["sig_perm"] and r["sig_bs"]
            print(f"{CN[tgt]+'/'+mname:<22s}{r['n_trades']:>7d}"
                  f"{r['real_alpha_bp']:>+10.2f}bp"
                  f"{r['perm_p']:>9.4f}{r['bs_ci'][0]:>+11.2f}bp  "
                  f"{'显著' if sig else '不显著'}")
        print("-" * 72)

    n_sig = sum(1 for r in out.values() if r["sig_perm"] and r["sig_bs"])
    print(f"\n通过双重检验的模型数: {n_sig}/{len(out)}")
    if n_sig == 0:
        print("\n★ 最终判定：没有任何模型在双重检验下显著。")
        print("  此前看到的'净盈利'与'alpha为正'，在考虑市场beta暴露、")
        print("  年度不稳定性与多重比较后，均无法与噪声区分。")
    else:
        print(f"\n★ 有 {n_sig} 个模型通过双重检验。")

    with open(os.path.join(OUT, "significance_test.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已存 {os.path.join(OUT, 'significance_test.json')}")


if __name__ == "__main__":
    main()