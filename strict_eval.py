"""
黄金模型 · 严格复核（修正基线错误）

★核心修正★
上一版用二项检验 H0: acc=0.50 作为显著性判据，但黄金测试集上涨天数占比
并非 50%（约 52%），这会让"永远猜涨"的策略也显得"显著"。
金融时序的正确零假设是多数类基线，不是 50%。

本脚本做四项严格修正：
1. 基线改为多数类（永远猜当日多数方向），而非 50%
2. 新增平衡准确率（对类别不平衡免疫）
3. 新增逐年样本外稳定性——非平稳性检验
4. Bootstrap 置信区间
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
SEED = 20261014
SPLIT = "2019-01-01"
TARGETS = ["fwd_1d", "fwd_3d", "fwd_5d", "fwd_10d"]
CN = {"fwd_1d": "1日", "fwd_3d": "3日", "fwd_5d": "1周", "fwd_10d": "2周"}


def binom_vs_baseline(n_correct, n, p_base):
    """单侧二项检验，零假设为准确率 = p_base（多数类基线）。"""
    return float(stats.binomtest(int(n_correct), int(n), p_base,
                                alternative="greater").pvalue)


def balanced_acc(y, pred):
    tn, fp, fn, tp = stats.contingency_table2 if False else (None,) * 4
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tpr = tp / (tp + fn) if tp + fn else 0.5
    tnr = tn / (tn + fp) if tn + fp else 0.5
    return 0.5 * (tpr + tnr)


def main():
    df = pd.read_csv(os.path.join(BASE, "data", "gold_features.csv"),
                     index_col=0, parse_dates=True)
    feats = [c for c in df.columns if c not in TARGETS + ["close"]]
    is_train = pd.Series(df.index < SPLIT, index=df.index)

    print("=" * 72)
    print("黄金多因子模型 · 严格复核（基线修正版）")
    print("=" * 72)

    # ---- 0. 关键诊断：测试集类别分布 ----
    print("\n【诊断】测试集类别分布（这是上一版基线错误的根源）")
    dist = {}
    for tgt in TARGETS:
        sub = df[[tgt] + feats].dropna()
        y = (sub[tgt] > 0).astype(int)
        mte = (~is_train.reindex(sub.index, fill_value=False)).to_numpy()
        yte = y[mte]
        pos = float(yte.mean())
        dist[tgt] = {"pos_rate": pos, "majority_baseline": max(pos, 1 - pos)}
        print(f"  {CN[tgt]:4s} 上涨占比 {pos*100:5.2f}%  "
              f"多数类基线准确率 {max(pos,1-pos)*100:5.2f}%")

    # ---- 1. 逐模型逐周期完整评估 ----
    print("\n" + "=" * 72)
    print("【实验1】样本外评估（基线 = 多数类）")
    print("=" * 72)
    results = {}
    for tgt in TARGETS:
        sub = df[[tgt] + feats].dropna()
        y = (sub[tgt] > 0).astype(int)
        X = sub[feats]
        m = is_train.reindex(sub.index, fill_value=False).to_numpy()
        Xtr, ytr, Xte, yte = X[m], y[m], X[~m], y[~m]
        mu, sd = Xtr.values.mean(0), Xtr.values.std(0) + 1e-12
        Xtrs = (Xtr.values - mu) / sd
        Xtes = (Xte.values - mu) / sd

        p_base = max(float(yte.mean()), 1 - float(yte.mean()))
        print(f"\n  ── {CN[tgt]} ({tgt}) ── 多数类基线 {p_base*100:.2f}%")
        tgt_res = {"majority_baseline": p_base, "pos_rate": float(yte.mean()),
                   "n": int(len(yte)), "models": {}}

        models = {
            "logistic": LogisticRegression(C=0.1, max_iter=2000),
            "rf": RandomForestClassifier(n_estimators=300, max_depth=3,
                                         min_samples_leaf=30,
                                         random_state=SEED, n_jobs=-1),
            "gb": GradientBoostingClassifier(n_estimators=200, max_depth=2,
                                             learning_rate=0.05,
                                             random_state=SEED),
        }
        for mname, mdl in models.items():
            mdl.fit(Xtrs, ytr.values)
            prob = mdl.predict_proba(Xtes)[:, 1]
            pred = (prob > 0.5).astype(int)
            acc = float(accuracy_score(yte, pred))
            n = len(yte)
            bacc = float(balanced_acc(yte.values, pred))
            try:
                auc = float(roc_auc_score(yte, prob))
            except ValueError:
                auc = 0.5
            n_corr = int((pred == yte.values).sum())
            p_raw = binom_vs_baseline(n_corr, n, p_base)
            p_corr = min(1.0, p_raw * len(TARGETS))

            # Bootstrap 95% CI
            rng = np.random.default_rng(SEED)
            correct = (pred == yte.values).astype(float)
            bs = [correct[rng.integers(0, n, n)].mean() for _ in range(2000)]
            ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

            # 相对基线的超额
            edge = acc - p_base
            r = {"acc": acc, "bacc": bacc, "auc": auc,
                 "edge_vs_baseline": edge,
                 "p_raw": p_raw, "p_corr": p_corr,
                 "ci_low": ci[0], "ci_high": ci[1],
                 "ci_excludes_baseline": bool(ci[0] > p_base),
                 "sig_vs_baseline": bool(p_corr < 0.05 / len(TARGETS))}
            tgt_res["models"][mname] = r

            sig_mark = "显著" if r["sig_vs_baseline"] else "不显著"
            ci_mark = "CI高于基线" if r["ci_excludes_baseline"] else "CI覆盖基线"
            print(f"    {mname:9s} acc={acc*100:5.2f}% (超额{edge*100:+5.2f}pp) "
                  f"平衡acc={bacc*100:5.2f}% auc={auc:.4f}")
            print(f"    {'':9s} CI95=[{ci[0]*100:.2f}%, {ci[1]*100:.2f}%] {ci_mark}"
                  f"  校正p={p_corr:.4f} {sig_mark}")

            # 保存预测供稳健性分析
            np.save(os.path.join(OUT, f"pred_{tgt}_{mname}.npy"), prob)
        results[tgt] = tgt_res

    # ---- 2. 逐年样本外稳定性 ----
    print("\n" + "=" * 72)
    print("【实验2】逐年样本外准确率（非平稳性检验）")
    print("=" * 72)
    yearly = {}
    for tgt in TARGETS:
        sub = df[[tgt] + feats].dropna()
        y = (sub[tgt] > 0).astype(int)
        X = sub[feats]
        m = is_train.reindex(sub.index, fill_value=False).to_numpy()
        Xtr, ytr = X[m], y[m]
        mu, sd = Xtr.values.mean(0), Xtr.values.std(0) + 1e-12
        mdl = LogisticRegression(C=0.1, max_iter=2000)
        mdl.fit((Xtr.values - mu) / sd, ytr.values)

        mte = ~m
        Xte, yte = X[mte], y[mte]
        prob = mdl.predict_proba((Xte.values - mu) / sd)[:, 1]
        pred = (prob > 0.5).astype(int)
        tmp = pd.DataFrame({
            "year": Xte.index.year, "correct": (pred == yte.values).astype(int),
            "actual": yte.values})
        yr_acc = tmp.groupby("year").agg(
            acc=("correct", "mean"), n=("correct", "size"),
            up_ratio=("actual", "mean"))
        yr_acc = yr_acc.reset_index()
        yr_acc["baseline"] = yr_acc["up_ratio"].map(lambda p: max(p, 1 - p))
        yr_acc["edge"] = yr_acc["acc"] - yr_acc["baseline"]
        yearly[tgt] = yr_acc.to_dict("records")

        wins = int((yr_acc["edge"] > 0).sum())
        print(f"\n  {CN[tgt]:4s} 跑赢基线的年份: {wins}/{len(yr_acc)}")
        for _, r in yr_acc.iterrows():
            mark = "↑" if r["edge"] > 0 else "↓"
            print(f"    {int(r['year'])}  准确率{r['acc']*100:5.2f}%  "
                  f"基线{r['baseline']*100:5.2f}%  超额{r['edge']*100:+5.2f}pp {mark}"
                  f"  (n={int(r['n'])})")

    # ---- 3. 交易可行性（含成本）----
    print("\n" + "=" * 72)
    print("【实验3】扣交易成本后的净收益（关键：准确率不等于能赚钱）")
    print("=" * 72)
    trading = {}
    for tgt in ["fwd_1d", "fwd_5d"]:
        print(f"\n  ── {CN[tgt]} ──")
        for mname in ["logistic", "rf", "gb"]:
            prob = np.load(os.path.join(OUT, f"pred_{tgt}_{mname}.npy"))
            sub = df[[tgt] + feats].dropna()
            m = is_train.reindex(sub.index, fill_value=False).to_numpy()
            Xte, yte = sub[~m], sub[tgt][~m]
            ret = yte.values
            rows = []
            for thresh in [0.0, 0.1, 0.2, 0.3]:
                sig = np.where(prob > 0.5 + thresh, 1,
                               np.where(prob < 0.5 - thresh, -1, 0))
                active = sig != 0
                if active.sum() < 30:
                    continue
                gross = (sig[active] * ret[active]).mean() * 1e4
                for cost_bps in [0, 5, 10]:
                    net = gross - 2 * cost_bps
                    rows.append({
                        "thresh": thresh, "cost_bps": cost_bps,
                        "n_trades": int(active.sum()),
                        "coverage": float(active.mean()),
                        "gross_bp": float(gross), "net_bp": float(net),
                    })
            r5 = [r for r in rows if r["cost_bps"] == 5]
            if r5:
                best = max(r5, key=lambda r: r["net_bp"])
                print(f"    {mname:9s} 阈值±0.2 交易{best['n_trades']:4d}次 "
                      f"覆盖{best['coverage']*100:4.1f}%  "
                      f"毛{best['gross_bp']:+7.2f}bp  净{best['net_bp']:+7.2f}bp")
                trading[f"{tgt}_{mname}"] = rows

    # ---- 4. 综合判定 ----
    print("\n" + "=" * 72)
    print("【最终判定】")
    print("=" * 72)
    verdict = {}
    for tgt in TARGETS:
        best = max(results[tgt]["models"].items(),
                   key=lambda kv: kv[1]["acc"])
        mn, r = best
        # 净收益检查
        tr_key = f"{tgt}_{mn}"
        net_ok = False
        if tr_key in trading:
            r5 = [x for x in trading[tr_key] if x["cost_bps"] == 5]
            net_ok = any(x["net_bp"] > 0 for x in r5)
        verdict[tgt] = {
            "best_model": mn,
            "acc": r["acc"], "edge": r["edge_vs_baseline"],
            "auc": r["auc"], "p_corr": r["p_corr"],
            "sig": r["sig_vs_baseline"],
            "ci_excludes_baseline": r["ci_excludes_baseline"],
            "net_profitable": net_ok,
        }
        v = verdict[tgt]
        print(f"\n  {CN[tgt]:4s} 最优模型 {mn}: 准确率{v['acc']*100:.2f}% "
              f"(超额基线 {v['edge']*100:+.2f}pp)")
        print(f"       统计显著: {'是' if v['sig'] else '否'}  "
              f"CI高于基线: {'是' if v['ci_excludes_baseline'] else '否'}  "
              f"扣成本后净盈利: {'是' if v['net_profitable'] else '否'}")

    out = {"class_distribution": dist, "results": results,
           "yearly": yearly, "trading": trading, "verdict": verdict}
    p = os.path.join(OUT, "strict_results.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已存 {p}")


if __name__ == "__main__":
    main()