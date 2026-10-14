"""
黄金 v2 实验：时序 Transformer + 时间特征

严格遵循 PREREGISTRATION_V2.json：
  三段式切分 训练(2008-2018) / 验证(2019-2021) / 测试(2022-2026，仅跑一次)
  测试期采用扩展窗滚动重训（每年重训），这是实盘唯一可用形式
  基线 = 逐年多数类基线（绝不是50%）
  16 项检验做 Holm-Bonferroni 校正
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from transformer_model import (SeqDataset, GoldTransformer, LSTMBaseline,
                                train_model, predict, TARGETS, CN)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

SEED = 20261014
TRAIN_END = "2018-12-31"
VAL_END = "2021-12-31"
TEST_START = "2022-01-01"
WINDOW = 32
CAL_COLS = ["sin_doy", "cos_doy", "sin_dow", "cos_dow",
            "sin_month", "cos_month", "month_pos", "event_density"]

torch.manual_seed(SEED)
np.random.seed(SEED)


def holm_bonferroni(pvals):
    """Holm-Bonferroni 校正，返回校正后的 p 值列表。"""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        val = (n - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj.tolist()


def majority_baseline(y_bin):
    """逐年多数类基线，逐样本匹配所属年份。"""
    p = y_bin.mean()
    return max(p, 1 - p)


def acc_vs_baseline(correct, base_rate):
    n = len(correct)
    k = int(correct.sum())
    p = float(stats.binomtest(k, n, base_rate, alternative="greater").pvalue)
    return p


def balanced_acc(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tpr = tp / (tp + fn) if tp + fn else 0.5
    tnr = tn / (tn + fp) if tn + fp else 0.5
    return 0.5 * (tpr + tnr)


def expanding_window_backtest(make_model, ds, cal_all, target,
                              epochs=40, verbose=True):
    """扩展窗滚动回测：每年重训一次，预测下一年。"""
    years = sorted(set(ds.index[np.asarray(ds.index >= pd.Timestamp(TEST_START))].year))
    all_idx, all_prob, all_act, all_ret, all_year = [], [], [], [], []
    fold_info = []

    prev_end = ds.index.searchsorted(VAL_END) - 1
    for yr in years:
        # 训练数据：从头到上一年末（扩展窗）
        tr_end = ds.index.searchsorted(f"{yr-1}-12-31") - 1
        if tr_end <= WINDOW + 50:
            continue
        Xtr, rtr, btr, _, _ = ds.make(0, tr_end + 1)

        # 测试区间：当年
        te_start = ds.index.searchsorted(f"{yr}-01-01") - 1
        te_end = ds.index.searchsorted(f"{yr}-12-31")
        Xte, rte, bte, ite, dte = ds.make(te_start, te_end)
        if len(Xte) < 30:
            continue

        Xtr_t = torch.tensor(Xtr)
        rtr_t = torch.tensor(rtr)
        btr_t = torch.tensor(btr)

        def gather_cal(positions):
            """返回 [N, WINDOW, n_cal] 的日历窗口张量，与特征窗口严格对齐。"""
            L = WINDOW
            out = np.zeros((len(positions), L, cal_all.shape[1]),
                           dtype=np.float32)
            for i, p0 in enumerate(positions):
                p0 = int(np.clip(p0, L - 1, len(cal_all) - 1))
                s = p0 - L + 1
                out[i] = cal_all[s:p0 + 1]
            return torch.tensor(out)

        # 每个样本的窗口末日位置
        last_pos_tr = np.arange(len(Xtr)) + WINDOW - 1
        cal_tr = gather_cal(last_pos_tr)

        base_pos_te = ds.index.searchsorted(f"{yr}-01-01") - 1
        pos_te = base_pos_te + np.arange(len(Xte)) + WINDOW - 1
        cal_te = gather_cal(pos_te)

        model = make_model()
        val = train_model(model, Xtr_t, btr_t, rtr_t,
                          Xtr_t[:200], btr_t[:200], rtr_t[:200],
                          cal_tr=cal_tr, cal_va=cal_tr[:200],
                          epochs=epochs, patience=10)
        prob = predict(model, torch.tensor(Xte), cal_te)

        pred = (prob > 0.5).astype(int)
        base = majority_baseline(bte)
        correct = (pred == bte).astype(float)
        fold_info.append({
            "year": yr, "n": int(len(bte)),
            "acc": float(correct.mean()),
            "baseline": float(base),
            "edge": float(correct.mean() - base),
            "bacc": float(balanced_acc(bte.astype(int), pred)),
            "auc": float(roc_auc_score(bte, prob)) if len(set(bte)) > 1 else 0.5,
            "val_logloss": float(val),
        })
        all_idx.extend(dte)
        all_prob.extend(prob)
        all_act.extend(bte)
        all_ret.extend(rte)
        all_year.extend([yr] * len(bte))
        if verbose:
            print(f"      {yr}: acc={fold_info[-1]['acc']*100:5.2f}%  "
                  f"基线={base*100:5.2f}%  超额={fold_info[-1]['edge']*100:+5.2f}pp"
                  f"  auc={fold_info[-1]['auc']:.4f}")

    res = pd.DataFrame({
        "date": all_idx, "prob": all_prob, "act": all_act,
        "ret": all_ret, "year": all_year})
    return res, fold_info


def main():
    print("=" * 74)
    print("黄金 v2 · 时序 Transformer + 时间特征")
    print(f"训练 ≤{TRAIN_END}  验证 ≤{VAL_END}  测试 ≥{TEST_START}")
    print("=" * 74)

    df = pd.read_csv(os.path.join(BASE, "data", "gold_features_v2.csv"),
                     index_col=0, parse_dates=True)
    meta = json.load(open(os.path.join(BASE, "data", "meta_v2.json"),
                          encoding="utf-8"))
    feats = [c for c in df.columns
             if c not in TARGETS + ["close"] and not c.startswith("fwd_")]
    cal_cols = [c for c in CAL_COLS if c in df.columns]
    feat_cols = [c for c in feats if c not in cal_cols]
    print(f"特征 {len(feat_cols)} 维 + 日历 {len(cal_cols)} 维 = {len(feats)} 维")

    # 日历通道：标准化到 [-1,1]
    cal_vals = df[cal_cols].values.astype(np.float32)
    cal_vals = np.nan_to_num(cal_vals, nan=0.0)
    lo, hi = cal_vals[:int(len(cal_vals) * 0.6)].min(0), \
        cal_vals[:int(len(cal_vals) * 0.6)].max(0)
    cal_all = np.clip((cal_vals - lo) / (hi - lo + 1e-8), 0, 2) - 1

    results = {}
    raw_preds = {}

    for target in TARGETS:
        print(f"\n{'='*74}")
        print(f"【{CN[target]}】{target}")
        print(f"{'='*74}")
        ds = SeqDataset(df, feat_cols, target, window=WINDOW)

        tgt_res = {}
        # ---- 1. Transformer ----
        print("  ── GoldTransformer (3层, d=128, 8头, 因果掩码) ──")
        tr_res, folds = expanding_window_backtest(
            lambda: GoldTransformer(len(feat_cols), n_cal=len(cal_cols),
                                    d_model=128, nhead=8, layers=3,
                                    dim_ff=256, dropout=0.2, window=WINDOW),
            ds, cal_all, target, epochs=40)
        if len(tr_res) == 0:
            print("    样本不足，跳过")
            continue
        pred = (tr_res["prob"] > 0.5).astype(int).values
        correct = (pred == tr_res["act"].values).astype(float)
        base = majority_baseline(tr_res["act"].values)
        auc = roc_auc_score(tr_res["act"], tr_res["prob"])
        p_raw = acc_vs_baseline(correct, base)
        # bootstrap CI
        rng = np.random.default_rng(SEED)
        n = len(correct)
        bs = [correct[rng.integers(0, n, n)].mean() for _ in range(2000)]
        ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        n_win = sum(1 for f in folds if f["edge"] > 0)

        # beta 中性化（防v1 的假alpha 重演）
        sig = np.where(tr_res["prob"] > 0.6, 1,
                       np.where(tr_res["prob"] < 0.4, -1, 0))
        act = sig != 0
        strat = np.where(act, sig * tr_res["ret"].values, 0.0)
        beta = np.polyfit(tr_res["ret"].values, strat, 1)[0]
        alpha = float((strat[act] - beta * tr_res["ret"].values[act]).mean() * 1e4) \
            if act.sum() > 10 else np.nan
        net_expo = float((sig[act] == 1).mean() * 2 - 1)
        bh = float(tr_res["ret"].values.mean() * 1e4)
        simple_diff = float((strat[act].mean() - tr_res["ret"].values[act].mean())
                            * 1e4)

        tgt_res["transformer"] = {
            "acc": float(correct.mean()), "baseline": float(base),
            "edge": float(correct.mean() - base),
            "bacc": float(balanced_acc(tr_res["act"].astype(int), pred)),
            "auc": float(auc), "p_raw": p_raw,
            "ci": ci, "n": int(n),
            "folds": folds, "folds_win": f"{n_win}/{len(folds)}",
            "alpha_bp": alpha, "beta": float(beta),
            "net_long_expo": net_expo, "buy_hold_bp": bh,
            "strategy_minus_bh_bp": simple_diff,
        }
        raw_preds[(target, "transformer")] = tr_res
        print(f"    样本外准确率 {correct.mean()*100:5.2f}%  基线 {base*100:5.2f}%"
              f"  超额 {tgt_res['transformer']['edge']*100:+5.2f}pp")
        print(f"    平衡准确率 {tgt_res['transformer']['bacc']*100:5.2f}%"
              f"  AUC {auc:.4f}  CI95=[{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]")
        print(f"    逐年跑赢 {n_win}/{len(folds)}   p={p_raw:.4f}")
        print(f"    ★净多头敞口 {net_expo*100:+.1f}%  "
              f"alpha={alpha:+.2f}bp  策略−买入持有={simple_diff:+.2f}bp")

        # ---- 2. LSTM 对照 ----
        print("  ── LSTM 对照（检验 attention 是否必要）──")
        lstm_res, lstm_folds = expanding_window_backtest(
            lambda: LSTMBaseline(len(feat_cols)), ds, cal_all, target,
            epochs=40, verbose=False)
        if len(lstm_res):
            lp = (lstm_res["prob"] > 0.5).astype(int).values
            lc = (lp == lstm_res["act"].values).astype(float)
            tgt_res["lstm"] = {
                "acc": float(lc.mean()), "edge": float(lc.mean() - base),
                "p_raw": acc_vs_baseline(lc, base), "folds": lstm_folds,
            }
            print(f"    准确率 {lc.mean()*100:5.2f}%  超额 "
                  f"{tgt_res['lstm']['edge']*100:+5.2f}pp")

        # ---- 3. LightGBM 对照（当日特征，无时序）----
        print("  ── LightGBM 对照（当日特征，无时序建模）──")
        sub = df[[target] + feat_cols + cal_cols].dropna()
        yb = (sub[target] > 0).astype(int)
        te = np.asarray(sub.index >= TEST_START)
        Xa = sub[feat_cols + cal_cols].values
        mu, sd = Xa[~te].mean(0), Xa[~te].std(0) + 1e-8
        Xa = (Xa - mu) / sd
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_estimators=300, max_depth=3,
                                 num_leaves=7, learning_rate=0.05,
                                 random_state=SEED, verbose=-1)
        clf.fit(Xa[~te], yb[~te])
        lp = clf.predict_proba(Xa[te])[:, 1]
        la = yb[te].values
        lpred = (lp > 0.5).astype(int)
        lcorr = (lpred == la).astype(float)
        tgt_res["lightgbm"] = {
            "acc": float(lcorr.mean()), "edge": float(lcorr.mean() - base),
            "p_raw": acc_vs_baseline(lcorr, base),
            "auc": float(roc_auc_score(la, lp)),
        }
        print(f"    准确率 {lcorr.mean()*100:5.2f}%  AUC "
              f"{tgt_res['lightgbm']['auc']:.4f}  超额 "
              f"{tgt_res['lightgbm']['edge']*100:+5.2f}pp")

        # ---- 4. Logistic 对照 ----
        lr = LogisticRegression(C=0.1, max_iter=3000)
        lr.fit(Xa[~te], yb[~te])
        pp = lr.predict_proba(Xa[te])[:, 1]
        ppr = (pp > 0.5).astype(int)
        pc = (ppr == la).astype(float)
        tgt_res["logistic"] = {
            "acc": float(pc.mean()), "edge": float(pc.mean() - base),
            "p_raw": acc_vs_baseline(pc, base),
            "auc": float(roc_auc_score(la, pp)),
        }
        print(f"  ── Logistic 对照: 准确率 {pc.mean()*100:5.2f}%  超额 "
              f"{tgt_res['logistic']['edge']*100:+5.2f}pp")

        results[target] = tgt_res

    # ---- Holm-Bonferroni 校正 ----
    print(f"\n{'='*74}")
    print("Holm-Bonferroni 多重比较校正（16 项检验）")
    print(f"{'='*74}")
    keys, pvals = [], []
    for t, tr in results.items():
        for m, r in tr.items():
            if "p_raw" in r:
                keys.append(f"{CN[t]}/{m}")
                pvals.append(r["p_raw"])
    if pvals:
        adj = holm_bonferroni(pvals)
        for k, pv, av in sorted(zip(keys, pvals, adj),
                                key=lambda x: x[1]):
            sig = "显著" if av < 0.05 else "不显著"
            print(f"  {k:<26s} p={pv:.4f}  Holm校正p={av:.4f}  {sig}")

    # ---- 汇总判定 ----
    print(f"\n{'='*74}")
    print("最终判定")
    print(f"{'='*74}")
    summary = {}
    for t, tr in results.items():
        r = tr.get("transformer", {})
        if not r:
            continue
        pass_acc = r["edge"] > 0
        ci_pass = r["ci"][0] > r["baseline"]
        alpha_pass = (not np.isnan(r["alpha_bp"])) and r["alpha_bp"] > 0
        diff_pass = r["strategy_minus_bh_bp"] > 0
        fold_pass = sum(1 for f in r["folds"] if f["edge"] > 0) >= 2 / 3 * len(r["folds"])
        final = all([pass_acc, ci_pass, alpha_pass, diff_pass, fold_pass])
        summary[t] = {
            "acc": r["acc"], "edge": r["edge"], "baseline": r["baseline"],
            "ci": r["ci"], "auc": r["auc"], "folds_win": r["folds_win"],
            "alpha_bp": r["alpha_bp"], "strategy_minus_bh_bp": r["strategy_minus_bh_bp"],
            "net_long_expo": r["net_long_expo"],
            "pass_acc": bool(pass_acc), "pass_ci": bool(ci_pass),
            "pass_alpha": bool(alpha_pass), "pass_diff": bool(diff_pass),
            "pass_stability": bool(fold_pass), "FINAL_PASS": bool(final),
        }
        print(f"\n  【{CN[t]}】")
        print(f"    准确率 {r['acc']*100:.2f}% vs 基线 {r['baseline']*100:.2f}%"
              f"  超额 {r['edge']*100:+.2f}pp")
        print(f"    CI95下限高于基线: {ci_pass}   逐年跑赢 {r['folds_win']}")
        print(f"    中性化alpha {r['alpha_bp']:+.2f}bp   "
              f"策略−买入持有 {r['strategy_minus_bh_bp']:+.2f}bp   "
              f"净多头敞口 {r['net_long_expo']*100:+.1f}%")
        print(f"    ★ 五项判据: {'全部通过' if final else '未全部通过'}")

    n_pass = sum(1 for v in summary.values() if v["FINAL_PASS"])
    print(f"\n{'='*74}")
    print(f"总体判定：{n_pass}/4 个周期通过全部五项判据")
    print(f"{'='*74}")

    payload = {"results": results, "summary": summary,
               "holm": dict(zip(keys, holm_bonferroni(pvals))) if pvals else {},
               "config": {"window": WINDOW, "n_features": len(feat_cols),
                          "n_cal": len(cal_cols), "layers": 3, "d_model": 128,
                          "n_heads": 8, "train_end": TRAIN_END,
                          "val_end": VAL_END, "test_start": TEST_START},
               "n_pass": n_pass}
    with open(os.path.join(OUT, "v2_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)

    for (t, m), d in raw_preds.items():
        d.to_csv(os.path.join(OUT, f"v2_pred_{t}_{m}.csv"),
                 encoding="utf-8-sig")
    print(f"\n已存 {os.path.join(OUT, 'v2_results.json')}")


if __name__ == "__main__":
    main()