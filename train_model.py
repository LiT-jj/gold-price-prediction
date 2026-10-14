"""
黄金多因子预测模型 · 训练与样本外验证

严格遵循 PREREGISTRATION.json 的判据。
时间序列切分，测试集全程封存，只在最后跑一次。
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

SEED = 20261014
TARGETS = ["fwd_1d", "fwd_3d", "fwd_5d", "fwd_10d"]
HORIZON_CN = {"fwd_1d": "1日", "fwd_3d": "3日", "fwd_5d": "1周", "fwd_10d": "2周"}
TECH_FEATS = ["au_r1", "au_r3", "au_r5", "au_r10", "au_r21",
              "au_vol5", "au_vol20", "au_ma_ratio", "au_rsi14",
              "gap", "intraday", "hl_range"]
ALPHA_CORR = 0.05 / len(TARGETS)   # Bonferroni

SPLIT = "2019-01-01"


def logloss_const(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def make_models():
    return {
        "logistic": lambda: Pipeline([
            ("sc", StandardScaler()),
            ("m", LogisticRegression(C=0.1, max_iter=2000)),
        ]),
        "rf": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=3, min_samples_leaf=30,
            random_state=SEED, n_jobs=-1),
        "gb": lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            random_state=SEED),
    }


def walkforward_select(Xtr, ytr, model_name, n_splits=3):
    """训练集内部时序CV选超参——不打乱。"""
    from sklearn.model_selection import TimeSeriesSplit
    Xa = Xtr.values
    ya = ytr.values
    grid = {
        "logistic": [{"C": c} for c in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]],
        "rf": [{"max_depth": d} for d in [2, 3, 4]],
        "gb": [{"learning_rate": lr, "n_estimators": n}
               for lr in [0.02, 0.05] for n in [150, 300]],
    }[model_name]

    best, best_score = None, -np.inf
    n = len(Xa)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for params in grid:
        scores = []
        for tr_idx, va_idx in tscv.split(Xa):
            mu, sd = Xa[tr_idx].mean(0), Xa[tr_idx].std(0) + 1e-12
            Xtr2 = (Xa[tr_idx] - mu) / sd
            Xva2 = (Xa[va_idx] - mu) / sd
            if model_name == "logistic":
                m = LogisticRegression(C=params["C"], penalty="l2", max_iter=2000)
            elif model_name == "rf":
                m = RandomForestClassifier(
                    n_estimators=300, max_depth=params["max_depth"],
                    min_samples_leaf=30, random_state=SEED, n_jobs=-1)
            else:
                m = GradientBoostingClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=2, learning_rate=params["learning_rate"],
                    random_state=SEED)
            m.fit(Xtr2, ya[tr_idx])
            p = m.predict_proba(Xva2)[:, 1]
            acc = ((p > 0.5).astype(int) == ya[va_idx]).mean()
            scores.append(acc)
        s = float(np.mean(scores))
        if s > best_score:
            best_score, best = s, params
    return best, best_score


def fit_one(Xtr, ytr, Xte, model_name, params, standardize):
    Xa, Xb = Xtr.values.copy(), Xte.values.copy()
    if standardize:
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-12
        Xa = (Xa - mu) / sd
        Xb = (Xb - mu) / sd
    if model_name == "logistic":
        m = LogisticRegression(C=params["C"], penalty="l2", max_iter=2000)
    elif model_name == "rf":
        m = RandomForestClassifier(n_estimators=300, max_depth=params["max_depth"],
                                   min_samples_leaf=30, random_state=SEED, n_jobs=-1)
    else:
        m = GradientBoostingClassifier(n_estimators=params["n_estimators"],
                                       max_depth=2,
                                       learning_rate=params["learning_rate"],
                                       random_state=SEED)
    m.fit(Xa, ytr.values)
    return m, Xa, Xb


def evaluate(yte, prob):
    pred = (prob > 0.5).astype(int)
    acc = accuracy_score(yte, pred)
    try:
        auc = roc_auc_score(yte, prob)
    except ValueError:
        auc = 0.5
    base_p = float(yte.mean())
    return {
        "acc": float(acc),
        "auc": float(auc),
        "ll": logloss_const(yte.values, prob),
        "ll_const": logloss_const(yte.values, base_p),
        "n": int(len(yte)),
        "n_pos": int(yte.sum()),
    }


def binomial_p(acc, n):
    """单侧检验：H0: 准确率 = 0.5"""
    k = int(round(acc * n))
    return float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue)


def main():
    df = pd.read_csv(os.path.join(DATA, "gold_features.csv"),
                     index_col=0, parse_dates=True)
    pre = json.load(open(os.path.join(BASE, "PREREGISTRATION.json"), encoding="utf-8"))

    feats = [c for c in df.columns if c not in TARGETS + ["close"]]
    is_train = pd.Series(df.index < SPLIT, index=df.index)
    tr_mask = is_train.to_numpy()
    te_mask = ~tr_mask
    results = {}
    preds_store = {}

    def split_masks(index):
        """把切分掩码对齐到任意（可能更短的）子集索引上。"""
        m = is_train.reindex(index, fill_value=False)
        return m.to_numpy(), ~m.to_numpy()

    print("=" * 70)
    print("黄金多因子预测 · 训练与样本外验证")
    print(f"训练集 {tr_mask.sum()} 期 | 测试集 {te_mask.sum()} 期 "
          f"| 切分点 {SPLIT}")
    print("=" * 70)

    # ---------- 主实验：全因子，四个周期 ----------
    print("\n[实验1] 全因子模型（57维）")
    for tgt in TARGETS:
        sub = df[[tgt] + feats].dropna()
        y = (sub[tgt] > 0).astype(int)
        X = sub[feats]
        mtr, mte = split_masks(sub.index)
        Xtr, ytr = X[mtr], y[mtr]
        Xte, yte = X[mte], y[mte]
        if len(ytr) < 500 or len(yte) < 100:
            continue
        print(f"\n  ── {HORIZON_CN[tgt]} ({tgt}) ──  训练{len(ytr)} 测试{len(yte)}")
        tgt_res = {}
        for mname in ["logistic", "rf", "gb"]:
            params, cvscore = walkforward_select(Xtr, ytr, mname)
            m, Xa, Xb = fit_one(Xtr, ytr, Xte, mname, params, mname == "logistic")
            prob = m.predict_proba(Xb)[:, 1]
            r = evaluate(yte, prob)
            r["params"] = params
            r["cv_acc"] = cvscore
            r["p_value"] = binomial_p(r["acc"], r["n"])
            r["p_corr"] = min(1.0, r["p_value"] * len(TARGETS))
            r["pass_acc"] = r["acc"] > 0.53
            r["pass_sig"] = r["p_corr"] < ALPHA_CORR
            tgt_res[mname] = r
            flag = "✓通过" if (r["pass_acc"] and r["pass_sig"]) else "✗未达"
            print(f"    {mname:9s} acc={r['acc']*100:5.2f}%  auc={r['auc']:.4f}  "
                  f"ll={r['ll']:.4f}(基线{r['ll_const']:.4f})  "
                  f"p={r['p_value']:.4f} 校正p={r['p_corr']:.4f}  {flag}")
            preds_store[(tgt, mname)] = {
                "idx": Xte.index, "prob": prob, "actual": yte.values,
                "actual_ret": sub.loc[Xte.index, tgt].values}
        results[tgt] = tgt_res

    # ---------- 对照2：纯技术面基线 ----------
    print("\n[实验2] 对照组 —— 仅黄金自身动量（12维技术面）")
    tech_results = {}
    for tgt in ["fwd_1d", "fwd_3d", "fwd_5d"]:
        sub = df[[tgt] + TECH_FEATS].dropna()
        y = (sub[tgt] > 0).astype(int)
        X = sub[TECH_FEATS]
        mtr, _ = split_masks(sub.index)
        Xtr, ytr, Xte, yte = X[mtr], y[mtr], X[~mtr], y[~mtr]
        params, cvscore = walkforward_select(Xtr, ytr, "logistic")
        m, Xa, Xb = fit_one(Xtr, ytr, Xte, "logistic", params, True)
        prob = m.predict_proba(Xb)[:, 1]
        r = evaluate(yte, prob)
        r["p_value"] = binomial_p(r["acc"], r["n"])
        r["p_corr"] = min(1.0, r["p_value"] * len(TARGETS))
        tech_results[tgt] = r
        print(f"  {HORIZON_CN[tgt]:4s} acc={r['acc']*100:5.2f}%  auc={r['auc']:.4f}  p={r['p_value']:.4f}")

    # ---------- 对照3：打乱标签（20次） ----------
    print("\n[实验3] 对照组 —— 训练集内打乱标签，重复20次（检验是否只是噪声拟合）")
    shuffle_results = {}
    rng = np.random.default_rng(SEED)
    tgt = "fwd_1d"
    sub = df[[tgt] + feats].dropna()
    y = (sub[tgt] > 0).astype(int)
    X = sub[feats]
    mtr, _ = split_masks(sub.index)
    Xtr, ytr, Xte, yte = X[mtr], y[mtr], X[~mtr], y[~mtr]
    Xa0 = Xtr.values
    mu, sd = Xa0.mean(0), Xa0.std(0) + 1e-12
    Xb0 = (Xte.values - mu) / sd
    accs = []
    for i in range(20):
        ysh = pd.Series(rng.permutation(ytr.values))
        m = LogisticRegression(C=0.1, penalty="l2", max_iter=2000)
        m.fit((Xa0 - mu) / sd, ysh)
        accs.append(accuracy_score(yte, (m.predict_proba(Xb0)[:, 1] > 0.5).astype(int)))
    accs = np.array(accs)
    real_acc = results[tgt]["logistic"]["acc"]
    p95 = float(np.percentile(accs, 95))
    shuffle_results = {"accs": accs.tolist(), "p95": p95, "real_acc": float(real_acc),
                       "beat_p95": bool(real_acc > p95)}
    print(f"  打乱标签 20次样本外准确率: 均值{accs.mean()*100:.2f}%  "
          f"标准差{accs.std()*100:.2f}%  95分位 {p95*100:.2f}%  最大 {accs.max()*100:.2f}%")
    print(f"  真实标签模型: {real_acc*100:.2f}%")
    print(f"  判定: {'真实模型优于打乱95分位' if real_acc > p95 else '真实模型未超过打乱95分位 —— 学到的可能是噪声'}")

    # ---------- 对照4：随机特征 ----------
    print("\n[实验4] 对照组 —— 特征替换为随机数（结构相同）")
    rnd_accs = []
    for i in range(20):
        Xr = pd.DataFrame(rng.standard_normal(Xtr.shape), columns=Xtr.columns,
                          index=Xtr.index)
        params, _ = walkforward_select(Xr, ytr, "logistic")
        m, Xa, Xb = fit_one(Xr, ytr, Xte, "logistic", params, True)
        prob = m.predict_proba(Xb)[:, 1]
        rnd_accs.append(accuracy_score(yte, (prob > 0.5).astype(int)))
    rnd_accs = np.array(rnd_accs)
    rp95 = float(np.percentile(rnd_accs, 95))
    random_feat = {"accs": rnd_accs.tolist(), "p95": rp95,
                   "real_acc": float(real_acc), "beat_p95": bool(real_acc > rp95)}
    print(f"  随机特征 20次: 均值{rnd_accs.mean()*100:.2f}% 95分位 {rp95*100:.2f}%")
    print(f"  判定: {'真实特征优于随机95分位' if real_acc > rp95 else '真实特征未超过随机95分位'}")

    # ---------- 特征重要性 ----------
    print("\n[实验5] 特征系数（logistic，1日，全训练集标准化）")
    mu, sd = Xtr.values.mean(0), Xtr.values.std(0) + 1e-12
    lr = LogisticRegression(C=0.1, penalty="l2", max_iter=2000)
    lr.fit((Xtr.values - mu) / sd, ytr.values)
    coef = pd.Series(lr.coef_[0], index=feats).sort_values(key=abs, ascending=False)
    print("  Top15 绝对系数:")
    for k, v in coef.head(15).items():
        print(f"    {k:>16s}  {v:+.4f}")

    # ---------- 交易可行性粗估 ----------
    print("\n[实验6] 交易成本后粗估（1日方向，仅作量级参考）")
    cost_bps = 5  # 单边 5bp
    for mname in ["logistic", "rf", "gb"]:
        if tgt not in results or mname not in results[tgt]:
            continue
        pr = preds_store[(tgt, mname)]
        signal = (pr["prob"] > 0.6).astype(int) - (pr["prob"] < 0.4).astype(int)
        mask = signal != 0
        if mask.sum() < 20:
            print(f"  {mname}: 信号过少")
            continue
        ret = pr["actual_ret"][mask] * signal[mask]
        gross_bp = ret.mean() * 1e4
        net_bp = gross_bp - 2 * cost_bps
        print(f"  {mname:9s} 交易{len(ret)}次 毛收益{gross_bp:+.2f}bp "
              f"扣双边{cost_bps*2}bp后 {net_bp:+.2f}bp")

    # ---------- 汇总判定 ----------
    print("\n" + "=" * 70)
    print("预注册判据汇总")
    print("=" * 70)
    main_r = results.get("fwd_1d", {}).get("logistic", {})
    verdicts = {
        "主目标_1日_准确率": main_r.get("acc"),
        "阈值_0.53": main_r.get("pass_acc"),
        "校正p_0.0125": main_r.get("pass_sig"),
        "优于打乱标签95分位": shuffle_results["beat_p95"],
        "优于随机特征95分位": random_feat["beat_p95"],
        "优于纯技术面基线": (tech_results.get("fwd_1d", {}).get("acc", 0)
                            > main_r.get("acc", 1)),
    }
    for k, v in verdicts.items():
        if isinstance(v, float):
            print(f"  {k:22s} {v*100:.2f}%")
        else:
            print(f"  {k:22s} {v}")
    final_pass = bool(verdicts["阈值_0.53"] and verdicts["校正p_0.0125"]
                      and verdicts["优于打乱标签95分位"] and verdicts["优于随机特征95分位"])
    print(f"\n  ★ 最终判定: {'H1成立 —— 存在样本外可重复预测力' if final_pass else 'H1不成立 —— 未发现稳定样本外预测力'}")

    out = {
        "results": results,
        "technical_baseline": tech_results,
        "shuffle_control": shuffle_results,
        "random_feature_control": random_feat,
        "feature_coef": coef.head(25).to_dict(),
        "verdicts": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                     for k, v in verdicts.items()},
        "final_pass": final_pass,
        "alpha_corrected": ALPHA_CORR,
        "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
        "split_date": SPLIT,
    }
    p = os.path.join(OUT, "model_results.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果已存 {p}")

    # 预测明细供报告使用
    rows = []
    for (tg, mn), pr in preds_store.items():
        rows.append(pd.DataFrame({
            "date": pr["idx"], "target": tg, "model": mn,
            "prob": pr["prob"], "actual": pr["actual"], "ret": pr["actual_ret"]}))
    pd.concat(rows).to_csv(os.path.join(OUT, "oos_predictions.csv"),
                           index=False, encoding="utf-8-sig")
    print(f"预测明细已存 {os.path.join(OUT, 'oos_predictions.csv')}")


if __name__ == "__main__":
    main()