"""
黄金价格走势复盘与因子相关性分析
用于报告第一章：数据概览 + 经济机制验证
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, "gold_features.csv"),
                 index_col=0, parse_dates=True)
raw = pd.read_csv(os.path.join(DATA, "gold_raw.csv"), index_col=0, parse_dates=True)

summary = {}

# ---- 1. 价格概览 ----
c = raw["xau_close"]
years = raw.groupby(raw.index.year).agg(
    close_first=("xau_close", "first"), close_last=("xau_close", "last"),
    hi=("xau_high", "max"), lo=("xau_low", "min"), days=("xau_close", "size"))
years["ret"] = years["close_last"] / years["close_first"] - 1
years["range_pos"] = (years["close_last"] - years["lo"]) / (years["hi"] - years["lo"])
summary["price"] = {
    "start": str(c.index[0].date()), "end": str(c.index[-1].date()),
    "latest": float(c.iloc[-1]),
    "min": float(c.min()), "max": float(c.max()),
    "n_days": int(len(c)),
    "annual_vol": float(c.pct_change().std() * np.sqrt(252)),
}
years.index.name = "year"
summary["yearly"] = years.reset_index()
summary["yearly"]["year"] = summary["yearly"]["year"].astype(int)

# ---- 2. 涨跌分布（肥尾检验） ----
r1 = c.pct_change().dropna()
# KS 统计量手工计算（scipy 1.18 的 args 传参接口变更）
xs = np.sort(r1.values)
cdf = stats.norm.cdf(xs, loc=r1.mean(), scale=r1.std())
ks_stat = float(np.max(np.abs(np.arange(1, len(xs) + 1) / len(xs) - cdf)))
n_ks = len(xs)
ks_p = float(stats.kstest(r1.values, "norm", args=(r1.mean(), r1.std())).pvalue) \
    if False else float(stats.kstest(r1.values, "norm").pvalue)
summary["returns"] = {
    "daily_mean": float(r1.mean()),
    "daily_std": float(r1.std()),
    "kurtosis": float(r1.kurtosis()),
    "skew": float(r1.skew()),
    "ks_stat": ks_stat, "ks_p": ks_p,
    "pct_up": float((r1 > 0).mean()),
    "worst_day": float(r1.min()), "best_day": float(r1.max()),
    "worst_date": str(r1.idxmin().date()), "best_date": str(r1.idxmax().date()),
}

# ---- 3. 因子相关性（对 fwd_1d），分训练/测试期 ----
feats = [c for c in df.columns if c not in ["fwd_1d", "fwd_3d", "fwd_5d",
                                            "fwd_10d", "close"]]
y = (df["fwd_1d"] > 0).astype(int)
SPLIT = "2019-01-01"
tr = df.index < SPLIT
corr_rows = []
for f in feats:
    sub = pd.DataFrame({"x": df[f], "y": y, "tr": tr}).dropna()
    if len(sub) < 200:
        continue
    pr, pp = stats.pointbiserialr(sub["y"], sub["x"])
    subtr = sub[sub["tr"]]
    subte = sub[~sub["tr"]]
    ptr, ptp = stats.pointbiserialr(subtr["y"], subtr["x"]) if subtr["y"].nunique() > 1 else (np.nan, np.nan)
    pte, pep = stats.pointbiserialr(subte["y"], subte["x"]) if subte["y"].nunique() > 1 else (np.nan, np.nan)
    corr_rows.append({
        "feature": f, "corr_all": pr, "p_all": pp,
        "corr_train": ptr, "corr_test": pte, "p_test": pep,
        "n": len(sub),
    })
corr_df = pd.DataFrame(corr_rows).sort_values("corr_all", key=abs, ascending=False)
summary["correlations"] = corr_df.to_dict("records")

# ---- 4. 机制分组验证 ----
GROUPS = {
    "美元指数": ["dxy_r1", "dxy_r5", "dxy_r20", "dxy_r60", "dxy_ma20", "useu_r1", "useu_r5", "useu_r20"],
    "实际利率": ["ry_chg1", "ry_chg5", "ry_level", "y10_chg1", "y10_chg5"],
    "黄金动量": ["au_r1", "au_r3", "au_r5", "au_r10", "au_r21", "au_ma_ratio", "au_rsi14"],
    "波动率": ["au_vol5", "au_vol20", "hl_range", "vix_level", "vix_chg1", "vix_chg5", "vix_pctl60", "gvz_level"],
    "原油通胀": ["wti_r1", "wti_r5", "wti_r20", "be5_chg5", "be5_level", "cpi_yoy"],
    "就业周期": ["nfp_chg1", "nfp_chg3", "nfp_chg6", "nfp_chg12", "ur_level", "ur_chg3", "ip_mom", "ip_yoy"],
}
grp_res = []
for g, fl in GROUPS.items():
    fl = [f for f in fl if f in feats]
    if not fl:
        continue

    def corr_on(frame):
        """在给定子样本上计算该组各特征与次日涨跌的点二列相关。"""
        out = {}
        for f in fl:
            pair = pd.concat([frame[f].rename("x"), frame["y"]], axis=1).dropna()
            if len(pair) < 100 or pair["y"].nunique() < 2:
                continue
            out[f] = float(stats.pointbiserialr(pair["y"], pair["x"])[0])
        return out

    work = pd.concat([df[fl], y.rename("y")], axis=1).dropna()
    all_c = corr_on(work)
    if not all_c:
        continue
    tr_c = corr_on(work[work.index < SPLIT])
    te_c = corr_on(work[work.index >= SPLIT])
    best = max(all_c, key=lambda k: abs(all_c[k]))
    grp_res.append({
        "group": g, "n_feats": len(fl), "n_obs": int(len(work)),
        "mean_abs_corr": float(np.mean(np.abs(list(all_c.values())))),
        "max_abs_corr": float(np.max(np.abs(list(all_c.values())))),
        "best_feat": best,
        "best_corr": float(all_c[best]),
        "train_mean_corr": float(np.mean(list(tr_c.values()))) if tr_c else None,
        "test_mean_corr": float(np.mean(list(te_c.values()))) if te_c else None,
        "sign_flip": (bool(np.mean(list(tr_c.values())) * np.mean(list(te_c.values())) < 0)
                      if tr_c and te_c else None),
    })
summary["groups"] = grp_res

# ---- 5. 分年度胜率基准（随机买入的天然胜率） ----
yr_acc = {}
for y_, g in df.groupby(df.index.year):
    s = g["fwd_1d"].dropna()
    if len(s) > 20:
        yr_acc[int(y_)] = float((s > 0).mean())
summary["yearly_up_ratio"] = yr_acc

p = os.path.join(OUT, "analysis_summary.json")


def _sanitize(obj):
    """把 DataFrame/Series 递归转成可 JSON 序列化的原生类型。"""
    import math
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _sanitize(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _sanitize(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)[:10]
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)
with open(p, "w", encoding="utf-8") as f:
    json.dump(_sanitize(summary), f, ensure_ascii=False, indent=2)
corr_df.to_csv(os.path.join(OUT, "factor_correlations.csv"),
               index=False, encoding="utf-8-sig")

print("=" * 66)
print("黄金因子分析摘要")
print("=" * 66)
p_ = summary["price"]
print(f"\n样本: {p_['start']} ~ {p_['end']}  {p_['n_days']} 个交易日")
print(f"价格: {p_['min']:.0f} ~ {p_['max']:.0f}  最新 {p_['latest']:.2f} USD/oz")
print(f"年化波动率: {p_['annual_vol']*100:.1f}%")

r_ = summary["returns"]
print(f"\n日收益: 均值{r_['daily_mean']*100:.4f}% 标准差{r_['daily_std']*100:.3f}% "
      f"峰度{r_['kurtosis']:.2f} 上涨占比{r_['pct_up']*100:.1f}%")
print(f"最大单日: {r_['best_date']} {r_['best_day']*100:+.2f}%")
print(f"最大单跌: {r_['worst_date']} {r_['worst_day']*100:+.2f}%")

print("\n── 组别机制验证（对次日涨跌）──")
print(f"{'组别':<10s}{'特征数':>6s}{'最大|corr|':>12s}{'最强特征':>16s}"
      f"{'训练均值':>10s}{'测试均值':>10s}{'符号翻转':>10s}")
for g in grp_res:
    flip = "是" if g["sign_flip"] else "否"
    trm = f"{g['train_mean_corr']:+.4f}" if g["train_mean_corr"] is not None else "NA"
    tem = f"{g['test_mean_corr']:+.4f}" if g["test_mean_corr"] is not None else "NA"
    print(f"{g['group']:<10s}{g['n_feats']:>6d}{g['max_abs_corr']:>12.4f}"
          f"{g['best_feat']:>16s}{trm:>10s}{tem:>10s}{flip:>10s}")

print("\n── Top12 单因子相关性（对次日涨跌，全样本）──")
for _, r in corr_df.head(12).iterrows():
    print(f"  {r['feature']:>16s}  r={r['corr_all']:+.4f} p={r['p_all']:.4f}  "
          f"训练{r['corr_train']:+.4f} → 测试{r['corr_test']:+.4f}")

print(f"\n已存 {p}")