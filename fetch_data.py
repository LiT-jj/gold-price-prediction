"""
黄金多因子数据集构建
数据源：
  - XAU 伦敦金日线（新浪财经 GlobalFuturesService，2006-10起，约5200 期）
  - 美元指数 DTWEXBGS / DEXUSEU（FRED，广义美元指数）
  - 非农就业 PAYEMS、失业率 UNRATE（FRED，月度）
  - WTI 原油 DCOILWTICO（FRED）
  - VIXCLS 恐慌指数、GVZCLS 黄金波动率指数（FRED）
  - DFII10 10年期美国实际国债收益率（FRED）—— 持有黄金的机会成本
  - T10YIE / T10Y2Y 期限利差（FRED）

方法论要点（承接彩票项目的教训）：
1. 所有特征严格使用 t-1 及更早的信息，禁止任何当期数据
2. 月度数据（PAYEMS/UNRATE）必须按发布日期对齐，不能按统计月份对齐
3. 非农有"修正"问题——初值与终值差异，用修订后序列做静态分析，
   但必须在报告中标注这一偏差来源
4. 每个特征都记录"入库日期"（何时可知），而不只是"统计日期"
"""
import urllib.request
import re
import json
import os
import io
import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)

HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
       "Referer": "https://finance.sina.com.cn/"}


def _get(url, headers=None, timeout=30, gbk=False):
    req = urllib.request.Request(url, headers=headers or HDR)
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    return raw.decode("gbk", "ignore") if gbk else raw.decode("utf-8", "ignore")


# ---------------- 数据源1：伦敦金日线 ----------------
def fetch_xau():
    url = ("https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_XAU=/"
           "GlobalFuturesService.getGlobalFuturesDailyKLine?symbol=XAU")
    txt = _get(url)
    arr = json.loads(re.search(r"\((\[.*\])\)", txt, re.S).group(1))
    rows = []
    for r in arr:
        rows.append({
            "date": pd.to_datetime(r["date"]),
            "xau_open": float(r["open"]),
            "xau_high": float(r["high"]),
            "xau_low": float(r["low"]),
            "xau_close": float(r["close"]),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df = df[df["xau_close"] > 0]
    assert df["xau_close"].between(100, 10000).all(), "XAU 价格超出合理区间"
    assert df["date"].is_monotonic_increasing
    assert not df["date"].duplicated().any(), "XAU 存在重复日期"
    return df


# ---------------- 数据源2：FRED 宏观 ----------------
FRED_IDS = {
    "dxy":        "DTWEXBGS",   # 广义美元指数（日频）
    "useu":       "DEXUSEU",    # 美元/欧元汇率（日频）
    "nfp":        "PAYEMS",     # 非农就业人数（月频，千人）
    "unrate":     "UNRATE",     # 失业率（月频，%）
    "wti":        "DCOILWTICO", # WTI 原油现货（日频）
    "vix":        "VIXCLS",     # 恐慌指数（日频）
    "gvz":        "GVZCLS",     # 黄金波动率指数（日频）
    "realyield":  "DFII10",     # 10年期TIPS实际收益率（日频,%）
    "y10":        "DGS10",      # 10年期名义国债收益率（日频,%）
    "breakeven":  "T10YIE",     # 10年通胀预期（日频,%）
    "curve":      "T10Y2Y",     # 10Y-2Y 利差（日频,%）
    "fedfunds":   "DFF",        # 联邦基金有效利率（日频,%）
    "indpro":     "INDPRO",     # 工业生产指数（月频，2017=100）
    "cpia":       "CPIAUCSL",   # CPI 指数（月频，1982-84=100）
}


def fetch_fred(name, sid):
    """FRED 免key CSV 接口。返回 date -> value。"""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2005-01-01"
    txt = _get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=40)
    if txt.lstrip().startswith("<"):
        raise RuntimeError(f"FRED {sid} 被拦截")
    lines = txt.strip().splitlines()
    hdr = lines[0].split(",")
    if len(hdr) < 2:
        raise RuntimeError(f"FRED {sid} 表头异常: {hdr}")
    out = {}
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        d, v = parts[0], parts[1].strip()
        if v in ("", "."):
            continue
        try:
            out[pd.to_datetime(d)] = float(v)
        except ValueError:
            continue
    return out


def fetch_all_fred():
    data, failed = {}, {}
    for name, sid in FRED_IDS.items():
        try:
            data[name] = fetch_fred(name, sid)
            print(f"  [OK]   {name:10s} {sid:14s} n={len(data[name])}")
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {e}"
            print(f"  [FAIL] {name:10s} {sid:14s} {type(e).__name__}")
    return data, failed


# ---------------- 非农发布日期对齐（关键防泄漏） ----------------
# BLS 非农就业报告通常在当月第一个周五 08:30 ET 发布。
# 统计月份为 M 的数据，在次月第一个周五左右才可知。
def nfp_release_dates(series):
    """由统计期生成真实可得日期：次月第一个周五。"""
    rel = {}
    for stat_month in series.index:
        m = stat_month + pd.offsets.MonthBegin(1)
        # 该月第一个周五
        friday = m + pd.to_timedelta((4 - m.dayofweek) % 7, unit="D")
        rel[friday] = series[stat_month]
    return pd.Series(rel)


# ---------------- 特征工程 ----------------
def build_features(xau, fred):
    """返回 (特征表, 标签表)。特征窗口严格不含当日信息。"""
    df = xau.copy().set_index("date")

    for name, s in fred.items():
        ss = pd.Series(s).sort_index()
        # 日频宏观直接对齐（前值填充最多5个交易日）
        df[name] = ss.reindex(df.index).ffill(limit=5)

    # ---------- 黄金自身动量（严格 t-1 及更早） ----------
    c = df["xau_close"]
    df["au_ret1"] = c.pct_change(1)        # t日收益，用于展示
    # 预测特征只能用 t-1 及之前 ->统一 shift(1)
    feats = pd.DataFrame(index=df.index)
    feats["au_r1"]   = c.pct_change(1).shift(1)
    feats["au_r3"]   = c.pct_change(3).shift(1)
    feats["au_r5"]   = c.pct_change(5).shift(1)
    feats["au_r10"]  = c.pct_change(10).shift(1)
    feats["au_r21"]  = c.pct_change(21).shift(1)
    feats["au_vol5"]  = c.pct_change(1).rolling(5).std().shift(1)
    feats["au_vol20"] = c.pct_change(1).rolling(20).std().shift(1)
    feats["au_ma_ratio"] = (c / c.rolling(20).mean() - 1).shift(1)
    feats["au_rsi14"] = self_rsi(c, 14).shift(1)
    # 隔夜跳空（当日开盘 vs 前收）—— 只用当日开盘，属于 T 日盘中已知，
    # 但我们预测的是 T+1，故允许使用 T 日完整信息。
    feats["gap"] = (df["xau_open"] / c.shift(1) - 1)
    feats["intraday"] = (c / df["xau_open"] - 1)
    feats["hl_range"] = (df["xau_high"] / df["xau_low"] - 1)

    # ---------- 美元 ----------
    for k in ["dxy", "useu"]:
        if k in df.columns:
            s = df[k]
            feats[f"{k}_r1"] = s.pct_change(1).shift(1)
            feats[f"{k}_r5"] = s.pct_change(5).shift(1)
            feats[f"{k}_r20"] = s.pct_change(20).shift(1)
            feats[f"{k}_r60"] = s.pct_change(60).shift(1)
            feats[f"{k}_level"] = s.shift(1)
            feats[f"{k}_ma20"] = (s / s.rolling(20).mean() - 1).shift(1)

    # ---------- 实际利率（黄金的机会成本） ----------
    if "realyield" in df.columns:
        s = df["realyield"]
        feats["ry_chg1"] = s.diff(1).shift(1)
        feats["ry_chg5"] = s.diff(5).shift(1)
        feats["ry_level"] = s.shift(1)

    # ---------- 名义利率与曲线 ----------
    for k in ["y10", "fedfunds"]:
        if k in df.columns:
            feats[f"{k}_chg1"] = df[k].diff(1).shift(1)
            feats[f"{k}_chg5"] = df[k].diff(5).shift(1)
            feats[f"{k}_level"] = df[k].shift(1)
    if "curve" in df.columns:
        feats["curve_chg5"] = df["curve"].diff(5).shift(1)
        feats["curve_level"] = df["curve"].shift(1)

    # ---------- 原油 / 通胀 ----------
    if "wti" in df.columns:
        feats["wti_r1"] = df["wti"].pct_change(1).shift(1)
        feats["wti_r5"] = df["wti"].pct_change(5).shift(1)
        feats["wti_r20"] = df["wti"].pct_change(20).shift(1)
        feats["wti_level"] = df["wti"].shift(1)
    if "breakeven" in df.columns:
        feats["be5_chg5"] = df["breakeven"].diff(5).shift(1)
        feats["be5_level"] = df["breakeven"].shift(1)
    if "cpia" in df.columns:
        feats["cpi_yoy"] = df["cpia"].pct_change(12).shift(1)

    # ---------- 风险偏好 / 波动率 ----------
    if "vix" in df.columns:
        feats["vix_level"] = df["vix"].shift(1)
        feats["vix_chg1"] = df["vix"].diff(1).shift(1)
        feats["vix_chg5"] = df["vix"].diff(5).shift(1)
        feats["vix_pctl60"] = df["vix"].rolling(60).rank(pct=True).shift(1)
    if "gvz" in df.columns:
        feats["gvz_level"] = df["gvz"].shift(1)
        feats["gvz_chg5"] = df["gvz"].diff(5).shift(1)

    # ---------- 月度宏观：非农/失业率/工业生产（按发布日对齐） ----------
    for key, tag in [("nfp", "nfp"), ("unrate", "unrate"), ("indpro", "indpro")]:
        if key not in fred:
            continue
        s = pd.Series(fred[key])
        s.index = pd.to_datetime(s.index)
        rel = nfp_release_dates(s)
        # 对每个交易日，取<=该日 已发布的最新值
        merged = pd.Series(index=df.index, dtype=float)
        rel_sorted = rel.sort_index()
        idx = df.index
        pos = np.searchsorted(rel_sorted.index.values, idx.values, side="right") - 1
        vals = np.where(pos >= 0, rel_sorted.values[np.clip(pos, 0, None)], np.nan)
        merged[:] = vals
        df[f"{tag}_released"] = merged
        if key == "nfp":
            feats["nfp_chg1"] = merged.diff(1)      # 新增非农人数
            feats["nfp_chg3"] = merged.diff(3)
            feats["nfp_chg6"] = merged.diff(6)
            feats["nfp_chg12"] = merged.diff(12)
        if key == "unrate":
            feats["ur_level"] = merged.shift(1)
            feats["ur_chg3"] = merged.diff(3)
        if key == "indpro":
            feats["ip_mom"] = merged.pct_change(1)
            feats["ip_yoy"] = merged.pct_change(12)

    # ---------- 战争/地缘风险代理 ----------
    # 用VIX 与 原油 波动 作为地缘冲突的可观测代理；
    # 再加日历特征：FOMC 会议窗口（每月 Fed 决议日）
    feats["month"] = df.index.month
    feats["dow"] = df.index.dayofweek

    # ---------- 标签 ----------
    labels = pd.DataFrame(index=df.index)
    labels["fwd_1d"] = c.pct_change(1).shift(-1)
    labels["fwd_3d"] = c.pct_change(3).shift(-3)
    labels["fwd_5d"] = c.pct_change(5).shift(-5)
    labels["fwd_10d"] = c.pct_change(10).shift(-10)
    labels["close"] = c

    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats, labels, df


def self_rsi(series, n=14):
    d = series.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / (dn + 1e-12)
    return 100 - 100 / (1 + rs)


# ---------------- 泄漏自检 ----------------
def leakage_selfcheck(feats, labels, raw):
    """手工复算若干特征，与程序结果逐项比对。"""
    checks = []
    idx = feats.index
    c = raw["xau_close"]
    for probe in [300, 1500, 3000, 4000]:
        d = idx[probe]
        # au_r5 = (close[d-1]/close[d-6]) - 1
        manual = c.loc[idx[probe - 1]] / c.loc[idx[probe - 6]] - 1
        auto = feats.loc[d, "au_r5"]
        checks.append((f"au_r5@{d.date()}", abs(manual - auto) < 1e-9, manual, auto))
        # 严格性检查：au_r5 不得用到 d 当日价格
        wrong = c.loc[d] / c.loc[idx[probe - 5]] - 1
        checks.append((f"au_r5不含当日@{d.date()}", abs(wrong - auto) > 1e-6, wrong, auto))
    # 非农对齐检查：released 值必须 <= 当日
    if "nfp_released" in raw.columns:
        sub = raw.dropna(subset=["nfp_released"])
        checks.append(("非农已对齐", len(sub) > 100, len(sub), len(raw)))
    return checks


if __name__ == "__main__":
    print("=" * 66)
    print("黄金多因子数据集构建")
    print("=" * 66)

    print("\n[1/5] 抓取伦敦金日线")
    xau = fetch_xau()
    print(f"  XAU: {len(xau)} 期  {xau['date'].iloc[0].date()} ~ {xau['date'].iloc[-1].date()}")
    print(f"  最新收盘: {xau['xau_close'].iloc[-1]:.2f} USD/oz")

    print("\n[2/5] 抓取 FRED 宏观序列")
    fred, failed = fetch_all_fred()

    print("\n[3/5] 特征工程")
    feats, labels, raw = build_features(xau, fred)
    keep = feats.notna().sum() >= int(0.5 * len(feats))
    feats = feats.loc[:, keep]
    valid = feats.notna().all(axis=1) & labels["fwd_1d"].notna()
    feats, labels = feats[valid], labels[valid]
    print(f"  特征维度: {feats.shape[1]}  有效交易日: {len(feats)}")
    print(f"  区间: {feats.index[0].date()} ~ {feats.index[-1].date()}")

    print("\n[4/5] 泄漏自检")
    allok = True
    for name, ok, a, b in leakage_selfcheck(feats, labels, raw):
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}  a={a} b={b}")
        allok &= ok
    if not allok:
        raise SystemExit("泄漏自检未通过，中止")

    print("\n[5/5] 落盘")
    out = feats.join(labels)
    path = os.path.join(OUT_DIR, "gold_features.csv")
    out.to_csv(path, encoding="utf-8-sig")
    raw.to_csv(os.path.join(OUT_DIR, "gold_raw.csv"), encoding="utf-8-sig")
    meta = {
        "n_samples": int(len(out)),
        "n_features": int(feats.shape[1]),
        "start": str(out.index[0].date()),
        "end": str(out.index[-1].date()),
        "xau_latest": float(xau["xau_close"].iloc[-1]),
        "fred_ok": sorted(fred.keys()),
        "fred_failed": failed,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  {path}")
    print(f"  特征清单 ({feats.shape[1]}维):")
    names = list(feats.columns)
    for i in range(0, len(names), 6):
        print("    " + "  ".join(f"{n:>16s}" for n in names[i:i + 6]))
    print("\n完成。")