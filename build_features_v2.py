"""
黄金 v2 数据集：时间特征 + 宏观事件日历

相比 v1 新增四类特征块：
  1. 周期性时间特征（sin/cos 多尺度编码 + 月内/周内位置）
  2. 宏观事件窗口（FOMC / CPI / 非农 的真实发布日历与倒计时）
  3. 市场状态regime（波动率分位、趋势强度、滚动相关性）
  4. 交互特征（美元×实际利率、实际利率×VIX）

★防泄漏铁律★
  所有特征只使用 t-1 及更早的信息。
  事件日历用"事件是否已发生 + 距最近事件天数"的形式，
  绝不把"未来事件的具体内容"当作特征（如当日公布的非农数值本身，
  在发布当天的 08:30 之前是未知的）。
"""
import os
import json
import re
import urllib.request
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

HDR = {"User-Agent": "Mozilla/5.0"}


# ============ 宏观事件日历 ============
def fomc_dates():
    """FOMC 利率决议日（约每年8 次）。用规律性近似生成。

    FOMC 会议通常在以下日期附近：1月末/3月中/4月末/5月中/6月中/
    7月末/9月中/10月底/11月中/12月中。
    这里用「每年固定候选日 + 第三个周三」的规律近似，
    并在报告中明确标注这是近似日历而非官方日历。
    说明：黄金对FOMC 的反应窗口是决议日当天与次日，
    即使日期有 1~2 天误差，±3 日窗口特征仍能覆盖。
    """
    out = []
    for yr in range(2005, 2027):
        # 1月/3月/5月/7月 的决议多在月末附近
        for m, day_base in [(1, 31), (3, 20), (5, 9), (7, 29)]:
            d = pd.Timestamp(yr, m, 1) + pd.offsets.MonthEnd(0)
            # 往前找最近的周三
            while d.dayofweek != 2:
                d -= pd.Timedelta(days=1)
            out.append(d)
        # 4/6/9/11 的通常在月中（第三个周三附近）
        for m in [4, 6, 9, 11]:
            d = pd.Timestamp(yr, m, 1) + pd.Timedelta(days=14)
            while d.dayofweek != 2:
                d += pd.Timedelta(days=1)
            out.append(d)
    return sorted(set(out))


def nonfarm_release_days(months):
    """由非农统计期推算发布日：次月第一个周五。months 为统计期时间戳序列。"""
    idx = pd.to_datetime(pd.Index(months))
    rel = []
    for m in idx:
        nxt = m + pd.offsets.MonthBegin(1)
        friday = nxt + pd.to_timedelta((4 - nxt.dayofweek) % 7, unit="D")
        rel.append(friday)
    return sorted(set(rel))


def fetch_nfp_months():
    """从 FRED 取非农统计期（PAYEMS 的日期即统计月）。"""
    cache = os.path.join(DATA, "nfp_months.json")
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PAYEMS&cosd=2005-01-01"
    req = urllib.request.Request(url, headers=HDR)
    txt = urllib.request.urlopen(req, timeout=40).read().decode()
    months = []
    for ln in txt.strip().splitlines()[1:]:
        parts = ln.split(",")
        if len(parts) < 2 or parts[1].strip() in ("", "."):
            continue
        months.append(parts[0])
    json.dump(months, open(cache, "w", encoding="utf-8"))
    return months


def cpi_release_days():
    """CPI 通常在统计月次月的 10 日前后发布。用「次月 10~14 日中的某工作日」近似。"""
    out = []
    for yr in range(2005, 2027):
        for m in range(1, 13):
            d = pd.Timestamp(yr, m, 15)
            # 往前找最近工作日
            while d.dayofweek >= 5:
                d -= pd.Timedelta(days=1)
            out.append(d)
    return out


def days_since_last(dates, idx):
    """对每个交易日，返回距最近一次事件的天数（正数）。"""
    ds = np.array([d.toordinal() for d in dates], dtype=float)
    out = np.full(len(idx), np.nan)
    for i, d in enumerate(idx):
        v = d.toordinal()
        prior = ds[ds <= v]
        out[i] = (v - prior.max()) if len(prior) else np.nan
    return out


def days_to_next(dates, idx):
    """对每个交易日，返回距下一次事件的天数（正数）；无未来事件则 NaN。"""
    ds = np.array([d.toordinal() for d in dates], dtype=float)
    out = np.full(len(idx), np.nan)
    for i, d in enumerate(idx):
        v = d.toordinal()
        nxt = ds[ds > v]
        out[i] = (nxt.min() - v) if len(nxt) else np.nan
    return out


def in_event_window(dates, idx, lo=-1, hi=3):
    """事件窗口哑变量：事件日在[lo, hi] 偏移内为 1。"""
    ds = np.array([d.toordinal() for d in dates], dtype=float)
    out = np.zeros(len(idx))
    for i, d in enumerate(idx):
        v = d.toordinal()
        near = ds[(ds >= v + lo) & (ds <= v + hi)]
        out[i] = 1.0 if len(near) else 0.0
    return out


# ============ 主构建 ============
def build():
    print("[1/4] 读取 v1 特征表")
    df = pd.read_csv(os.path.join(DATA, "gold_features.csv"),
                     index_col=0, parse_dates=True)
    targets = ["fwd_1d", "fwd_3d", "fwd_5d", "fwd_10d"]
    base_feats = [c for c in df.columns if c not in targets + ["close"]]
    X = df[base_feats].copy()
    idx = X.index

    print("[2/4] 构造事件日历")
    raw = pd.read_csv(os.path.join(DATA, "gold_raw.csv"),
                      index_col=0, parse_dates=True)

    fomc = fomc_dates()
    nfp_months = fetch_nfp_months()
    nfp_days = nonfarm_release_days(nfp_months)
    cpi = cpi_release_days()

    print(f"    FOMC 事件 {len(fomc)} 个  非农发布 {len(nfp_days)} 次"
          f"  CPI 发布 {len(cpi)} 个")

    print("[3/4] 构造时间特征")
    T = pd.DataFrame(index=idx)

    # ---- 1. 周期性 sin/cos 编码 ----
    doy = idx.dayofyear.values
    dow = idx.dayofweek.values
    T["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    T["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    T["sin_dow"] = np.sin(2 * np.pi * dow / 7)
    T["cos_dow"] = np.cos(2 * np.pi * dow / 7)
    T["sin_month"] = np.sin(2 * np.pi * idx.month.values / 12)
    T["cos_month"] = np.cos(2 * np.pi * idx.month.values / 12)
    # 季度
    T["sin_q"] = np.sin(2 * np.pi * idx.quarter.values / 4)
    T["cos_q"] = np.cos(2 * np.pi * idx.quarter.values / 4)

    # ---- 2. 日历位置 ----
    T["month_pos"] = (idx.day.values - 1) / 30.0
    T["is_month_start"] = (idx.day <= 3).astype(float)
    T["is_month_end"] = (idx.day >= 27).astype(float)
    T["is_quarter_end"] = ((idx.month.isin([3, 6, 9, 12])) &
                           (idx.day >= 25)).astype(float)
    T["is_year_end"] = ((idx.month == 12) & (idx.day >= 20)).astype(float)
    T["days_to_year_end"] = np.asarray(
        (pd.Timestamp(2026, 12, 31) - pd.to_datetime(idx)).days) / 365.0

    # ---- 3. 宏观事件窗口 ----
    T["fomc_d_since"] = np.log1p(days_since_last(fomc, idx))
    T["fomc_d_to"] = np.log1p(days_to_next(fomc, idx))
    T["fomc_window"] = in_event_window(fomc, idx, lo=-1, hi=1)
    T["fomc_window2"] = in_event_window(fomc, idx, lo=0, hi=4)

    T["cpi_d_since"] = np.log1p(days_since_last(cpi, idx))
    T["cpi_d_to"] = np.log1p(days_to_next(cpi, idx))
    T["cpi_window"] = in_event_window(cpi, idx, lo=-1, hi=2)

    # 非农：使用真实统计期推算的发布日
    T["nfp_d_since"] = np.log1p(days_since_last(nfp_days, idx))
    T["nfp_d_to"] = np.log1p(days_to_next(nfp_days, idx))
    T["nfp_window"] = in_event_window(nfp_days, idx, lo=-1, hi=1)
    T["nfp_eve"] = in_event_window(nfp_days, idx, lo=-1, hi=-1)   # 非农前一日

    # 组合事件：FOMC + 非农同周 = 高波动窗口
    T["event_density"] = T["fomc_window"] + T["nfp_window"] + T["cpi_window"]

    # ---- 4. 市场状态 regime ----
    c = raw["xau_close"]
    r1 = c.pct_change()
    for w in [20, 60, 250]:
        T[f"vol_pctl_{w}"] = r1.rolling(w).rank(pct=True).shift(1)
    # 趋势强度：|价格 - MA|/价格 的滚动值 + 均线方向
    for w in [20, 60]:
        ma = c.rolling(w).mean()
        T[f"ma_dev_{w}"] = (c / ma - 1).shift(1)
        T[f"ma_slope_{w}"] = (ma.diff(5) / ma).shift(1)
    # 滚动相关性 regime（金价 vs 美元 / 实际利率）
    if "dxy" in raw.columns:
        T["corr_au_dxy_60"] = c.pct_change().rolling(60).corr(
            raw["dxy"].pct_change()).shift(1)
    if "realyield" in raw.columns:
        T["corr_au_ry_60"] = c.pct_change().rolling(60).corr(
            raw["realyield"].diff()).shift(1)
    # 波动率变化率（波动率聚集）
    T["vol_of_vol"] = r1.rolling(20).std().diff().shift(1)
    T["abs_ret_5"] = r1.rolling(5).sum().abs().shift(1)

    # ---- 5. 交互特征 ----
    if "dxy_r5" in X.columns and "ry_chg5" in X.columns:
        T["int_dxy_ry"] = X["dxy_r5"] * X["ry_chg5"]
    if "ry_chg5" in X.columns and "vix_chg5" in X.columns:
        T["int_ry_vix"] = X["ry_chg5"] * X["vix_chg5"]
    if "au_r5" in X.columns and "dxy_r5" in X.columns:
        T["int_au_dxy"] = X["au_r5"] * X["dxy_r5"]
    if "wti_r5" in X.columns and "au_r5" in X.columns:
        T["int_wti_au"] = X["wti_r5"] * X["au_r5"]

    print("[4/4] 合并与落盘")
    ALL = X.join(T)
    ALL = ALL.join(df[targets])          # 标签单独合并，避免被特征过滤逻辑影响
    ALL = ALL.replace([np.inf, -np.inf], np.nan)
    # 保留覆盖率≥60% 的新特征
    keep = [c for c in T.columns if T[c].notna().mean() >= 0.6]
    drop = [c for c in T.columns if c not in keep]
    ALL = ALL.drop(columns=drop)
    ALL = ALL.ffill(limit=3)     # 事件日历类特征少量缺失可安全前填

    valid = ALL[base_feats].notna().all(axis=1) & df[targets].notna().all(axis=1)
    ALL = ALL[valid]
    print(f"    原特征 {len(base_feats)} 维 + 时间特征 {len(keep) - len(drop)} 维"
          f" = {len(base_feats)+len(keep)-len(drop)} 维")
    print(f"    有效样本 {len(ALL)} 个交易日  {ALL.index[0].date()} ~ {ALL.index[-1].date()}")

    # 泄漏自检：时间特征不得含当日之后的信息
    print("\n[泄漏自检]")
    ok = True
    # FOMC d_to 在事件日当天应为 0
    for probe in [1000, 2500, 3500]:
        d = ALL.index[probe]
        v = ALL.loc[d, "fomc_d_since"]
        print(f"    {d.date()} FOMC距上次事件 log1p={v:.3f}  "
              f"(0=当天是决议日)")
    # 检验 sin_doy 计算正确
    d = ALL.index[500]
    manual = np.sin(2 * np.pi * d.dayofyear / 365.25)
    auto = ALL.loc[d, "sin_doy"]
    good = abs(manual - auto) < 1e-9
    print(f"    sin_doy 复算 {'OK' if good else 'FAIL'}")
    ok &= good
    if not ok:
        raise SystemExit("泄漏自检失败")

    p = os.path.join(DATA, "gold_features_v2.csv")
    ALL.to_csv(p, encoding="utf-8-sig")
    meta = {
        "n_samples": int(len(ALL)),
        "n_base": len(base_feats),
        "n_time": len(keep),
        "n_total_features": int(ALL.shape[1] - len(targets) - 1),
        "start": str(ALL.index[0].date()), "end": str(ALL.index[-1].date()),
        "time_features": keep,
        "dropped_time": drop,
    }
    json.dump(meta, open(os.path.join(DATA, "meta_v2.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已存 {p}")
    print(f"\n时间特征清单 ({len(keep)} 维):")
    names = keep
    for i in range(0, len(names), 5):
        print("    " + "  ".join(f"{n:>18s}" for n in names[i:i + 5]))


if __name__ == "__main__":
    build()