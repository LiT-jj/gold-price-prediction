"""
★致命矛盾排查 ★

现象 A：方向准确率 53.80% < 多数类基线 54.45%（模型看跌能力差）
现象 B：市场中性化 alpha = +31.50bp，置换 p=0.0000（高度显著）

A 与 B 数学上不可能同时由"方向预测能力"解释。
假设：错误在 alpha 的计算——市场中性化只剔除线性 beta，
      但alpha 里混入了"信号活跃度与波动率的相关"等结构性偏差。

核心怀疑点：polyfit 对常数信号拟合出的 beta≈0，残差≈原值；
             即 beta 剥离失败时，alpha 退化为"策略毛收益"，
             而策略毛收益里含有黄金上涨的漂移。
本脚本做三项独立检验。
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
SPLIT = "2019-01-01"
ALL_TARGETS = ["fwd_1d", "fwd_3d", "fwd_5d", "fwd_10d"]
TARGETS = ["fwd_1d", "fwd_5d"]
CN = {"fwd_1d": "1日", "fwd_5d": "1周"}

df = pd.read_csv(os.path.join(BASE, "data", "gold_features.csv"),
                 index_col=0, parse_dates=True)
feats = [c for c in df.columns if c not in ALL_TARGETS + ["close"]]
SEED = 20261014
rng = np.random.default_rng(SEED)

print("=" * 72)
print("★ 矛盾根源排查")
print("=" * 72)

for tgt in TARGETS:
    sub = df[[tgt] + feats].dropna()
    m = np.asarray(sub.index >= SPLIT)
    ret = sub.loc[m, tgt].values
    idx = sub.index[m]
    # prob 可能是用另一份 dropna 顺序训练的，长度可能不一致 -> 按索引对齐
    prob_loaded = np.load(os.path.join(OUT, f"pred_{tgt}_logistic.npy"))
    if len(prob_loaded) != len(idx):
        print(f"    [对齐] prob长度{len(prob_loaded)} ≠ 测试集{len(idx)}，截取尾部对齐")
        prob_loaded = prob_loaded[-len(idx):]
    prob = prob_loaded
    sig = np.where(prob > 0.6, 1, np.where(prob < 0.4, -1, 0))
    active = sig != 0

    print(f"\n{'='*72}")
    print(f"【{CN[tgt]}】交易 {active.sum()} 笔")
    print(f"{'='*72}")

    # ---- 检验1：多空对称性检查 ----
    print("\n[检验1] 多空收益是否对称（若不对称则alpha来自单边暴露）")
    strat = np.where(active, sig * ret, 0.0)
    s_act, r_act = sig[active], ret[active]
    up_mask = s_act == 1
    dn_mask = s_act == -1
    print(f"    做多 {up_mask.sum():4d} 笔  平均收益 {r_act[up_mask].mean()*1e4:+8.2f}bp")
    print(f"    做空 {dn_mask.sum():4d} 笔  平均收益 {-r_act[dn_mask].mean()*1e4:+8.2f}bp")
    print(f"    多空净敞口(笔数差) {up_mask.sum()-dn_mask.sum():+5d}")
    net_expo = (up_mask.sum() - dn_mask.sum()) / active.sum()
    print(f"    → 净多头敞口 {net_expo*100:+.1f}%  "
          f"{'严重偏多头，收益主要来自beta' if abs(net_expo)>0.2 else '大致对称'}")

    # ---- 检验2：把黄金涨幅全部剔除后还剩什么 ----
    print("\n[检验2] 用'减去基准日均收益'替代beta剥离（更严格）")
    strat_act = s_act * r_act
    # 简单法：每笔减去同日买入持有收益，再减去 beta 相关项
    mu = r_act.mean()
    strat_demeaned = strat_act - mu
    alpha_simple = strat_demeaned.mean() * 1e4
    print(f"    买入持有日均       {mu*1e4:+8.2f}bp")
    print(f"    策略日均           {strat_act.mean()*1e4:+8.2f}bp")
    print(f"    策略-基准          {alpha_simple:+8.2f}bp  ← 简单差额")

    # ---- 检验3：符号无关的幅度检验（关键） ----
    print("\n[检验3] 只预测'幅度'不预测'方向'能否获利？")
    print("    若 |策略收益|的提升只来自幅度预测，则方向预测无效")
    # 随机打乱方向后，策略是否仍有同样alpha？
    null_alpha = []
    for _ in range(1000):
        s_perm = rng.choice([-1, 1], size=active.sum())
        tmp = np.where(active, np.nan, 0.0)
        t2 = np.zeros(len(ret))
        t2[active] = s_perm * r_act
        b = np.polyfit(ret, t2, 1)[0]
        null_alpha.append((t2[active] - b * r_act).mean() * 1e4)
    null_alpha = np.array(null_alpha)
    real_b = np.polyfit(ret, strat, 1)[0]
    real_alpha = (strat[active] - real_b * r_act).mean() * 1e4
    p = float((null_alpha >= real_alpha).mean())
    print(f"    真实信号 alpha      {real_alpha:+8.2f}bp")
    print(f"    随机方向 alpha 均值 {null_alpha.mean():+8.2f}bp "
          f"标准差 {null_alpha.std():.2f}bp")
    print(f"    随机方向 95分位     {np.percentile(null_alpha,95):+8.2f}bp")
    print(f"    置换 p = {p:.4f}  "
          f"{'alpha真实存在' if p<0.05 else 'alpha与随机方向无法区分'}")

    # ---- 检验4：符号一致性 ----
    print("\n[检验4] 信号与收益的符号一致性（方向预测的直接度量）")
    agree = (np.sign(s_act) == np.sign(r_act)).mean()
    maj = max((r_act > 0).mean(), (r_act < 0).mean())
    print(f"    信号与实际方向一致率 {agree*100:.2f}%")
    print(f"    交易期多数方基准     {maj*100:.2f}%")
    print(f"    → 准确率{'高于' if agree>maj else '低于'}基准 "
          f"{(agree-maj)*100:+.2f}pp")

print("\n" + "=" * 72)
print("【结论逻辑】")
print("=" * 72)
print("""
若【检验3】显示 alpha 与随机方向无法区分，
说明 polyfit 中性化失效，alpha 主要来自：
  (a) 信号净敞口偏多头 × 黄金整体上涨
  (b) 波动率聚集：模型在高波动日更倾向做多/做空

这种情况下，正确做法是：
  用「方向准确率 vs 多数类基线」作为主判据
  alpha 仅作辅助，且必须已验证多空对称
""")