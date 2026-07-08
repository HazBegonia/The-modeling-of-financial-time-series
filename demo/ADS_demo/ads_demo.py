# -*- coding: utf-8 -*-
"""
ADS (Activity–Direction–Size) 分解模型 —— 一条龙完整流程
逐笔价格变化 ΔP -> 拆成 (动不动 A / 往哪动 D / 动多大 S) -> 条件分解似然 -> 极大似然 -> 预测下一笔 ΔP 的完整概率分布

  Tsay《金融时间序列分析》第 5 章 高频(逐笔)数据, 价格变化的分解模型 (式 5.21–5.28):
    ΔP_i = P_i - P_{i-1} = A_i · D_i · S_i                                      (5.21–5.23)
        A_i ∈ {0,1}      动不动 (Activity)
        D_i ∈ {+1,-1}    往哪动 (Direction, 仅当 A_i=1)
        S_i ∈ {1,2,3,…}  动多大/跳几格 (Size, 仅当 A_i=1)
    条件分解: P(A_i,D_i,S_i | F_{i-1}) = P(A_i|F) · P(D_i|A_i,F) · P(S_i|D_i,A_i,F)  (5.24)
    各配一个简单模型:
        A: logit                                                                (5.25)
        D: logit, 依赖"上一笔方向" -> 抓买卖价差跳动的反转效应                       (5.26)
        S-1: 几何分布(按方向 j 分两套, λ 也可带解释变量)                            (5.27–5.28)
    似然: L = ∏ [(1-p_i)^{1-A_i} · (p_i·P(D_i)·P(S_i))^{A_i}]
    极大似然: 因条件分解 -> 三个子模型可分开估 (① 用全样本; ②③ 只用"动了"的子样本).

  本质: 把"难建模的离散价格变化"拆成"动不动 / 往哪动 / 动多大"三个独立小问题,
        条件概率连乘成似然, 极大似然估计, 最后组合出下一笔 ΔP 的完整概率分布(而非点预测).

  工具说明: A/D 的 logit 用 statsmodels.Logit (给 z 值便于解读); S 的几何回归
    statsmodels 无现成 GLM 族 -> 手写极大似然(scipy BFGS, 由 hess_inv 取标准误); 预测分布手写.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")               # Windows 控制台强制 UTF-8
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                                  # 不弹窗, 直接存图
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

SEP = "=" * 70


def logistic(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def geom_fit(s, x):
    """几何回归极大似然: S-1 ~ Geom(λ), logit(λ)=θ0+θ1·x.
       P(S=k)=λ(1-λ)^{k-1} (k>=1), 故 logL = Σ[ln λ + (S-1)ln(1-λ)].
       返回 (θ, 标准误, 最大对数似然 llf)."""
    s = np.asarray(s, float); x = np.asarray(x, float)

    def nll(theta):
        lam = np.clip(logistic(theta[0] + theta[1] * x), 1e-9, 1 - 1e-9)
        return -np.sum(np.log(lam) + (s - 1.0) * np.log(1.0 - lam))

    res = minimize(nll, x0=np.array([0.5, 0.0]), method="BFGS")
    se = np.sqrt(np.clip(np.diag(res.hess_inv), 0, None))
    return res.x, se, -res.fun


# ====================================================================
# 步骤0  数据生成: 用已知的 ADS 真值逐笔造价格变化 ΔP (假装不知道参数)
#   解释变量取自历史: prevA=上一笔动没动, prevD=上一笔方向∈{-1,0,+1}, prevS=上一笔大小
#   ① 活动 A: logit(p)=bA0+bA1·prevA+bA2·prevS    (bA1>0 -> 成交活跃度有聚集性)
#   ② 方向 D: logit(P(D=+1))=gD0+gD1·prevD         (gD1<0 -> 反转: 上涨后倾向回跌, 即买卖价差跳动)
#   ③ 大小 S: S-1~Geom(λ_j), logit(λ_j)=ej0+ej1·prevS, 分上(U)/下(D)两套 -> size 分布按方向不同
# ====================================================================
rng = np.random.default_rng(42)
bA = np.array([0.30, 0.60, 0.05])                      # 活动 logit 系数 (常数, prevA, prevS)
gD = np.array([0.00, -0.80])                           # 方向 logit 系数 (常数, prevD)  gD1<0=反转
eU = np.array([0.90, -0.12])                           # 上行 size 的 λ_U logit 系数 (常数, prevS)
eD = np.array([0.60, -0.10])                           # 下行 size 的 λ_D logit 系数 (常数, prevS)

n, burn = 4000, 600                                    # 取 4000 笔, 前 600 笔烧入丢弃
N = n + burn
A = np.zeros(N, dtype=int); D = np.zeros(N, dtype=int); S = np.zeros(N, dtype=int)
for i in range(1, N):
    pa, pd_, ps = A[i - 1], D[i - 1], S[i - 1]         # 上一笔的 (A,D,S) 作解释变量
    p = logistic(bA[0] + bA[1] * pa + bA[2] * ps)
    if rng.random() < p:                               # ① 动不动
        A[i] = 1
        delta = logistic(gD[0] + gD[1] * pd_)
        D[i] = 1 if rng.random() < delta else -1       # ② 往哪动
        e = eU if D[i] == 1 else eD
        lam = logistic(e[0] + e[1] * ps)
        S[i] = rng.geometric(lam)                       # ③ 动多大 (numpy geometric: P(k)=λ(1-λ)^{k-1}, k>=1)
    # A_i=0 时 D_i=S_i=0 (不动)

A, D, S = A[burn:], D[burn:], S[burn:]
dP = A * D * S                                          # 逐笔价格变化 ΔP = A·D·S
price = 100.0 + np.cumsum(dP)                           # 还原一条 tick 价格路径 (起点 100)
n_obs = len(dP)

# 建模样本: 用到 prevA/prevD/prevS, 故从 i=1 起 (i=0 的"上一笔"落在烧入边界, 丢弃)
idx = np.arange(1, n_obs)
prevA, prevD, prevS = A[idx - 1], D[idx - 1], S[idx - 1]
Ai, Di, Si, dPi = A[idx], D[idx], S[idx], dP[idx]
h = 800                                                 # 留最后 800 笔做样本外预测
split = np.where(idx < n_obs - h, "train", "test")
tr, te = split == "train", split == "test"

pd.DataFrame({"i": idx, "price": np.round(price[idx], 0).astype(int), "dP": dPi,
              "A": Ai, "D": Di, "S": Si,
              "prevA": prevA, "prevD": prevD, "prevS": prevS, "split": split}
             ).to_csv("ads_data.csv", index=False, encoding="utf-8")

act_rate = Ai.mean()
up_rate = (Di[Ai == 1] == 1).mean()
print(SEP); print("步骤0  数据生成 (ADS 真值)")
print(f"  ① 活动 A: logit(p)   = {bA[0]} + {bA[1]}*prevA + {bA[2]}*prevS")
print(f"  ② 方向 D: logit(δ)   = {gD[0]} + ({gD[1]})*prevD    (gD1<0 = 反转/买卖价差跳动)")
print(f"  ③ 大小 S: logit(λ_U) = {eU[0]} + ({eU[1]})*prevS  |  logit(λ_D) = {eD[0]} + ({eD[1]})*prevS")
print(f"  样本 n={n_obs} 笔 (建模用 {len(idx)}; train={tr.sum()}, test={te.sum()})")
print(f"  序列预览 ΔP(前16): {dPi[:16]}")
print(f"  活动率 P(A=1)={act_rate:.1%}, 动时上涨占比 P(D=+1|A=1)={up_rate:.1%}, "
      f"ΔP 范围[{dPi.min()},{dPi.max()}]")
print("  已导出 -> ads_data.csv")

# 图1: 价格路径 + 逐笔 ΔP + ΔP 的离散分布 (展示"难建模的离散数据")
fig, ax = plt.subplots(3, 1, figsize=(11, 9))
W = 250                                                 # 展示窗口
ax[0].plot(price[idx][:W], lw=0.9, color="steelblue")
ax[0].set_title("Reconstructed tick price path (P_i = P_{i-1} + dP_i)")
ax[0].set_ylabel("price")
seg = dPi[:W]; xs = np.arange(W)
ax[1].vlines(xs[seg > 0], 0, seg[seg > 0], color="tomato", lw=1.2, label="up (D=+1)")
ax[1].vlines(xs[seg < 0], 0, seg[seg < 0], color="steelblue", lw=1.2, label="down (D=-1)")
ax[1].scatter(xs[seg == 0], np.zeros((seg == 0).sum()), s=6, color="gray", label="no move (A=0)")
ax[1].axhline(0, color="k", lw=.6); ax[1].set_title("Tick-by-tick price change dP = A*D*S")
ax[1].set_ylabel("dP (ticks)"); ax[1].legend(loc="upper right", fontsize=8)
vals, cnts = np.unique(dPi, return_counts=True)
ax[2].bar(vals, cnts / cnts.sum(), width=0.8, color="purple", alpha=.75, edgecolor="k")
ax[2].set_title("Empirical distribution of dP (discrete, spike at 0)")
ax[2].set_xlabel("dP (ticks)"); ax[2].set_ylabel("probability")
plt.tight_layout(); plt.savefig("ads_fig1_decompose.png", dpi=110); plt.close()

# ====================================================================
# 步骤1  拆分: 一笔 ΔP <-> 一组 (A,D,S)  (式 5.21–5.23)
# ====================================================================
print(SEP); print("步骤1  把每笔 ΔP 拆成 (A 动不动, D 往哪动, S 动多大)  (5.21–5.23)")
print(f"   {'i':>4} {'dP':>5} | {'A':>2} {'D':>3} {'S':>3}")
for k in range(8):
    j = idx[k]
    dline = f"{Di[k]:+d}" if Ai[k] else "—"
    sline = f"{Si[k]}" if Ai[k] else "—"
    print(f"   {j:>4} {dPi[k]:>+5d} | {Ai[k]:>2} {dline:>3} {sline:>3}")
print("   -> 把'怪数据'(带 0、有正负、跳格不一)翻译成三个规整小变量, 各自好建模")

# ====================================================================
# 步骤2  条件分解: 一个难的联合分布 -> 三个独立简单模型 (式 5.24)
#   P(A,D,S|F) = P(A|F) · P(D|A,F) · P(S|D,A,F)
#   关键红利: 三块可用不同子样本分开估 -> ① 全样本; ② 只用动了的; ③ 动了的再按方向分
# ====================================================================
act = Ai == 1
print(SEP); print("步骤2  条件分解 P(A,D,S|F)=P(A|F)·P(D|A,F)·P(S|D,A,F)  (5.24)")
print(f"  ① A 模型样本 = 全部交易            : {len(Ai)} 笔 (train {tr.sum()})")
print(f"  ② D 模型样本 = 只用'动了'(A=1)     : {act.sum()} 笔 (train {(act & tr).sum()})")
print(f"  ③ S 模型样本 = 动了再按方向分 U/D  : 上行 {(act & (Di==1)).sum()} / 下行 {(act & (Di==-1)).sum()}")
print("  -> 联合分布拆成三个互不干扰的子问题, 可分别极大似然 (步骤3-4)")

# ====================================================================
# 步骤3  给三块各配一个模型并拟合 (式 5.25–5.28)
#   ① A: logit (全 train)          ② D: logit (train 中 A=1)      ③ S: 几何回归 (按方向, 手写 MLE)
# ====================================================================
print(SEP); print("步骤3  三块建模 (① logit  ② logit  ③ 几何分布)  (5.25–5.28)")

# ① 活动 logit:  endog=A, exog=[1, prevA, prevS]
XA_tr = sm.add_constant(np.column_stack([prevA[tr], prevS[tr]]))
A_res = sm.Logit(Ai[tr], XA_tr).fit(disp=0)
print(f"  ① A-logit (5.25):  logit(p) = {A_res.params[0]:+.3f} {A_res.params[1]:+.3f}*prevA "
      f"{A_res.params[2]:+.3f}*prevS   (McFadden R²={A_res.prsquared:.3f})")

# ② 方向 logit (仅 A=1):  endog=(D==+1), exog=[1, prevD]
m2 = act & tr
XD_tr = sm.add_constant(prevD[m2])
D_res = sm.Logit((Di[m2] == 1).astype(int), XD_tr).fit(disp=0)
print(f"  ② D-logit (5.26):  logit(δ) = {D_res.params[0]:+.3f} {D_res.params[1]:+.3f}*prevD "
      f"   (McFadden R²={D_res.prsquared:.3f})")

# ③ size 几何回归 (仅 A=1, 按方向分 U/D):  S-1~Geom(λ_j), logit(λ_j)=ej0+ej1*prevS
mU = act & tr & (Di == 1); mD = act & tr & (Di == -1)
eU_hat, eU_se, llU = geom_fit(Si[mU], prevS[mU])
eD_hat, eD_se, llD_s = geom_fit(Si[mD], prevS[mD])
print(f"  ③ S-Geom (5.27-28): logit(λ_U) = {eU_hat[0]:+.3f} {eU_hat[1]:+.3f}*prevS  "
      f"(均值 size_U≈{1/logistic(eU_hat[0]):.2f})")
print(f"                      logit(λ_D) = {eD_hat[0]:+.3f} {eD_hat[1]:+.3f}*prevS  "
      f"(均值 size_D≈{1/logistic(eD_hat[0]):.2f})")

bA_hat, gD_hat = A_res.params, D_res.params

# ====================================================================
# 步骤4  似然 + 极大似然估计 (参数估计 vs 真值)
#   L = ∏[(1-p)^{1-A}·(p·P(D)·P(S))^A]; 取对数后三块可加 -> 分开极大化 (步骤3 已完成)
# ====================================================================
llf = A_res.llf + D_res.llf + llU + llD_s              # 联合对数似然 = 三块之和
print(SEP); print("步骤4  极大似然结果 (估计 vs 真值)")
rows = [
    ("A:常数",   bA_hat[0], bA[0]), ("A:prevA", bA_hat[1], bA[1]), ("A:prevS", bA_hat[2], bA[2]),
    ("D:常数",   gD_hat[0], gD[0]), ("D:prevD", gD_hat[1], gD[1]),
    ("S_U:常数", eU_hat[0], eU[0]), ("S_U:prevS", eU_hat[1], eU[1]),
    ("S_D:常数", eD_hat[0], eD[0]), ("S_D:prevS", eD_hat[1], eD[1]),
]
print(f"   {'参数':<10}{'估计':>9}{'真值':>9}")
for nm, est, tru in rows:
    print(f"   {nm:<10}{est:>+9.3f}{tru:>+9.3f}")
print(f"  联合对数似然 logL = llA({A_res.llf:.1f}) + llD({D_res.llf:.1f}) "
      f"+ llS_U({llU:.1f}) + llS_D({llD_s:.1f}) = {llf:.1f}")
print("  注: 因条件分解(5.24), 三块各自极大化即得联合 MLE, 互不影响.")

# ====================================================================
# 步骤5  诊断与解读 (系数显著性 + 反转效应 + size 方向不对称 + 残差白噪声)
# ====================================================================
print(SEP); print("步骤5  诊断与解读")
# 5a 系数显著性 (z 检验)
print("  [5a] 关键系数显著性:")
print(f"       A:prevA  z={A_res.tvalues[1]:+.2f} (p={A_res.pvalues[1]:.2g}) -> "
      + ("活动有聚集性 [OK]" if A_res.pvalues[1] < 0.05 and A_res.params[1] > 0 else "不显著"))
print(f"       D:prevD  z={D_res.tvalues[1]:+.2f} (p={D_res.pvalues[1]:.2g}) -> "
      + ("系数为负=反转效应显著 [OK]" if D_res.pvalues[1] < 0.05 and D_res.params[1] < 0 else "不显著/非反转"))
zU = eU_hat[1] / eU_se[1]; zD = eD_hat[1] / eD_se[1]
print(f"       S_U:prevS z={zU:+.2f},  S_D:prevS z={zD:+.2f}  (size 随上一笔大小的持续性)")
# 5b 反转效应解读: 比较上一笔涨/跌后的"再涨概率"
pUp_after_up = logistic(gD_hat[0] + gD_hat[1] * (+1))
pUp_after_dn = logistic(gD_hat[0] + gD_hat[1] * (-1))
print(f"  [5b] 反转效应(§5.2/5.3 买卖价差跳动): P(涨|上一笔涨)={pUp_after_up:.1%} "
      f"< P(涨|上一笔跌)={pUp_after_dn:.1%} -> 涨完倾向跌, 跌完倾向涨 (负相关)")
# 5c size 方向不对称
print(f"  [5c] size 方向不对称: 估计均值 上行≈{1/logistic(eU_hat[0]):.2f} 格, "
      f"下行≈{1/logistic(eD_hat[0]):.2f} 格 (允许上/下分布不同 -> 用了两套 λ)")
# 5d 残差: A 模型 Pearson 残差是否还有自相关 (好模型应无剩余结构)
p_fit = A_res.predict(XA_tr)
pear = (Ai[tr] - p_fit) / np.sqrt(p_fit * (1 - p_fit))
lb_p = acorr_ljungbox(pear, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
print(f"  [5d] A 模型 Pearson 残差 Ljung-Box(10) p={lb_p:.3g} -> "
      + ("无剩余自相关, 活动模型充分 [OK]" if lb_p > 0.05 else "仍有自相关, 可加滞后项 [NO]"))

# 图2: 三块各自的"数据 vs 模型" + 图3: 诊断
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
# ① 活动聚集性: 经验 P(A=1) by prevA
ea = [Ai[tr][prevA[tr] == v].mean() for v in (0, 1)]
ma = [logistic(bA_hat[0] + bA_hat[1] * v + bA_hat[2] * prevS[tr].mean()) for v in (0, 1)]
xb = np.arange(2)
ax[0].bar(xb - 0.18, ea, 0.36, label="empirical", color="steelblue")
ax[0].bar(xb + 0.18, ma, 0.36, label="model", color="tomato", alpha=.8)
ax[0].set_xticks(xb); ax[0].set_xticklabels(["prevA=0", "prevA=1"])
ax[0].set_title("(1) Activity P(A=1): clustering"); ax[0].set_ylabel("P(A=1)"); ax[0].legend()
# ② 反转: 经验 P(up) by prevD
ev = [-1, 0, 1]
ed = [(Di[m2][prevD[m2] == v] == 1).mean() if (prevD[m2] == v).any() else np.nan for v in ev]
md = [logistic(gD_hat[0] + gD_hat[1] * v) for v in ev]
xb = np.arange(3)
ax[1].bar(xb - 0.18, ed, 0.36, label="empirical", color="steelblue")
ax[1].bar(xb + 0.18, md, 0.36, label="model", color="tomato", alpha=.8)
ax[1].axhline(0.5, color="gray", ls=":")
ax[1].set_xticks(xb); ax[1].set_xticklabels(["prevD=-1", "prevD=0", "prevD=+1"])
ax[1].set_title("(2) Direction P(up): reversal (down-sloping)"); ax[1].set_ylabel("P(D=+1)"); ax[1].legend()
# ③ size 几何拟合 (上/下)
kk = np.arange(1, 9)
for m_, e_, c_, lab in [(mU, eU_hat, "tomato", "up"), (mD, eD_hat, "steelblue", "down")]:
    emp = np.array([(Si[m_] == k).mean() for k in kk])
    lam = logistic(e_[0] + e_[1] * prevS[m_].mean())
    fit = lam * (1 - lam) ** (kk - 1)
    ax[2].plot(kk, emp, "o", color=c_, label=f"{lab} empirical")
    ax[2].plot(kk, fit, "-", color=c_, alpha=.7, label=f"{lab} geom fit")
ax[2].set_title("(3) Size S: geometric (decreasing)"); ax[2].set_xlabel("size (ticks)")
ax[2].set_ylabel("probability"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("ads_fig2_models.png", dpi=110); plt.close()

# 图3: 校准 (A、D) + size 残差(经验-拟合)
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
for axi, res, X, y, title in [(ax[0], A_res, XA_tr, Ai[tr], "A calibration"),
                              (ax[1], D_res, XD_tr, (Di[m2] == 1).astype(int), "D calibration")]:
    ph = res.predict(X); bins = np.linspace(0, 1, 11)
    bi = np.clip(np.digitize(ph, bins) - 1, 0, 9)
    bx, by = [], []
    for b in range(10):
        m = bi == b
        if m.sum() > 5:
            bx.append(ph[m].mean()); by.append(y[m].mean())
    axi.plot([0, 1], [0, 1], "k--", lw=.8)
    axi.plot(bx, by, "o-", color="purple")
    axi.set_title(title); axi.set_xlabel("predicted prob"); axi.set_ylabel("observed freq")
kk = np.arange(1, 9)
for m_, e_, c_, lab in [(mU, eU_hat, "tomato", "up"), (mD, eD_hat, "steelblue", "down")]:
    emp = np.array([(Si[m_] == k).mean() for k in kk])
    lam = logistic(e_[0] + e_[1] * prevS[m_].mean())
    fit = lam * (1 - lam) ** (kk - 1)
    ax[2].bar(kk + (0.18 if lab == "up" else -0.18), emp - fit, 0.36, color=c_, label=f"{lab} (emp-fit)")
ax[2].axhline(0, color="k", lw=.6); ax[2].set_title("Size fit residual (emp - geom)")
ax[2].set_xlabel("size (ticks)"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("ads_fig3_diagnostics.png", dpi=110); plt.close()

# ====================================================================
# 步骤6  预测: 给定 F_{i-1}, 依次组合 -> 下一笔 ΔP 的完整概率分布 (而非点预测)
#   P(ΔP=0)=1-p;  P(ΔP=+k)=p·δ·λ_U(1-λ_U)^{k-1};  P(ΔP=-k)=p·(1-δ)·λ_D(1-λ_D)^{k-1}
#   样本外评估三档: [活动]A 命中  [方向]D 命中  [整体]点预测 RMSE + 预测对数似然
# ====================================================================
KMAX = 8


def predictive_dist(pa, pd_, ps):
    """给定上一笔 (A,D,S), 返回下一笔 ΔP 取值与概率 (式 5.24 三块组合)."""
    p = logistic(bA_hat[0] + bA_hat[1] * pa + bA_hat[2] * ps)
    delta = logistic(gD_hat[0] + gD_hat[1] * pd_)
    lamU = logistic(eU_hat[0] + eU_hat[1] * ps); lamD = logistic(eD_hat[0] + eD_hat[1] * ps)
    ks = np.arange(1, KMAX + 1)
    pUp = p * delta * lamU * (1 - lamU) ** (ks - 1)
    pDn = p * (1 - delta) * lamD * (1 - lamD) ** (ks - 1)
    vals = np.concatenate([-ks[::-1], [0], ks])
    probs = np.concatenate([pDn[::-1], [1 - p], pUp])
    return vals, probs


# 预测原点 = 最后一笔 train 的 (A,D,S)
o = np.where(tr)[0][-1]
vals, probs = predictive_dist(Ai[o], Di[o], Si[o])
p0 = logistic(bA_hat[0] + bA_hat[1] * Ai[o] + bA_hat[2] * Si[o])
d0 = logistic(gD_hat[0] + gD_hat[1] * Di[o])
exp_dP = float(np.sum(vals * probs))
print(SEP); print("步骤6  预测: 下一笔 ΔP 的完整概率分布")
print(f"  预测原点 F: 上一笔 (A,D,S)=({Ai[o]},{Di[o]},{Si[o]})")
print(f"  ① 会不会动 p̂={p0:.1%}  ② 往哪动 P(涨|动)={d0:.1%}  ③ 动多大 -> 几何分布")
print(f"  下一笔 ΔP 概率分布 (式 5.24 组合):")
for v, pr in zip(vals, probs):
    if pr > 0.005:
        bar = "█" * int(pr * 120)
        print(f"     ΔP={v:>+3d}: {pr:6.1%} {bar}")
print(f"  -> 点预测(期望) E[ΔP]={exp_dP:+.3f}; 但 ADS 给的是整条分布, 信息远多于一个点")

# 样本外 one-step 评估 (用实际历史协变量做滚动一步预测)
pte = logistic(bA_hat[0] + bA_hat[1] * prevA[te] + bA_hat[2] * prevS[te])
dte = logistic(gD_hat[0] + gD_hat[1] * prevD[te])
lamU_te = logistic(eU_hat[0] + eU_hat[1] * prevS[te])
lamD_te = logistic(eD_hat[0] + eD_hat[1] * prevS[te])
# [活动]
predA = (pte >= 0.5).astype(int); accA = (predA == Ai[te]).mean()
baseA = max(Ai[te].mean(), 1 - Ai[te].mean())
# [方向] 仅实际动了的
mte = Ai[te] == 1
predU = dte[mte] >= 0.5; actU = Di[te][mte] == 1; accD = (predU == actU).mean()
baseD = max(actU.mean(), 1 - actU.mean())
# [整体] 点预测 E[ΔP] 与 RMSE; 朴素=预测 0
EdP = pte * (dte / lamU_te - (1 - dte) / lamD_te)
rmse = np.sqrt(np.mean((dPi[te] - EdP) ** 2))
naive = np.sqrt(np.mean(dPi[te] ** 2))
# 预测对数似然/笔 (完整分布的合理性打分): 模型 vs 无条件常数基准
def avg_loglik(p_, d_, lU, lD):
    a = Ai[te]; dd = Di[te]; ss = Si[te]
    ll = np.where(a == 0, np.log(1 - p_),
                  np.log(p_) + np.where(dd == 1, np.log(d_), np.log(1 - d_))
                  + np.where(dd == 1, np.log(lU) + (ss - 1) * np.log(1 - lU),
                             np.log(lD) + (ss - 1) * np.log(1 - lD)))
    return ll.mean()
ll_model = avg_loglik(pte, dte, lamU_te, lamD_te)
# 无条件基准: 用 train 的常数概率
p_c = Ai[tr].mean(); d_c = (Di[act & tr] == 1).mean()
lU_c = 1 / Si[mU].mean(); lD_c = 1 / Si[mD].mean()
ll_base = avg_loglik(np.full(te.sum(), p_c), np.full(te.sum(), d_c),
                     np.full(te.sum(), lU_c), np.full(te.sum(), lD_c))
print("  样本外评估 (test, one-step):")
print(f"     [活动] A 命中率={accA:.1%}  (基准=恒猜多数类 {baseA:.1%})")
print(f"     [方向] D 命中率={accD:.1%}  (基准 {baseD:.1%}; 反转使方向可预测)")
print(f"     [整体] 点预测 RMSE={rmse:.3f} vs 朴素(预测0) RMSE={naive:.3f} -> "
      + ("更优 [OK]" if rmse < naive else "未胜过"))
print(f"     [分布] 预测对数似然/笔 模型={ll_model:.3f} vs 无条件基准={ll_base:.3f} -> "
      + ("条件分布更优 [OK]" if ll_model > ll_base else "未胜过"))

pd.DataFrame({"dP_value": vals, "prob": np.round(probs, 5)}
             ).to_csv("ads_forecast.csv", index=False, encoding="utf-8")
print("  已导出 下一笔 ΔP 预测分布 -> ads_forecast.csv")

# 图4: 左=下一笔 ΔP 预测分布; 右=test 经验分布 vs 模型平均预测分布(验证整条分布)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
cols = ["tomato" if v > 0 else ("steelblue" if v < 0 else "gray") for v in vals]
ax[0].bar(vals, probs, color=cols, edgecolor="k", alpha=.8)
ax[0].set_title(f"Predicted dist. of next dP  (origin: prev A,D,S=({Ai[o]},{Di[o]},{Si[o]}))")
ax[0].set_xlabel("dP (ticks)"); ax[0].set_ylabel("probability")
ax[0].axvline(exp_dP, color="green", ls="--", lw=1.2, label=f"E[dP]={exp_dP:+.2f}")
ax[0].legend()
# 右: 模型隐含的 test 边际分布(逐笔预测分布求平均) vs 经验
ks = np.arange(1, KMAX + 1)
mUp = (pte[:, None] * dte[:, None] * lamU_te[:, None] * (1 - lamU_te[:, None]) ** (ks - 1)).mean(0)
mDn = (pte[:, None] * (1 - dte[:, None]) * lamD_te[:, None] * (1 - lamD_te[:, None]) ** (ks - 1)).mean(0)
m_vals = np.concatenate([-ks[::-1], [0], ks])
m_probs = np.concatenate([mDn[::-1], [(1 - pte).mean()], mUp])
emp = pd.Series(dPi[te]).value_counts(normalize=True)
emp_p = np.array([emp.get(v, 0.0) for v in m_vals])
xb = np.arange(len(m_vals))
ax[1].bar(xb - 0.2, emp_p, 0.4, label="empirical (test)", color="steelblue")
ax[1].bar(xb + 0.2, m_probs, 0.4, label="model avg predictive", color="tomato", alpha=.8)
ax[1].set_xticks(xb); ax[1].set_xticklabels(m_vals, fontsize=8)
ax[1].set_title("Test dP: empirical vs model-implied (full distribution check)")
ax[1].set_xlabel("dP (ticks)"); ax[1].set_ylabel("probability"); ax[1].legend()
plt.tight_layout(); plt.savefig("ads_fig4_forecast.png", dpi=110); plt.close()

# ====================================================================
print(SEP)
print(f"完成. 全流程: 拆分(A,D,S) -> 条件分解(5.24) -> 三块建模(logit/logit/几何) -> "
      f"极大似然 -> 反转效应[{'OK' if D_res.pvalues[1]<0.05 and D_res.params[1]<0 else 'NO'}] -> "
      f"预测下一笔 ΔP 完整分布")
print(f"揭晓真值: A:{bA.tolist()}, D:{gD.tolist()}, S_U:{eU.tolist()}, S_D:{eD.tolist()}")
print("图: ads_fig1_decompose / ads_fig2_models / ads_fig3_diagnostics / ads_fig4_forecast (.png)")
