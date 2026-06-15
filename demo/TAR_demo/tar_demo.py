# -*- coding: utf-8 -*-
"""
TAR (门限自回归) 建模 —— 一条龙完整流程 (SETAR, 按"正负号"分两个离散状态)

  TAR 与 AR 的关系 (本 demo 的主角):
    AR(p):       y_t = c + phi_1 y_{t-1} + ... + phi_p y_{t-p} + e_t      (全程一套系数)
    SETAR(2;1,1):                                                         (按状态切两套系数)
        y_{t-1} <= r :  y_t = c^(L) + phi^(L) y_{t-1} + e_t   (状态 L)
        y_{t-1} >  r :  y_t = c^(H) + phi^(H) y_{t-1} + e_t   (状态 H)
    门限变量取自序列自身的滞后 y_{t-d} (自激励 self-exciting); d=延迟, r=门限.
    本 demo 把门限定成"正负号": d=1, r=0 —— 上一步是负就走 L, 是正就走 H.

  为什么需要 TAR: 条件均值在 r 处有"跳变/折点", 线性 AR 一条直线抓不住 ->
    两段各有自己的中心(一正一负)和弱自相关 -> 序列在正/负两个离散电平间切换.

  六步法 (下游一律用"定出来的门限/延迟", 不写死):
    第0步 准备数据    : 用 SETAR 真值造序列 y_t (全局平稳), 留尾部做样本外
    第1步 平稳+线性破绽: ADF 验平稳; 先拟合线性 AR(1) 当基准, 看它抓不住门限
    第2步 门限效应检验 : 门限已知(r=0) -> Chow/LR-F 检验 线性AR vs 两段TAR, 要不要 TAR
    第3步 门限&延迟定位: 网格搜 (d,r) 最小化 SSR -> 验证数据自己也选出 d=1,r≈0
    第4步 分段估计     : 给定 d=1,r=0, 两段各跑一次 OLS -> 参数 vs 真值, 每段 sigma
    第5步 模型检验     : 合并残差查白噪声(Ljung-Box) + 分布; 不过关 -> 升阶/改门限
    第6步 使用模型     : 1步=状态确定直接算; 多步=蒙特卡洛模拟 -> 点预测+区间(可能双峰)

  工具说明: statsmodels 无现成 TAR. 但给定 (d,r) 后, TAR 估计 = 按状态切样本各做一次 OLS,
    完全透明; 多步预测因 E[g(y)]!=g(E[y]) 必须"模拟+平均"(蒙特卡洛), 故全部手写.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")               # Windows 控制台强制 UTF-8
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                                  # 不弹窗, 直接存图
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

SEP = "=" * 70

# ---------- 小工具: OLS / 取滞后对齐 / 给定门限分段拟合 ----------
def ols(yv, Xv):                                       # 最小二乘, 返回 (beta, 残差, SSR)
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid = yv - Xv @ beta
    return beta, resid, float(resid @ resid)

def make_arrays(a, d, start):                          # 对齐出 (y_t, y_{t-1}, y_{t-d}); start=公共起点保证跨 d 可比
    a = np.asarray(a, float); idx = np.arange(start, len(a))
    return a[idx], a[idx - 1], a[idx - d]              # Y(响应), Xlag(回归用 y_{t-1}), Z(门限变量 y_{t-d})

def tar_fit(Y, Xlag, Z, r, min_frac=0.15):             # 按 Z<=r 切两段各做 AR(1) OLS
    mL = Z <= r; mH = ~mL
    if min(mL.sum(), mH.sum()) < max(10, int(min_frac * len(Y))):
        return np.nan, None                            # 某状态样本太少 -> 该门限不可用
    ssr, parts = 0.0, {}
    for m, key in [(mL, "L"), (mH, "H")]:
        X = np.column_stack([np.ones(m.sum()), Xlag[m]])
        b, res, s = ols(Y[m], X)
        ssr += s; parts[key] = dict(b=b, ssr=s, n=int(m.sum()))
    return ssr, parts

# ====================================================================
# 步骤0  数据生成: SETAR(2;1,1), 门限=正负号(d=1,r=0), 两段各弱 AR、中心一正一负
#   负状态 L (y_{t-1}<=0): y_t = -3.0 + 0.5 y_{t-1} + e   -> 局部中心 -6.0 (AR更强,更黏更深)
#   正状态 H (y_{t-1}> 0): y_t = +4.0 + 0.3 y_{t-1} + e   -> 局部中心 +5.7 (AR更弱,浅而短)
#   平稳(遍历)条件 Petruccelli-Woolford: phi_L<1, phi_H<1, phi_L*phi_H<1 -> 0.5,0.3,0.15 满足
# ====================================================================
cL, phiL = -3.0, 0.5
cH, phiH = +4.0, 0.3
SIGMA = 4.5
R_TRUE, D_TRUE = 0.0, 1
n, burn = 900, 300
mL, mH = cL / (1 - phiL), cH / (1 - phiH)               # 两段局部中心(吸引点)

rng = np.random.default_rng(7)
eps = rng.normal(0, SIGMA, n + burn)
y = np.zeros(n + burn)
for t in range(1, n + burn):
    if y[t - 1] <= R_TRUE:
        y[t] = cL + phiL * y[t - 1] + eps[t]            # 上一步为负 -> 走 L
    else:
        y[t] = cH + phiH * y[t - 1] + eps[t]            # 上一步为正 -> 走 H
y = pd.Series(y[burn:]).reset_index(drop=True)

h = 50                                                  # 留最后 50 个点做样本外预测
train, test = y.iloc[:-h], y.iloc[-h:]
regime = np.where(np.r_[np.nan, y.values[:-1]] <= 0, "L", "H"); regime[0] = "-"  # 生成各点的状态(由 y_{t-1} 定)
pd.DataFrame({"t": np.arange(len(y)), "y": np.round(y.values, 3), "regime": regime,
              "split": ["train"] * len(train) + ["test"] * len(test)}
             ).to_csv("tar_data.csv", index=False, encoding="utf-8")

switches = int(np.sum(np.sign(y.values[1:]) != np.sign(y.values[:-1])))
print(SEP); print("步骤0  数据生成 (SETAR 真值)")
print(f"  真值 L(y<=0): y_t = {cL} + {phiL}*y(t-1) + e   中心 {mL:+.1f}")
print(f"  真值 H(y> 0): y_t = {cH} + {phiH}*y(t-1) + e   中心 {mH:+.1f}   sigma={SIGMA}")
print(f"  样本 n={len(y)} (train={len(train)}, test={len(test)})")
print(f"  序列预览(前12): {np.round(y.values[:12], 1)}")
print(f"  范围[{y.min():.1f}, {y.max():.1f}], 正值占比 {(y>0).mean():.0%}, 翻号 {switches}/{len(y)-1} 次")
print("  已导出 -> tar_data.csv")

# ====================================================================
# 步骤1  平稳性检验 + 线性 AR(1) 的"破绽" (为什么一条直线不够)
# ====================================================================
adf_stat, adf_p, *_ = adfuller(train, autolag="AIC")
Y0 = train.values[1:]; X0 = train.values[:-1]           # (y_t, y_{t-1}) 对, d=1 全样本
b_lin, res_lin, ssr0 = ols(Y0, np.column_stack([np.ones_like(X0), X0]))
print(SEP); print("步骤1  平稳性(ADF) + 线性 AR(1) 基准")
print(f"  ADF={adf_stat:.3f}, p={adf_p:.3g} -> "
      + ("平稳, 可建模 [OK]" if adf_p < 0.05 else "不平稳 [NO]"))
print(f"  线性 AR(1) 基准: y_t = {b_lin[0]:+.2f} {b_lin[1]:+.2f}*y(t-1) + e   (全程一套系数)")
print(f"  -> 它把正负两段强行用一条直线拟合, 抓不住 0 处的跳变 (下一步用检验证实)")

fig, ax = plt.subplots(3, 1, figsize=(10, 9))
tt = np.arange(len(train)); pos = train.values > 0
ax[0].plot(tt, train.values, lw=.5, color="gray", zorder=1)
ax[0].scatter(tt[pos], train.values[pos], s=9, color="tomato", label="y>0  (state H)", zorder=2)
ax[0].scatter(tt[~pos], train.values[~pos], s=9, color="steelblue", label="y<=0 (state L)", zorder=2)
ax[0].axhline(0, color="k", lw=.8, ls="--")
ax[0].set_title("TAR series colored by sign — two discrete regimes")
ax[0].legend(loc="upper right", fontsize=8)
plot_acf(train, lags=25, ax=ax[1], zero=False); ax[1].set_title("ACF (sign persists -> positive autocorr)")
plot_pacf(train, lags=25, ax=ax[2], method="ywm", zero=False); ax[2].set_title("PACF")
plt.tight_layout(); plt.savefig("tar_fig1_identify.png", dpi=110); plt.close()

# ====================================================================
# 步骤2  门限效应检验 (要不要 TAR): 门限已知 r=0 -> Chow/LR-F 检验
#   H0(线性): 全程一套 AR(1) (受限, k0=2);  H1(TAR): 两段各一套 (非受限, k1=4)
#   F = [(SSR0-SSR1)/(k1-k0)] / [SSR1/(n-k1)] ~ F(2, n-4)  (门限已知时为标准 F 分布)
# ====================================================================
ssr1, parts = tar_fit(Y0, X0, X0, 0.0)                  # 非受限: 按 y_{t-1} 正负切两段
k0, k1 = 2, 4; df1, df2 = k1 - k0, len(Y0) - k1
F = ((ssr0 - ssr1) / df1) / (ssr1 / df2)
pF = stats.f.sf(F, df1, df2)
print(SEP); print("步骤2  门限效应检验 (线性 AR vs 两段 TAR, 门限 r=0 已知)")
print(f"  受限(线性AR1)  SSR0={ssr0:.1f} (k0={k0})")
print(f"  非受限(两段TAR) SSR1={ssr1:.1f} (k1={k1})  -> 切两段后残差平方和明显下降")
print(f"  F({df1},{df2})={F:.2f}, p={pF:.3g} -> "
      + ("门限效应显著, 确需 TAR [OK]" if pF < 0.05 else "不显著, 线性 AR 足矣 [NO]"))
print("  注: 门限未知时该检验有 Davies 问题(非标准分布), 要用 Tsay(1989)/Hansen 自助法; 本例门限按正负号已定, 用标准 F.")

# ====================================================================
# 步骤3  门限&延迟定位 + 定阶: 网格搜 (d,r) 最小化 SSR, 验证数据自己也选 d=1,r≈0
#   即使知道按正负号(r=0,d=1), 也搜一遍确认"数据驱动"的最优门限确实落在 0 附近.
# ====================================================================
DMAX = 3; START = DMAX                                  # 各 d 用公共起点, SSR 才可比
search = {}
for d in range(1, DMAX + 1):
    Yd, Xd, Zd = make_arrays(train.values, d, START)
    lo, hi = np.quantile(Zd, 0.15), np.quantile(Zd, 0.85)   # 修剪 15%~85%, 保证两段样本足够
    gridd = np.linspace(lo, hi, 161)
    ssrd = np.array([tar_fit(Yd, Xd, Zd, r)[0] for r in gridd])
    kmin = int(np.nanargmin(ssrd))
    search[d] = dict(grid=gridd, ssr=ssrd, r=gridd[kmin], ssrmin=ssrd[kmin])
best_d = min(search, key=lambda d: search[d]["ssrmin"])
print(SEP); print("步骤3  门限 & 延迟网格搜索 (确认 d=1, r≈0)")
print(f"  {'延迟d':>5} {'最优门限r*':>12} {'最小SSR':>12}")
for d in range(1, DMAX + 1):
    mk = "  <- 最优" if d == best_d else ""
    print(f"  {d:>5} {search[d]['r']:>12.2f} {search[d]['ssrmin']:>12.1f}{mk}")
print(f"  -> 数据驱动最优: d={best_d}, r*={search[best_d]['r']:.2f}  "
      + ("≈ 按正负号(d=1,r=0)一致 [OK]" if best_d == 1 and abs(search[best_d]['r']) < 1.5
         else "与正负号略有出入"))

fig, axx = plt.subplots(1, 2, figsize=(12, 4.8))
g1 = search[1]
axx[0].plot(g1["grid"], g1["ssr"], color="purple", lw=1.5)
axx[0].axvline(g1["r"], color="red", ls="--", lw=1.2, label=f"argmin r*={g1['r']:.2f}")
axx[0].axvline(0, color="green", ls=":", lw=1.2, label="sign rule r=0")
axx[0].set_xlabel("candidate threshold r"); axx[0].set_ylabel("total SSR")
axx[0].set_title("Threshold search (d=1): SSR bottoms near 0"); axx[0].legend()
axx[1].scatter(X0, Y0, s=8, alpha=.35, color="gray")
bL, bH = parts["L"]["b"], parts["H"]["b"]
xs_L = np.linspace(X0.min(), 0, 50); xs_H = np.linspace(0, X0.max(), 50)
axx[1].plot(xs_L, bL[0] + bL[1] * xs_L, color="steelblue", lw=2.2, label=f"L: {bL[0]:+.1f}{bL[1]:+.2f}·y")
axx[1].plot(xs_H, bH[0] + bH[1] * xs_H, color="tomato", lw=2.2, label=f"H: {bH[0]:+.1f}{bH[1]:+.2f}·y")
axx[1].axvline(0, color="green", ls=":", lw=1.2); axx[1].axhline(0, color="k", lw=.5)
axx[1].set_xlabel("y_{t-1}"); axx[1].set_ylabel("y_t")
axx[1].set_title("Conditional mean JUMPS at 0 (one line can't fit)"); axx[1].legend()
plt.tight_layout(); plt.savefig("tar_fig2_threshold.png", dpi=110); plt.close()

# ====================================================================
# 步骤4  分段估计 (给定 d=1, r=0, 两段各一次 OLS): 参数 vs 真值, 每段 sigma
# ====================================================================
nL, nH = parts["L"]["n"], parts["H"]["n"]
sigL = np.sqrt(parts["L"]["ssr"] / (nL - 2)); sigH = np.sqrt(parts["H"]["ssr"] / (nH - 2))
print(SEP); print("步骤4  分段 OLS 估计 (参数估计 vs 真值)")
print(f"  状态 L (y_{{t-1}}<=0, n={nL}): c={bL[0]:+.2f} (真值 {cL}),  phi={bL[1]:+.3f} (真值 {phiL}),  sigma={sigL:.2f}")
print(f"  状态 H (y_{{t-1}}> 0, n={nH}): c={bH[0]:+.2f} (真值 {cH}),  phi={bH[1]:+.3f} (真值 {phiH}),  sigma={sigH:.2f}")
print(f"  估计局部中心: L={bL[0]/(1-bL[1]):+.1f} (真值 {mL:+.1f}),  H={bH[0]/(1-bH[1]):+.1f} (真值 {mH:+.1f})")

# ====================================================================
# 步骤5  模型检验: 合并残差查白噪声 (Ljung-Box) + 分布
# ====================================================================
maskL0 = X0 <= 0
fitted0 = np.where(maskL0, bL[0] + bL[1] * X0, bH[0] + bH[1] * X0)
resid0 = pd.Series(Y0 - fitted0)
lb_p = acorr_ljungbox(resid0, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
sk, ku = stats.skew(resid0), stats.kurtosis(resid0, fisher=False)
print(SEP); print("步骤5  模型检验 (合并残差)")
print(f"  Ljung-Box(10) p={lb_p:.3g} -> "
      + ("残差无自相关, 近似白噪声, 模型充分 [OK]" if lb_p > 0.05 else "残差仍有自相关, 升阶/改门限 [NO]"))
print(f"  残差 mean={resid0.mean():+.2f}, std={resid0.std():.2f} (真值 sigma={SIGMA}); 偏度={sk:+.2f}, 峰度={ku:.2f}(正态3)")

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
ax[0, 0].plot(resid0.values, lw=.7); ax[0, 0].axhline(0, color="r", ls="--")
ax[0, 0].set_title("Residuals over time")
plot_acf(resid0, lags=25, ax=ax[0, 1], zero=False); ax[0, 1].set_title("Residual ACF (white noise?)")
ax[1, 0].hist(resid0, bins=30, edgecolor="k", alpha=.7); ax[1, 0].set_title("Residual histogram")
stats.probplot(resid0, dist="norm", plot=ax[1, 1]); ax[1, 1].set_title("Q-Q plot")
plt.tight_layout(); plt.savefig("tar_fig3_residuals.png", dpi=110); plt.close()

# ====================================================================
# 步骤6  使用模型: 蒙特卡洛多步预测 (非线性必须"模拟+平均")
#   从最后一个观测出发, 按估计的 TAR 递推 + 高斯噪声, 模拟大量路径;
#   点预测=各步路径均值, 区间=各步分位数. 远期预测分布趋于"双峰"(对应两个状态).
# ====================================================================
NP = 5000                                               # 模拟路径数
y_last = train.values[-1]; reg_last = "L" if y_last <= 0 else "H"
yp = np.full(NP, y_last); sims = np.zeros((NP, h))
for k in range(h):
    mL_ = yp <= 0; z = rng.standard_normal(NP)
    yp = np.where(mL_, bL[0] + bL[1] * yp + sigL * z, bH[0] + bH[1] * yp + sigH * z)
    sims[:, k] = yp
fc_mean = sims.mean(0)
lo95, hi95 = np.quantile(sims, 0.025, axis=0), np.quantile(sims, 0.975, axis=0)
inside = (test.values >= lo95) & (test.values <= hi95)
rmse = np.sqrt(np.mean((test.values - fc_mean) ** 2))
naive = np.sqrt(np.mean((test.values - y_last) ** 2))   # 朴素基准: 一直用最后一个值
gmean = train.mean()                                    # 全局(无条件)均值
print(SEP); print(f"步骤6  蒙特卡洛多步预测 ({NP} 条路径, {h} 步)")
print(f"  起点 y_T={y_last:+.1f} (处于状态 {reg_last}); 全局均值≈{gmean:+.2f}")
print(f"  点预测 fc(1)={fc_mean[0]:+.2f} -> fc({h})={fc_mean[-1]:+.2f} (随状态变模糊, 趋向全局均值)")
print(f"  RMSE={rmse:.2f}  vs  朴素(last-value) RMSE={naive:.2f} -> "
      + ("TAR 更优 [OK]" if rmse < naive else "未胜过基准"))
print(f"  样本外 95% 区间覆盖率={inside.mean():.0%} (理想≈95%)")
print(f"  远期预测分布呈双峰(两个状态各一峰), 见 fig4 下图 —— 这是 TAR 区别于 AR 的关键证据")

pd.DataFrame({"step": np.arange(1, h + 1), "mc_mean": np.round(fc_mean, 3),
              "lo95": np.round(lo95, 3), "hi95": np.round(hi95, 3),
              "actual": np.round(test.values, 3), "in_interval": inside}
             ).to_csv("tar_forecast.csv", index=False, encoding="utf-8")
print("  已导出每步预测(点/区间/实际) -> tar_forecast.csv")

fig, axx = plt.subplots(2, 1, figsize=(11, 8))
tail = 80; xx = np.arange(h)
axx[0].plot(range(-tail, 0), train.iloc[-tail:].values, color="gray", lw=.8, label="train (tail)")
axx[0].plot(xx, test.values, color="black", lw=1, marker=".", ms=4, label="actual")
axx[0].plot(xx, fc_mean, "b-", lw=1.4, label="MC mean forecast")
axx[0].fill_between(xx, lo95, hi95, color="red", alpha=.15, label="95% MC interval")
axx[0].axhline(gmean, color="green", ls=":", lw=1, label=f"global mean={gmean:.1f}")
axx[0].axhline(0, color="k", lw=.5); axx[0].set_xlabel("forecast horizon")
axx[0].set_title("Out-of-sample: Monte-Carlo forecast (mean + dynamic interval)")
axx[0].legend(loc="upper right", fontsize=8)
axx[1].hist(sims[:, 1], bins=45, density=True, alpha=.5, color="steelblue", label="h=2 (near start regime)")
axx[1].hist(sims[:, -1], bins=45, density=True, alpha=.5, color="tomato", label=f"h={h} (long run, bimodal)")
for m_, c_ in [(bL[0] / (1 - bL[1]), "steelblue"), (bH[0] / (1 - bH[1]), "tomato")]:
    axx[1].axvline(m_, color=c_, ls="--", lw=1.2)
axx[1].axvline(0, color="green", ls=":", lw=1)
axx[1].set_xlabel("y"); axx[1].set_ylabel("density")
axx[1].set_title("Predictive density: unimodal (short) -> bimodal (long), peaks at two regime centers")
axx[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("tar_fig4_forecast.png", dpi=110); plt.close()

# ====================================================================
print(SEP)
print(f"完成. 全流程: 平稳[OK] -> 线性AR不够 -> 门限F检验显著 -> 搜出 d={best_d},r≈{search[best_d]['r']:.1f} "
      f"-> 分段OLS估计 -> 残差白噪声[{'OK' if lb_p>0.05 else 'NO'}] -> 蒙特卡洛预测(双峰)")
print(f"揭晓真值: SETAR(2;1,1), L: {cL}+{phiL}y, H: {cH}+{phiH}y, sigma={SIGMA}")
print("图: tar_fig1_identify / tar_fig2_threshold / tar_fig3_residuals / tar_fig4_forecast (.png)")
