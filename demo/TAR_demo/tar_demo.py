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

  六步法 (下游一律用"定出来的门限/延迟/阶", 不写死):
    第0步 准备数据    : 用 SETAR 真值造序列 y_t (全局平稳), 留尾部做样本外
    第1步 平稳+线性破绽: ADF 验平稳; 先拟合线性 AR(1) 当基准, 看它抓不住门限
    第2步 门限非线性检验: (2a) Tsay(1989) 排列自回归 TAR-F —— 门限未知时的标准检验(书本主检验);
                          (2b) Chow/结构突变 F —— 门限已知 r=0 时的简化确认(正负号外生设定)
    第3步 识别 d,r,p   : (3a) 网格搜 (d,r) 最小化 SSR 定门限/延迟; (3b) 各段 AIC 定段内阶 p
    第4步 分段估计     : 给定 d,r,(pL,pH), 两段各跑一次 OLS -> 参数 vs 真值, 每段 sigma
    第5步 模型检验     : 合并残差查白噪声(Ljung-Box) + 分布; 不过关 -> 升阶/改门限
    第6步 使用模型     : 1步=状态确定直接算; 多步=蒙特卡洛模拟 -> 点预测+区间(可能双峰)
                          + 三档评估: 大小(RMSE)/分布(覆盖率)/方向(区制命中 2x2 卡方表)

  工具说明: statsmodels 无现成 TAR. 但给定 (d,r,p) 后, TAR 估计 = 按状态切样本各做一次 OLS,
    完全透明; Tsay 门限检验(排列自回归+递归残差)与多步蒙特卡洛预测亦全部手写.
    多步预测因 E[g(y)]!=g(E[y]) 必须"模拟+平均"(蒙特卡洛).
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

# ---------- 小工具: OLS / 滞后设计阵 / 分段拟合 / AIC / Tsay 门限检验 ----------
def ols(yv, Xv):                                       # 最小二乘, 返回 (beta, 残差, SSR)
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid = yv - Xv @ beta
    return beta, resid, float(resid @ resid)

def lag_matrix(a, ii, p):                              # 在样本点 ii 处堆 [1, y_{t-1}, ..., y_{t-p}]
    return np.column_stack([np.ones(len(ii))] + [a[ii - j] for j in range(1, p + 1)])

def tar_fit(a, d, r, pL, pH, start, min_frac=0.15):    # 按 y_{t-d}<=r 切两段, 各做 AR(pL)/AR(pH) OLS
    a = np.asarray(a, float); idx = np.arange(start, len(a))   # start=公共起点, 保证跨 d,p 可比
    Y = a[idx]; Z = a[idx - d]; mL = Z <= r
    if min(mL.sum(), (~mL).sum()) < max(10, int(min_frac * len(idx))):
        return np.nan, None                            # 某状态样本太少 -> 该门限不可用
    ssr, parts = 0.0, {}
    for m, key, p in [(mL, "L", pL), (~mL, "H", pH)]:
        ii = idx[m]; b, res, s = ols(Y[m], lag_matrix(a, ii, p))
        ssr += s; parts[key] = dict(b=b, ssr=s, n=int(m.sum()), p=p)
    return ssr, parts

def aic_of(ssr, n, k):                                 # 高斯 AIC (k=参数个数=p+1); 越小越好
    return n * np.log(ssr / n) + 2 * k

def tsay_tar_F(a, p, d, start):
    """Tsay(1989) 排列自回归门限非线性检验 (门限未知时的标准做法, 书里 4.2 的 TAR-F):
       按门限变量 y_{t-d} 升序排列(arranged autoregression) -> 递归最小二乘求"标准化预测残差"
       -> 残差对回归元 [1, y_{t-1..t-p}] 做辅助回归取 F.
       直觉: 若线性 AR(p) 已足够, 标准化预测残差应与过去无关(系数全 0); 有门限则排序后
            前/后段服从不同模型, 残差会与回归元相关 -> F 变大. 避开了"门限未知"的 Davies 问题.
       返回 (F, p值, dfn, dfd); 还可比较不同 d 的 F 来挑延迟(F 最大者)."""
    a = np.asarray(a, float); idx = np.arange(start, len(a))
    Y = a[idx]; X = lag_matrix(a, idx, p); Z = a[idx - d]
    order = np.argsort(Z, kind="mergesort")            # 按门限变量排序 -> 排列自回归
    Xs, Ys = X[order], Y[order]; N, k = Xs.shape       # k = p+1 (含常数项)
    b0 = max(k + 2, N // 10 + p)                        # 递归起点(Tsay 建议 ~ n/10+p, 须 > 参数数)
    XtX = Xs[:b0].T @ Xs[:b0]; Xty = Xs[:b0].T @ Ys[:b0]
    ehat, Xrec = [], []
    for m in range(b0, N):                             # 逐点递归: 用前 m 个点估计, 预测第 m 点
        beta = np.linalg.solve(XtX, Xty)
        xm = Xs[m]; w = np.linalg.solve(XtX, xm)       # w = (X'X)^{-1} x_m
        ehat.append((Ys[m] - xm @ beta) / np.sqrt(1.0 + xm @ w))   # 标准化(递归)预测残差
        Xrec.append(xm)
        XtX += np.outer(xm, xm); Xty += xm * Ys[m]     # 把第 m 点纳入, 递归更新
    ehat = np.asarray(ehat); Xrec = np.asarray(Xrec)
    _, _, ssr_aux = ols(ehat, Xrec)                    # 辅助回归: 标准化残差 ~ [1, 滞后]
    sst = float(ehat @ ehat)                           # 未中心化总平方和 (H0: 系数全 0 -> 预测 0)
    dfn = k; dfd = len(ehat) - k
    F = ((sst - ssr_aux) / dfn) / (ssr_aux / dfd)
    return float(F), float(stats.f.sf(F, dfn, dfd)), dfn, dfd

def coef_str(b, p):                                    # 把系数串成可读式子 c + phi1*y(t-1) + ...
    s = f"c={b[0]:+.2f}"
    for i in range(1, p + 1):
        s += f", phi{i}={b[i]:+.3f}"
    return s

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
# 步骤2  门限非线性检验 (要不要 TAR)
#   2a  Tsay(1989) 排列自回归 TAR-F: 门限"未知"时的标准检验(书本主检验), 同时给出延迟 d.
#       —— 把门限当未知, 用排序+递归预测残差绕开 Davies 问题(零假设下 r 不可识别).
#   2b  Chow/结构突变 F: 门限"已知" r=0 时的简化确认(本 demo 正负号是外生设定, 故可直接用标准 F).
# ====================================================================
P_TEST = 1                                              # 检验用基准阶(与步骤1 线性 AR(1) 一致)
DMAX = 3                                                # 候选最大延迟
START = max(DMAX, 4)                                    # 公共起点(>=最大延迟与最大候选阶), 保证可比
tsay = {}
for d in range(1, DMAX + 1):
    F, pF_, dfn, dfd = tsay_tar_F(train.values, P_TEST, d, START)
    tsay[d] = dict(F=F, p=pF_, dfn=dfn, dfd=dfd)
d_tsay = max(tsay, key=lambda d: tsay[d]["F"])          # F 最大者 -> 推荐延迟

print(SEP); print("步骤2  门限非线性检验 (要不要 TAR)")
print("  [2a] Tsay(1989) 排列自回归 TAR-F (门限未知时的标准检验; 同时挑延迟 d)")
print(f"       {'延迟d':>5} {'TAR-F':>10} {'p值':>12}")
for d in range(1, DMAX + 1):
    mk = "  <- F最大, 选此 d" if d == d_tsay else ""
    print(f"       {d:>5} {tsay[d]['F']:>10.2f} {tsay[d]['p']:>12.3g}{mk}")
print(f"       -> d={d_tsay} 处 F 最大且 p={tsay[d_tsay]['p']:.3g}"
      + ("(≪0.05): 拒绝线性, 存在门限非线性, 延迟 d=1 [OK]" if tsay[d_tsay]['p'] < 0.05
         else ": 未拒绝线性 [NO]"))

# 2b  门限已知 r=0 的 Chow/结构突变 F (与受限模型同一 start=1 样本, 两段各 AR(1))
mL0 = X0 <= 0; ssr1 = 0.0
for m in (mL0, ~mL0):
    _, _, s = ols(Y0[m], np.column_stack([np.ones(m.sum()), X0[m]])); ssr1 += s
k0, k1 = 2, 4; df1, df2 = k1 - k0, len(Y0) - k1
F_chow = ((ssr0 - ssr1) / df1) / (ssr1 / df2)
p_chow = stats.f.sf(F_chow, df1, df2)
print("  [2b] Chow/结构突变 F (门限已知 r=0, 正负号外生设定下的简化确认)")
print(f"       受限(线性AR1) SSR0={ssr0:.1f} (k0={k0}); 非受限(两段) SSR1={ssr1:.1f} (k1={k1})")
print(f"       F({df1},{df2})={F_chow:.2f}, p={p_chow:.3g} -> "
      + ("与 Tsay 一致, 门限效应显著 [OK]" if p_chow < 0.05 else "不显著 [NO]"))
print("       注: Chow-F 假设两段同方差(本例真值同为 4.5, 不冲突); 若两段方差确不同, 严格应改用异方差稳健(White)版.")
print("       注: 门限若未知, Chow-F 有 Davies 问题(非标准分布) -> 须以 2a 的 Tsay(1989)/Hansen 自助为准.")

# ====================================================================
# 步骤3  识别 d, r, p (书里 4.3 要同时定这三样)
#   3a  网格搜 (d,r) 最小化 SSR -> 验证数据自己也选出 d=1, r≈0 (与正负号一致)
#   3b  给定 d=1,r=0, 各段用 AIC 选段内 AR 阶 p (允许两段不同阶; 真值各为 1 阶)
# ====================================================================
search = {}
for d in range(1, DMAX + 1):
    a = train.values; idx = np.arange(START, len(a)); Zd = a[idx - d]
    lo, hi = np.quantile(Zd, 0.15), np.quantile(Zd, 0.85)   # 修剪 15%~85%, 保证两段样本足够
    gridd = np.linspace(lo, hi, 161)
    ssrd = np.array([tar_fit(a, d, r, 1, 1, START)[0] for r in gridd])   # 搜门限时用基准阶 1
    kmin = int(np.nanargmin(ssrd))
    search[d] = dict(grid=gridd, ssr=ssrd, r=gridd[kmin], ssrmin=ssrd[kmin])
best_d = min(search, key=lambda d: search[d]["ssrmin"])
print(SEP); print("步骤3  识别 d, r, p")
print("  [3a] (d,r) 网格搜索 (最小化总 SSR, 确认 d=1, r≈0)")
print(f"       {'延迟d':>5} {'最优门限r*':>12} {'最小SSR':>12}")
for d in range(1, DMAX + 1):
    mk = "  <- 最优" if d == best_d else ""
    print(f"       {d:>5} {search[d]['r']:>12.2f} {search[d]['ssrmin']:>12.1f}{mk}")
print(f"       -> 数据驱动最优: d={best_d}, r*={search[best_d]['r']:.2f}  "
      + ("≈ 按正负号(d=1,r=0)一致 [OK]" if best_d == 1 and abs(search[best_d]['r']) < 1.5
         else "与正负号略有出入"))

# 操作上仍按"正负号"定门限/延迟(外生设定), 与上面搜索结果一致
D_USE, R_USE = 1, 0.0
# 3b  各段 AIC 选阶 (给定 d=1, r=0)
PMAX = 4
a = train.values; idx = np.arange(START, len(a)); Zf = a[idx - D_USE]; mLf = Zf <= R_USE
print("  [3b] 段内 AR 阶 (给定 d=1,r=0; 各段独立按 AIC 选阶, 真值各为 1 阶)")
sel_p = {}
for m, key in [(mLf, "L"), (~mLf, "H")]:
    ii = idx[m]; Yreg = a[idx][m]; nreg = len(Yreg); aics = []
    for p in range(1, PMAX + 1):
        _, _, s = ols(Yreg, lag_matrix(a, ii, p)); aics.append(aic_of(s, nreg, p + 1))
    pbest = int(np.argmin(aics)) + 1; sel_p[key] = pbest
    tab_s = "  ".join(f"p={p}:AIC={aics[p-1]:.1f}" for p in range(1, PMAX + 1))
    print(f"       状态 {key} (n={nreg}): {tab_s}  -> 选 p{key}={pbest}")
pL, pH = sel_p["L"], sel_p["H"]
print(f"       -> 段内阶: pL={pL}, pH={pH} (真值各为 1 阶)")

# 给定 d,r,(pL,pH) 拟合最终模型
_, parts = tar_fit(train.values, D_USE, R_USE, pL, pH, START)
bL, bH = parts["L"]["b"], parts["H"]["b"]

fig, axx = plt.subplots(1, 2, figsize=(12, 4.8))
g1 = search[1]
axx[0].plot(g1["grid"], g1["ssr"], color="purple", lw=1.5)
axx[0].axvline(g1["r"], color="red", ls="--", lw=1.2, label=f"argmin r*={g1['r']:.2f}")
axx[0].axvline(0, color="green", ls=":", lw=1.2, label="sign rule r=0")
axx[0].set_xlabel("candidate threshold r"); axx[0].set_ylabel("total SSR")
axx[0].set_title("Threshold search (d=1): SSR bottoms near 0"); axx[0].legend()
axx[1].scatter(X0, Y0, s=8, alpha=.35, color="gray")
xs_L = np.linspace(X0.min(), 0, 50); xs_H = np.linspace(0, X0.max(), 50)
axx[1].plot(xs_L, bL[0] + bL[1] * xs_L, color="steelblue", lw=2.2, label=f"L: {bL[0]:+.1f}{bL[1]:+.2f}·y")
axx[1].plot(xs_H, bH[0] + bH[1] * xs_H, color="tomato", lw=2.2, label=f"H: {bH[0]:+.1f}{bH[1]:+.2f}·y")
axx[1].axvline(0, color="green", ls=":", lw=1.2); axx[1].axhline(0, color="k", lw=.5)
axx[1].set_xlabel("y_{t-1}"); axx[1].set_ylabel("y_t")
axx[1].set_title("Conditional mean JUMPS at 0 (one line can't fit)"); axx[1].legend()
plt.tight_layout(); plt.savefig("tar_fig2_threshold.png", dpi=110); plt.close()

# ====================================================================
# 步骤4  分段估计 (给定 d=1, r=0, 阶 pL,pH; 两段各一次 OLS): 参数 vs 真值, 每段 sigma
# ====================================================================
nL, nH = parts["L"]["n"], parts["H"]["n"]
sigL = np.sqrt(parts["L"]["ssr"] / (nL - (pL + 1))); sigH = np.sqrt(parts["H"]["ssr"] / (nH - (pH + 1)))
ctrL = bL[0] / (1 - np.sum(bL[1:1 + pL])); ctrH = bH[0] / (1 - np.sum(bH[1:1 + pH]))
print(SEP); print("步骤4  分段 OLS 估计 (参数估计 vs 真值)")
print(f"  状态 L (y_{{t-1}}<=0, n={nL}, AR({pL})): {coef_str(bL, pL)}   sigma={sigL:.2f}  (真值 c={cL}, phi1={phiL})")
print(f"  状态 H (y_{{t-1}}> 0, n={nH}, AR({pH})): {coef_str(bH, pH)}   sigma={sigH:.2f}  (真值 c={cH}, phi1={phiH})")
print(f"  估计局部中心: L={ctrL:+.1f} (真值 {mL:+.1f}),  H={ctrH:+.1f} (真值 {mH:+.1f})")

# ====================================================================
# 步骤5  模型检验: 合并残差查白噪声 (Ljung-Box) + 分布
# ====================================================================
fitted0 = np.empty(len(idx))
for m, key in [(mLf, "L"), (~mLf, "H")]:
    b = parts[key]["b"]; p = parts[key]["p"]; ii = idx[m]
    fitted0[m] = lag_matrix(a, ii, p) @ b
resid0 = pd.Series(a[idx] - fitted0)
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
# 步骤6  使用模型: 蒙特卡洛多步预测 (非线性必须"模拟+平均") + 三档评估
#   从最后一个观测出发, 按估计的 TAR 递推 + 高斯噪声, 模拟大量路径;
#   点预测=各步路径均值, 区间=各步分位数. 远期预测分布趋于"双峰"(对应两个状态).
#   评估三档(书 4.4.2): (大小) RMSE; (分布) 区间覆盖率; (方向) 区制命中 2x2 卡方表.
# ====================================================================
NP = 5000                                               # 模拟路径数
bufw = max(pL, pH, D_USE)                                # 路径状态缓冲宽度(够取最大滞后与门限)
y_last = train.values[-1]; reg_last = "L" if y_last <= 0 else "H"
buf = np.tile(train.values[-bufw:][::-1], (NP, 1))       # buf[:,0]=y_T, [:,1]=y_{T-1}, ...
sims = np.zeros((NP, h))
for kk in range(h):
    zt = buf[:, D_USE - 1]                               # 门限变量 y_{t-d}; d=1 -> buf[:,0]
    mL_ = zt <= R_USE; z = rng.standard_normal(NP)
    predL = bL[0] + sum(bL[j + 1] * buf[:, j] for j in range(pL)) + sigL * z
    predH = bH[0] + sum(bH[j + 1] * buf[:, j] for j in range(pH)) + sigH * z
    ynext = np.where(mL_, predL, predH)
    buf = np.column_stack([ynext, buf[:, :-1]])          # 状态前移一格
    sims[:, kk] = ynext
fc_mean = sims.mean(0)
lo95, hi95 = np.quantile(sims, 0.025, axis=0), np.quantile(sims, 0.975, axis=0)
inside = (test.values >= lo95) & (test.values <= hi95)
rmse = np.sqrt(np.mean((test.values - fc_mean) ** 2))
naive = np.sqrt(np.mean((test.values - y_last) ** 2))   # 朴素基准: 一直用最后一个值
gmean = train.mean()                                    # 全局(无条件)均值

# 方向档: 各步预测落 H(正)的概率 -> 多数表决预测区制, 与实际比 2x2 + 卡方(对照抛硬币)
pH_path = (sims > 0).mean(0)                             # P(y_{T+k}>0)
pred_pos = pH_path >= 0.5
act_pos = test.values > 0
tab = np.array([[int(np.sum(pred_pos & act_pos)),    int(np.sum(pred_pos & ~act_pos))],
                [int(np.sum(~pred_pos & act_pos)),   int(np.sum(~pred_pos & ~act_pos))]])
hit = float(np.mean(pred_pos == act_pos))
base = float(max(act_pos.mean(), 1 - act_pos.mean()))   # 恒猜多数类(抛硬币基准)的命中率
if tab.sum(0).min() == 0 or tab.sum(1).min() == 0:      # 某行/列全 0 -> 卡方退化
    chi2v, pchi = np.nan, np.nan
else:
    chi2v, pchi, _, _ = stats.chi2_contingency(tab, correction=True)

print(SEP); print(f"步骤6  蒙特卡洛多步预测 ({NP} 条路径, {h} 步) + 三档评估")
print(f"  起点 y_T={y_last:+.1f} (处于状态 {reg_last}); 全局均值≈{gmean:+.2f}")
print(f"  点预测 fc(1)={fc_mean[0]:+.2f} -> fc({h})={fc_mean[-1]:+.2f} (随状态变模糊, 趋向全局均值)")
print(f"  [大小] RMSE={rmse:.2f}  vs  朴素(last-value) RMSE={naive:.2f} -> "
      + ("TAR 更优 [OK]" if rmse < naive else "未胜过基准"))
print(f"  [分布] 样本外 95% 区间覆盖率={inside.mean():.0%} (理想≈95%)")
print(f"  [方向] 区制(正负号)命中 2x2 表 (行=预测, 列=实际):")
print(f"             实际H(+)  实际L(-)")
print(f"      预测H(+) {tab[0,0]:>7d} {tab[0,1]:>9d}")
print(f"      预测L(-) {tab[1,0]:>7d} {tab[1,1]:>9d}")
print(f"         命中率={hit:.0%}  (抛硬币基准=恒猜多数类={base:.0%});  "
      + ((f"卡方={chi2v:.2f}, p={pchi:.3g} -> "
          + ("方向显著优于抛硬币 [OK]" if pchi < 0.05 else "未显著优于抛硬币(远期方向本就难) [NO]"))
         if not np.isnan(pchi) else "卡方退化(某类样本太少)"))
print(f"  远期预测分布呈双峰(两个状态各一峰), 见 fig4 下图 —— 这是 TAR 区别于 AR 的关键证据")

pd.DataFrame({"step": np.arange(1, h + 1), "mc_mean": np.round(fc_mean, 3),
              "lo95": np.round(lo95, 3), "hi95": np.round(hi95, 3),
              "actual": np.round(test.values, 3), "in_interval": inside,
              "p_up": np.round(pH_path, 3), "pred_pos": pred_pos, "actual_pos": act_pos}
             ).to_csv("tar_forecast.csv", index=False, encoding="utf-8")
print("  已导出每步预测(点/区间/实际/方向) -> tar_forecast.csv")

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
for m_, c_ in [(ctrL, "steelblue"), (ctrH, "tomato")]:
    axx[1].axvline(m_, color=c_, ls="--", lw=1.2)
axx[1].axvline(0, color="green", ls=":", lw=1)
axx[1].set_xlabel("y"); axx[1].set_ylabel("density")
axx[1].set_title("Predictive density: unimodal (short) -> bimodal (long), peaks at two regime centers")
axx[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("tar_fig4_forecast.png", dpi=110); plt.close()

# ---- fig5: 两个新增检验的可视化 (Tsay 门限检验 by 延迟 + 方向/区制命中) ----
fig, axx = plt.subplots(1, 2, figsize=(12, 4.6))
ds = list(range(1, DMAX + 1)); Fs = [tsay[d]["F"] for d in ds]
crit = stats.f.ppf(0.95, tsay[1]["dfn"], tsay[1]["dfd"])
axx[0].bar([str(d) for d in ds], Fs, color="purple", alpha=.75)
axx[0].axhline(crit, color="red", ls="--", lw=1.2, label=f"5% crit≈{crit:.2f}")
axx[0].set_xlabel("delay d"); axx[0].set_ylabel("Tsay(1989) TAR-F")
axx[0].set_title("Threshold test by delay (peaks at d=1)"); axx[0].legend()
xx1 = np.arange(1, h + 1)
axx[1].plot(xx1, pH_path, color="tomato", lw=1.6, label="forecast P(y>0)")
axx[1].axhline(0.5, color="gray", ls=":", lw=1)
axx[1].scatter(xx1[act_pos], np.ones(act_pos.sum()), s=16, color="tomato", marker="^", label="actual H (y>0)")
axx[1].scatter(xx1[~act_pos], np.zeros((~act_pos).sum()), s=16, color="steelblue", marker="v", label="actual L (y<=0)")
axx[1].set_ylim(-0.08, 1.08); axx[1].set_xlabel("forecast horizon"); axx[1].set_ylabel("P(regime H)")
axx[1].set_title(f"Direction/regime hit={hit:.0%} (base {base:.0%}, chi2 p={pchi:.2g})")
axx[1].legend(fontsize=7, loc="center right")
plt.tight_layout(); plt.savefig("tar_fig5_eval.png", dpi=110); plt.close()

# ====================================================================
print(SEP)
print(f"完成. 全流程: 平稳[OK] -> 线性AR不够 -> Tsay TAR-F 显著(选 d={d_tsay}) & Chow-F 确认"
      f" -> 搜出 d={best_d},r≈{search[best_d]['r']:.1f} & AIC 定阶 pL={pL},pH={pH}"
      f" -> 分段OLS估计 -> 残差白噪声[{'OK' if lb_p>0.05 else 'NO'}] -> 蒙特卡洛预测(双峰)+三档评估")
print(f"揭晓真值: SETAR(2;1,1), L: {cL}+{phiL}y, H: {cH}+{phiH}y, sigma={SIGMA}")
print("图: tar_fig1_identify / tar_fig2_threshold / tar_fig3_residuals / tar_fig4_forecast / tar_fig5_eval (.png)")
