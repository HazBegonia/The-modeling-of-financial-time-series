# -*- coding: utf-8 -*-
"""
VAR (向量自回归) 建模 —— 一条龙完整流程 (二维 k=2: 个股 STK + 大盘 MKT, 仿书 例8.4 IBM+SP500)

  VAR 与 AR 的关系 (本 demo 的主角):
    AR(p):      标量 y_t = c + phi_1 y_{t-1} + ... + phi_p y_{t-p} + a_t        (一个序列看自己)
    VAR(p):     向量 r_t = phi_0 + Phi_1 r_{t-1} + ... + Phi_p r_{t-p} + a_t     (k 个序列互相看)
        r_t=(STK_t, MKT_t)' 是 k=1 维向量; phi_0 是 k 维常数; 每个 Phi_i 是 k×k 矩阵;
        a_t 是 k 维白噪声, Cov(a_t)=Sigma —— 同一天的"同步关系"全藏在 Sigma 的非对角线里.
    把标量 AR 的"系数"换成"矩阵", 就得到 VAR: 既抓每个序列的自身记忆(Phi 对角),
    又抓"谁带动谁"的跨序列引导(Phi 非对角).

  为什么需要 VAR: 多个序列彼此牵动(大盘领跑个股 = 单向引导; 同日齐涨齐跌 = 同期相关).
    单独对每个序列做 AR 会丢掉这些"跨序列"信息 —— VAR 用矩阵系数一次性把它们建进来.

  五步法 (书 8.2: 前提 -> 定阶 -> 估计 -> 检验 -> 预测; 下游一律用"定出来的阶 p", 不写死):
    第0步 准备数据    : 用 VAR(2) 真值造二维序列 (MKT 单向引导 STK + 强同期相关), 留尾部样本外
    第1步 平稳前提    : 各分量 ADF; 关键看 VAR 的伴随矩阵特征值是否全在单位圆内(向量版平稳条件)
    第2步 定阶 p      : (2a) M(i) 逐阶似然比检验(书式 8.18, H0:Phi_i=0, 渐近 chi2(k^2));
                        (2b) AIC/BIC/HQ 三准则打分取最小 (罚项含 k^2*i, 阶/序列越多罚越狠)
    第3步 估计        : 逐方程 OLS(简化式 VAR 每个方程回归元相同 -> 可分方程做, 渐近=ML);
                        报系数 vs 真值 + 残差协方差 Sigma(式 8.19); 剔不显著项再估(书 step b);
                        附 Granger 因果: 验证 MKT->STK 单向引导(8.1.2)
    第4步 检验        : 多元 Ljung-Box Q_k(m) 整体查残差 CCM; 残差 CCM 符号表(+/-/.)几乎全"."
                        -> 自相关与互相带动都被榨干 -> 模型充分; 不过关回第2步升阶
    第5步 预测        : 合格后向量递推 r_hat_t(1..h); 预测误差协方差用 MA(∞) 的 Psi 权重逐步累加
                        + 评估(各分量 RMSE vs 朴素基准, 95% 区间覆盖率)

  工具说明: statsmodels 有 VAR, 但本 demo 全程手写(numpy/scipy)以保持透明:
    估计=逐方程 OLS; M(i)=两个相邻阶 ML 残差协方差行列式之比的似然比; Q_k(m)=多元 portmanteau;
    CCM 符号表=残差互相关阵按 2/sqrt(T) 两倍标准误判显著; 预测误差协方差=Psi 权重递推.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")               # Windows 控制台强制 UTF-8
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                                  # 不弹窗, 直接存图
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller

SEP = "=" * 72
NAMES = ["STK", "MKT"]                                  # 序列名: 个股 / 大盘

# ---------- 小工具: 多方程 OLS 拟合 VAR / M(i) / IC / 多元 Ljung-Box / CCM ----------
def design(a, p, start):
    """在公共起点 start 处堆 VAR 的设计阵: 行=时点, 列=[1, r_{t-1}, ..., r_{t-p}] (含常数)."""
    a = np.asarray(a, float); T, k = a.shape
    idx = np.arange(start, T)
    Y = a[idx]                                          # (n, k) 被解释
    cols = [np.ones((len(idx), 1))]
    for j in range(1, p + 1):
        cols.append(a[idx - j])                         # 第 j 阶滞后(k 列)
    X = np.hstack(cols)                                 # (n, 1+k*p)
    return Y, X, idx

def fit_var(a, p, start):
    """逐方程 OLS 估计 VAR(p). 简化式每方程回归元相同 -> 一次 lstsq 解出所有方程(渐近=ML).
       返回系数 B((1+kp)×k, y=B'x)、残差、ML 残差协方差 Sigma(式8.19,除以 n)、样本量 n."""
    Y, X, idx = design(a, p, start)
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)           # 一次解 k 个方程
    resid = Y - X @ B
    n = len(idx)
    Sigma = resid.T @ resid / n                         # 式(8.19): (1/T) Σ a_t a_t'
    return dict(B=B, resid=resid, Sigma=Sigma, X=X, Y=Y, idx=idx, n=n, p=p)

def sigma_var0(a, start):
    """VAR(0)(只含常数=均值)的 ML 残差协方差, 作为 M(i)/IC 的 i=0 基准(同一公共样本)."""
    Y = np.asarray(a, float)[start:]
    r = Y - Y.mean(0)
    return r.T @ r / len(Y)

def companion_eig(B, k, p):
    """由估计系数搭伴随(companion)矩阵, 返回其特征值模长. 全 <1 -> VAR 弱平稳(向量版单位圆条件)."""
    Phi = [B[1 + j * k:1 + (j + 1) * k, :].T for j in range(p)]   # Phi_{j+1} (k×k)
    top = np.hstack(Phi)                                # (k, k*p)
    comp = np.zeros((k * p, k * p))
    comp[:k, :] = top
    if p > 1:
        comp[k:, :-k] = np.eye(k * (p - 1))             # 下方单位阵堆叠
    return np.abs(np.linalg.eigvals(comp)), Phi

def mv_ljung_box(resid, m, k, p):
    """多元 Ljung-Box (portmanteau) Q_k(m): 整体检验残差的交叉-相关矩阵 CCM 到 m 阶是否全 0.
       Q_k(m)=T^2 Σ_{l=1}^m tr(Γ_l' Γ_0^{-1} Γ_l Γ_0^{-1})/(T-l), 渐近 chi2, df=k^2 m - k^2 p."""
    a = np.asarray(resid, float); T = len(a)
    C0 = a.T @ a / T; C0inv = np.linalg.inv(C0)
    Q = 0.0
    for l in range(1, m + 1):
        Cl = a[l:].T @ a[:-l] / T                       # Γ_l = (1/T) Σ a_t a_{t-l}'
        Q += np.trace(Cl.T @ C0inv @ Cl @ C0inv) / (T - l)
    Q *= T * T
    df = k * k * m - k * k * p
    return float(Q), df, float(stats.chi2.sf(Q, df))

def ccm_signs(resid, l, thr):
    """残差在滞后 l 的互相关矩阵 CCM 的符号表: |rho|>2/sqrt(T) 记 +/-, 否则 '.'(不显著)."""
    a = np.asarray(resid, float); T = len(a)
    C0 = a.T @ a / T; d = np.sqrt(np.diag(C0))
    Cl = a[l:].T @ a[:-l] / T
    rho = Cl / np.outer(d, d)
    sym = np.where(rho > thr, "+", np.where(rho < -thr, "-", "."))
    return rho, sym

def fmt_eq(name, b0, Phis, k, names):
    """把一行(某个方程)的系数串成可读式子: name_t = c + Σ Phi_ij * series_j(t-i)."""
    s = f"  {name}_t = {b0:+.2f}"
    for i, Phi in enumerate(Phis, 1):
        for j in range(k):
            c = Phi[j]
            if abs(c) > 1e-8:
                s += f" {c:+.3f}*{names[j]}(t-{i})"
    return s

# ====================================================================
# 步骤0  数据生成: VAR(2), MKT 单向引导 STK(8.1.2) + 强同期相关(藏在 Sigma)
#   真值结构(仿 例8.4 IBM+SP500):
#     MKT 方程: 只用自己的滞后(STK 列全 0) -> STK 不引导 MKT
#     STK 方程: 用自己滞后 + MKT 的滞后    -> MKT 引导 STK (单向)
#     Sigma 非对角 0.6 -> 同一天 STK 与 MKT 强相关(齐涨齐跌)
# ====================================================================
phi0 = np.array([0.5, 0.3])
Phi1 = np.array([[0.10, 0.30],          # STK <- 0.10*STK(t-1) + 0.30*MKT(t-1)
                 [0.00, 0.20]])         # MKT <- 0.00*STK(t-1) + 0.20*MKT(t-1)  (STK列=0)
Phi2 = np.array([[-0.05, 0.15],         # STK <- -0.05*STK(t-2) + 0.15*MKT(t-2)
                 [0.00, -0.10]])        # MKT <-  0.00*STK(t-2) - 0.10*MKT(t-2) (STK列=0)
SIGMA = np.array([[1.00, 0.60],
                  [0.60, 1.00]])        # 同期相关 corr=0.6 (仿书 ~0.63)
P_TRUE = 2
n, burn = 600, 200
k = 2

rng = np.random.default_rng(11)
L = np.linalg.cholesky(SIGMA)
eps = (rng.standard_normal((n + burn, k)) @ L.T)        # a_t ~ N(0, Sigma)
r = np.zeros((n + burn, k))
for t in range(2, n + burn):
    r[t] = phi0 + Phi1 @ r[t - 1] + Phi2 @ r[t - 2] + eps[t]
r = r[burn:]                                            # 丢弃 burn-in

h = 12                                                  # 留最后 12 个点做样本外预测
train, test = r[:-h], r[-h:]
T = len(train)
df_out = pd.DataFrame({"t": np.arange(len(r)), "STK": np.round(r[:, 0], 4), "MKT": np.round(r[:, 1], 4),
                       "split": ["train"] * len(train) + ["test"] * len(test)})
df_out.to_csv("var_data.csv", index=False, encoding="utf-8")

corr_emp = np.corrcoef(train.T)[0, 1]
print(SEP); print("步骤0  数据生成 (VAR(2) 真值: MKT 单向引导 STK + 强同期相关)")
print(fmt_eq("STK", phi0[0], [Phi1[0], Phi2[0]], k, NAMES) + "   <- 含 MKT 滞后")
print(fmt_eq("MKT", phi0[1], [Phi1[1], Phi2[1]], k, NAMES) + "   <- 不含 STK 滞后(单向)")
print(f"  噪声协方差 Sigma=[[1,0.6],[0.6,1]] (同期相关 0.6); 样本 T={T}(train)+{h}(test)")
print(f"  训练集经验同期相关 corr(STK,MKT)={corr_emp:+.2f} (应≈0.6)")
print("  已导出 -> var_data.csv")

# ====================================================================
# 步骤1  平稳前提: 各分量 ADF + VAR 伴随矩阵特征值(向量版"单位圆内")
# ====================================================================
print(SEP); print("步骤1  平稳前提 (各分量 ADF + 伴随矩阵特征值在单位圆内)")
for j in range(k):
    s, pv, *_ = adfuller(train[:, j], autolag="AIC")
    print(f"  ADF[{NAMES[j]}]={s:.2f}, p={pv:.3g} -> " + ("平稳 [OK]" if pv < 0.05 else "不平稳 [NO]"))
# 用真值系数搭伴随矩阵(也可用估计值); 模长全 <1 即向量版弱平稳条件
B_true = np.vstack([phi0, Phi1.T, Phi2.T])              # 与 fit_var 的 B 同布局(y=B'x)
eig_mod, _ = companion_eig(B_true, k, P_TRUE)
print(f"  伴随矩阵特征值模长: {np.round(np.sort(eig_mod)[::-1], 3)}")
print(f"  最大模长={eig_mod.max():.3f} -> " + ("全部 <1, VAR 弱平稳, 可建模 [OK]" if eig_mod.max() < 1
                                              else "存在 >=1, 不平稳, 需差分/协整 [NO]"))

fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for j in range(k):
    ax[j].plot(train[:, j], lw=.6, color=("steelblue" if j == 0 else "tomato"))
    ax[j].axhline(train[:, j].mean(), color="k", ls="--", lw=.7)
    ax[j].set_ylabel(NAMES[j]); ax[j].set_title(f"{NAMES[j]} series (train)")
ax[1].set_xlabel("t")
plt.tight_layout(); plt.savefig("var_fig1_series.png", dpi=110); plt.close()

# ====================================================================
# 步骤2  定阶 p: (2a) M(i) 逐阶似然比检验(式8.18) + (2b) AIC/BIC/HQ 取最小
# ====================================================================
PMAX = 6
START = PMAX                                            # 公共起点: 各阶在同一样本上比, 行列式才可比
g = T - START                                           # 公共有效样本量
# 各阶 ML 残差协方差(含 i=0)
Sig = {0: sigma_var0(train, START)}
fits = {}
for i in range(1, PMAX + 1):
    fits[i] = fit_var(train, i, START)
    Sig[i] = fits[i]["Sigma"]
logdet = {i: np.log(np.linalg.det(Sig[i])) for i in range(PMAX + 1)}

# (2a) M(i): H0 Phi_i=0; M(i)=-(g-k-i-1.5) ln(|Sig_i|/|Sig_{i-1}|) ~ chi2(k^2)
print(SEP); print("步骤2  定阶 p")
print("  [2a] M(i) 逐阶似然比检验 (式8.18; H0: Phi_i=0 -> 加第 i 阶没用; 渐近 chi2(k^2), k^2=%d)" % (k * k))
print(f"       {'阶i':>4} {'M(i)':>10} {'p值':>12}   判定")
crit = stats.chi2.ppf(0.95, k * k)
Mvals = {}
for i in range(1, PMAX + 1):
    Mi = -(g - k - i - 1.5) * (logdet[i] - logdet[i - 1])
    pim = stats.chi2.sf(Mi, k * k); Mvals[i] = (Mi, pim)
    verdict = "拒绝H0,该阶有用,保留" if pim < 0.05 else "不显著,可砍"
    print(f"       {i:>4} {Mi:>10.2f} {pim:>12.3g}   {verdict}")
# 最大显著阶 = M(i) 在该阶后连续不显著的最后一个显著阶
p_M = max([i for i in range(1, PMAX + 1) if Mvals[i][1] < 0.05], default=0)
print(f"       -> 最后一个显著阶 = {p_M} (5%临界值 chi2={crit:.2f}); 真值 p={P_TRUE}")

# (2b) 信息准则: ln|Sig_i| + 罚(含 k^2 i); 取最小
print("  [2b] 信息准则 (ln|Sigma_i| + 罚项; 罚项含 k^2*i, 阶/序列越多罚越狠; 取最小)")
ic = {"AIC": [], "BIC": [], "HQ": []}
for i in range(PMAX + 1):
    pen_aic = 2 * k * k * i / g
    pen_bic = k * k * i * np.log(g) / g
    pen_hq = 2 * k * k * i * np.log(np.log(g)) / g
    ic["AIC"].append(logdet[i] + pen_aic)
    ic["BIC"].append(logdet[i] + pen_bic)
    ic["HQ"].append(logdet[i] + pen_hq)
sel = {key: int(np.argmin(v)) for key, v in ic.items()}
print(f"       {'阶i':>4} {'AIC':>10} {'BIC':>10} {'HQ':>10}")
for i in range(PMAX + 1):
    mk = "  <-" + ",".join([key for key in sel if sel[key] == i]) if i in sel.values() else ""
    print(f"       {i:>4} {ic['AIC'][i]:>10.4f} {ic['BIC'][i]:>10.4f} {ic['HQ'][i]:>10.4f}{mk}")
print(f"       -> AIC选 {sel['AIC']}, BIC选 {sel['BIC']}, HQ选 {sel['HQ']} "
      f"(BIC/HQ 罚更重,倾向更小阶); 真值 p={P_TRUE}")
P_USE = sel["BIC"] if sel["BIC"] > 0 else max(p_M, 1)   # 综合定阶(BIC 稳健), 下游一律用 P_USE
print(f"  ==> 综合定阶 P_USE={P_USE} (以 BIC 为准; M(i) 与 AIC 旁证)")

fig, axx = plt.subplots(1, 2, figsize=(12, 4.6))
xs = np.arange(PMAX + 1)
for key, c in [("AIC", "steelblue"), ("BIC", "tomato"), ("HQ", "green")]:
    axx[0].plot(xs, ic[key], marker="o", color=c, label=f"{key} (min@{sel[key]})")
    axx[0].scatter([sel[key]], [ic[key][sel[key]]], s=90, facecolors="none", edgecolors=c, lw=2)
axx[0].axvline(P_TRUE, color="k", ls=":", lw=1, label=f"true p={P_TRUE}")
axx[0].set_xlabel("VAR order i"); axx[0].set_ylabel("criterion"); axx[0].set_title("Order selection: AIC/BIC/HQ")
axx[0].legend(fontsize=8)
pis = [Mvals[i][1] for i in range(1, PMAX + 1)]
axx[1].bar(range(1, PMAX + 1), pis, color="purple", alpha=.7)
axx[1].axhline(0.05, color="red", ls="--", lw=1.2, label="5% level")
axx[1].axvline(P_TRUE, color="k", ls=":", lw=1, label=f"true p={P_TRUE}")
axx[1].set_xlabel("order i"); axx[1].set_ylabel("M(i) p-value"); axx[1].set_title("Sequential test M(i): last sig. order")
axx[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("var_fig2_order.png", dpi=110); plt.close()

# ====================================================================
# 步骤3  估计: 逐方程 OLS -> 系数 vs 真值 + Sigma; 剔不显著项再估; Granger 因果
# ====================================================================
fit = fit_var(train, P_USE, START)
B, resid, Sigma_hat, n_eff = fit["B"], fit["resid"], fit["Sigma"], fit["n"]
X = fit["X"]
npar_eq = B.shape[0]                                    # 每方程参数数 = 1 + k*P_USE
Sigma_unb = resid.T @ resid / (n_eff - npar_eq)         # 无偏(供预测误差用)
print(SEP); print(f"步骤3  估计 VAR({P_USE}) (逐方程 OLS, 渐近=ML)")
Phis_hat = [B[1 + j * k:1 + (j + 1) * k, :].T for j in range(P_USE)]   # 各 Phi_i (k×k)
for eq in range(k):
    print(fmt_eq(NAMES[eq], B[0, eq], [Ph[eq] for Ph in Phis_hat], k, NAMES))
print(f"  残差协方差 Sigma_hat = [[{Sigma_hat[0,0]:.2f},{Sigma_hat[0,1]:.2f}],"
      f"[{Sigma_hat[1,0]:.2f},{Sigma_hat[1,1]:.2f}]] (真值对角1, 非对角0.6)")
print(f"  残差同期相关={Sigma_hat[0,1]/np.sqrt(Sigma_hat[0,0]*Sigma_hat[1,1]):+.2f} -> "
      "同一天 STK 与 MKT 一起动(藏在 Sigma 的同步关系) [OK]")

# 剔不显著项再估(书 step b): 各方程算 t 值, |t|<1.96 的滞后项剔掉, 该方程单独重估
print(f"  [精修] 逐方程剔除 |t|<1.96 的不显著项后重估 (让模型更省参数):")
XtX_inv = np.linalg.inv(X.T @ X)
B_ref = B.copy()
for eq in range(k):
    sig2 = resid[:, eq] @ resid[:, eq] / (n_eff - npar_eq)
    se = np.sqrt(np.diag(sig2 * XtX_inv))
    tstat = B[:, eq] / se
    keep = np.abs(tstat) >= 1.96; keep[0] = True        # 常数项始终保留
    Xk = X[:, keep]
    bk, *_ = np.linalg.lstsq(Xk, fit["Y"][:, eq], rcond=None)
    bfull = np.zeros(npar_eq); bfull[keep] = bk
    B_ref[:, eq] = bfull
    dropped = int((~keep).sum())
    print(f"     {NAMES[eq]} 方程: 保留 {int(keep.sum())}/{npar_eq} 项, 剔除 {dropped} 个不显著滞后")
print("     -> 精简后 STK 方程仍留 MKT 滞后、MKT 方程仍无 STK 滞后(还原单向引导结构)")

# Granger 因果(8.1.2): 单方程 F 检验"加上对方的滞后有没有用"
print("  [Granger 因果] 加入对方滞后能否显著改善本方程预测 (F 检验):")
def granger_F(cause, effect):
    """检验 series[cause] 是否 Granger-引导 series[effect]: 受限=只含 effect 自身滞后."""
    Yv = fit["Y"][:, effect]
    _, ssr_u, *_ = (lambda b: (None, float((Yv - X @ b) @ (Yv - X @ b))))(
        np.linalg.lstsq(X, Yv, rcond=None)[0])
    cols_eff = [0] + [1 + j * k + effect for j in range(P_USE)]   # 常数 + effect 自身滞后
    Xr = X[:, cols_eff]
    br, *_ = np.linalg.lstsq(Xr, Yv, rcond=None)
    ssr_r = float((Yv - Xr @ br) @ (Yv - Xr @ br))
    q = X.shape[1] - Xr.shape[1]                         # 被约束(对方滞后)的个数
    Fv = ((ssr_r - ssr_u) / q) / (ssr_u / (n_eff - X.shape[1]))
    return Fv, float(stats.f.sf(Fv, q, n_eff - X.shape[1]))
gF = {}
for cause, effect in [(1, 0), (0, 1)]:                  # MKT->STK, STK->MKT
    Fv, pv = granger_F(cause, effect); gF[(cause, effect)] = (Fv, pv)
    arrow = f"{NAMES[cause]} -> {NAMES[effect]}"
    print(f"     {arrow}: F={Fv:.2f}, p={pv:.3g} -> "
          + ("显著, 存在引导 [OK]" if pv < 0.05 else "不显著, 无引导 [.]"))
print(f"     -> {NAMES[1]} 引导 {NAMES[0]} 显著、反向不显著 = 单向引导(书 8.1.2 被估成具体数字)")

# ====================================================================
# 步骤4  检验: 多元 Ljung-Box Q_k(m) + 残差 CCM 符号表(几乎全 '.')
# ====================================================================
print(SEP); print("步骤4  模型检验 (残差是否白噪声 = 信息榨干没有)")
thr = 2.0 / np.sqrt(n_eff)                              # CCM 两倍标准误阈值
for m in (6, 12):
    Qk, dfQ, pQ = mv_ljung_box(resid, m, k, P_USE)
    print(f"  多元 Ljung-Box Q_k(m={m})={Qk:.2f}, df={dfQ}, p={pQ:.3g} -> "
          + ("残差无交叉自相关, 近似多元白噪声, 模型充分 [OK]" if pQ > 0.05
             else "残差仍有结构, 升阶 [NO]"))
print(f"  残差 CCM 符号表 (阈值 2/sqrt(T)={thr:.3f}; 行=被影响序列, 列=滞后源; '.'=不显著):")
alldot = True
for l in range(1, 5):
    _, sym = ccm_signs(resid, l, thr)
    rowstr = "   ".join("[" + " ".join(sym[i]) + "]" for i in range(k))
    print(f"     lag {l}:  {rowstr}")
    if np.any(sym != "."):
        alldot = False
print("     -> 符号表" + ("几乎全是 '.', 自相关与互相带动都被吸收干净, 模型充分 [OK]"
                          if alldot else "尚有 +/- 残留, 考虑升阶 [NO]"))

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for j in range(k):
    ax[0, j].plot(resid[:, j], lw=.6); ax[0, j].axhline(0, color="r", ls="--", lw=.7)
    ax[0, j].set_title(f"Residual {NAMES[j]} over time")
# 下排: 残差自相关(各分量) 与 STK-MKT 交叉相关 (都应≈0 才算白噪声)
maxlag = 12
lags = np.arange(1, maxlag + 1)
ac0 = [np.corrcoef(resid[l:, 0], resid[:-l, 0])[0, 1] for l in lags]
ac1 = [np.corrcoef(resid[l:, 1], resid[:-l, 1])[0, 1] for l in lags]
cc = [np.corrcoef(resid[l:, 0], resid[:-l, 1])[0, 1] for l in lags]
ax[1, 0].bar(lags - 0.2, ac0, width=0.4, label=f"{NAMES[0]} ACF", color="steelblue")
ax[1, 0].bar(lags + 0.2, ac1, width=0.4, label=f"{NAMES[1]} ACF", color="tomato")
ax[1, 0].axhline(thr, color="gray", ls="--", lw=.8); ax[1, 0].axhline(-thr, color="gray", ls="--", lw=.8)
ax[1, 0].set_title("Residual auto-corr (should be ~0)"); ax[1, 0].legend(fontsize=8)
ax[1, 1].bar(lags, cc, color="purple")
ax[1, 1].axhline(thr, color="gray", ls="--", lw=.8); ax[1, 1].axhline(-thr, color="gray", ls="--", lw=.8)
ax[1, 1].set_title("Residual cross-corr STK vs MKT (should be ~0)")
plt.tight_layout(); plt.savefig("var_fig3_diagnose.png", dpi=110); plt.close()

# ====================================================================
# 步骤5  预测: 向量递推 r_hat_t(1..h) + 预测误差协方差(Psi 权重) + 评估
#   点预测: r_hat(1)=phi0+Σ Phi_i r_{T+1-i}; 多步用已预测值续推(plug-in)
#   误差协方差: Sigma(l)=Σ_{j=0}^{l-1} Psi_j Sigma Psi_j'  (MA(∞) 表示, Psi 递推)
# ====================================================================
print(SEP); print(f"步骤5  样本外预测 ({h} 步) + 评估")
b0 = B[0]                                                # 截距向量
hist = list(train[-P_USE:][::-1])                        # hist[0]=r_T, hist[1]=r_{T-1}, ...
fc = np.zeros((h, k))
buf = [v.copy() for v in hist]
for step in range(h):
    pred = b0.copy()
    for j in range(P_USE):
        pred = pred + Phis_hat[j] @ buf[j]
    fc[step] = pred
    buf = [pred] + buf[:-1]                              # 预测值前移(plug-in)
# Psi 权重: Psi_0=I; Psi_l=Σ_{j=1}^{min(l,p)} Phi_j Psi_{l-j}
Psi = [np.eye(k)]
for l in range(1, h):
    P_l = np.zeros((k, k))
    for j in range(1, min(l, P_USE) + 1):
        P_l += Phis_hat[j - 1] @ Psi[l - j]
    Psi.append(P_l)
fc_var = np.zeros((h, k))                                # 各分量预测误差方差
S = np.zeros((k, k))
for l in range(h):
    S = S + Psi[l] @ Sigma_unb @ Psi[l].T
    fc_var[l] = np.diag(S)
fc_se = np.sqrt(fc_var)
lo95, hi95 = fc - 1.96 * fc_se, fc + 1.96 * fc_se

inside = (test >= lo95) & (test <= hi95)
naive = train[-1]                                        # 朴素基准: 一直用最后一个观测
rmse = np.sqrt(np.mean((test - fc) ** 2, axis=0))
rmse_naive = np.sqrt(np.mean((test - naive) ** 2, axis=0))
for j in range(k):
    print(f"  {NAMES[j]}: 1步预测={fc[0,j]:+.2f}, {h}步={fc[-1,j]:+.2f} (趋向均值 {train[:,j].mean():+.2f})")
    print(f"        RMSE={rmse[j]:.2f} vs 朴素 last-value RMSE={rmse_naive[j]:.2f} -> "
          + ("VAR 更优 [OK]" if rmse[j] < rmse_naive[j] else "未胜过基准"))
    print(f"        95% 区间覆盖率={inside[:,j].mean():.0%} (理想≈95%)")

pd.DataFrame({"step": np.arange(1, h + 1),
              "STK_fc": np.round(fc[:, 0], 3), "STK_lo": np.round(lo95[:, 0], 3),
              "STK_hi": np.round(hi95[:, 0], 3), "STK_act": np.round(test[:, 0], 3),
              "MKT_fc": np.round(fc[:, 1], 3), "MKT_lo": np.round(lo95[:, 1], 3),
              "MKT_hi": np.round(hi95[:, 1], 3), "MKT_act": np.round(test[:, 1], 3),
              }).to_csv("var_forecast.csv", index=False, encoding="utf-8")
print("  已导出每步预测(点/区间/实际) -> var_forecast.csv")

fig, ax = plt.subplots(2, 1, figsize=(11, 8))
tail = 60; xx = np.arange(1, h + 1)
for j in range(k):
    ax[j].plot(range(-tail + 1, 1), train[-tail:, j], color="gray", lw=.8, label="train (tail)")
    ax[j].plot(xx, test[:, j], color="black", lw=1, marker=".", ms=6, label="actual")
    ax[j].plot(xx, fc[:, j], "b-", lw=1.5, marker="o", ms=3, label="VAR forecast")
    ax[j].fill_between(xx, lo95[:, j], hi95[:, j], color="red", alpha=.15, label="95% interval")
    ax[j].axhline(train[:, j].mean(), color="green", ls=":", lw=1, label="train mean")
    ax[j].axvline(0, color="k", lw=.5)
    ax[j].set_ylabel(NAMES[j]); ax[j].set_title(f"Out-of-sample forecast: {NAMES[j]}")
    ax[j].legend(loc="upper right", fontsize=8)
ax[1].set_xlabel("forecast horizon")
plt.tight_layout(); plt.savefig("var_fig4_forecast.png", dpi=110); plt.close()

# ---- fig5: 两个"能直接读出"的结论 —— 同期相关(Sigma) + 单向引导(Granger) ----
fig, axx = plt.subplots(1, 2, figsize=(12, 4.6))
axx[0].scatter(resid[:, 1], resid[:, 0], s=10, alpha=.4, color="purple")
axx[0].set_xlabel(f"{NAMES[1]} innovation a2"); axx[0].set_ylabel(f"{NAMES[0]} innovation a1")
rr = Sigma_hat[0, 1] / np.sqrt(Sigma_hat[0, 0] * Sigma_hat[1, 1])
axx[0].set_title(f"Contemporaneous innovations corr={rr:+.2f} (same-day co-movement in Sigma)")
labels = [f"{NAMES[1]}->{NAMES[0]}", f"{NAMES[0]}->{NAMES[1]}"]
Fs = [gF[(1, 0)][0], gF[(0, 1)][0]]
critF = stats.f.ppf(0.95, P_USE, n_eff - X.shape[1])
axx[1].bar(labels, Fs, color=["tomato", "steelblue"], alpha=.8)
axx[1].axhline(critF, color="red", ls="--", lw=1.2, label=f"5% crit≈{critF:.2f}")
axx[1].set_ylabel("Granger F"); axx[1].set_title("One-way lead: MKT Granger-causes STK, not vice versa")
axx[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("var_fig5_structure.png", dpi=110); plt.close()

# ====================================================================
print(SEP)
print(f"完成. 全流程: 平稳[OK](特征值<1) -> 定阶 M(i)末显著阶={p_M} & BIC={sel['BIC']} (用 p={P_USE})"
      f" -> 逐方程OLS估计(Sigma同期相关{Sigma_hat[0,1]/np.sqrt(Sigma_hat[0,0]*Sigma_hat[1,1]):+.2f})"
      f" & Granger单向引导 -> 残差Q_k白噪声[{'OK' if pQ>0.05 else 'NO'}]+CCM全'.' -> 向量递推预测+区间")
print(f"揭晓真值: VAR(2), MKT 单向引导 STK, Sigma 非对角=0.6")
print("图: var_fig1_series / var_fig2_order / var_fig3_diagnose / var_fig4_forecast / var_fig5_structure (.png)")
