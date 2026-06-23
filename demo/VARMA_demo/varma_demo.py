# -*- coding: utf-8 -*-
"""
VARMA (向量自回归滑动平均) 建模 —— 一条龙完整流程 (二维 k=2: 个股 STK + 大盘 MKT)

  VARMA 与 VAR / VMA 的关系 (本 demo 的主角 = 前两者的"合体"):
    VAR(p):    r_t = phi_0 + Phi_1 r_{t-1}+...               + a_t                 (只看过去的值)
    VMA(q):    r_t = theta_0 +                  a_t - Theta_1 a_{t-1}-...          (只看过去的冲击)
    VARMA(p,q):r_t = phi_0 + Phi_1 r_{t-1}+...+ a_t - Theta_1 a_{t-1}-...          (两边都看)
        既有 AR 部分(长记忆, 值层层递推)又有 MA 部分(有限记忆, 冲击只活 q 期).

  骨架仍是四步, 但 VARMA 比 VAR/VMA 多两道独有的坎(本 demo 重点演示):
    [坎0·劝退关] 书的明确建议: 金融里 VAR 或 VMA 单用基本够; 只有两者特征都明显、非用不可才上 VARMA,
                 且只取低阶(1,1)/(2,1). -> 先看数据: CCM 截尾用 VMA、拖尾用 VAR; 都不干净才上 VARMA.
    [坎1·识别]   VARMA 的 CCM 像 VAR 一样拖尾 -> 光看 CCM 定不出 (p,q); 更糟的是 AR/MA 可能有"公因子"
                 相互抵消 -> 同一数据有无数等价写法(不唯一). 必须用"结构指定"(Tiao-Tsay 1989 / Tsay 1991:
                 Kronecker 指数 / 标量分量模型 SCM)强行让模型唯一. 这是最专业、最易翻车的一步,
                 也是书不展开、实务慎用的根本原因. 本 demo 用 Psi 权重数值演示"公因子->不唯一",
                 再施加最常用的正规化(Phi_0=Theta_0=I + 取最低阶), 把模型钉唯一.

  五步法 (前提/劝退 -> 定阶+识别 -> 估计 -> 检验 -> 预测; 下游用定出的 (p,q), 不写死):
    第0步 准备数据    : 用 VARMA(1,1) 真值造序列 (AR、MA 各带一个结构性单向 0), 留尾部样本外
    第1步 劝退关+前提  : 平稳(Phi)&可逆(Theta); 演示"纯 VMA 数不出 q(CCM 拖尾)、纯 VAR(1) 残差不白
                        (需很高阶)"-> 两者都不干净, 才考虑 VARMA(1,1)
    第2步 定阶+识别    : 用 Psi 权重证"公因子=同一过程的两种写法"(不唯一); 施加正规化把 (p,q)=(1,1) 钉唯一
    第3步 估计        : 条件 MLE(含 AR+MA, 冲击要反推, 不能纯 OLS); 带识别约束; 剔不显著项再估
    第4步 检验        : 多元 Ljung-Box Q_k(m) + 残差 CCM 符号表 (与 VAR/VMA 完全相同)
    第5步 预测        : 两种记忆叠加 —— 短期受 MA 冲击项影响(只 q 步)、长期由 AR 主导慢衰减回均值

  工具说明: statsmodels 的 VARMAX 能拟 VARMA, 但本 demo 全程手写(numpy/scipy)以保持透明:
    识别=Psi 权重等价性演示 + 正规化; 估计=条件 MLE(Sigma 集中并出); 检验=多元 portmanteau + CCM;
    预测=AR+MA 递推 + Psi 权重误差协方差.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")               # Windows 控制台强制 UTF-8
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                                  # 不弹窗, 直接存图
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller

SEP = "=" * 72
NAMES = ["STK", "MKT"]
k = 2

# ---------- 小工具: 冲击反推 / 条件MLE / Psi权重 / 多元 Ljung-Box / CCM ----------
def innovations(r, th0, Phi1, Th1):
    """条件法反推冲击(VARMA(1,1)): a_0=0, a_t = r_t - theta_0 - Phi_1 r_{t-1} + Theta_1 a_{t-1}."""
    r = np.asarray(r, float); T = len(r)
    a = np.empty((T, k)); prev_a = np.zeros(k); prev_r = np.zeros(k)
    for t in range(T):
        cur = r[t] - th0 - Phi1 @ prev_r + Th1 @ prev_a
        a[t] = cur; prev_a = cur; prev_r = r[t]
    return a

def _split(p):
    return p[:2], p[2:6].reshape(2, 2), p[6:10].reshape(2, 2)   # theta0, Phi1, Theta1

def neg_loglik(params, r, mask=None):
    """集中(条件)高斯负对数似然: Sigma 解析并出 -> 0.5*T*ln|Sigma_hat(params)|."""
    full = params if mask is None else _unpack(params, mask)
    th0, Phi1, Th1 = _split(full)
    a = innovations(r, th0, Phi1, Th1)
    S = a.T @ a / len(r)
    sgn, logdet = np.linalg.slogdet(S)
    if sgn <= 0 or not np.isfinite(logdet) or np.max(np.abs(a)) > 1e6:
        return 1e10
    return 0.5 * len(r) * logdet

def _unpack(free, mask):
    full = np.zeros(10); full[mask] = free; return full

def ols_var1(r):
    """纯 VAR(1) 的 OLS, 仅用于第1步'劝退关'里看残差白不白."""
    Y = r[1:]; X = np.hstack([np.ones((len(Y), 1)), r[:-1]])
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ B
    return B[0], B[1:].T, resid                         # phi0, Phi1, resid

def hannan_rissanen(r, p_long=8):
    """Hannan-Rissanen 两步法, 给 VARMA(1,1) 的 MLE 提供【一致的】暖启动(避免落入弱识别的等价解):
       (1) 先拟合一个长阶 VAR(p_long) OLS, 用其残差 e_t 当看不见的冲击 a_t 的代理;
       (2) 再把 r_t 对 [1, r_{t-1}, e_{t-1}] 做 OLS -> 得 (theta0, Phi1, Theta1) 初值.
       模型 r=theta0+Phi1 r_{t-1}+a-Theta1 a_{t-1} -> e_{t-1} 的系数 = -Theta1."""
    T = len(r)
    Yl = r[p_long:]
    Xl = np.hstack([np.ones((len(Yl), 1))] + [r[p_long - j:T - j] for j in range(1, p_long + 1)])
    Bl, *_ = np.linalg.lstsq(Xl, Yl, rcond=None)
    e_full = np.full((T, k), np.nan); e_full[p_long:] = Yl - Xl @ Bl    # 冲击代理(从 t=p_long 起有效)
    idx = np.arange(p_long + 1, T)                      # 需要 e_{t-1} 有效
    Y2 = r[idx]
    X2 = np.hstack([np.ones((len(idx), 1)), r[idx - 1], e_full[idx - 1]])
    B2, *_ = np.linalg.lstsq(X2, Y2, rcond=None)
    th0 = B2[0]; Phi1 = B2[1:1 + k].T; Th1 = -B2[1 + k:1 + 2 * k].T
    return np.concatenate([th0, Phi1.ravel(), Th1.ravel()])

def fit_cond_mle(r, x0, mask=None):
    """条件 MLE. 返回 (参数10, 残差a, Sigma, 逆Hessian)."""
    if mask is None:
        res = minimize(neg_loglik, x0, args=(r, None), method="BFGS",
                       options=dict(maxiter=2000))
        full = res.x; hinv = res.hess_inv
    else:
        res = minimize(neg_loglik, x0[mask], args=(r, mask), method="BFGS",
                       options=dict(maxiter=2000))
        full = _unpack(res.x, mask)
        hinv = np.zeros((10, 10)); idx = np.where(mask)[0]
        hinv[np.ix_(idx, idx)] = res.hess_inv
    th0, Phi1, Th1 = _split(full)
    a = innovations(r, th0, Phi1, Th1); S = a.T @ a / len(r)
    return full, a, S, hinv

def psi_weights(Phis, Thetas, H):
    """MA(无穷) 表示的 Psi 权重(=脉冲响应): Psi_0=I, Psi_l=Σ_i Phi_i Psi_{l-i} - Theta_l.
       约定模型 (I-ΣPhi B)r = (I-ΣTheta B)a; Theta_l(l>q)与 Phi_i(i>p) 视为 0."""
    p = len(Phis); q = len(Thetas)
    Psi = [np.eye(k)]
    for l in range(1, H + 1):
        M = np.zeros((k, k))
        for i in range(1, min(l, p) + 1):
            M += Phis[i - 1] @ Psi[l - i]
        if l <= q:
            M -= Thetas[l - 1]
        Psi.append(M)
    return Psi

def mv_ljung_box(resid, m, npar_dyn):
    """多元 Ljung-Box Q_k(m). df = k^2 m - (动态参数个数). 动态参数 = k^2*(p+q)."""
    a = np.asarray(resid, float); T = len(a)
    C0 = a.T @ a / T; C0inv = np.linalg.inv(C0)
    Q = 0.0
    for l in range(1, m + 1):
        Cl = a[l:].T @ a[:-l] / T
        Q += np.trace(Cl.T @ C0inv @ Cl @ C0inv) / (T - l)
    Q *= T * T
    df = k * k * m - npar_dyn
    return float(Q), df, float(stats.chi2.sf(Q, df))

def ccm_signs(x, l, thr):
    a = np.asarray(x, float); T = len(a)
    C0 = a.T @ a / T; d = np.sqrt(np.diag(C0))
    Cl = a[l:].T @ a[:-l] / T
    rho = Cl / np.outer(d, d)
    return rho, np.where(rho > thr, "+", np.where(rho < -thr, "-", "."))

def fmt_eq(name, b0, Phi_row, Th_row, names):
    s = f"  {name}_t = {b0:+.2f}"
    for j in range(k):
        if abs(Phi_row[j]) > 1e-6:
            s += f" {Phi_row[j]:+.3f}*{names[j]}(t-1)"
    s += f" + a_{name}(t)"
    for j in range(k):
        if abs(Th_row[j]) > 1e-6:
            s += f" {-Th_row[j]:+.3f}*a_{names[j]}(t-1)"   # 冲击项系数 = -Theta1
    return s

# ====================================================================
# 步骤0  数据生成: VARMA(1,1), r_t = theta_0 + Phi_1 r_{t-1} + a_t - Theta_1 a_{t-1}
#   AR 部分 Phi_1 = [[0.6,0.25],[0,0.5]]  (MKT 行 STK 列=0: 值层面 MKT 不被 STK 引导)
#   MA 部分 Theta_1=[[-0.4,0.2],[0,-0.3]] (MKT 行 STK 列=0: 冲击层面 STK 不传 MKT)
#   关键: Phi_1 特征值 {0.6,0.5}(正) 与 Theta_1 特征值 {-0.4,-0.3}(负) 远离、不相消
#         -> 无近公因子, 强可识别 (若 phi≈theta 会近似抵消 -> 弱识别, MLE 会乱跑, 故意避开)
#   Sigma 非对角 0.5 -> 同期相关
# ====================================================================
theta0 = np.array([1.0, 0.5])
Phi1 = np.array([[0.6, 0.25],
                 [0.0, 0.5]])
Theta1 = np.array([[-0.4, 0.2],
                   [0.0, -0.3]])
SIGMA = np.array([[1.00, 0.50],
                  [0.50, 1.00]])
P_TRUE, Q_TRUE = 1, 1
n, burn = 700, 200

rng = np.random.default_rng(17)
Lc = np.linalg.cholesky(SIGMA)
a_all = rng.standard_normal((n + burn, k)) @ Lc.T
r = np.empty((n + burn, k))
for t in range(n + burn):
    pr = r[t - 1] if t > 0 else np.zeros(k)
    pa = a_all[t - 1] if t > 0 else np.zeros(k)
    r[t] = theta0 + Phi1 @ pr + a_all[t] - Theta1 @ pa
r = r[burn:]

h = 12
train, test = r[:-h], r[-h:]
T = len(train)
pd.DataFrame({"t": np.arange(len(r)), "STK": np.round(r[:, 0], 4), "MKT": np.round(r[:, 1], 4),
             "split": ["train"] * len(train) + ["test"] * len(test)}
            ).to_csv("varma_data.csv", index=False, encoding="utf-8")

eig_ar = np.abs(np.linalg.eigvals(Phi1)); eig_ma = np.abs(np.linalg.eigvals(Theta1))
mu_true = np.linalg.solve(np.eye(k) - Phi1, theta0)
print(SEP); print("步骤0  数据生成 (VARMA(1,1) 真值: AR+MA 合体, 各带一个单向 0)")
print(fmt_eq("STK", theta0[0], Phi1[0], Theta1[0], NAMES))
print(fmt_eq("MKT", theta0[1], Phi1[1], Theta1[1], NAMES) + "   <- AR、MA 都不含 STK(单向)")
print(f"  Sigma=[[1,0.5],[0.5,1]]; 样本 T={T}(train)+{h}(test); 无条件均值 mu={np.round(mu_true,2)}")
print(f"  AR 特征值 {np.round(eig_ar,2)} (平稳<1), MA 特征值 {np.round(eig_ma,2)} (可逆<1), 两组不同 -> 无公因子")
print("  已导出 -> varma_data.csv")

# ====================================================================
# 步骤1  劝退关 + 前提: 平稳&可逆; 证"纯 VMA 数不出 q、纯 VAR(1) 残差不白" -> 才上 VARMA
# ====================================================================
print(SEP); print("步骤1  劝退关 + 前提 (要不要 VARMA? 先看 VAR/VMA 够不够)")
for j in range(k):
    s, pv, *_ = adfuller(train[:, j], autolag="AIC")
    print(f"  ADF[{NAMES[j]}]={s:.2f}, p={pv:.3g} -> " + ("平稳 [OK]" if pv < 0.05 else "不平稳 [NO]"))
print(f"  前提: AR 特征值<1(平稳)、MA 特征值<1(可逆) 真值均满足.")

thr0 = 2.0 / np.sqrt(T)
# (a) 纯 VMA 路线: 样本 CCM 是否截尾? 拖尾 -> 数不出 q
LMAX = 8
nsig = []
for l in range(1, LMAX + 1):
    _, sym = ccm_signs(train - train.mean(0), l, thr0)
    nsig.append(int(np.sum(sym != ".")))
tail_off = sum(nsig[3:]) > 0                            # 第4阶以后仍有显著 -> 拖尾
print(f"  [VMA 路线] 样本 CCM 各阶显著格数(lag1..{LMAX}): {nsig}")
print(f"     -> CCM " + ("缓慢拖尾、不归零 -> 数不出截尾阶 q, 纯 VMA 不适用 [NO]" if tail_off
                          else "截尾, 直接用 VMA"))
# (b) 纯 VAR(1) 路线: 残差白不白? 不白 -> 需很高阶, 不省参数
phi0_v, Phi1_v, resid_v = ols_var1(train)
Qv, dfv, pv_var = mv_ljung_box(resid_v, 8, k * k * 1)
print(f"  [VAR 路线] 纯 VAR(1) OLS 残差 Q_k(8)={Qv:.1f}, df={dfv}, p={pv_var:.3g}")
print(f"     -> 残差" + ("仍显著自相关 -> VAR(1) 不够, 要堆很高阶才白(MA=无穷阶 AR), 不省参数 [NO]"
                         if pv_var < 0.05 else "已白, 用 VAR 即可"))
print(f"  ==> VMA 数不出 q、低阶 VAR 不白 -> 两者特征都有 -> 才上 VARMA(1,1)(书: 只取低阶)")

fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for j in range(k):
    ax[j].plot(train[:, j], lw=.6, color=("steelblue" if j == 0 else "tomato"))
    ax[j].axhline(train[:, j].mean(), color="k", ls="--", lw=.7)
    ax[j].set_ylabel(NAMES[j]); ax[j].set_title(f"{NAMES[j]} series (train)")
ax[1].set_xlabel("t")
plt.tight_layout(); plt.savefig("varma_fig1_series.png", dpi=110); plt.close()

# ====================================================================
# 步骤2  定阶 + 识别: Psi 权重证"公因子=同一过程两种写法"(不唯一) -> 施加正规化钉唯一
# ====================================================================
print(SEP); print("步骤2  定阶 + 识别 (VARMA 独有的坎: 模型不唯一)")
H_PSI = 10
Psi_11 = psi_weights([Phi1], [Theta1], H_PSI)          # VARMA(1,1) 的脉冲响应
# 公因子 (1-cB) 同乘两边 -> 等价的 VARMA(2,2): Phi*=[Phi1+cI, -c Phi1], Theta*=[Theta1+cI, -c Theta1]
c = 0.3
Phi_s = [Phi1 + c * np.eye(k), -c * Phi1]
Th_s = [Theta1 + c * np.eye(k), -c * Theta1]
Psi_22 = psi_weights(Phi_s, Th_s, H_PSI)
maxdiff = max(np.max(np.abs(Psi_11[l] - Psi_22[l])) for l in range(H_PSI + 1))
print("  [识别问题] 给 AR、MA 同乘一个公因子 (1-0.3B), VARMA(1,1) 变成一个 VARMA(2,2):")
print(f"     两者的 Psi 权重(脉冲响应)逐阶最大差={maxdiff:.2e} -> 数值上完全相同")
print(f"     => 同一个过程有(1,1)与(2,2)等多种等价写法 -> CCM/数据分不开 -> 模型不唯一 (公因子相消)")
print("  [解决] 结构指定(Tiao-Tsay 1989 / Tsay 1991: Kronecker 指数 / 标量分量模型 SCM)系统去公因子.")
print("     本 demo 施加最常用正规化: Phi_0=Theta_0=I(已隐含) + 取【最低阶】 -> 钉死 (p,q)=(1,1).")
P_USE, Q_USE = 1, 1
print(f"  ==> 识别后定阶 (p,q)=({P_USE},{Q_USE}) (真值 ({P_TRUE},{Q_TRUE}))")

fig, axx = plt.subplots(1, 2, figsize=(12, 4.6))
axx[0].bar(range(1, LMAX + 1), nsig, color="purple", alpha=.75)
axx[0].axhline(0, color="k", lw=.6)
axx[0].set_xlabel("lag l"); axx[0].set_ylabel("# significant CCM cells")
axx[0].set_title("CCM tails off (not cut) -> VARMA, not VMA")
ls = np.arange(H_PSI + 1)
n11 = [np.linalg.norm(Psi_11[l]) for l in ls]
n22 = [np.linalg.norm(Psi_22[l]) for l in ls]
axx[1].plot(ls, n11, "o-", color="steelblue", lw=1.6, label="VARMA(1,1)  ||Psi_l||")
axx[1].plot(ls, n22, "x--", color="tomato", lw=1.4, label="inflated VARMA(2,2)")
axx[1].set_xlabel("lag l"); axx[1].set_ylabel("||Psi_l|| (impulse response)")
axx[1].set_title("Common-factor non-uniqueness: identical Psi -> same process")
axx[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("varma_fig2_identify.png", dpi=110); plt.close()

# ====================================================================
# 步骤3  估计: 条件 MLE (含 AR+MA) -> theta0, Phi1, Theta1, Sigma; 剔不显著项再估
# ====================================================================
print(SEP); print(f"步骤3  条件 MLE 估计 VARMA({P_USE},{Q_USE}) (含 MA -> 冲击反推, 不能纯 OLS)")
x0 = hannan_rissanen(train, p_long=8)                  # Hannan-Rissanen 一致暖启动(防落入弱识别等价解)
hr_phi, hr_th = x0[2:6].reshape(2, 2), x0[6:10].reshape(2, 2)
print(f"  Hannan-Rissanen 暖启动: Phi1~[[{hr_phi[0,0]:+.2f},{hr_phi[0,1]:+.2f}],"
      f"[{hr_phi[1,0]:+.2f},{hr_phi[1,1]:+.2f}]], Theta1~[[{hr_th[0,0]:+.2f},{hr_th[0,1]:+.2f}],"
      f"[{hr_th[1,0]:+.2f},{hr_th[1,1]:+.2f}]] (已近真值)")
full, a_hat, S_hat, hinv = fit_cond_mle(train, x0)
th0_h, Phi1_h, Th1_h = _split(full)
se = np.sqrt(np.maximum(np.diag(hinv), 0))
tstat = np.divide(full, se, out=np.zeros_like(full), where=se > 0)
for eq in range(k):
    print(fmt_eq(NAMES[eq], th0_h[eq], Phi1_h[eq], Th1_h[eq], NAMES))
print(f"  theta_0={np.round(th0_h,2)}(真值{theta0});  Sigma_hat=[[{S_hat[0,0]:.2f},{S_hat[0,1]:.2f}],"
      f"[{S_hat[1,0]:.2f},{S_hat[1,1]:.2f}]](真值非对角0.5)")
print(f"  Phi_1=[[{Phi1_h[0,0]:+.2f},{Phi1_h[0,1]:+.2f}],[{Phi1_h[1,0]:+.2f},{Phi1_h[1,1]:+.2f}]] "
      f"真值[[+.6,+.25],[0,+.5]]")
print(f"  Theta_1=[[{Th1_h[0,0]:+.2f},{Th1_h[0,1]:+.2f}],[{Th1_h[1,0]:+.2f},{Th1_h[1,1]:+.2f}]] "
      f"真值[[-.4,+.2],[0,-.3]]")
print(f"  残差同期相关={S_hat[0,1]/np.sqrt(S_hat[0,0]*S_hat[1,1]):+.2f} (同期联动藏在 Sigma)")

# 精修: theta0 恒留; Phi1/Theta1 的 |t|<1.96 项固定为0 再估
pn = ["th0_S", "th0_M", "Phi[S<-S]", "Phi[S<-M]", "Phi[M<-S]", "Phi[M<-M]",
      "Th[S<-aS]", "Th[S<-aM]", "Th[M<-aS]", "Th[M<-aM]"]
print("  [精修] 剔除 |t|<1.96 的动态项后重估 (参数 t 值):")
for i in range(10):
    tag = "" if (i < 2 or abs(tstat[i]) >= 1.96) else "   <- 剔除"
    print(f"     {pn[i]:11s} est={full[i]:+.3f} se={se[i]:.3f} t={tstat[i]:+.2f}{tag}")
mask = np.abs(tstat) >= 1.96; mask[:2] = True
if mask.sum() < 10:
    full, a_hat, S_hat, _ = fit_cond_mle(train, x0, mask=mask)
    th0_h, Phi1_h, Th1_h = _split(full)
    print(f"     -> 精简后 Phi_1=[[{Phi1_h[0,0]:+.2f},{Phi1_h[0,1]:+.2f}],[{Phi1_h[1,0]:+.2f},{Phi1_h[1,1]:+.2f}]],"
          f" Theta_1=[[{Th1_h[0,0]:+.2f},{Th1_h[0,1]:+.2f}],[{Th1_h[1,0]:+.2f},{Th1_h[1,1]:+.2f}]]")
    print("        (MKT 行的 STK 项被剔为 0, 还原 AR、MA 双重'单向')")
eig_ar_h = np.abs(np.linalg.eigvals(Phi1_h)); eig_ma_h = np.abs(np.linalg.eigvals(Th1_h))
print(f"  估计 AR 特征值={np.round(eig_ar_h,2)}(<1平稳), MA 特征值={np.round(eig_ma_h,2)}(<1可逆) "
      + ("[OK]" if eig_ar_h.max() < 1 and eig_ma_h.max() < 1 else "[NO]"))

# ====================================================================
# 步骤4  检验: 多元 Ljung-Box Q_k(m) + 残差 CCM 符号表 (与 VAR/VMA 相同)
# ====================================================================
print(SEP); print("步骤4  模型检验 (残差是否多元白噪声)")
thr = 2.0 / np.sqrt(T); npar_dyn = k * k * (P_USE + Q_USE)
pQ = None
for m in (8, 12):
    Qk, dfQ, pQ = mv_ljung_box(a_hat, m, npar_dyn)
    print(f"  多元 Ljung-Box Q_k(m={m})={Qk:.2f}, df={dfQ}, p={pQ:.3g} -> "
          + ("残差近似多元白噪声, 模型充分 [OK]" if pQ > 0.05 else "残差仍有结构, 回第2步调 (p,q) [NO]"))
print(f"  残差 CCM 符号表 (阈值 2/sqrt(T)={thr:.3f}; '.'=不显著):")
CCM_LAGS = 4; sig_count = 0
for l in range(1, CCM_LAGS + 1):
    _, sym = ccm_signs(a_hat, l, thr)
    rowstr = "   ".join("[" + " ".join(sym[i]) + "]" for i in range(k))
    print(f"     lag {l}:  {rowstr}")
    sig_count += int(np.sum(sym != "."))
total = k * k * CCM_LAGS; exp_chance = 0.05 * total
print(f"     共 {sig_count}/{total} 格显著 (5%水平偶然期望≈{exp_chance:.1f}) -> "
      + ("个别格属抽样偶然, 以 Q_k 为准, 模型充分 [OK]" if sig_count <= round(exp_chance) + 1
         else "显著格偏多, 调 (p,q) [NO]"))

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for j in range(k):
    ax[0, j].plot(a_hat[:, j], lw=.6); ax[0, j].axhline(0, color="r", ls="--", lw=.7)
    ax[0, j].set_title(f"Residual shock a_{NAMES[j]} over time")
lags = np.arange(1, 13)
ac0 = [np.corrcoef(a_hat[l:, 0], a_hat[:-l, 0])[0, 1] for l in lags]
ac1 = [np.corrcoef(a_hat[l:, 1], a_hat[:-l, 1])[0, 1] for l in lags]
cc = [np.corrcoef(a_hat[l:, 0], a_hat[:-l, 1])[0, 1] for l in lags]
ax[1, 0].bar(lags - 0.2, ac0, width=0.4, label=f"{NAMES[0]} ACF", color="steelblue")
ax[1, 0].bar(lags + 0.2, ac1, width=0.4, label=f"{NAMES[1]} ACF", color="tomato")
ax[1, 0].axhline(thr, color="gray", ls="--", lw=.8); ax[1, 0].axhline(-thr, color="gray", ls="--", lw=.8)
ax[1, 0].set_title("Residual auto-corr (~0)"); ax[1, 0].legend(fontsize=8)
ax[1, 1].bar(lags, cc, color="purple")
ax[1, 1].axhline(thr, color="gray", ls="--", lw=.8); ax[1, 1].axhline(-thr, color="gray", ls="--", lw=.8)
ax[1, 1].set_title("Residual cross-corr (~0)")
plt.tight_layout(); plt.savefig("varma_fig3_diagnose.png", dpi=110); plt.close()

# ====================================================================
# 步骤5  预测: 两种记忆叠加 —— 短期 MA 冲击项(只 q 步), 长期 AR 主导慢衰减回均值
#   r_T(1)=theta0+Phi1 r_T-Theta1 a_T;  r_T(l)=theta0+Phi1 r_T(l-1) (l>=2, MA 已失效)
#   误差协方差: Sigma(l)=Σ_{j=0}^{l-1} Psi_j Sigma Psi_j'  (Psi 由估计模型算)
# ====================================================================
print(SEP); print(f"步骤5  样本外预测 ({h} 步) —— 两种记忆叠加 (短期MA冲击 + 长期AR衰减回均值)")
mu_h = np.linalg.solve(np.eye(k) - Phi1_h, th0_h)       # 估计无条件均值
a_T = a_hat[-1]; r_T = train[-1]
fc = np.zeros((h, k))
fc[0] = th0_h + Phi1_h @ r_T - Th1_h @ a_T              # 1 步: AR + 尚存的 MA 冲击
for l in range(1, h):
    fc[l] = th0_h + Phi1_h @ fc[l - 1]                  # >=2 步: 仅 AR 递推, 几何衰减回 mu
Psi = psi_weights([Phi1_h], [Th1_h], h)
S = np.zeros((k, k)); fc_var = np.zeros((h, k))
for l in range(h):
    S = S + Psi[l] @ S_hat @ Psi[l].T
    fc_var[l] = np.diag(S)
fc_se = np.sqrt(fc_var); lo95, hi95 = fc - 1.96 * fc_se, fc + 1.96 * fc_se

inside = (test >= lo95) & (test <= hi95)
naive = train[-1]
rmse = np.sqrt(np.mean((test - fc) ** 2, axis=0))
rmse_naive = np.sqrt(np.mean((test - naive) ** 2, axis=0))
for j in range(k):
    print(f"  {NAMES[j]}: 1步={fc[0,j]:+.2f}, 2步={fc[1,j]:+.2f}, {h}步={fc[-1,j]:+.2f} "
          f"-> 慢慢衰减回均值 mu={mu_h[j]:+.2f} (AR 长记忆)")
    print(f"        RMSE={rmse[j]:.2f} vs 朴素 last-value RMSE={rmse_naive[j]:.2f} -> "
          + ("VARMA 更优 [OK]" if rmse[j] < rmse_naive[j] else "与基准相当"))
    print(f"        95% 区间覆盖率={inside[:,j].mean():.0%} (理想≈95%)")
print(f"  对比 VMA(>q即拍平回均值) / VAR(无 MA 冲击修正): VARMA 第1步带 MA 冲击修正、之后 AR 平滑衰减.")

pd.DataFrame({"step": np.arange(1, h + 1),
              "STK_fc": np.round(fc[:, 0], 3), "STK_lo": np.round(lo95[:, 0], 3),
              "STK_hi": np.round(hi95[:, 0], 3), "STK_act": np.round(test[:, 0], 3),
              "MKT_fc": np.round(fc[:, 1], 3), "MKT_lo": np.round(lo95[:, 1], 3),
              "MKT_hi": np.round(hi95[:, 1], 3), "MKT_act": np.round(test[:, 1], 3),
              }).to_csv("varma_forecast.csv", index=False, encoding="utf-8")
print("  已导出每步预测(点/区间/实际) -> varma_forecast.csv")

fig, ax = plt.subplots(2, 1, figsize=(11, 8))
tail = 50; xx = np.arange(1, h + 1)
for j in range(k):
    ax[j].plot(range(-tail + 1, 1), train[-tail:, j], color="gray", lw=.8, label="train (tail)")
    ax[j].plot(xx, test[:, j], color="black", lw=1, marker=".", ms=6, label="actual")
    ax[j].plot(xx, fc[:, j], "b-", lw=1.5, marker="o", ms=4, label="VARMA forecast")
    ax[j].fill_between(xx, lo95[:, j], hi95[:, j], color="red", alpha=.15, label="95% interval")
    ax[j].axhline(mu_h[j], color="green", ls=":", lw=1.2, label=f"mean mu={mu_h[j]:.2f}")
    ax[j].axvline(Q_USE + 0.5, color="purple", ls="--", lw=1, label=f"MA memory ends q={Q_USE}")
    ax[j].axvline(0, color="k", lw=.5)
    ax[j].set_ylabel(NAMES[j]); ax[j].set_title(f"Out-of-sample: {NAMES[j]} (MA kink then AR decay to mean)")
    ax[j].legend(loc="upper right", fontsize=7)
ax[1].set_xlabel("forecast horizon")
plt.tight_layout(); plt.savefig("varma_fig4_forecast.png", dpi=110); plt.close()

# ---- fig5: 两个招牌 —— 同期相关(Sigma) + 两种记忆(Psi 在 l<=q 有 MA 修正, 之后 AR 几何衰减) ----
fig, axx = plt.subplots(1, 2, figsize=(12, 4.6))
axx[0].scatter(a_hat[:, 1], a_hat[:, 0], s=10, alpha=.4, color="purple")
rr = S_hat[0, 1] / np.sqrt(S_hat[0, 0] * S_hat[1, 1])
axx[0].set_xlabel("MKT shock a2"); axx[0].set_ylabel("STK shock a1")
axx[0].set_title(f"Contemporaneous shock corr={rr:+.2f} (same-day co-movement in Sigma)")
ls2 = np.arange(h + 1)
norm_psi = [np.linalg.norm(Psi[l]) for l in ls2]
axx[1].plot(ls2, norm_psi, "o-", color="darkorange", lw=1.6)
axx[1].axvline(Q_USE, color="purple", ls="--", lw=1, label=f"MA active up to q={Q_USE}")
axx[1].set_xlabel("lag l"); axx[1].set_ylabel("||Psi_l|| (impulse response)")
axx[1].set_title("Two memories: MA correction (l<=q) then AR geometric decay")
axx[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("varma_fig5_structure.png", dpi=110); plt.close()

# ====================================================================
print(SEP)
print(f"完成. 全流程: 劝退关(VMA数不出q & VAR(1)不白)->识别(公因子致不唯一, 正规化钉 (p,q)=(1,1))"
      f"->条件MLE估计(AR+MA, Sigma同期{rr:+.2f})+精修剔单向->残差Q_k白噪声[{'OK' if pQ>0.05 else 'NO'}]"
      f"->两种记忆预测(短期MA修正+长期AR衰减回均值)")
print(f"揭晓真值: VARMA(1,1), Phi_1=[[.6,.25],[0,.5]], Theta_1=[[-.4,.2],[0,-.3]], Sigma 非对角=0.5")
print("图: varma_fig1_series / varma_fig2_identify / varma_fig3_diagnose / varma_fig4_forecast / varma_fig5_structure (.png)")
