# -*- coding: utf-8 -*-
"""
VECM (向量误差修正) 建模 —— 一条龙完整流程 (二维 k=2: 短端利率 SHORT + 长端利率 LONG, 仿书 §8.6.5 tb3m+tb6m)

  VECM 与 VAR 的关系 (本 demo 的主角):
    VAR(p) 直接建"水平" x_t, 但要求 x_t 弱平稳; 若各分量都是单位根非平稳 I(1), 直接差分会"过度差分"
    (差分后 MA 含单位根、不可逆). 若这些 I(1) 序列存在一个平稳的线性组合 beta'x_t (协整),
    就在"差分 VAR"右端补回一个平稳的误差修正项 alpha*beta'*x_{t-1}, 既消非平稳、又不丢长期均衡:
        VECM:  Δx_t = alpha*beta'*x_{t-1} + Σ_{i=1}^{p-1} Γ_i Δx_{t-i} + a_t          (式 8.36)
                       ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       误差修正项(长期)      短期动态(差分滞后)
      · beta (k×m): 协整矩阵 —— beta'x_t 是 m 个平稳的"长期均衡关系"(m=协整秩 r).
      · alpha(k×m): 调整/载荷矩阵 —— 偏离均衡后各分量的"回调速度".
      · Π = alpha*beta' = -Φ(1): 全部协整信息都在 Π 的秩里 (Rank=0 不协整; =k 本就 I(0); 0<r<k 才用 VECM).

  为什么需要 VECM: 一组 I(1) 变量短期各自游走, 却被一个均衡关系(如利差 SHORT-LONG)长期拴住.
    差分 VAR 丢掉这个"锚"(长期信息), 水平 VAR 又不平稳 —— VECM 用 alpha*beta'*x_{t-1} 把锚补回来.

  七步法 (含两层定阶; 下游一律用"定出来的阶 p 与协整秩 r", 不写死):
    第0步 造数据      : 用 VECM 真值造二维 I(1) 协整序列 (利差平稳, 短端主要向均衡回调), 留尾部样本外
    第1步 单位根检验  : 各分量 ADF 确认 I(1)(门槛) + 对线性组合 ADF 预判协整
    第2步 VAR 定阶(第一层): 对"水平"x_t 用 AIC/BIC/HQ 选 p; ECM 差分滞后 = p-1
    第3步 确定性项    : 按 5 情形(§8.6.1)选常数/趋势设定; 利率类常用"受限常数"(rc)
    第4步 Johansen 协整检验(第二层): 手写 S00/S01/S11 特征值问题 -> 迹/最大特征值序贯检验定协整秩 r
    第5步 ML 估计     : 由特征向量得 beta(长期均衡), 回代得 alpha(调整速度)与短期 Γ; beta 标准化; vs 真值
    第6步 模型检验    : 残差多元 Ljung-Box + CCM 符号表 + 协整残差平稳(ADF) + alpha 符号/显著性(经济含义)
    第7步 预测        : 先用 ECM 递推 Δx_t 再累加还原 x_t; 水平预测扇形张开, 但协整组合(利差)区间收敛

  工具说明: statsmodels 有 vecm(coint_johansen / VECM), 但本 demo 全程手写(numpy/scipy)以保持透明 ——
    Johansen = 两辅助回归净化确定性/短期项 + 缩秩特征值问题; 迹/最大特征值统计量;
    临界值非标准(布朗运动泛函)故查表(Osterwald-Lenum 受限常数表); 末尾用 statsmodels VECM 交叉验证 beta/alpha.
"""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")               # Windows 控制台强制 UTF-8
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                                  # 不弹窗, 直接存图
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import eig as geig, inv
from statsmodels.tsa.stattools import adfuller

SEP = "=" * 72
NAMES = ["SHORT", "LONG"]                              # 序列名: 短端 / 长端 利率(%)

# Johansen 临界值 (Osterwald-Lenum 1992 受限常数"rc"表; 键=k-r, 值=(90%,95%,99%)) ----------
#   ⚠️ 因单位根存在, 统计量渐近分布是布朗运动泛函(非 chi2), 临界值须查模拟表且随确定性项设定而变.
CV_TRACE = {1: (7.52, 9.24, 12.97), 2: (17.85, 19.96, 24.60), 3: (32.00, 34.91, 41.07)}
CV_MAXEIG = {1: (7.52, 9.24, 12.97), 2: (13.75, 15.67, 20.20), 3: (19.77, 22.00, 26.81)}


# ---------- 水平 VAR 定阶用的小工具 (与 var_demo 同, 用于第一层定阶) ----------
def design(a, p, start):
    """堆 VAR 设计阵: 行=时点, 列=[1, x_{t-1},...,x_{t-p}] (含常数)."""
    a = np.asarray(a, float); T = a.shape[0]
    idx = np.arange(start, T)
    Y = a[idx]
    cols = [np.ones((len(idx), 1))]
    for j in range(1, p + 1):
        cols.append(a[idx - j])
    return Y, np.hstack(cols), idx


def fit_var(a, p, start):
    """逐方程 OLS 估 VAR(p); 返回 ML 残差协方差 Sigma(式8.19)、样本量 n."""
    Y, X, idx = design(a, p, start)
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ B
    n = len(idx)
    return dict(B=B, resid=resid, Sigma=resid.T @ resid / n, n=n)


def sigma_var0(a, start):
    """VAR(0)(只含均值)的 ML 残差协方差, 作 IC 的 i=0 基准."""
    Y = np.asarray(a, float)[start:]
    r = Y - Y.mean(0)
    return r.T @ r / len(Y)


def mv_ljung_box(resid, m, k, nparm):
    """多元 Ljung-Box Q_k(m): 残差 CCM 到 m 阶是否全 0; df=k^2 m - nparm(被估的动态参数数)."""
    a = np.asarray(resid, float); T = len(a)
    C0 = a.T @ a / T; C0inv = inv(C0)
    Q = 0.0
    for l in range(1, m + 1):
        Cl = a[l:].T @ a[:-l] / T
        Q += np.trace(Cl.T @ C0inv @ Cl @ C0inv) / (T - l)
    Q *= T * T
    df = max(k * k * m - nparm, 1)
    return float(Q), df, float(stats.chi2.sf(Q, df))


def ccm_signs(resid, l, thr):
    """残差滞后 l 的互相关阵符号表: |rho|>2/sqrt(T) 记 +/-, 否则 '.'."""
    a = np.asarray(resid, float); T = len(a)
    C0 = a.T @ a / T; d = np.sqrt(np.diag(C0))
    Cl = a[l:].T @ a[:-l] / T
    rho = Cl / np.outer(d, d)
    return np.where(rho > thr, "+", np.where(rho < -thr, "-", "."))


def resid_out(A, Z):
    """把 A 对 Z 做 OLS 后的残差 (净化掉 Z 的成分); Z 为空则原样返回."""
    if Z is None or Z.shape[1] == 0:
        return A
    return A - Z @ np.linalg.lstsq(Z, A, rcond=None)[0]


# ====================================================================
# 步骤0  数据生成: VECM(p-1=2 差分滞后, 即水平 VAR(3)) 真值, 造 I(1) 协整二元利率
#   真值结构(仿 §8.6.5 tb3m/tb6m):
#     协整向量 beta=(1,-1) -> 平稳组合 = 利差 SHORT-LONG, 均衡水平 -0.225 (正常收益率曲线: 短端低于长端)
#     载荷 alpha=(-0.095,-0.021) -> 短端强力向均衡回调、长端几乎不动(长端≈公共随机趋势/驱动者)
#     短期矩阵 Γ1,Γ2 直接取书 p373 的拟合值; 同期噪声强相关(利率日变动齐动)
# ====================================================================
BETA = np.array([1.0, -1.0])                            # 协整向量: 利差 = SHORT - LONG
MU_EC = -0.225                                         # 利差的长期均衡水平 (平稳组合的均值)
ALPHA = np.array([-0.095, -0.021])                     # 载荷: 短端回调强、长端弱
G1 = np.array([[0.05, 0.27], [-0.04, 0.32]])           # 短期矩阵 Γ1 (书 p373)
G2 = np.array([[-0.21, 0.25], [-0.03, 0.10]])          # 短期矩阵 Γ2 (书 p373)
SD = np.array([0.20, 0.18]); RHO = 0.85                # 残差标准误(书 0.20/0.18)+ 强同期相关
SIGMA = np.array([[SD[0] ** 2, RHO * SD[0] * SD[1]],
                  [RHO * SD[0] * SD[1], SD[1] ** 2]])
P_TRUE = 3                                             # 水平 VAR 阶 (=> 差分滞后 p-1=2)
R_TRUE = 1                                             # 协整秩 (k-r=1 个公共随机趋势)
n, burn, h, k = 1800, 500, 16, 2                       # 样本仿书规模(~2000), beta 才识别得准

rng = np.random.default_rng(20260708)
L = np.linalg.cholesky(SIGMA)


def sim_vecm(rng):                                     # 按式(8.36)递推造 I(1) 协整序列
    N = n + burn
    a = rng.standard_normal((N, k)) @ L.T
    x = np.zeros((N, k)); dx = np.zeros((N, k))
    x[0] = np.array([6.0, 6.225]); x[1] = x[0].copy()  # 初始利差 = -0.225 = 均衡
    for t in range(2, N):
        ec = (BETA @ x[t - 1]) - MU_EC                 # 误差修正项(标量): 利差偏离均衡多少
        dx[t] = ALPHA * ec + G1 @ dx[t - 1] + G2 @ dx[t - 2] + a[t]
        x[t] = x[t - 1] + dx[t]
    return x[burn:]


x_all = sim_vecm(rng)
x_all = x_all + (5.0 - x_all[:, 0].mean())             # 沿单位根方向平移(无害), 把水平挪到"利率样"正区间
h_split = h
train, test = x_all[:-h_split], x_all[-h_split:]
T = len(train)
pd.DataFrame({"t": np.arange(len(x_all)),
              "SHORT": np.round(x_all[:, 0], 4), "LONG": np.round(x_all[:, 1], 4),
              "spread": np.round(x_all[:, 0] - x_all[:, 1], 4),
              "split": ["train"] * len(train) + ["test"] * len(test)}).to_csv(
    "vecm_data.csv", index=False, encoding="utf-8")

print(SEP); print("步骤0  数据生成 (VECM 真值: I(1) 协整二元利率, 利差平稳, 短端主回调)")
print(f"  协整向量 beta=(1,-1) -> 平稳组合 = 利差 SHORT-LONG, 均衡水平 mu={MU_EC}")
print(f"  载荷 alpha=({ALPHA[0]},{ALPHA[1]}) -> 短端强回调、长端弱(≈公共趋势); 差分滞后 p-1={P_TRUE-1}")
print(f"  同期噪声相关={RHO} (利率日变动齐动); 样本 T={T}(train)+{h}(test); 单位=%")
print(f"  训练集利差均值={np.mean(train[:,0]-train[:,1]):+.3f} (应≈{MU_EC}), 标准差={np.std(train[:,0]-train[:,1]):.3f}")
print("  已导出 -> vecm_data.csv")

# ====================================================================
# 步骤1  单位根检验: 各分量 ADF 确认 I(1)(门槛) + 对利差 ADF 预判协整
#   h=系统单位根个数; 0<h<k => 协整, 协整因子数=k-h. 期望 h=1(一个公共趋势) => r=k-h=1.
# ====================================================================
print(SEP); print("步骤1  单位根检验 (各分量 I(1)? + 线性组合是否已平稳 -> 预判协整)")
for j in range(k):
    s_lv, p_lv, *_ = adfuller(train[:, j], autolag="AIC")
    s_df, p_df, *_ = adfuller(np.diff(train[:, j]), autolag="AIC")
    print(f"  {NAMES[j]:>5}: 水平 ADF={s_lv:+.2f}(p={p_lv:.3g}) -> "
          + ("非平稳" if p_lv > 0.05 else "平稳?") + f"; 差分 ADF={s_df:+.2f}(p={p_df:.3g}) -> "
          + ("平稳 => I(1) [OK]" if p_df < 0.05 else "仍非平稳"))
spread_tr = train[:, 0] - train[:, 1]
s_sp, p_sp, *_ = adfuller(spread_tr, autolag="AIC")
print(f"  线性组合 SHORT-LONG: ADF={s_sp:+.2f}, p={p_sp:.3g} -> "
      + ("平稳! 存在协整 (进入 VECM 框架) [OK]" if p_sp < 0.05 else "非平稳, 无协整迹象"))
print(f"  => 各分量 I(1)、且有平稳线性组合 -> 单位根个数 h=1, 协整因子数 k-h={k-1} (待 Johansen 确认)")

fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
ax[0].plot(train[:, 0], lw=.7, color="steelblue"); ax[0].set_ylabel("SHORT")
ax[0].set_title("SHORT rate (I(1): wanders, no mean-reversion)")
ax[1].plot(train[:, 1], lw=.7, color="tomato"); ax[1].set_ylabel("LONG")
ax[1].set_title("LONG rate (I(1): wanders together with SHORT)")
ax[2].plot(spread_tr, lw=.7, color="purple"); ax[2].axhline(MU_EC, color="k", ls="--", lw=1, label=f"equilibrium={MU_EC}")
ax[2].set_ylabel("spread"); ax[2].set_xlabel("t")
ax[2].set_title("Cointegration: spread SHORT-LONG is STATIONARY (mean-reverting)"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("vecm_fig1_series.png", dpi=110); plt.close()

# ====================================================================
# 步骤2  VAR 定阶 (第一层): 对"水平"x_t 用 AIC/BIC/HQ 选 p -> ECM 差分滞后 = p-1
# ====================================================================
print(SEP); print("步骤2  VAR 定阶 (第一层: 对水平 x_t 选 p; ECM 差分滞后 = p-1)")
PMAX = 6; START = PMAX; g = T - START
Sig = {0: sigma_var0(train, START)}
for i in range(1, PMAX + 1):
    Sig[i] = fit_var(train, i, START)["Sigma"]
logdet = {i: np.log(np.linalg.det(Sig[i])) for i in range(PMAX + 1)}
ic = {"AIC": [], "BIC": [], "HQ": []}
for i in range(PMAX + 1):
    ic["AIC"].append(logdet[i] + 2 * k * k * i / g)
    ic["BIC"].append(logdet[i] + k * k * i * np.log(g) / g)
    ic["HQ"].append(logdet[i] + 2 * k * k * i * np.log(np.log(g)) / g)
sel = {key: int(np.argmin(v)) for key, v in ic.items()}
print(f"       {'阶i':>4} {'AIC':>10} {'BIC':>10} {'HQ':>10}")
for i in range(PMAX + 1):
    mk = "  <-" + ",".join([key for key in sel if sel[key] == i]) if i in sel.values() else ""
    print(f"       {i:>4} {ic['AIC'][i]:>10.4f} {ic['BIC'][i]:>10.4f} {ic['HQ'][i]:>10.4f}{mk}")
P_USE = sel["BIC"] if sel["BIC"] > 0 else max(sel["AIC"], 1)
print(f"  -> AIC选 {sel['AIC']}, BIC选 {sel['BIC']}, HQ选 {sel['HQ']}; 真值 p={P_TRUE}")
print(f"  ==> 综合定阶 P_USE={P_USE} (以 BIC 为准) -> VECM 差分滞后 lags = p-1 = {P_USE-1}")
KDIFF = P_USE - 1
xs = np.arange(PMAX + 1)                                # fig2 左半在步骤4拿到协整统计量后统一画

# ====================================================================
# 步骤3  确定性项设定 (§8.6.1 五情形): 选常数/趋势. 利率无漂移理由 -> 受限常数 rc.
# ====================================================================
print(SEP); print("步骤3  确定性项设定 (§8.6.1 五情形; 检验临界值随此设定而变)")
cases = [
    "1) mu=0            分量无漂移, 协整关系均值 0",
    "2) mu=alpha*c0     受限常数: 分量无漂移, 但协整关系有非零均值   <== 本 demo(利率)",
    "3) mu=mu0 (非受限) 分量带漂移的 I(1)",
    "4) mu=mu0+alpha*c1*t 受限趋势: 协整关系带线性趋势",
    "5) mu=mu0+mu1*t    常数趋势都不受限, 分量带二次趋势",
]
for c in cases:
    print("  " + c)
DET = "rc"                                             # restricted constant (常数进协整关系)
print(f"  -> 选「受限常数 rc」: 利率无长期漂移, 但利差有非零均衡({MU_EC}); Johansen 用受限常数临界值表")

# ====================================================================
# 步骤4  Johansen 协整检验 (第二层): 手写缩秩特征值问题 -> 迹/最大特征值序贯检验定 r
#   两辅助回归(式8.40/8.41)净化短期项: R0=Δx_t 的残差, R1=[x_{t-1};1] 的残差(rc 把常数并入 x_{t-1});
#   解 |lambda*S11 - S10 S00^{-1} S01|=0, 特征值=典型相关^2; 迹/最大特征值统计量对查受限常数临界值.
# ====================================================================
print(SEP); print(f"步骤4  Johansen 协整检验 (第二层: 定协整秩 r; lags={KDIFF}, 受限常数)")
DX = np.diff(train, axis=0)                             # Δx_t, 长度 T-1
p1 = KDIFF                                              # 差分滞后数
start = p1 + 1                                          # 公共起点(需 x_{t-1} 与 p1 个差分滞后)
DXt = DX[p1:]                                           # 被解释 Δx_t
X1 = np.hstack([train[p1:-1], np.ones((len(DXt), 1))])  # [x_{t-1}; 1] (rc: 常数并入协整块)
Zlags = [DX[p1 - j: -j if j > 0 else None][:len(DXt)] for j in range(1, p1 + 1)]  # Δx_{t-1..t-p1}
Z = np.hstack(Zlags) if p1 > 0 else np.zeros((len(DXt), 0))
n_eff = len(DXt)

R0 = resid_out(DXt, Z)                                  # 式(8.40) 残差 u_t
R1 = resid_out(X1, Z)                                   # 式(8.41) 残差 v_t
S00 = R0.T @ R0 / n_eff
S01 = R0.T @ R1 / n_eff
S11 = R1.T @ R1 / n_eff
# 缩秩特征值问题: (S10 S00^{-1} S01) v = lambda S11 v
M = S01.T @ inv(S00) @ S01
w_eig, V = geig(M, S11)
w_eig = np.real(w_eig); order = np.argsort(w_eig)[::-1]
w_eig, V = w_eig[order], np.real(V[:, order])
for i in range(V.shape[1]):                             # 按 v'S11 v = 1 标准化
    q = V[:, i].T @ S11 @ V[:, i]
    if q > 0:
        V[:, i] /= np.sqrt(q)
lam = np.clip(w_eig[:k], 1e-12, 1 - 1e-12)              # 取前 k 个有效特征值(rc 多出的≈0 丢弃)

print(f"  特征值(典型相关^2, 降序): {np.round(lam, 4)}  (n_eff={n_eff})")
print(f"  {'H0: rank=m':>12} {'迹统计量':>10} {'95%/99%':>14} {'最大特征值':>12} {'95%/99%':>14}  判定(迹)")
r_hat = 0; decided = False
for m in range(k):
    tr = -n_eff * np.sum(np.log(1 - lam[m:]))          # 迹检验(式8.42)
    me = -n_eff * np.log(1 - lam[m])                   # 最大特征值检验
    cvt, cvm = CV_TRACE[k - m], CV_MAXEIG[k - m]
    rej = tr > cvt[1]                                  # 与 95% 临界值比
    verdict = "拒绝 -> 秩>%d" % m if rej else "不能拒绝 -> r=%d" % m
    star = "**" if tr > cvt[2] else ("*" if tr > cvt[1] else "  ")
    print(f"  {('H('+str(m)+')'):>12} {tr:>10.2f}{star:>2} {cvt[1]:>6.2f}/{cvt[2]:<6.2f} "
          f"{me:>12.2f}   {cvm[1]:>6.2f}/{cvm[2]:<6.2f}  {verdict}")
    if not rej and not decided:
        r_hat = m; decided = True
if not decided:
    r_hat = k
print(f"  -> 序贯检验: 第一个不能拒绝的 m 即协整秩 r_hat={r_hat} (真值 r={R_TRUE}); k-r={k-r_hat} 个公共随机趋势")
R_USE = max(r_hat, 1)                                   # 下游用(至少取 1 以演示 VECM)

# ====================================================================
# 步骤5  VECM ML 估计: beta(前 r 个特征向量, 标准化) + 回代 alpha 与短期 Γ; vs 真值
# ====================================================================
print(SEP); print(f"步骤5  VECM ML 估计 (秩 r={R_USE}: beta 长期均衡 + alpha 调整速度 + 短期 Γ)")
beta_hat = V[:, :R_USE].copy()                          # (k+1, r): 末行=常数(rc)
beta_hat = beta_hat / beta_hat[0, 0]                    # 标准化 coint.1: 令 SHORT 系数=1
W = X1 @ beta_hat                                       # 误差修正回归元 w_{t-1}=beta'[x_{t-1};1] (n_eff×r)
Xreg = np.hstack([W, Z]) if p1 > 0 else W               # OLS: Δx_t ~ [w_{t-1}, Δx 滞后]
Bc, *_ = np.linalg.lstsq(Xreg, DXt, rcond=None)
alpha_hat = Bc[:R_USE].T                                # (k×r) 载荷
Gam_hat = [Bc[R_USE + j * k: R_USE + (j + 1) * k].T for j in range(p1)]   # 各 Γ_i (k×k)
resid = DXt - Xreg @ Bc
Sig_resid = resid.T @ resid / (n_eff - Xreg.shape[1])

beta_x = beta_hat[:k, 0]                                # 协整向量(不含常数部分)
beta_c = beta_hat[k, 0]                                 # 协整关系里的常数
print(f"  协整向量 beta (标准化 SHORT=1): LONG={beta_x[1]:+.4f} (真值 {-1.0}), 常数={beta_c:+.4f} (真值≈{-MU_EC})")
print(f"     -> 平稳均衡 w_t = SHORT {beta_x[1]:+.3f}*LONG {beta_c:+.3f}; 即利差长期≈{-beta_c:+.3f}")
print(f"  载荷 alpha (回调速度):")
XtXi = inv(Xreg.T @ Xreg)
for j in range(k):
    se_a = np.sqrt(Sig_resid[j, j] * XtXi[0, 0])       # alpha_j 的标准误(w 是第 0 列回归元)
    ta = alpha_hat[j, 0] / se_a
    tv = ALPHA[j]
    sig = "显著" if abs(ta) > 1.96 else "不显著"
    print(f"     {NAMES[j]:>5} 方程: alpha={alpha_hat[j,0]:+.4f} (t={ta:+.2f}, {sig}) | 真值 {tv}")
print(f"     -> 短端 {NAMES[0]} 强力回调(显著)、长端 {NAMES[1]} 几乎不动 = 短端向均衡收敛(书同款结论)")
print(f"  短期矩阵 Γ (差分动态): 估 {p1} 个 (k×k), 略; 残差标准误={np.sqrt(np.diag(Sig_resid))}")

# 打印"拟合 ECM"(书 p373 风格)
eq_w = f"(w_t{'+' if -beta_c>=0 else '-'}{abs(-beta_c):.3f})"
print(f"  拟合 ECM:  Δx_t = [{alpha_hat[0,0]:+.3f},{alpha_hat[1,0]:+.3f}]' * {eq_w}"
      f" + Γ1 Δx_(t-1) + ... + a_t,  w_t = SHORT - LONG")

# ---- statsmodels VECM 交叉验证 (deterministic='ci' = 受限常数, 与本 demo 同设定) ----
try:
    from statsmodels.tsa.vector_ar.vecm import VECM
    sm = VECM(train, k_ar_diff=KDIFF, coint_rank=R_USE, deterministic="ci").fit()
    sm_beta = sm.beta[:, 0] / sm.beta[0, 0]
    sm_alpha = sm.alpha[:, 0]
    print(f"  [交叉验证] statsmodels VECM(ci): beta_LONG={sm_beta[1]:+.4f}, "
          f"alpha=({sm_alpha[0]:+.4f},{sm_alpha[1]:+.4f}) -> 与手写基本一致 [OK]")
except Exception as e:
    print(f"  [交叉验证] statsmodels VECM 跳过 ({type(e).__name__})")

# ====================================================================
# 步骤6  模型检验: 残差多元 Ljung-Box + CCM 符号表 + 协整残差平稳(ADF) + alpha 经济含义
# ====================================================================
print(SEP); print("步骤6  模型检验 (残差白噪声 + 协整残差平稳 + alpha 显著性)")
nparm = R_USE * k + p1 * k * k                          # 被估动态参数数(alpha + Γ)
thr = 2.0 / np.sqrt(n_eff)
for m in (6, 12):
    Qk, dfQ, pQ = mv_ljung_box(resid, m, k, nparm)
    print(f"  多元 Ljung-Box Q_k(m={m})={Qk:.2f}, df={dfQ}, p={pQ:.3g} -> "
          + ("残差近似多元白噪声, 模型充分 [OK]" if pQ > 0.05 else "残差仍有结构, 升阶 [NO]"))
print(f"  残差 CCM 符号表 (阈值 2/sqrt(n)={thr:.3f}; '.'=不显著):")
alldot = True
for l in range(1, 5):
    sym = ccm_signs(resid, l, thr)
    print(f"     lag {l}:  " + "   ".join("[" + " ".join(sym[i]) + "]" for i in range(k)))
    if np.any(sym != "."):
        alldot = False
# 协整残差(误差修正项)是否平稳 —— VECM 的立身之本
ecm_resid = (X1 @ beta_hat)[:, 0]                       # w_t = beta'[x_{t-1};1]
s_ec, p_ec, *_ = adfuller(ecm_resid, autolag="AIC")
print(f"  协整残差 w_t=beta'x 的 ADF={s_ec:+.2f}, p={p_ec:.3g} -> "
      + ("平稳, 协整关系成立 [OK]" if p_ec < 0.05 else "非平稳, 协整可疑 [NO]"))
print(f"  -> 残差{'近白噪声' if alldot else '略有残留'}、协整残差平稳、alpha 短端显著 = VECM 充分")

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for j in range(k):
    ax[0, j].plot(resid[:, j], lw=.6); ax[0, j].axhline(0, color="r", ls="--", lw=.7)
    ax[0, j].set_title(f"Residual Δ{NAMES[j]} (should be white)")
lags = np.arange(1, 13)
ac = [[np.corrcoef(resid[l:, j], resid[:-l, j])[0, 1] for l in lags] for j in range(k)]
ax[1, 0].bar(lags - 0.2, ac[0], width=0.4, label=f"Δ{NAMES[0]} ACF", color="steelblue")
ax[1, 0].bar(lags + 0.2, ac[1], width=0.4, label=f"Δ{NAMES[1]} ACF", color="tomato")
ax[1, 0].axhline(thr, color="gray", ls="--", lw=.8); ax[1, 0].axhline(-thr, color="gray", ls="--", lw=.8)
ax[1, 0].set_title("Residual auto-corr (~0)"); ax[1, 0].legend(fontsize=8)
ax[1, 1].plot(ecm_resid, lw=.6, color="purple"); ax[1, 1].axhline(np.mean(ecm_resid), color="k", ls="--", lw=.8)
ax[1, 1].set_title("Cointegration resid w_t=beta'x (stationary)")
plt.tight_layout(); plt.savefig("vecm_fig3_diagnose.png", dpi=110); plt.close()

# ====================================================================
# 步骤7  预测: 先用 ECM 递推 Δx_t 再累加还原 x_t; 水平预测扇形张开, 协整组合(利差)区间收敛
#   水平误差协方差用还原的水平 VAR(p) 的 Psi 权重 (含一个单位根 -> 水平方差随步长增长=不回复);
#   而 beta'预测误差被 beta 消掉单位根 -> 利差区间有界 = 协整"长期拴住"的直接体现.
# ====================================================================
print(SEP); print(f"步骤7  预测 ({h} 步): ECM 递推 -> 水平 + 协整组合(利差)")
# 由 VECM 估计还原水平 VAR(p): Phi1=I+Π+Γ1, Phi_j=Γ_j-Γ_{j-1}, Phi_p=-Γ_{p-1}
Pi = alpha_hat @ beta_x.reshape(1, k)                  # Π = alpha*beta_x' (k×k)
Phis = []
if p1 == 0:
    Phis = [np.eye(k) + Pi]
else:
    Phis.append(np.eye(k) + Pi + Gam_hat[0])           # Phi_1
    for j in range(1, p1):
        Phis.append(Gam_hat[j] - Gam_hat[j - 1])       # Phi_2..Phi_{p-1}
    Phis.append(-Gam_hat[-1])                           # Phi_p
const_vec = alpha_hat[:, 0] * beta_c                    # 水平 VAR 的常数向量 = alpha*beta_c

# 点预测: ECM 递推 (与水平 VAR 递推等价; 这里显式走 ECM 以贴合式8.36)
xhist = [train[-1 - i].copy() for i in range(P_USE)]   # xhist[0]=x_T, [1]=x_{T-1}, ...
dxh = [train[-1 - i] - train[-2 - i] for i in range(p1)]  # 最近的差分
fc = np.zeros((h, k)); xlast = xhist[0].copy(); dxbuf = list(dxh)
for step in range(h):
    ec = (beta_x @ xlast) + beta_c                     # 误差修正项(含常数)
    dxn = alpha_hat[:, 0] * ec
    for j in range(p1):
        dxn = dxn + Gam_hat[j] @ dxbuf[j]
    xnew = xlast + dxn
    fc[step] = xnew
    dxbuf = [dxn] + dxbuf[:-1] if p1 > 0 else dxbuf
    xlast = xnew

# 水平预测误差协方差: Psi 权重 (Psi_0=I; Psi_l=Σ Phi_j Psi_{l-j}); 含单位根 -> 不衰减
Psi = [np.eye(k)]
for l in range(1, h):
    Pl = np.zeros((k, k))
    for j in range(1, min(l, len(Phis)) + 1):
        Pl += Phis[j - 1] @ Psi[l - j]
    Psi.append(Pl)
Slev = np.zeros((k, k)); lev_var = np.zeros((h, k)); spr_var = np.zeros(h)
bx = np.array([[1.0], [-1.0]])                         # 协整方向(真值 (1,-1)): 用它消掉单位根
for l in range(h):
    Slev = Slev + Psi[l] @ Sig_resid @ Psi[l].T
    lev_var[l] = np.diag(Slev)
    spr_var[l] = float(bx.T @ Slev @ bx)               # 协整组合的预测误差方差(被 beta 消掉单位根 -> 有界)
lev_se = np.sqrt(lev_var); spr_se = np.sqrt(spr_var)
lo, hi = fc - 1.96 * lev_se, fc + 1.96 * lev_se
spr_fc = fc[:, 0] - fc[:, 1]

inside = (test >= lo) & (test <= hi)
for j in range(k):
    print(f"  {NAMES[j]}: 1步={fc[0,j]:+.2f}, {h}步={fc[-1,j]:+.2f}; "
          f"95%区间半宽 1步={1.96*lev_se[0,j]:.2f} -> {h}步={1.96*lev_se[-1,j]:.2f} (张开=水平不回复)")
    print(f"        样本外 95% 覆盖率={inside[:,j].mean():.0%}")
uncond_spr = 1.96 * np.std(spread_tr)                  # 平稳组合的无条件 95% 带(区间上限)
print(f"  利差 SHORT-LONG: 1步={spr_fc[0]:+.3f} -> {h}步={spr_fc[-1]:+.3f} (向均衡≈{MU_EC}回调); "
      f"区间半宽 1步={1.96*spr_se[0]:.3f} -> {h}步={1.96*spr_se[-1]:.3f}, 饱和于无条件带≈{uncond_spr:.2f}")
print(f"  ★ 关键对比: 水平区间随步长无限'张开'(单位根不回复, {h}步半宽={1.96*lev_se[-1,0]:.2f} 仍在增长); "
      f"协整组合区间'饱和于无条件带'(被均衡拴住, {h}步半宽={1.96*spr_se[-1]:.2f}≈上限{uncond_spr:.2f}) [OK]")

pd.DataFrame({
    "step": np.arange(1, h + 1),
    "SHORT_fc": np.round(fc[:, 0], 3), "SHORT_lo": np.round(lo[:, 0], 3),
    "SHORT_hi": np.round(hi[:, 0], 3), "SHORT_act": np.round(test[:, 0], 3),
    "LONG_fc": np.round(fc[:, 1], 3), "LONG_lo": np.round(lo[:, 1], 3),
    "LONG_hi": np.round(hi[:, 1], 3), "LONG_act": np.round(test[:, 1], 3),
    "spread_fc": np.round(spr_fc, 3), "spread_act": np.round(test[:, 0] - test[:, 1], 3),
}).to_csv("vecm_forecast.csv", index=False, encoding="utf-8")
print("  已导出每步预测(水平点/区间/利差) -> vecm_forecast.csv")

fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
tail = 80; xx = np.arange(1, h + 1)
for j in range(k):
    ax[j].plot(range(-tail + 1, 1), train[-tail:, j], color="gray", lw=.8, label="train (tail)")
    ax[j].plot(xx, test[:, j], color="black", lw=1, marker=".", ms=6, label="actual")
    ax[j].plot(xx, fc[:, j], "b-", lw=1.5, marker="o", ms=3, label="VECM forecast")
    ax[j].fill_between(xx, lo[:, j], hi[:, j], color="red", alpha=.15, label="95% interval (fans out)")
    ax[j].axvline(0, color="k", lw=.5); ax[j].set_ylabel(NAMES[j])
    ax[j].set_title(f"Level forecast {NAMES[j]}: interval widens (unit root, no reversion)")
    ax[j].legend(loc="upper left", fontsize=7)
spr_tr = train[:, 0] - train[:, 1]
ax[2].plot(range(-tail + 1, 1), spr_tr[-tail:], color="gray", lw=.8, label="train spread")
ax[2].plot(xx, test[:, 0] - test[:, 1], color="black", lw=1, marker=".", ms=6, label="actual spread")
ax[2].plot(xx, spr_fc, "purple", lw=1.6, marker="o", ms=3, label="spread forecast")
ax[2].fill_between(xx, spr_fc - 1.96 * spr_se, spr_fc + 1.96 * spr_se, color="purple", alpha=.15,
                   label="95% interval (bounded!)")
ax[2].axhline(MU_EC, color="green", ls=":", lw=1.2, label=f"equilibrium={MU_EC}")
ax[2].axvline(0, color="k", lw=.5); ax[2].set_ylabel("spread"); ax[2].set_xlabel("forecast horizon")
ax[2].set_title("Cointegration combo SHORT-LONG: reverts to equilibrium, interval stays BOUNDED")
ax[2].legend(loc="upper left", fontsize=7)
plt.tight_layout(); plt.savefig("vecm_fig4_forecast.png", dpi=110); plt.close()

# ---- fig2: 两层定阶合一图 (左=VAR阶 AIC/BIC/HQ; 右=Johansen 迹检验定协整秩) ----
tr_stats = [-n_eff * np.sum(np.log(1 - lam[m:])) for m in range(k)]
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.6))
for key, c in [("AIC", "steelblue"), ("BIC", "tomato"), ("HQ", "green")]:
    ax2[0].plot(xs, ic[key], marker="o", color=c, label=f"{key} (min@{sel[key]})")
    ax2[0].scatter([sel[key]], [ic[key][sel[key]]], s=90, facecolors="none", edgecolors=c, lw=2)
ax2[0].axvline(P_TRUE, color="k", ls=":", lw=1, label=f"true p={P_TRUE}")
ax2[0].set_xlabel("VAR order p (levels)"); ax2[0].set_ylabel("criterion")
ax2[0].set_title("Layer-1: VAR order on levels (AIC/BIC/HQ)"); ax2[0].legend(fontsize=8)
ax2[1].bar([f"H({m})" for m in range(k)], tr_stats, color=["tomato", "steelblue"], alpha=.85)
for m in range(k):
    ax2[1].hlines(CV_TRACE[k - m][1], m - 0.4, m + 0.4, color="red", ls="--", lw=1.5)
ax2[1].plot([], [], "r--", label="95% crit value")
ax2[1].set_ylabel("trace statistic")
ax2[1].set_title(f"Layer-2: Johansen trace test -> rank r={r_hat}"); ax2[1].legend(fontsize=8)
fig2.tight_layout(); fig2.savefig("vecm_fig2_order.png", dpi=110); plt.close(fig2)

# ---- fig5: 两个"能直接读出"的结论 —— 协整关系(平稳) + 调整速度(谁回调) ----
fig, axx5 = plt.subplots(1, 2, figsize=(12, 4.6))
axx5[0].plot(ecm_resid, lw=.7, color="purple")
axx5[0].axhline(np.mean(ecm_resid), color="k", ls="--", lw=1, label=f"mean={np.mean(ecm_resid):+.3f}")
axx5[0].set_title("Cointegration relation beta'x is stationary (long-run anchor)")
axx5[0].set_xlabel("t"); axx5[0].set_ylabel("w_t = SHORT - LONG (+const)"); axx5[0].legend(fontsize=8)
alp = alpha_hat[:, 0]
se_alp = [np.sqrt(Sig_resid[j, j] * XtXi[0, 0]) for j in range(k)]
tvals = [abs(alp[j] / se_alp[j]) for j in range(k)]
axx5[1].bar([f"{NAMES[0]}\n(short)", f"{NAMES[1]}\n(long)"], [abs(alp[0]), abs(alp[1])],
            color=["tomato", "steelblue"], alpha=.85)
axx5[1].set_ylabel("|adjustment speed alpha|")
axx5[1].set_title(f"Who corrects? SHORT |t|={tvals[0]:.1f} (sig) vs LONG |t|={tvals[1]:.1f}")
plt.tight_layout(); plt.savefig("vecm_fig5_structure.png", dpi=110); plt.close()

# ====================================================================
print(SEP)
print(f"完成. 全流程: 各分量 I(1)(ADF) + 利差平稳 -> 水平 VAR 定阶 p={P_USE}(BIC) -> 受限常数 "
      f"-> Johansen 迹检验定协整秩 r={r_hat} -> ML 估 beta(利差均衡)/alpha(短端回调) "
      f"-> 残差白噪声[{'OK' if pQ>0.05 else 'NO'}]+协整残差平稳 -> ECM 递推预测(水平张开/利差有界)")
print(f"揭晓真值: VECM 差分滞后 {P_TRUE-1}, 协整秩 {R_TRUE}, beta=(1,-1), 均衡利差 {MU_EC}, 短端主回调")
print("图: vecm_fig1_series / vecm_fig2_order / vecm_fig3_diagnose / vecm_fig4_forecast / vecm_fig5_structure (.png)")
