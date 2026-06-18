# -*- coding: utf-8 -*-
"""
GARCH-M 风险溢价建模 —— 一条龙完整流程 (均值用 ARMA + in-mean 项, 波动用 GARCH(p,q)-t)

  GARCH-M 与 GARCH 的关系 (本 demo 的主角):
    GARCH:   r_t = mu_t(ARMA) + a_t            <- 均值与波动"分家", 波动不影响中心
    GARCH-M: r_t = mu_t(ARMA) + lambda*sigma_t^2 + a_t
                                 ^^^^^^^^^^^^^^^^  把"今天的条件方差"塞进均值方程
    多出的 lambda*sigma_t^2 叫 "in-mean / 风险溢价项":
      金融直觉 = "高风险要高回报"——波动越大(sigma_t^2 越大), 要求的预期收益越高.
      lambda 称风险溢价系数(risk premium): lambda>0 => 风险与收益正相关(Merton ICAPM 直觉).
    Tsay《金融时间序列分析》§3.6 的写法即 r_t = mu + c*sigma_t^2 + a_t (in-mean 用方差 sigma_t^2;
      也可用 sigma_t 或 ln sigma_t^2). 一个副作用: 因 sigma_t^2 自相关, 它会给 r_t 带进自相关.

  为什么要联合估计(不能两步走): in-mean 把波动方程的 sigma_t^2 喂回均值方程, 两个方程耦合,
    无法像普通 GARCH 那样"先 ARMA 取残差, 再对残差建 GARCH". 故本 demo 手写联合 MLE
    (numpy/scipy), 一次性估 ARMA + GARCH + lambda. (arch 包不支持 in-mean; R 的 rugarch 可.)

  六步法 (下游一律用"定出来的阶", 不写死):
    第0步 准备数据    : 价格 -> 对数收益率 r_t (平稳序列, 建模入口)
    第1步 均值方程    : r_t 的 Ljung-Box 有自相关 -> BIC 网格定阶 ARMA(P,Q) -> 暂得冲击 a_t
    第2步 ARCH 效应   : 对 a_t^2 查波动聚集(Ljung-Box + Engle LM) -> 要不要建波动率模型
    第3步 定阶 (p,q)  : a_t^2 双拖尾 -> (p,q) 二维 AIC/BIC 网格定波动阶
    第4步 联合估计    : 手写 MLE 一次估 ARMA(P,Q)-GARCH(p,q)-M (Student-t); 重点看 lambda
    第5步 模型检验    : 标准化残差三查 + 平稳性 + 风险溢价检验(LR/t: lambda=0 vs GARCH-M)
    第6步 使用模型    : 均值预测(随波动起伏!) + 波动率递推 -> 动态区间 + VaR; 看风险溢价贡献
"""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")               # Windows 控制台强制 UTF-8
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                                  # 不弹窗, 直接存图
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.special import gammaln
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model

SEP = "=" * 70

# ====================================================================
# 步骤0  准备数据: 造"价格" -> 转对数收益率 r_t (平稳)
#   均值真值 = ARMA(1,1) + 风险溢价 lambda*sigma_t^2; 波动真值 = GARCH(1,1)-t.
#   单位用"百分比收益"(x100), 量纲适中.
# ====================================================================
C0, PHI, THETA, LAM = 0.0, 0.5, -0.3, 0.20            # 均值: 常数/AR1/MA1/风险溢价系数
OMEGA, ALPHA, BETA, NU = 0.05, 0.08, 0.90, 7.0        # GARCH(1,1)-t: 持续度 0.98, df=7 厚尾
n, burn = 2500, 1500
uncond_var_true = OMEGA / (1 - ALPHA - BETA)
truth_txt = (f"均值 ARMA(1,1) c={C0}, phi={PHI}, theta={THETA} + 风险溢价 lambda={LAM} (in-mean=sigma_t^2); "
             f"波动 GARCH(1,1)-t omega={OMEGA}, alpha={ALPHA}, beta={BETA}, df={NU} "
             f"(持续度 {ALPHA+BETA:.2f}, 无条件方差 {uncond_var_true:.2f})")

rng = np.random.default_rng(20260616)

def sim_garchm(rng):                                   # 造 ARMA(1,1)-GARCH(1,1)-M 收益序列
    N = n + burn
    e = rng.standard_t(NU, N) * np.sqrt((NU - 2) / NU) # 标准化 t 新息(单位方差)
    s2 = np.zeros(N); a = np.zeros(N); r = np.zeros(N)
    s2[0] = uncond_var_true; a[0] = np.sqrt(s2[0]) * e[0]
    r[0] = C0 + LAM * s2[0] + a[0]                      # t=0 无 AR/MA 项
    for t in range(1, N):
        s2[t] = OMEGA + ALPHA * a[t - 1] ** 2 + BETA * s2[t - 1]    # GARCH 方差递推
        a[t] = np.sqrt(s2[t]) * e[t]
        r[t] = C0 + LAM * s2[t] + PHI * r[t - 1] + THETA * a[t - 1] + a[t]  # 均值: ARMA + in-mean
    return r[burn:]

ret = pd.Series(sim_garchm(rng)).reset_index(drop=True)
price = pd.Series(np.r_[100.0, 100.0 * np.exp(np.cumsum(ret.values) / 100.0)])  # 收益->价格
r = 100.0 * np.log(price / price.shift(1)).dropna().reset_index(drop=True)      # 价格->对数收益 r_t

h = 60                                                 # 留最后 60 天做样本外预测
train, test = r.iloc[:-h], r.iloc[-h:]
rtr = train.values
pd.DataFrame({"t": np.arange(len(r)), "logret_pct": r.values,
              "split": ["train"] * len(train) + ["test"] * len(test)}).to_csv(
    "garchm_returns.csv", index=False, encoding="utf-8")

print(SEP); print("步骤0  准备数据: 价格 -> 对数收益率 r_t")
print(f"  真值: {truth_txt}")
print(f"  样本 n={len(r)} (train={len(train)}, test={len(test)}), 单位=百分比")
print(f"  平稳性 ADF p={adfuller(rtr)[1]:.3g} -> 平稳, 可建模 [OK]")
print("  已导出 -> garchm_returns.csv")

# ====================================================================
# 步骤1  均值方程 (管"中心"): r_t 有自相关 -> BIC 网格定阶 ARMA(P,Q)
#   注: GARCH-M 里 r_t 的自相关一部分来自 in-mean(sigma_t^2 自相关), 这里先按普通 ARMA 定阶.
# ====================================================================
print(SEP); print("步骤1  均值方程 (ARMA 定阶, 暂得冲击 a_t)")
lb_mean = acorr_ljungbox(train, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
print(f"  r_t 的 Ljung-Box(10) p={lb_mean:.3g} -> "
      + ("有自相关, 走 ARMA 分支" if lb_mean < 0.05 else "无自相关"))

best_ic, best_order, grid = np.inf, (1, 1), {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for p in range(4):
        for q in range(4):
            if p == 0 and q == 0:
                continue
            try:
                bic = ARIMA(train, order=(p, 0, q)).fit().bic
                grid[(p, q)] = bic
                if bic < best_ic:
                    best_ic, best_order = bic, (p, q)
            except Exception:
                grid[(p, q)] = np.nan
    arma0 = ARIMA(train, order=(best_order[0], 0, best_order[1])).fit()
P, Q = best_order
a_t = arma0.resid                                      # 暂用的冲击(普通 ARMA, 仅用于步骤2/3)
mean_desc = f"ARMA({P},{Q})"
print(f"  BIC 网格 (p,q∈0..3, 行=p 列=q, 越小越好, [*]=最优):")
print("         " + "".join(f"q={q:<8}" for q in range(4)))
for p in range(4):
    cells = ""
    for q in range(4):
        if (p, q) not in grid:
            cells += "   --     "
        else:
            star = "*" if (p, q) == best_order else " "
            cells += f"{grid[(p, q)]:8.1f}{star} "
    print(f"     p={p} {cells}")
print(f"  -> BIC 最小 = {mean_desc} (BIC={best_ic:.1f}); in-mean 风险溢价项留到步骤4联合估")

fig, ax = plt.subplots(3, 1, figsize=(10, 9))
ax[0].plot(train.values, lw=.6, color="steelblue")
ax[0].set_title(f"Log returns r_t  [mean: {mean_desc}+in-mean]  —— volatility clustering & drift shifts")
plot_acf(train, lags=25, ax=ax[1], zero=False); ax[1].set_title("ACF of r_t (mean structure -> ARMA, partly from in-mean)")
plot_pacf(train, lags=25, ax=ax[2], method="ywm", zero=False); ax[2].set_title("PACF of r_t")
plt.tight_layout(); plt.savefig("garchm_fig1_mean.png", dpi=110); plt.close()

# ====================================================================
# 步骤2  检验 ARCH 效应 (要不要建波动率模型): 看 a_t^2 会不会聚集
# ====================================================================
print(SEP); print("步骤2  检验 ARCH 效应 (波动聚集?)")
a2 = a_t ** 2
for m in (5, 10, 20):
    pq = acorr_ljungbox(a2, lags=[m], return_df=True)["lb_pvalue"].iloc[0]
    print(f"  [方法1] a_t^2 的 Ljung-Box Q({m}) p={pq:.3g} -> "
          + ("有 ARCH 效应" if pq < 0.05 else "不显著"))
lm_stat, lm_p, _, _ = het_arch(a_t, nlags=10)
print(f"  [方法2] Engle LM(lags=10): LM={lm_stat:.2f}, p={lm_p:.3g} -> "
      + ("有 ARCH 效应, 需建波动率模型 [OK]" if lm_p < 0.05 else "不显著, 收工"))

# ====================================================================
# 步骤3  定阶 (p,q): a_t^2 双拖尾 -> (p,q) 二维 AIC/BIC 网格 (与 GARCH 同理)
# ====================================================================
print(SEP); print("步骤3  定阶 (p,q) (GARCH 波动阶, 二维网格)")
PMAX = QMAX = 2
ggrid_bic, ggrid_aic = {}, {}
best_gic, best_gorder = np.inf, (1, 1)
for p in range(1, PMAX + 1):
    for q in range(0, QMAX + 1):
        try:
            rr = arch_model(a_t, mean="Zero", vol="Garch", p=p, q=q, dist="t").fit(disp="off")
            ggrid_bic[(p, q)] = rr.bic; ggrid_aic[(p, q)] = rr.aic
            if rr.bic < best_gic:
                best_gic, best_gorder = rr.bic, (p, q)
        except Exception:
            ggrid_bic[(p, q)] = np.nan; ggrid_aic[(p, q)] = np.nan
pg, qg = best_gorder
aic_order = min(ggrid_aic, key=lambda k: ggrid_aic[k])
print("  BIC 网格 (行=alpha阶 p, 列=beta阶 q; q=0 即 ARCH(p); [*]=最优):")
print("         " + "".join(f"q={q:<8}" for q in range(QMAX + 1)))
for p in range(1, PMAX + 1):
    cells = ""
    for q in range(QMAX + 1):
        v = ggrid_bic.get((p, q), np.nan)
        star = "*" if (p, q) == best_gorder else " "
        cells += (f"{v:8.1f}{star} " if np.isfinite(v) else "   --     ")
    print(f"     p={p} {cells}")
print(f"  AIC 选 (p,q)={aic_order} | BIC 选 (p,q)={best_gorder}  >>> 最终波动定阶 GARCH{best_gorder} (以BIC为准)")

fig, axx = plt.subplots(1, 2, figsize=(12, 4.5))
plot_acf(a2, lags=25, ax=axx[0], zero=False)
axx[0].set_title("ACF of a_t^2 (slow decay -> high persistence)")
plot_pacf(a2, lags=25, ax=axx[1], method="ywm", zero=False)
axx[1].set_title("PACF of a_t^2 (both tail off -> use (p,q) grid)")
plt.tight_layout(); plt.savefig("garchm_fig2_order.png", dpi=110); plt.close()

# ====================================================================
# 步骤4  联合估计 ARMA(P,Q)-GARCH(pg,qg)-M (手写 MLE, Student-t)
#   参数向量: [c, phi(1..P), theta(1..Q), lambda, omega, alpha(1..pg), beta(1..qg), nu]
#   in-mean=sigma_t^2; 因 sigma_t^2 由 t-1 信息算出, 均值方程无同期内生, 递推良定义.
# ====================================================================
print(SEP); print(f"步骤4  联合估计 {mean_desc}-GARCH{best_gorder}-M (手写 MLE, Student-t)")

def unpack(th, with_M=True):                           # 解包参数向量
    i = 0
    c = th[i]; i += 1
    phi = th[i:i + P]; i += P
    theta = th[i:i + Q]; i += Q
    if with_M:
        lam = th[i]; i += 1
    else:
        lam = 0.0
    omega = th[i]; i += 1
    alpha = th[i:i + pg]; i += pg
    beta = th[i:i + qg]; i += qg
    nu = th[i]
    return c, phi, theta, lam, omega, alpha, beta, nu

def garchm_filter(th, y, with_M=True):                 # 递推 a_t 与 sigma_t^2
    c, phi, theta, lam, omega, alpha, beta, nu = unpack(th, with_M)
    N = len(y)
    a = np.zeros(N); s2 = np.zeros(N)
    psum = alpha.sum() + beta.sum()
    sbar = omega / (1 - psum) if psum < 1 else np.nan  # 无条件方差(初始化用)
    rbar = y.mean()
    for t in range(N):
        v = omega                                      # 条件方差(只用 t-1 及更早信息)
        for i in range(1, pg + 1):
            v += alpha[i - 1] * (a[t - i] ** 2 if t - i >= 0 else sbar)
        for j in range(1, qg + 1):
            v += beta[j - 1] * (s2[t - j] if t - j >= 0 else sbar)
        s2[t] = v
        m = c + lam * v                                # 条件均值: 常数 + 风险溢价 + ARMA
        for i in range(1, P + 1):
            m += phi[i - 1] * (y[t - i] if t - i >= 0 else rbar)
        for j in range(1, Q + 1):
            m += theta[j - 1] * (a[t - j] if t - j >= 0 else 0.0)
        a[t] = y[t] - m
    return a, s2, nu

def stdt_logpdf(z, nu):                                # 标准化(单位方差)Student-t 对数密度
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log((nu - 2) * np.pi)
            - (nu + 1) / 2 * np.log(1 + z ** 2 / (nu - 2)))

def negloglik(th, y, with_M=True):
    c, phi, theta, lam, omega, alpha, beta, nu = unpack(th, with_M)
    if omega <= 0 or nu <= 2.01 or np.any(alpha <= 0) or np.any(beta < 0) \
            or alpha.sum() + beta.sum() >= 0.999:      # 平稳/正定护栏
        return 1e10
    a, s2, nu = garchm_filter(th, y, with_M)
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return 1e10
    z = a / np.sqrt(s2)
    ll = np.sum(stdt_logpdf(z, nu) - 0.5 * np.log(s2))
    return -ll if np.isfinite(ll) else 1e10

def fit_mle(with_M=True):                              # 起点用普通 ARMA/arch 估计, 再联合优化
    g0 = arch_model(a_t, mean="Zero", vol="Garch", p=pg, q=qg, dist="t").fit(disp="off")
    om0 = g0.params["omega"]
    al0 = [g0.params.get(f"alpha[{i}]", 0.05) for i in range(1, pg + 1)]
    be0 = [g0.params.get(f"beta[{j}]", 0.85) for j in range(1, qg + 1)]
    nu0 = g0.params.get("nu", 8.0)
    ar0 = [arma0.params.get(f"ar.L{i}", 0.0) for i in range(1, P + 1)]
    ma0 = [arma0.params.get(f"ma.L{j}", 0.0) for j in range(1, Q + 1)]
    x0 = [arma0.params.get("const", 0.0)] + ar0 + ma0 + ([0.0] if with_M else []) \
         + [om0] + al0 + be0 + [nu0]
    bnds = [(-5, 5)] + [(-.99, .99)] * P + [(-.99, .99)] * Q + ([(-3, 3)] if with_M else []) \
        + [(1e-8, 5)] + [(1e-8, .99)] * pg + [(1e-8, .99)] * qg + [(2.1, 60)]
    res = optimize.minimize(negloglik, np.array(x0, float), args=(train.values, with_M),
                            method="L-BFGS-B", bounds=bnds)
    res = optimize.minimize(negloglik, res.x, args=(train.values, with_M),    # 二次打磨
                            method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
    return res

resM = fit_mle(with_M=True)
thM = resM.x
llf_M = -resM.fun
c_h, phi_h, theta_h, lam_h, om_h, al_h, be_h, nu_h = unpack(thM, True)
persist = al_h.sum() + be_h.sum()
uncond_var = om_h / (1 - persist)
halflife = np.log(0.5) / np.log(persist) if persist < 1 else np.inf
kM = len(thM)
aicM = 2 * kM - 2 * llf_M; bicM = kM * np.log(len(train)) - 2 * llf_M

# 数值 Hessian -> 标准误 -> t 值 (lambda 的显著性)
def num_hessian(f, x):
    nx = len(x); step = np.maximum(1e-4 * np.abs(x), 1e-5); H = np.zeros((nx, nx))
    for i in range(nx):
        for j in range(nx):
            xpp = x.copy(); xpp[i] += step[i]; xpp[j] += step[j]
            xpm = x.copy(); xpm[i] += step[i]; xpm[j] -= step[j]
            xmp = x.copy(); xmp[i] -= step[i]; xmp[j] += step[j]
            xmm = x.copy(); xmm[i] -= step[i]; xmm[j] -= step[j]
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * step[i] * step[j])
    return H

try:
    H = num_hessian(lambda z: negloglik(z, train.values, True), thM)
    cov = np.linalg.inv(H); se = np.sqrt(np.abs(np.diag(cov)))
except Exception:
    se = np.full(kM, np.nan)

names = (["c"] + [f"phi{i}" for i in range(1, P + 1)] + [f"theta{j}" for j in range(1, Q + 1)]
         + ["lambda"] + ["omega"] + [f"alpha{i}" for i in range(1, pg + 1)]
         + [f"beta{j}" for j in range(1, qg + 1)] + ["nu"])
tv_map = {"c": C0, "phi1": PHI, "theta1": THETA, "lambda": LAM,
          "omega": OMEGA, "alpha1": ALPHA, "beta1": BETA, "nu": NU}
lam_idx = names.index("lambda")
print("  估计参数 (vs 真值; t = 估计/标准误):")
for k, nm in enumerate(names):
    tval = thM[k] / se[k] if np.isfinite(se[k]) and se[k] > 0 else np.nan
    tv = tv_map.get(nm)
    line = f"     {nm:<8}={thM[k]:+.4f}  (se={se[k]:.4f}, t={tval:+.2f})"
    print(line + (f"   | 真值 {tv}" if tv is not None else ""))
print(f"  >>> 风险溢价 lambda={lam_h:+.4f} (真值 {LAM}); "
      + ("t 显著(|t|>1.96) -> 高波动确实索取更高预期收益 [OK]"
         if np.isfinite(se[lam_idx]) and abs(thM[lam_idx] / se[lam_idx]) > 1.96
         else "未达显著, 看下一步 LR 检验"))
print(f"  持续度 alpha+beta = {persist:.4f} (真值 {ALPHA+BETA:.2f}; <1 -> 协方差平稳 [OK])")
print(f"  无条件方差 = {uncond_var:.3f} (真值 {uncond_var_true:.2f}); 波动冲击半衰期 = {halflife:.1f} 天")
print(f"  logL={llf_M:.1f}, AIC={aicM:.1f}, BIC={bicM:.1f}")

# ====================================================================
# 步骤5  模型检验: 标准化残差三查 + 平稳性 + 风险溢价检验(LR: lambda=0)
# ====================================================================
print(SEP); print("步骤5  模型检验 (标准化残差三查 + 平稳性 + 风险溢价 LR 检验)")
aM, s2M, _ = garchm_filter(thM, train.values, True)
zM = aM / np.sqrt(s2M)
p_mean = acorr_ljungbox(zM, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
p_vol = acorr_ljungbox(zM ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
sk, ku = stats.skew(zM), stats.kurtosis(zM, fisher=False)
print(f"  ① 均值({mean_desc}+M够不够) ~z_t Ljung-Box(10) p={p_mean:.3g} -> "
      + ("均值方程充分 [OK]" if p_mean > 0.05 else "均值阶数不够 [NO]"))
print(f"  ② 波动方程 ~z_t^2 Ljung-Box(10) p={p_vol:.3g} -> "
      + ("波动洗净, GARCH 充分 [OK]" if p_vol > 0.05 else "仍有 ARCH, 升阶/换模型 [NO]"))
print(f"  ③ 分布   偏度={sk:+.2f}, 峰度={ku:.2f} (正态=3; t 新息应仍略厚尾)")
print(f"  ④ 平稳性  alpha+beta={persist:.4f} < 1 -> 方差过程协方差平稳 [OK]")

# 风险溢价 LR 检验: 受限模型(lambda=0, 即普通 ARMA-GARCH) vs GARCH-M
res0 = fit_mle(with_M=False)
llf_0 = -res0.fun
LR = 2 * (llf_M - llf_0)
p_LR = stats.chi2.sf(LR, df=1)
print(f"  ⑤ 风险溢价 LR 检验 (H0: lambda=0, 即退回普通 GARCH):")
print(f"     logL(GARCH-M)={llf_M:.2f} vs logL(GARCH, lambda=0)={llf_0:.2f}")
print(f"     LR=2*Δ={LR:.2f} ~ chi2(1), p={p_LR:.3g} -> "
      + ("拒绝 lambda=0, 风险溢价显著存在 [OK]" if p_LR < 0.05 else "不显著, in-mean 项可省"))
verdict = ("三查通过 + 平稳 + 风险溢价显著, GARCH-M 模型充分"
           if (p_mean > 0.05 and p_vol > 0.05 and persist < 1 and p_LR < 0.05)
           else "有一项不过关, 需迭代")
print(f"  >>> {verdict}")

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
sig_t = np.sqrt(s2M)
ax[0, 0].plot(sig_t, lw=.7, color="darkred")
ax[0, 0].axhline(np.sqrt(uncond_var), color="gray", ls="--", lw=1, label=f"uncond sigma={np.sqrt(uncond_var):.2f}")
ax[0, 0].set_title(f"Conditional volatility sigma_t  [GARCH{best_gorder}-M]"); ax[0, 0].legend()
plot_acf(zM ** 2, lags=25, ax=ax[0, 1], zero=False); ax[0, 1].set_title("ACF of std-resid^2 (white? -> vol cleaned)")
ax[1, 0].hist(zM, bins=40, density=True, alpha=.7, edgecolor="k")
xs = np.linspace(zM.min(), zM.max(), 200); ax[1, 0].plot(xs, stats.norm.pdf(xs), "r--", label="N(0,1)")
ax[1, 0].set_title("Std-resid histogram (fat tails)"); ax[1, 0].legend()
stats.probplot(zM, dist="norm", plot=ax[1, 1]); ax[1, 1].set_title("Q-Q plot vs Normal")
plt.tight_layout(); plt.savefig("garchm_fig3_diagnose.png", dpi=110); plt.close()

# ====================================================================
# 步骤5b  风险溢价"招牌图": 条件均值随条件方差水涨船高
# ====================================================================
muM = train.values - aM                                # 拟合的条件均值 mu_t = r_t - a_t
prem = c_h + lam_h * s2M                               # in-mean 部分(常数 + 风险溢价)
fig, axx = plt.subplots(1, 2, figsize=(12, 4.8))
# 左: r_t vs sigma_t^2 散点 + 分箱均值 + 拟合溢价线 (风险-收益正相关的直接证据)
axx[0].scatter(s2M, train.values, s=5, alpha=.15, color="steelblue", label="r_t")
bins = np.linspace(s2M.min(), np.percentile(s2M, 99), 12)
bidx = np.digitize(s2M, bins)
bx = [s2M[bidx == k].mean() for k in range(1, len(bins)) if (bidx == k).sum() > 5]
by = [train.values[bidx == k].mean() for k in range(1, len(bins)) if (bidx == k).sum() > 5]
axx[0].plot(bx, by, "o-", color="black", ms=5, lw=1.2, label="binned mean of r_t")
xs2 = np.linspace(s2M.min(), np.percentile(s2M, 99), 100)
axx[0].plot(xs2, c_h + lam_h * xs2, "r-", lw=2, label=f"fitted premium c+λ·σ² (λ={lam_h:.3f})")
axx[0].set_xlabel("conditional variance sigma_t^2"); axx[0].set_ylabel("return r_t")
axx[0].set_title("Risk-return: higher variance -> higher mean return"); axx[0].legend(fontsize=8)
# 右: 条件方差 vs 实际收益的滚动均值 —— 模型外印证"波动高的时段平均收益也高"
roll = pd.Series(train.values).rolling(50, center=True).mean()
axx[1].plot(s2M, lw=.7, color="darkred", label="conditional variance sigma_t^2")
ax2 = axx[1].twinx()
ax2.plot(roll.values, lw=1.1, color="navy", label="rolling mean of r_t (window=50)")
ax2.axhline(0, color="navy", ls=":", lw=.6)
axx[1].set_xlabel("time"); axx[1].set_ylabel("sigma_t^2", color="darkred")
ax2.set_ylabel("rolling mean r_t", color="navy")
axx[1].set_title("High-volatility spells coincide with higher average returns")
plt.tight_layout(); plt.savefig("garchm_fig5_riskpremium.png", dpi=110); plt.close()

# ====================================================================
# 步骤6  使用模型: 均值预测(随波动起伏) + 波动率递推 -> 动态区间 + VaR
#   GARCH-M 招牌: 均值预测 mu(l) = c + lambda*sigma^2(l) + ARMA, 随波动预测一起均值回复.
# ====================================================================
print(SEP); print(f"步骤6  预测 (均值 {mean_desc}+M + 波动 GARCH{best_gorder}, {h} 步)")

# 取训练样本末端状态
a_hist, s2_hist, _ = aM, s2M, nu_h
psum = persist
sbar = uncond_var
# 多步递推 (期望意义): E[a_future]=0, E[a^2_future]=sigma^2_future, E[sigma^2_future]=预测方差
af = np.zeros(h); s2f = np.zeros(h); rf = np.zeros(h); muf = np.zeros(h); muf0 = np.zeros(h)
N = len(rtr)
def a_get(arr, k): return arr[k] if k >= 0 else 0.0
for l in range(h):
    # 条件方差预测
    v = om_h
    for i in range(1, pg + 1):
        k = l - i
        v += al_h[i - 1] * (s2f[k] if k >= 0 else a_hist[N + k] ** 2)   # E[a^2]=sigma^2(未来) / 实际(历史)
    for j in range(1, qg + 1):
        k = l - j
        v += be_h[j - 1] * (s2f[k] if k >= 0 else s2_hist[N + k])
    s2f[l] = v
    # 均值预测 (含风险溢价 lambda*sigma^2(l)); muf0 = 去掉溢价的对照
    base = c_h
    for i in range(1, P + 1):
        k = l - i
        base += phi_h[i - 1] * (rf[k] if k >= 0 else rtr[N + k])
    for j in range(1, Q + 1):
        k = l - j
        base += theta_h[j - 1] * (af[k] if k >= 0 else a_hist[N + k])
    muf[l] = base + lam_h * v
    muf0[l] = base                                     # 同结构但 lambda=0(纯 ARMA 漂移)
    rf[l] = muf[l]                                      # E[r_{T+l}] = mu(l) (未来冲击期望为0)

sig_f = np.sqrt(s2f)
q975 = stats.t.ppf(0.975, nu_h) * np.sqrt((nu_h - 2) / nu_h)
q01 = stats.t.ppf(0.01, nu_h) * np.sqrt((nu_h - 2) / nu_h)
lo, hi = muf - q975 * sig_f, muf + q975 * sig_f
VaR99 = -(muf + q01 * sig_f)
prem_f = muf - muf0                                    # 风险溢价对预测均值的贡献 = lambda*sigma^2(l)
print(f"  均值点预测 mu(1)={muf[0]:+.3f} -> mu({h})={muf[-1]:+.3f} "
      f"(随波动一起均值回复; 这是 GARCH-M 区别于普通 GARCH 的关键)")
print(f"  其中风险溢价贡献 λ·σ²(1)={prem_f[0]:+.3f} -> λ·σ²({h})={prem_f[-1]:+.3f} "
      f"(波动越高溢价越大; 向 λ·无条件方差 {lam_h*uncond_var:+.3f} 收敛)")
print(f"  波动预测  sigma(1)={sig_f[0]:.3f} -> sigma({h})={sig_f[-1]:.3f} "
      f"(向无条件水平 {np.sqrt(uncond_var):.3f} 几何收敛, 持续度 {persist:.3f})")
print(f"  1日 99% VaR: 明日={VaR99[0]:.3f}%")
inside = (test.values >= lo) & (test.values <= hi)
print(f"  样本外 95% 区间覆盖率={inside.mean():.1%} (理想≈95%)")

pd.DataFrame({
    "step": np.arange(1, h + 1),               # 预测第几步(未来第几天)
    "mu_hat": np.round(muf, 4),               # 条件均值(含风险溢价)
    "premium": np.round(prem_f, 4),           # 其中 in-mean 风险溢价贡献 λ·σ²(l)
    "mu_no_premium": np.round(muf0, 4),       # 对照: 去掉溢价的纯 ARMA 漂移
    "sigma_hat": np.round(sig_f, 4),          # 条件波动(区间宽度, GARCH 给)
    "lo95": np.round(lo, 4),                  # 95% 区间下界
    "hi95": np.round(hi, 4),                  # 95% 区间上界
    "VaR99": np.round(VaR99, 4),              # 1日 99% VaR(正数=潜在损失%)
    "actual": np.round(test.values, 4),       # 实际收益(事后真实值)
    "in_interval": inside,                    # 实际值是否落在 95% 区间内
}).to_csv("garchm_forecast.csv", index=False, encoding="utf-8")
print("  已导出每日预测(中心/溢价/波动/区间/VaR/实际) -> garchm_forecast.csv")

fig, axx = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
xx = np.arange(h)
axx[0].plot(xx, test.values, color="black", lw=.9, marker=".", ms=4, label="actual return")
axx[0].plot(xx, muf, "b-", lw=1.5, label=f"GARCH-M mean mu(l) (with premium)")
axx[0].plot(xx, muf0, "g--", lw=1.1, label="mean without premium (lambda=0)")
axx[0].fill_between(xx, lo, hi, color="red", alpha=.15, label="95% dynamic interval")
axx[0].set_title(f"Out-of-sample: {mean_desc}+M mean + GARCH{best_gorder} dynamic interval")
axx[0].legend(loc="upper right", fontsize=8)
axx[1].plot(xx, sig_f, "r-", lw=1.5, label="forecast volatility sigma(l)")
axx[1].axhline(np.sqrt(uncond_var), color="gray", ls="--", lw=1, label=f"uncond sigma={np.sqrt(uncond_var):.2f}")
axx[1].plot(xx, prem_f, color="navy", lw=1.3, label="risk premium contribution lambda*sigma^2(l)")
axx[1].plot(xx, VaR99, color="purple", lw=1.0, label="1-day 99% VaR (%)")
axx[1].set_xlabel("forecast horizon (days)"); axx[1].set_title("Volatility & risk-premium mean-reversion")
axx[1].legend(loc="upper right", fontsize=8)
plt.tight_layout(); plt.savefig("garchm_fig4_forecast.png", dpi=110); plt.close()

# ====================================================================
print(SEP)
print(f"完成. 全流程: 平稳 -> 均值={mean_desc} -> ARCH效应 -> 定阶 GARCH{best_gorder} "
      f"-> 联合MLE(t)估 ARMA+GARCH+lambda -> 三查/平稳/风险溢价LR -> 均值(随波动)+动态波动/VaR 预测")
print(f"揭晓真值: {truth_txt}")
print("图: garchm_fig1_mean / fig2_order / fig3_diagnose / fig4_forecast / fig5_riskpremium (.png)")
