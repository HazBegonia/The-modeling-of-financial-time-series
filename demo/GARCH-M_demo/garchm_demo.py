# -*- coding: utf-8 -*-
"""
GARCH-M 风险溢价建模 —— 一条龙完整流程 (均值 ARMA + in-mean 风险溢价项; 波动 GARCH(m,s)-t, 联合 MLE)

  GARCH-M = GARCH-in-Mean. 金融核心直觉"高风险要高回报"(风险-收益权衡): 市场越动荡(波动越大),
    投资者要求的预期收益越高. 普通 GARCH 只让波动随时间变, 均值里却没有波动; GARCH-M 把"今天的
    条件波动"直接塞进均值方程("M"=in the mean), 让"高波动 -> 高预期收益"这件事能被估出来.

  ★ 模型 (式 3.23):
        r_t     = mu + c*sigma_t^2 + a_t,      a_t = sigma_t * eps_t      <- 均值多了 c*sigma^2
        sigma_t^2 = alpha0 + sum alpha_i a_{t-i}^2 + sum beta_j sigma_{t-j}^2   <- 波动就是普通 GARCH
      · c 叫风险溢价参数(risk premium): c>0 => 收益与其波动正相关(Merton ICAPM 直觉); c=0 退回普通 GARCH.
      · 风险溢价项 c*g(sigma_t) 有三种常见形式 (书表 3-2, 本 demo 步骤4 用 AIC/BIC 三选一):
            g(sigma)=sigma^2   (var.in.mean, 用条件方差)   <- 本 demo 真值
            g(sigma)=sigma     (sd.in.mean,  用条件标准差)
            g(sigma)=ln sigma^2(logvar.in.mean, 用对数条件方差)

  ⚠️ 一条必须记住的理论性质(书里明确强调):
      (3.23) 的 GARCH-M 蕴涵 r_t 存在序列相关, 而这种序列相关是由波动率过程 {sigma_t^2} 的序列相关
      导致的 —— 风险溢价的存在, 是历史收益率具有序列相关性的"另一个原因".
      => 对建模流程的直接影响: r_t 的自相关未必来自 ARMA 动态, 可能是被 c*sigma_t^2 制造出来的.
         所以定均值阶时不能盲目对着 r_t 的 ACF 硬套 ARMA, 稳妥做法是先设"简约均值"(常数或低阶 AR),
         把明显的线性相关留到步骤5 和风险溢价项一起联合判断.

  和普通 GARCH 的三点区别(一句话记忆):
      1. 均值里多一项 c*g(sigma_t)  —— 波动进均值.
      2. 必须"联合估计"          —— sigma_t^2 同时进均值和方差, 两方程耦合, 两步法(先 ARMA 取残差、
                                     再对残差建 GARCH)会有偏; 故本 demo 手写联合 MLE (arch 包不支持 in-mean).
      3. 会给 r_t 制造序列相关   —— 定均值阶时别把风险溢价造出来的相关误当成 ARMA.

  阶 (m,s) 不是默认 (1,1): 波动方程和普通 GARCH 一样"不好定" —— m(ARCH 阶)借 a_t^2 的 PACF 量级,
      s(GARCH 阶)只在低阶候选里靠 AIC/BIC + 残差诊断挑. (1,1) 是"选出来的结果", 不是前提.

  八步法 (下游一律用"定出来的阶/选出的风险溢价形式/联合估出的参数", 不写死):
    第0步 准备数据    : 价格 -> 对数收益率 r_t (平稳序列, 建模入口)
    第1步 均值初步设定: r_t 的 Ljung-Box 有相关 -> 简约 ARMA 初定阶(暂得冲击 a_t; 阶留到步骤5复核)
    第2步 检验 ARCH   : 对 a_t^2 查波动聚集(Ljung-Box + Engle LM) -> 有 ARCH 才谈得上 GARCH-M
    第3步 波动定阶(m,s): a_t^2 的 PACF + 低阶 (m,s) 二维 AIC/BIC 网格 (不默认 (1,1))
    第4步 选溢价形式  : in-mean 三选一(sigma^2/sigma/ln sigma^2) 比 AIC/BIC + 定新息分布(t)
    第5步 联合估计    : 手写 MLE 一次估 ARMA + GARCH + c (Student-t); 重点看风险溢价 c
    第6步 模型检验    : 标准化残差三查 + 平稳性 + 风险溢价检验(LR/t: c=0 vs GARCH-M)(也是定阶裁判)
    第7步 解释与应用  : 先波动率预测 -> 代回均值预测(随波动起伏!) -> 动态区间 + VaR; 解读风险溢价
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

# 风险溢价 in-mean 形式 g(sigma_t) (书表 3-2): 步骤4 用 AIC/BIC 三选一, 全流程一律走它 ----------
MFORMS = {"var": "sigma_t^2", "sd": "sigma_t", "log": "ln sigma_t^2"}    # key -> 展示名


def inmean(v, mform):
    """风险溢价项 g(sigma^2): v=条件方差 sigma_t^2. var=方差 / sd=标准差 / log=对数方差 (表 3-2)."""
    if mform == "sd":
        return np.sqrt(v)
    if mform == "log":
        return np.log(v)
    return v                                            # "var": 直接用 sigma_t^2 (本 demo 真值)


# ====================================================================
# 步骤0  准备数据: 造"价格" -> 转对数收益率 r_t (平稳)
#   均值真值 = ARMA(1,1) + 风险溢价 c*sigma_t^2 (var.in.mean); 波动真值 = GARCH(1,1)-t.
#   单位用"百分比收益"(x100), 量纲适中.
# ====================================================================
C0, PHI, THETA, LAM = 0.0, 0.5, -0.3, 0.20            # 均值: 常数/AR1/MA1/风险溢价参数 c
OMEGA, ALPHA, BETA, NU = 0.05, 0.08, 0.90, 7.0        # GARCH(1,1)-t: 持续度 0.98, df=7 厚尾
MFORM_TRUE = "var"                                    # 真值用 var.in.mean (c*sigma_t^2)
n, burn = 2500, 1500
uncond_var_true = OMEGA / (1 - ALPHA - BETA)
truth_txt = (f"均值 ARMA(1,1) c={C0}, phi={PHI}, theta={THETA} + 风险溢价 c={LAM} (in-mean={MFORMS[MFORM_TRUE]}); "
             f"波动 GARCH(1,1)-t omega={OMEGA}, alpha={ALPHA}, beta={BETA}, df={NU} "
             f"(持续度 {ALPHA+BETA:.2f}, 无条件方差 {uncond_var_true:.2f})")

rng = np.random.default_rng(20260616)


def sim_garchm(rng):                                   # 造 ARMA(1,1)-GARCH(1,1)-M 收益序列
    N = n + burn
    e = rng.standard_t(NU, N) * np.sqrt((NU - 2) / NU) # 标准化 t 新息(单位方差)
    s2 = np.zeros(N); a = np.zeros(N); r = np.zeros(N)
    s2[0] = uncond_var_true; a[0] = np.sqrt(s2[0]) * e[0]
    r[0] = C0 + LAM * inmean(s2[0], MFORM_TRUE) + a[0]         # t=0 无 AR/MA 项
    for t in range(1, N):
        s2[t] = OMEGA + ALPHA * a[t - 1] ** 2 + BETA * s2[t - 1]    # GARCH 方差递推
        a[t] = np.sqrt(s2[t]) * e[t]
        r[t] = (C0 + LAM * inmean(s2[t], MFORM_TRUE)              # 均值 = 常数 + 风险溢价 + ARMA
                + PHI * r[t - 1] + THETA * a[t - 1] + a[t])
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
# 步骤1  均值方程"初步设定"(简约): r_t 有相关 -> BIC 网格初定 ARMA 阶 (暂得冲击 a_t)
#   ⚠️ 关键: r_t 的自相关一部分来自 in-mean(c*sigma_t^2 高度持续 -> 灌进缓慢自相关), 不全是 ARMA.
#      故这里只"初步"定阶、保持简约; 真正的阶/参数留到步骤5 联合估 + 步骤6 诊断复核.
# ====================================================================
print(SEP); print("步骤1  均值方程初步设定 (简约 ARMA 初定阶, 暂得冲击 a_t)")
lb_mean = acorr_ljungbox(train, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
print(f"  r_t 的 Ljung-Box(10) p={lb_mean:.3g} -> "
      + ("有自相关(注意: 部分来自 in-mean, 非全是 ARMA)" if lb_mean < 0.05 else "无自相关"))

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
a_t = arma0.resid                                      # 暂用的冲击(简约 ARMA, 仅用于步骤2/3/4起点)
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
print(f"  -> BIC 初定 = {mean_desc} (BIC={best_ic:.1f}); 简约起步, in-mean 风险溢价项留到步骤5联合估")

fig, ax = plt.subplots(3, 1, figsize=(10, 9))
ax[0].plot(train.values, lw=.6, color="steelblue")
ax[0].set_title(f"Log returns r_t  [mean: {mean_desc}+in-mean]  —— clustering & premium-driven autocorr")
plot_acf(train, lags=25, ax=ax[1], zero=False); ax[1].set_title("ACF of r_t (mean structure -> ARMA, partly from in-mean)")
plot_pacf(train, lags=25, ax=ax[2], method="ywm", zero=False); ax[2].set_title("PACF of r_t")
plt.tight_layout(); plt.savefig("garchm_fig1_mean.png", dpi=110); plt.close()

# ====================================================================
# 步骤2  检验 ARCH 效应 (要不要建波动率模型): 看 a_t^2 会不会聚集
#   有 ARCH 效应(波动可预测) 才有引入 GARCH-M 的意义 —— sigma_t^2 得先"会动", 放进均值才有信息.
# ====================================================================
print(SEP); print("步骤2  检验 ARCH 效应 (波动聚集? 有 ARCH 才谈得上 GARCH-M)")
a2 = a_t ** 2
for m in (5, 10, 20):
    pq = acorr_ljungbox(a2, lags=[m], return_df=True)["lb_pvalue"].iloc[0]
    print(f"  [方法1] a_t^2 的 Ljung-Box Q({m}) p={pq:.3g} -> "
          + ("有 ARCH 效应" if pq < 0.05 else "不显著"))
lm_stat, lm_p, _, _ = het_arch(a_t, nlags=10)
print(f"  [方法2] Engle LM(lags=10): LM={lm_stat:.2f}, p={lm_p:.3g} -> "
      + ("有 ARCH 效应, 波动可预测, 值得上 GARCH-M [OK]" if lm_p < 0.05 else "不显著, 收工"))

# ====================================================================
# 步骤3  波动方程定阶 (m,s): a_t^2 的 PACF 量级定 m + 低阶 (m,s) 二维 AIC/BIC 网格
#   与普通 GARCH 同理: s(GARCH 阶)不好定, 只在低阶试; (1,1) 是选出来的, 不写死默认.
# ====================================================================
print(SEP); print("步骤3  波动方程定阶 (m,s) (a_t^2 PACF + 低阶二维网格; 不默认 (1,1))")
pacf_a2 = plot_pacf  # (画在图2); 数值上前几阶 PACF 量级给 m 的直觉
PMAX = QMAX = 2
ggrid_bic, ggrid_aic = {}, {}
best_gic, best_gorder = np.inf, (1, 1)
for p in range(1, PMAX + 1):                           # p = ARCH 阶 m
    for q in range(0, QMAX + 1):                       # q = GARCH 阶 s (q=0 即纯 ARCH)
        try:
            rr = arch_model(a_t, mean="Zero", vol="Garch", p=p, q=q, dist="t").fit(disp="off")
            ggrid_bic[(p, q)] = rr.bic; ggrid_aic[(p, q)] = rr.aic
            if rr.bic < best_gic:
                best_gic, best_gorder = rr.bic, (p, q)
        except Exception:
            ggrid_bic[(p, q)] = np.nan; ggrid_aic[(p, q)] = np.nan
pg, qg = best_gorder
aic_order = min(ggrid_aic, key=lambda k: ggrid_aic[k])
print("  BIC 网格 (行=ARCH阶 m=p, 列=GARCH阶 s=q; q=0 即 ARCH(p); [*]=最优):")
print("         " + "".join(f"q={q:<8}" for q in range(QMAX + 1)))
for p in range(1, PMAX + 1):
    cells = ""
    for q in range(QMAX + 1):
        v = ggrid_bic.get((p, q), np.nan)
        star = "*" if (p, q) == best_gorder else " "
        cells += (f"{v:8.1f}{star} " if np.isfinite(v) else "   --     ")
    print(f"     p={p} {cells}")
print(f"  AIC 选 (m,s)={aic_order} | BIC 选 (m,s)={best_gorder}  >>> 波动定阶 GARCH{best_gorder} (以BIC为准, 选出来的不是默认)")

fig, axx = plt.subplots(1, 2, figsize=(12, 4.5))
plot_acf(a2, lags=25, ax=axx[0], zero=False)
axx[0].set_title("ACF of a_t^2 (slow decay -> high persistence)")
plot_pacf(a2, lags=25, ax=axx[1], method="ywm", zero=False)
axx[1].set_title("PACF of a_t^2 (order m hint; tails off -> use (m,s) grid)")
plt.tight_layout(); plt.savefig("garchm_fig2_order.png", dpi=110); plt.close()

# ====================================================================
# 步骤4-6 的公共零件: 手写联合 MLE (ARMA(P,Q) + GARCH(pg,qg) + 风险溢价 c, Student-t)
#   参数向量: [c, phi(1..P), theta(1..Q), lambda(=c), omega, alpha(1..pg), beta(1..qg), nu]
#   in-mean 由 t-1 信息算出 sigma_t^2 再取 g(.), 均值方程无同期内生, 递推良定义.
# ====================================================================
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


def garchm_filter(th, y, with_M=True, mform="var"):    # 递推 a_t 与 sigma_t^2 (随 in-mean 形式)
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
        if v <= 0:
            return a, s2, nu                           # 交给护栏拒绝
        m = c + lam * inmean(v, mform)                 # 条件均值: 常数 + 风险溢价 g(sigma) + ARMA
        for i in range(1, P + 1):
            m += phi[i - 1] * (y[t - i] if t - i >= 0 else rbar)
        for j in range(1, Q + 1):
            m += theta[j - 1] * (a[t - j] if t - j >= 0 else 0.0)
        a[t] = y[t] - m
    return a, s2, nu


def stdt_logpdf(z, nu):                                # 标准化(单位方差)Student-t 对数密度
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log((nu - 2) * np.pi)
            - (nu + 1) / 2 * np.log(1 + z ** 2 / (nu - 2)))


def negloglik(th, y, with_M=True, mform="var"):
    c, phi, theta, lam, omega, alpha, beta, nu = unpack(th, with_M)
    if omega <= 0 or nu <= 2.01 or np.any(alpha <= 0) or np.any(beta < 0) \
            or alpha.sum() + beta.sum() >= 0.999:      # 平稳/正定护栏
        return 1e10
    a, s2, nu = garchm_filter(th, y, with_M, mform)
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return 1e10
    z = a / np.sqrt(s2)
    ll = np.sum(stdt_logpdf(z, nu) - 0.5 * np.log(s2))
    return -ll if np.isfinite(ll) else 1e10


def fit_mle(with_M=True, mform="var", polish=True):    # 起点用普通 ARMA/arch 估计, 再联合优化
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
    res = optimize.minimize(negloglik, np.array(x0, float), args=(train.values, with_M, mform),
                            method="L-BFGS-B", bounds=bnds)
    if polish:                                         # 二次打磨(仅最终模型用, 定选时省时)
        res = optimize.minimize(negloglik, res.x, args=(train.values, with_M, mform),
                                method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
    return res


def ic_of(res):                                        # 从优化结果算 (logL, AIC, BIC)
    llf = -res.fun; k = len(res.x)
    return llf, 2 * k - 2 * llf, k * np.log(len(train)) - 2 * llf


# ====================================================================
# 步骤4  选风险溢价形式 (表 3-2 三选一) + 新息分布
#   in-mean 用 sigma^2 / sigma / ln sigma^2 各估一遍, 比 AIC/BIC 挑; 新息用 Student-t (厚尾).
# ====================================================================
print(SEP); print("步骤4  选风险溢价形式 (表 3-2: sigma^2 / sigma / ln sigma^2) + 新息分布")
print("  [4a] in-mean 形式三选一 (低阶联合 MLE 比 AIC/BIC; 越小越好, [*]=最优):")
form_sel = {}
for key in ("var", "sd", "log"):
    rr = fit_mle(with_M=True, mform=key, polish=False)     # 定选阶段用快速拟合(不打磨)省时
    llf, aic, bic = ic_of(rr)
    lam_k = unpack(rr.x, True)[3]
    form_sel[key] = (aic, bic, lam_k)
use_form = min(form_sel, key=lambda k: form_sel[k][1])     # 以 BIC 为准
cmd = {"var": "var.in.mean", "sd": "sd.in.mean", "log": "logvar.in.mean"}
print(f"       {'形式':<16}{'g(sigma)':<14}{'AIC':>9}{'BIC':>10}{'   c_hat':>10}")
for key in ("var", "sd", "log"):
    aic, bic, lam_k = form_sel[key]
    star = "*" if key == use_form else " "
    print(f"       {cmd[key]:<16}{MFORMS[key]:<14}{aic:9.1f}{bic:9.1f}{star}{lam_k:+9.3f}")
print(f"       >>> BIC 选 {cmd[use_form]} (g={MFORMS[use_form]})"
      + ("  与真值一致 [OK]" if use_form == MFORM_TRUE else f"  (真值本是 {cmd[MFORM_TRUE]}, 三形式经验上难分辨)"))
kurt_probe = stats.kurtosis(a_t / a_t.std(), fisher=False)
print(f"  [4b] 新息分布: 冲击标准化峰度={kurt_probe:.2f} (正态=3) -> 厚尾, 取 Student-t "
      f"(正态低估尾部; 更尖峰可试 GED)")

# ====================================================================
# 步骤5  联合估计 ARMA(P,Q)-GARCH(pg,qg)-M (手写 MLE, Student-t, 用选定的 in-mean 形式)
#   ⚠️ 必须"联合": in-mean 把 sigma_t^2 喂回均值, 两方程耦合, 两步法(先ARMA后GARCH)会有偏.
# ====================================================================
print(SEP); print(f"步骤5  联合估计 {mean_desc}-GARCH{best_gorder}-M ({cmd[use_form]}, 手写 MLE, Student-t)")
resM = fit_mle(with_M=True, mform=use_form, polish=True)
thM = resM.x
llf_M, aicM, bicM = ic_of(resM)
c_h, phi_h, theta_h, lam_h, om_h, al_h, be_h, nu_h = unpack(thM, True)
persist = al_h.sum() + be_h.sum()
uncond_var = om_h / (1 - persist)
halflife = np.log(0.5) / np.log(persist) if persist < 1 else np.inf
kM = len(thM)


def num_hessian(f, x):                                 # 数值 Hessian -> 标准误 -> t 值
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
    H = num_hessian(lambda z: negloglik(z, train.values, True, use_form), thM)
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
lam_t = thM[lam_idx] / se[lam_idx] if np.isfinite(se[lam_idx]) and se[lam_idx] > 0 else np.nan
print(f"  >>> 风险溢价 c(lambda)={lam_h:+.4f} (真值 {LAM}); t={lam_t:+.2f} -> "
      + ("显著(|t|>1.96): 高波动确实索取更高预期收益 [OK]"
         if np.isfinite(lam_t) and abs(lam_t) > 1.96 else "未达显著, 看下一步 LR 检验"))
print(f"  持续度 alpha+beta = {persist:.4f} (真值 {ALPHA+BETA:.2f}; <1 -> 协方差平稳 [OK])")
print(f"  无条件方差 = {uncond_var:.3f} (真值 {uncond_var_true:.2f}); 波动冲击半衰期 = {halflife:.1f} 天")
print(f"  logL={llf_M:.1f}, AIC={aicM:.1f}, BIC={bicM:.1f}")

# ====================================================================
# 步骤6  模型检验: 标准化残差三查 + 平稳性 + 风险溢价检验(LR: c=0) (也是定阶裁判)
# ====================================================================
print(SEP); print("步骤6  模型检验 (标准化残差三查 + 平稳性 + 风险溢价 LR 检验)")
aM, s2M, _ = garchm_filter(thM, train.values, True, use_form)
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

# 风险溢价 LR 检验: 受限模型(c=0, 即普通 ARMA-GARCH) vs GARCH-M
res0 = fit_mle(with_M=False, mform=use_form, polish=True)
llf_0 = -res0.fun
LR = 2 * (llf_M - llf_0)
p_LR = stats.chi2.sf(LR, df=1)
print(f"  ⑤ 风险溢价 LR 检验 (H0: c=0, 即退回普通 GARCH):")
print(f"     logL(GARCH-M)={llf_M:.2f} vs logL(GARCH, c=0)={llf_0:.2f}")
print(f"     LR=2*Δ={LR:.2f} ~ chi2(1), p={p_LR:.3g} -> "
      + ("拒绝 c=0, 风险溢价显著存在 [OK]" if p_LR < 0.05 else "不显著, in-mean 项可省(退回 GARCH)"))
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

# 风险溢价"招牌图": 条件均值随条件方差水涨船高 (风险-收益正相关的直接证据)
muM = train.values - aM                                # 拟合的条件均值 mu_t = r_t - a_t
fig, axx = plt.subplots(1, 2, figsize=(12, 4.8))
axx[0].scatter(s2M, train.values, s=5, alpha=.15, color="steelblue", label="r_t")
bins = np.linspace(s2M.min(), np.percentile(s2M, 99), 12)
bidx = np.digitize(s2M, bins)
bx = [s2M[bidx == k].mean() for k in range(1, len(bins)) if (bidx == k).sum() > 5]
by = [train.values[bidx == k].mean() for k in range(1, len(bins)) if (bidx == k).sum() > 5]
axx[0].plot(bx, by, "o-", color="black", ms=5, lw=1.2, label="binned mean of r_t")
xs2 = np.linspace(s2M.min(), np.percentile(s2M, 99), 100)
axx[0].plot(xs2, c_h + lam_h * inmean(xs2, use_form), "r-", lw=2, label=f"fitted premium c+λ·g(σ) (λ={lam_h:.3f})")
axx[0].set_xlabel("conditional variance sigma_t^2"); axx[0].set_ylabel("return r_t")
axx[0].set_title("Risk-return: higher variance -> higher mean return"); axx[0].legend(fontsize=8)
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
# 步骤7  解释与应用(预测): 先波动率预测 -> 代回均值预测(随波动起伏) -> 动态区间 + VaR
#   GARCH-M 招牌: mu(l) = c + lambda*g(sigma^2(l)) + ARMA, 均值随波动预测一起均值回复.
#   ⚠️ 顺序: 均值预测依赖波动预测 sigma^2(l), 所以必须"先波动、后均值".
# ====================================================================
print(SEP); print(f"步骤7  预测 (先波动 GARCH{best_gorder} -> 代回均值 {mean_desc}+M, {h} 步)")
a_hist, s2_hist = aM, s2M
af = np.zeros(h); s2f = np.zeros(h); rf = np.zeros(h); muf = np.zeros(h); muf0 = np.zeros(h)
prem_f = np.zeros(h)
N = len(rtr)
for l in range(h):
    # (1) 先算条件方差预测 sigma^2(l): E[a^2_future]=sigma^2_future
    v = om_h
    for i in range(1, pg + 1):
        k = l - i
        v += al_h[i - 1] * (s2f[k] if k >= 0 else a_hist[N + k] ** 2)
    for j in range(1, qg + 1):
        k = l - j
        v += be_h[j - 1] * (s2f[k] if k >= 0 else s2_hist[N + k])
    s2f[l] = v
    # (2) 再把 sigma^2(l) 代回均值: 常数 + 风险溢价 g(sigma^2(l)) + ARMA
    base = c_h
    for i in range(1, P + 1):
        k = l - i
        base += phi_h[i - 1] * (rf[k] if k >= 0 else rtr[N + k])
    for j in range(1, Q + 1):
        k = l - j
        base += theta_h[j - 1] * (af[k] if k >= 0 else a_hist[N + k])
    prem_f[l] = lam_h * inmean(v, use_form)            # 风险溢价对均值的贡献
    muf[l] = base + prem_f[l]
    muf0[l] = base                                     # 对照: 去掉溢价的纯 ARMA 漂移
    rf[l] = muf[l]                                      # E[r_{T+l}] = mu(l) (未来冲击期望为0)

sig_f = np.sqrt(s2f)
q975 = stats.t.ppf(0.975, nu_h) * np.sqrt((nu_h - 2) / nu_h)
q01 = stats.t.ppf(0.01, nu_h) * np.sqrt((nu_h - 2) / nu_h)
lo, hi = muf - q975 * sig_f, muf + q975 * sig_f
VaR99 = -(muf + q01 * sig_f)
prem_inf = lam_h * inmean(uncond_var, use_form)        # 溢价的长期收敛值
print(f"  波动预测(先)  sigma(1)={sig_f[0]:.3f} -> sigma({h})={sig_f[-1]:.3f} "
      f"(向无条件水平 {np.sqrt(uncond_var):.3f} 几何收敛, 持续度 {persist:.3f})")
print(f"  均值预测(后)  mu(1)={muf[0]:+.3f} -> mu({h})={muf[-1]:+.3f} "
      f"(随波动一起均值回复; 这是 GARCH-M 区别于普通 GARCH 的关键)")
print(f"  其中风险溢价贡献 λ·g(σ²)(1)={prem_f[0]:+.3f} -> λ·g(σ²)({h})={prem_f[-1]:+.3f} "
      f"(波动越高溢价越大; 向 {prem_inf:+.3f} 收敛)")
print(f"  1日 99% VaR: 明日={VaR99[0]:.3f}%")
inside = (test.values >= lo) & (test.values <= hi)
print(f"  样本外 95% 区间覆盖率={inside.mean():.1%} (理想≈95%)")

pd.DataFrame({
    "step": np.arange(1, h + 1),               # 预测第几步(未来第几天)
    "mu_hat": np.round(muf, 4),               # 条件均值(含风险溢价)
    "premium": np.round(prem_f, 4),           # 其中 in-mean 风险溢价贡献 λ·g(σ²)(l)
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
axx[0].plot(xx, muf, "b-", lw=1.5, label="GARCH-M mean mu(l) (with premium)")
axx[0].plot(xx, muf0, "g--", lw=1.1, label="mean without premium (c=0)")
axx[0].fill_between(xx, lo, hi, color="red", alpha=.15, label="95% dynamic interval")
axx[0].set_title(f"Out-of-sample: {mean_desc}+M mean + GARCH{best_gorder} dynamic interval")
axx[0].legend(loc="upper right", fontsize=8)
axx[1].plot(xx, sig_f, "r-", lw=1.5, label="forecast volatility sigma(l)")
axx[1].axhline(np.sqrt(uncond_var), color="gray", ls="--", lw=1, label=f"uncond sigma={np.sqrt(uncond_var):.2f}")
axx[1].plot(xx, prem_f, color="navy", lw=1.3, label="risk premium contribution λ·g(σ²)(l)")
axx[1].plot(xx, VaR99, color="purple", lw=1.0, label="1-day 99% VaR (%)")
axx[1].set_xlabel("forecast horizon (days)"); axx[1].set_title("Volatility & risk-premium mean-reversion")
axx[1].legend(loc="upper right", fontsize=8)
plt.tight_layout(); plt.savefig("garchm_fig4_forecast.png", dpi=110); plt.close()

# ====================================================================
print(SEP)
print(f"完成. 全流程: 平稳 -> 简约均值={mean_desc} -> ARCH效应 -> 波动定阶 GARCH{best_gorder} "
      f"-> 选溢价形式 {cmd[use_form]} -> 联合MLE(t)估 ARMA+GARCH+c -> 三查/平稳/风险溢价LR "
      f"-> 先波动后均值 预测 + 动态区间/VaR")
print(f"揭晓真值: {truth_txt}")
print("图: garchm_fig1_mean / fig2_order / fig3_diagnose / fig4_forecast / fig5_riskpremium (.png)")
