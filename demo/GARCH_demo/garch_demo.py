# -*- coding: utf-8 -*-
"""
GARCH 波动率建模 —— 一条龙完整流程 (均值用 ARMA, 波动用 GARCH(p,q)-t)

  GARCH 与 ARCH 的关系 (本 demo 的主角):
    ARCH(m):     sigma_t^2 = alpha0 + sum_i alpha_i a_{t-i}^2          (只看过去冲击)
    GARCH(p,q):  sigma_t^2 = omega + sum alpha_i a_{t-i}^2 + sum beta_j sigma_{t-j}^2
                 多了 beta*sigma^2 项(把"昨天估出来的方差"也接进来) -> 等价无穷阶 ARCH,
                 用极少参数刻画"高持续"的波动聚集, 故实务里 GARCH(1,1) 远比高阶 ARCH 常用.
    持续度 persistence = alpha+beta (越接近1越"长记忆"); 半衰期 = ln(0.5)/ln(alpha+beta).

  六步法 (下游一律用"定出来的阶", 不写死):
    第0步 准备数据    : 价格 -> 对数收益率 r_t (平稳序列, 建模入口)
    第1步 均值方程    : r_t 的 Ljung-Box 有自相关 -> BIC 网格定阶并拟合 ARMA -> 冲击 a_t = r_t - mu_t
    第2步 ARCH 效应   : 对 a_t^2 查波动聚集(Ljung-Box Q(m) + Engle LM) -> 要不要建波动率模型
    第3步 定阶 (p,q)  : a_t^2 的 ACF/PACF 双拖尾 -> 用 (p,q) 二维 AIC/BIC 网格 (GARCH 与 ARMA 同理)
    第4步 估计参数    : eps_t ~ Student-t(厚尾), MLE 估 omega, alpha, beta, nu; 报 持续度/半衰期
    第5步 模型检验    : 标准化残差三查(均值/波动/分布) + 持续度<1 平稳性
    第6步 使用模型    : ARMA 点预测 mu(l) + 波动率递推 sigma^2(l) 向无条件水平收敛 -> 动态区间 + VaR

  工具说明: Python 的 arch 包均值只支持 AR(无 MA), 故 ARMA 走"两步法":
    statsmodels 建 ARMA 取残差 a_t -> arch 对 a_t 建 GARCH(mean='Zero'). 完全联合估可用 R 的 rugarch.
"""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")               # Windows 控制台强制 UTF-8
warnings.filterwarnings("ignore")                      # 屏蔽网格中个别阶的收敛警告
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                                  # 不弹窗, 直接存图
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model

SEP = "=" * 70

# ====================================================================
# 步骤0  准备数据: 造"价格" -> 转对数收益率 r_t (平稳)
#   均值真值 = ARMA(1,1); 波动真值 = GARCH(1,1)-t (alpha+beta=0.96, 高持续).
#   单位用"百分比收益"(x100), 量纲适中, 是 arch 包推荐做法.
# ====================================================================
omega, alpha_t, beta_t, nu = 0.04, 0.09, 0.90, 7.0     # GARCH(1,1)-t: alpha+beta=0.99 极高持续, df=7 厚尾
phi, theta, mu_true = 0.5, 0.3, 0.05                   # ARMA(1,1) 均值真值
n, burn = 2000, 1000
truth_txt = (f"均值 ARMA(1,1) phi={phi}, theta={theta}, mu={mu_true}; "
             f"波动 GARCH(1,1)-t omega={omega}, alpha={alpha_t}, beta={beta_t}, df={nu} "
             f"(持续度 alpha+beta={alpha_t+beta_t:.2f})")

rng = np.random.default_rng(8)

def sim_garch(rng):                                    # 造 GARCH(1,1)-t 冲击序列 a_t
    e = rng.standard_t(nu, n + burn) * np.sqrt((nu - 2) / nu)   # 标准化 t 新息(单位方差)
    a = np.zeros(n + burn); s2 = np.zeros(n + burn)
    s2[0] = omega / (1 - alpha_t - beta_t); a[0] = np.sqrt(s2[0]) * e[0]
    for t in range(1, n + burn):
        s2[t] = omega + alpha_t * a[t - 1] ** 2 + beta_t * s2[t - 1]   # GARCH(1,1) 方差递推
        a[t] = np.sqrt(s2[t]) * e[t]
    return a

a_full = sim_garch(rng)                                # 先造带波动聚集的冲击
x = np.zeros(n + burn); x[0] = a_full[0]
for t in range(1, n + burn):
    x[t] = phi * x[t - 1] + a_full[t] + theta * a_full[t - 1]      # 再把冲击灌进 ARMA(1,1) 均值
ret = mu_true + x[burn:]
ret = pd.Series(ret).reset_index(drop=True)

price = pd.Series(np.r_[100.0, 100.0 * np.exp(np.cumsum(ret.values) / 100.0)])  # 收益->价格
r = 100.0 * np.log(price / price.shift(1)).dropna().reset_index(drop=True)      # 价格->对数收益 r_t

h = 60                                                  # 留最后 60 天做样本外预测
train, test = r.iloc[:-h], r.iloc[-h:]
pd.DataFrame({"t": np.arange(len(r)), "logret_pct": r.values,
              "split": ["train"] * len(train) + ["test"] * len(test)}).to_csv(
    "garch_returns.csv", index=False, encoding="utf-8")

print(SEP); print("步骤0  准备数据: 价格 -> 对数收益率 r_t")
print(f"  真值: {truth_txt}")
print(f"  样本 n={len(r)} (train={len(train)}, test={len(test)}), 单位=百分比")
print(f"  平稳性 ADF p={adfuller(train)[1]:.3g} -> 平稳, 可建模 [OK]")
print("  已导出 -> garch_returns.csv")

# ====================================================================
# 步骤1  均值方程 (管"中心"): r_t 有自相关 -> BIC 网格定阶 ARMA -> 冲击 a_t
# ====================================================================
print(SEP); print("步骤1  均值方程 (ARMA 定 mu_t, 得冲击 a_t)")
lb_mean = acorr_ljungbox(train, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
print(f"  r_t 的 Ljung-Box(10) p={lb_mean:.3g} -> "
      + ("有自相关, 走 ARMA 分支" if lb_mean < 0.05 else "无自相关(此数据应不出现)"))

best_ic, best_order, grid = np.inf, (1, 1), {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")                    # 个别阶不收敛属正常, 静默
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
    arma = ARIMA(train, order=(best_order[0], 0, best_order[1])).fit()
a_t = arma.resid                                       # 残差 = 冲击
mean_desc = f"ARMA{best_order}"
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
print(f"  -> BIC 最小 = {mean_desc} (BIC={best_ic:.1f}); 拟合参数 (vs 真值):")
for nm, v in arma.params.items():
    tv = {"ar.L1": phi, "ma.L1": theta, "const": mu_true}.get(nm)
    print(f"     {nm:<8}={v:+.3f}" + (f"   | 真值 {tv}" if tv is not None else ""))

def mean_forecast(steps):                              # ARMA 点预测: 逐步回复到无条件均值
    return arma.get_forecast(steps=steps).predicted_mean.values
print(f"  均值模型={mean_desc}; 冲击 a_t = r_t - mu_t (下面研究它的波动)")

fig, ax = plt.subplots(3, 1, figsize=(10, 9))
ax[0].plot(train.values, lw=.6, color="steelblue")
ax[0].set_title(f"Log returns r_t  [mean: {mean_desc}]  —— note volatility clustering")
plot_acf(train, lags=25, ax=ax[1], zero=False); ax[1].set_title("ACF of r_t (mean structure -> ARMA)")
plot_pacf(train, lags=25, ax=ax[2], method="ywm", zero=False); ax[2].set_title("PACF of r_t")
plt.tight_layout(); plt.savefig("garch_fig1_mean.png", dpi=110); plt.close()

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
# 步骤3  定阶 (p,q): a_t^2 的 ACF/PACF 双拖尾 -> (p,q) 二维 AIC/BIC 网格
#   GARCH(p,q) 的 a_t^2 形似 ARMA(max(p,q), p) -> ACF/PACF 都拖尾, 读不出阶, 故走网格.
#   实务铁律: GARCH(1,1) 几乎是默认起点, 高阶很少更优.
# ====================================================================
print(SEP); print("步骤3  定阶 (p,q) (GARCH 阶数, 二维网格)")
PMAX = QMAX = 2
ggrid_bic, ggrid_aic = {}, {}
best_gic, best_gorder = np.inf, (1, 1)
for p in range(1, PMAX + 1):                           # GARCH 的 p>=1 (q=0 即退化为 ARCH)
    for q in range(0, QMAX + 1):
        try:
            rr = arch_model(a_t, mean="Zero", vol="Garch", p=p, q=q, dist="t").fit(disp="off")
            ggrid_bic[(p, q)] = rr.bic; ggrid_aic[(p, q)] = rr.aic
            if rr.bic < best_gic:
                best_gic, best_gorder = rr.bic, (p, q)
        except Exception:
            ggrid_bic[(p, q)] = np.nan; ggrid_aic[(p, q)] = np.nan
p_hat, q_hat = best_gorder
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
print(f"  AIC 选 (p,q)={aic_order} | BIC 选 (p,q)={best_gorder}  >>> 最终定阶 GARCH{best_gorder} (以BIC为准)")
print(f"  注: q=0 那列就是纯 ARCH; 通常含 beta 的 GARCH(1,1) 因'高持续'胜出, 印证 beta 项的价值.")

fig, axx = plt.subplots(1, 2, figsize=(12, 4.5))
plot_acf(a2, lags=25, ax=axx[0], zero=False)
axx[0].set_title("ACF of a_t^2 (slow decay -> high persistence)")
plot_pacf(a2, lags=25, ax=axx[1], method="ywm", zero=False)
axx[1].set_title("PACF of a_t^2 (both tail off -> use (p,q) grid)")
plt.tight_layout(); plt.savefig("garch_fig2_order.png", dpi=110); plt.close()

# ====================================================================
# 步骤4  估计参数: GARCH(p_hat,q_hat) on a_t, Student-t, MLE; 报 持续度/半衰期
# ====================================================================
print(SEP); print(f"步骤4  估计波动率参数 (GARCH{best_gorder} on a_t, Student-t, MLE)")
res = arch_model(a_t, mean="Zero", vol="Garch", p=p_hat, q=q_hat, dist="t").fit(disp="off")
alpha_sum = sum(res.params.filter(like="alpha"))
beta_sum = sum(res.params.filter(like="beta"))
persist = alpha_sum + beta_sum
uncond = np.sqrt(res.params["omega"] / (1 - persist))  # 无条件波动率
halflife = np.log(0.5) / np.log(persist) if persist < 1 else np.inf
print(f"  估计参数 (vs 真值):")
tv_map = {"omega": omega, "alpha[1]": alpha_t, "beta[1]": beta_t, "nu": nu}
for nm, v in res.params.items():
    tv = tv_map.get(nm)
    print(f"     {nm:<10}={v:+.4f}" + (f"   | 真值 {tv}" if tv is not None else ""))
print(f"  持续度 alpha+beta = {persist:.4f} (真值 {alpha_t+beta_t:.2f}; <1 -> 协方差平稳 [OK])")
print(f"  无条件波动率 sigma_uncond = {uncond:.3f}  (波动长期回复到的水平)")
print(f"  波动冲击半衰期 = {halflife:.1f} 天 (持续度越接近1, 半衰期越长 -> 波动记忆越久)")
print(f"  logL={res.loglikelihood:.1f}, AIC={res.aic:.1f}, BIC={res.bic:.1f}")

# ====================================================================
# 步骤5  模型检验: 标准化残差三查 + 平稳性; 不过关 -> 升阶/换分布
# ====================================================================
print(SEP); print("步骤5  模型检验 (标准化残差三查 + 平稳性)")
z = res.std_resid.dropna()
p_mean = acorr_ljungbox(z, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
p_vol = acorr_ljungbox(z ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
sk, ku = stats.skew(z), stats.kurtosis(z, fisher=False)
print(f"  ① 均值({mean_desc}够不够) ~a_t Ljung-Box(10) p={p_mean:.3g} -> "
      + ("均值方程充分 [OK]" if p_mean > 0.05 else "均值阶数不够, 改 ARMA [NO]"))
print(f"  ② 波动方程 ~a_t^2 Ljung-Box(10) p={p_vol:.3g} -> "
      + ("波动洗净, GARCH 充分 [OK]" if p_vol > 0.05 else "仍有 ARCH, 升阶/换模型 [NO]"))
print(f"  ③ 分布   偏度={sk:+.2f}, 峰度={ku:.2f} (正态=3; t 新息应仍略厚尾)")
print(f"  ④ 平稳性  alpha+beta={persist:.4f} < 1 -> 方差过程协方差平稳 [OK]")
verdict = "三查通过 + 平稳, GARCH 模型充分" if (p_mean > 0.05 and p_vol > 0.05 and persist < 1) \
    else "有一项不过关, 需迭代"
print(f"  >>> {verdict}")

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].plot(res.conditional_volatility, lw=.7, color="darkred")
ax[0, 0].axhline(uncond, color="gray", ls="--", lw=1, label=f"uncond sigma={uncond:.2f}")
ax[0, 0].set_title(f"Conditional volatility sigma_t  [GARCH{best_gorder}]"); ax[0, 0].legend()
plot_acf(z ** 2, lags=25, ax=ax[0, 1], zero=False); ax[0, 1].set_title("ACF of std-resid^2 (white? -> vol cleaned)")
ax[1, 0].hist(z, bins=40, density=True, alpha=.7, edgecolor="k")
xs = np.linspace(z.min(), z.max(), 200); ax[1, 0].plot(xs, stats.norm.pdf(xs), "r--", label="N(0,1)")
ax[1, 0].set_title("Std-resid histogram (fat tails)"); ax[1, 0].legend()
stats.probplot(z, dist="norm", plot=ax[1, 1]); ax[1, 1].set_title("Q-Q plot vs Normal")
plt.tight_layout(); plt.savefig("garch_fig3_diagnose.png", dpi=110); plt.close()

# ====================================================================
# 步骤6  使用模型: ARMA 均值点预测 + GARCH 波动率递推 -> 动态区间 + VaR
#   GARCH(1,1) 多步波动: sigma^2(l) = omega + (alpha+beta) * sigma^2(l-1)
#     -> 几何收敛到无条件方差 omega/(1-alpha-beta), 持续度决定收敛快慢.
# ====================================================================
print(SEP); print(f"步骤6  预测 (均值 {mean_desc} + 波动 GARCH{best_gorder}, {h} 步)")
mu_path = mean_forecast(h)
sig_path = np.sqrt(res.forecast(horizon=h, reindex=False).variance.values[-1])
nu_hat = res.params.get("nu", 8.0)
q975 = stats.t.ppf(0.975, nu_hat) * np.sqrt((nu_hat - 2) / nu_hat)
q01 = stats.t.ppf(0.01, nu_hat) * np.sqrt((nu_hat - 2) / nu_hat)
lo, hi = mu_path - q975 * sig_path, mu_path + q975 * sig_path
VaR99 = -(mu_path + q01 * sig_path)
print(f"  均值点预测 mu(1)={mu_path[0]:+.3f} -> mu({h})={mu_path[-1]:+.3f} (ARMA 逐步回复到长期均值)")
print(f"  波动预测  sigma(1)={sig_path[0]:.3f} -> sigma({h})={sig_path[-1]:.3f} "
      f"(向无条件水平 {uncond:.3f} 几何收敛, 持续度 {persist:.3f} 决定收敛快慢)")
print(f"  1日 99% VaR: 明日={VaR99[0]:.3f}%  (区间随市场松紧动态伸缩)")
inside = (test.values >= lo) & (test.values <= hi)
print(f"  样本外 95% 区间覆盖率={inside.mean():.1%} (理想≈95%)")

pd.DataFrame({
    "step": np.arange(1, h + 1),               # 预测第几步(未来第几天)
    "mu_hat": np.round(mu_path, 4),            # 条件均值(中心, ARMA 给)
    "sigma_hat": np.round(sig_path, 4),        # 条件波动(区间宽度, GARCH 给)
    "lo95": np.round(lo, 4),                   # 95% 区间下界
    "hi95": np.round(hi, 4),                   # 95% 区间上界
    "VaR99": np.round(VaR99, 4),               # 1日 99% VaR(正数=潜在损失%)
    "actual": np.round(test.values, 4),        # 实际收益(事后真实值, 用于对照)
    "in_interval": inside,                     # 实际值是否落在 95% 区间内
}).to_csv("garch_forecast.csv", index=False, encoding="utf-8")
print("  已导出每日预测(中心/波动/区间/VaR/实际) -> garch_forecast.csv")

fig, axx = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
xx = np.arange(h)
axx[0].plot(xx, test.values, color="black", lw=.9, marker=".", ms=4, label="actual return")
axx[0].plot(xx, mu_path, "b-", lw=1.3, label=f"{mean_desc} mean forecast")
axx[0].fill_between(xx, lo, hi, color="red", alpha=.15, label="95% dynamic interval")
axx[0].set_title(f"Out-of-sample: mean {mean_desc} + vol GARCH{best_gorder}"); axx[0].legend(loc="upper right")
axx[1].plot(xx, sig_path, "r-", lw=1.5, label="forecast volatility sigma(l)")
axx[1].axhline(uncond, color="gray", ls="--", lw=1, label=f"uncond sigma={uncond:.2f}")
axx[1].plot(xx, VaR99, color="purple", lw=1.2, label="1-day 99% VaR (%)")
axx[1].set_xlabel("forecast horizon (days)"); axx[1].set_title("Volatility mean-reversion & VaR forecast")
axx[1].legend(loc="upper right")
plt.tight_layout(); plt.savefig("garch_fig4_forecast.png", dpi=110); plt.close()

# ====================================================================
print(SEP)
print(f"完成. 全流程: 平稳 -> 均值={mean_desc} -> ARCH效应 -> 定阶 GARCH{best_gorder} "
      f"-> MLE(t) -> 三查/平稳 -> 均值+动态波动/VaR 预测")
print(f"揭晓真值: {truth_txt}")
print("图: garch_fig1_mean / garch_fig2_order / garch_fig3_diagnose / garch_fig4_forecast (.png)")
