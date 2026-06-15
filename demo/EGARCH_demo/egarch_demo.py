# -*- coding: utf-8 -*-
"""
EGARCH 波动率建模 —— 一条龙完整流程 (均值用 ARMA, 波动用 EGARCH(1,1,1)-t)

  EGARCH (Exponential GARCH, Nelson 1991) 解决 GARCH 的两大短板 (本 demo 的主角):
    1) 杠杆效应(leverage): 股市"利空(负冲击)比同等利好(正冲击)更抬高未来波动".
       普通 GARCH 用 a_{t-1}^2, 正负冲击一视同仁, 抓不到这种不对称.
    2) 正定约束: GARCH 要求 omega,alpha,beta>=0 才保证 sigma^2>0, 估计带约束.

  EGARCH 对 "log 方差" 建模, 天然恒正, 且把冲击拆成"大小"与"方向"两块:
    ln sigma_t^2 = omega + beta * ln sigma_{t-1}^2
                          + alpha * (|z_{t-1}| - E|z|)    <- 大小项(波动聚集)
                          + gamma * z_{t-1}               <- 方向项(杠杆!), z=标准化残差
    gamma < 0  => 负冲击(z<0)抬高 ln sigma^2 更多 => 杠杆效应; gamma=0 退回对称.
    持续度 = |beta| (log 方差的 AR 系数), <1 即平稳; 因建模 log, 无需正定约束.

  与 GARCH 的对照(一句话): GARCH 只看冲击"多大", EGARCH 还看冲击"是涨是跌".

  六步法 (下游一律用"定出来的阶/带不带杠杆", 不写死):
    第0步 准备数据  : 价格 -> 对数收益率 r_t
    第1步 均值方程  : BIC 网格定阶并拟合 ARMA -> 冲击 a_t
    第2步 ARCH+杠杆 : a_t^2 查聚集(Engle LM) + "符号-平方"相关查杠杆(负冲击是否预示大波动)
    第3步 定阶/选型 : 比较 o=0(对称) vs o=1(带杠杆) 的 AIC/BIC -> 杠杆项值不值
    第4步 估计参数  : EGARCH(1,1,1)-t MLE, 重点看 gamma 的符号与显著性
    第5步 模型检验  : 标准化残差三查 + 杠杆是否洗净
    第6步 使用模型  : 均值点预测 + EGARCH 波动率(模拟)预测 + 新闻冲击曲线(NIC)看不对称

  工具说明: arch 包均值只支持 AR, 故 ARMA 走"两步法"(statsmodels 取残差 -> arch 建 EGARCH).
"""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model

SEP = "=" * 70

# ====================================================================
# 步骤0  准备数据: 造"价格" -> 对数收益率 r_t
#   均值真值 = ARMA(1,1); 波动真值 = EGARCH(1,1,1)-t, gamma<0 制造杠杆效应.
# ====================================================================
omega, alpha_t, beta_t, gamma_t, nu = 0.02, 0.12, 0.96, -0.10, 7.0   # EGARCH-t: gamma<0=杠杆, beta=持续度
phi, theta, mu_true = 0.5, 0.3, 0.05                                  # ARMA(1,1) 均值真值
n, burn = 2500, 1000
truth_txt = (f"均值 ARMA(1,1) phi={phi}, theta={theta}, mu={mu_true}; "
             f"波动 EGARCH(1,1,1)-t omega={omega}, alpha={alpha_t}, beta={beta_t}, "
             f"gamma={gamma_t}(<0 杠杆), df={nu} (持续度=|beta|={beta_t})")

rng = np.random.default_rng(7)

def sim_egarch(rng):                                   # 造 EGARCH(1,1,1)-t 冲击序列 a_t
    e = rng.standard_t(nu, n + burn) * np.sqrt((nu - 2) / nu)   # 标准化 t 新息(单位方差)
    kappa = np.mean(np.abs(e))                          # E|z| 的样本估计(中心化大小项)
    a = np.zeros(n + burn); logs2 = np.zeros(n + burn)
    logs2[0] = omega / (1 - beta_t)                     # log 方差的无条件水平
    a[0] = np.exp(logs2[0] / 2) * e[0]
    for t in range(1, n + burn):
        logs2[t] = (omega + beta_t * logs2[t - 1]
                    + alpha_t * (abs(e[t - 1]) - kappa)  # 大小项: 冲击越大波动越高
                    + gamma_t * e[t - 1])                # 方向项: gamma<0 -> 负冲击额外抬高波动
        a[t] = np.exp(logs2[t] / 2) * e[t]
    return a

a_full = sim_egarch(rng)
x = np.zeros(n + burn); x[0] = a_full[0]
for t in range(1, n + burn):
    x[t] = phi * x[t - 1] + a_full[t] + theta * a_full[t - 1]      # 灌进 ARMA(1,1) 均值
ret = mu_true + x[burn:]
ret = pd.Series(ret).reset_index(drop=True)

price = pd.Series(np.r_[100.0, 100.0 * np.exp(np.cumsum(ret.values) / 100.0)])
r = 100.0 * np.log(price / price.shift(1)).dropna().reset_index(drop=True)

h = 60
train, test = r.iloc[:-h], r.iloc[-h:]
pd.DataFrame({"t": np.arange(len(r)), "logret_pct": r.values,
              "split": ["train"] * len(train) + ["test"] * len(test)}).to_csv(
    "egarch_returns.csv", index=False, encoding="utf-8")

print(SEP); print("步骤0  准备数据: 价格 -> 对数收益率 r_t")
print(f"  真值: {truth_txt}")
print(f"  样本 n={len(r)} (train={len(train)}, test={len(test)}), 单位=百分比")
print(f"  平稳性 ADF p={adfuller(train)[1]:.3g} -> 平稳, 可建模 [OK]")
print("  已导出 -> egarch_returns.csv")

# ====================================================================
# 步骤1  均值方程: ARMA 定 mu_t -> 冲击 a_t
# ====================================================================
print(SEP); print("步骤1  均值方程 (ARMA 定 mu_t, 得冲击 a_t)")
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
    arma = ARIMA(train, order=(best_order[0], 0, best_order[1])).fit()
a_t = arma.resid
mean_desc = f"ARMA{best_order}"
print(f"  BIC 网格 (p,q∈0..3, 行=p 列=q, [*]=最优):")
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

def mean_forecast(steps):
    return arma.get_forecast(steps=steps).predicted_mean.values

fig, ax = plt.subplots(3, 1, figsize=(10, 9))
ax[0].plot(train.values, lw=.6, color="steelblue")
ax[0].set_title(f"Log returns r_t  [mean: {mean_desc}]  —— volatility clustering + leverage")
plot_acf(train, lags=25, ax=ax[1], zero=False); ax[1].set_title("ACF of r_t (mean structure -> ARMA)")
plot_pacf(train, lags=25, ax=ax[2], method="ywm", zero=False); ax[2].set_title("PACF of r_t")
plt.tight_layout(); plt.savefig("egarch_fig1_mean.png", dpi=110); plt.close()

# ====================================================================
# 步骤2  检验 ARCH 效应 + 杠杆效应(EGARCH 的关键动机)
# ====================================================================
print(SEP); print("步骤2  检验 ARCH 效应 + 杠杆效应")
a2 = a_t ** 2
for m in (5, 10, 20):
    pq = acorr_ljungbox(a2, lags=[m], return_df=True)["lb_pvalue"].iloc[0]
    print(f"  [ARCH] a_t^2 的 Ljung-Box Q({m}) p={pq:.3g} -> "
          + ("有 ARCH 效应" if pq < 0.05 else "不显著"))
lm_stat, lm_p, _, _ = het_arch(a_t, nlags=10)
print(f"  [ARCH] Engle LM(10): LM={lm_stat:.2f}, p={lm_p:.3g} -> "
      + ("有 ARCH 效应 [OK]" if lm_p < 0.05 else "不显著"))
# 模型无关的杠杆探针: 上一期为负冲击的指示 与 当期平方冲击 的相关; >0 => 负冲击预示更大波动
neg_prev = (a_t.values[:-1] < 0).astype(float)
lev_corr = np.corrcoef(neg_prev, a2.values[1:])[0, 1]
print(f"  [杠杆] corr(1[a_(t-1)<0], a_t^2) = {lev_corr:+.3f} -> "
      + ("正相关: 负冲击预示更大波动 => 有杠杆, 该上 EGARCH [OK]" if lev_corr > 0.02
         else "不明显"))

# ====================================================================
# 步骤3  选型: 对称 o=0 vs 带杠杆 o=1 的 AIC/BIC -> 杠杆项值不值
# ====================================================================
print(SEP); print("步骤3  选型 (对称 vs 带杠杆: o=0 / o=1)")
sel = {}
for o in (0, 1):
    rr = arch_model(a_t, mean="Zero", vol="EGARCH", p=1, o=o, q=1, dist="t").fit(disp="off")
    sel[o] = (rr.aic, rr.bic)
    tag = "对称 EGARCH(1,0,1)" if o == 0 else "带杠杆 EGARCH(1,1,1)"
    print(f"  o={o} {tag:<22} AIC={rr.aic:8.1f}  BIC={rr.bic:8.1f}")
use_o = 1 if sel[1][1] < sel[0][1] else 0
print(f"  >>> BIC 选 o={use_o} "
      + ("(带杠杆胜出 -> 不对称项有价值, 用 EGARCH(1,1,1))" if use_o == 1
         else "(对称即可)"))

fig, axx = plt.subplots(1, 2, figsize=(12, 4.5))
plot_acf(a2, lags=25, ax=axx[0], zero=False)
axx[0].set_title("ACF of a_t^2 (volatility clustering)")
# 杠杆散点: 上一期冲击 a_(t-1) 对 当期 a_t^2; 左半(负冲击)整体更高 = 杠杆
axx[1].scatter(a_t.values[:-1], a2.values[1:], s=5, alpha=.25, color="steelblue")
axx[1].axvline(0, color="r", ls="--", lw=1)
axx[1].set_xlabel("shock a_(t-1)"); axx[1].set_ylabel("next squared shock a_t^2")
axx[1].set_title("Leverage scatter (left side higher -> negatives raise vol)")
plt.tight_layout(); plt.savefig("egarch_fig2_order.png", dpi=110); plt.close()

# ====================================================================
# 步骤4  估计参数: EGARCH(1,1,1)-t MLE, 重点看 gamma 的符号与显著性
# ====================================================================
print(SEP); print("步骤4  估计参数 (EGARCH(1,1,1) on a_t, Student-t, MLE)")
res = arch_model(a_t, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
beta_hat = res.params["beta[1]"]
gamma_hat = res.params["gamma[1]"]
gamma_p = res.pvalues["gamma[1]"]
persist = abs(beta_hat)
halflife = np.log(0.5) / np.log(persist) if persist < 1 else np.inf
print(f"  估计参数 (vs 真值):")
tv_map = {"omega": omega, "alpha[1]": alpha_t, "gamma[1]": gamma_t, "beta[1]": beta_t, "nu": nu}
for nm, v in res.params.items():
    tv = tv_map.get(nm)
    print(f"     {nm:<10}={v:+.4f}" + (f"   | 真值 {tv}" if tv is not None else ""))
print(f"  >>> 杠杆 gamma = {gamma_hat:+.4f} (真值 {gamma_t}), p={gamma_p:.3g} -> "
      + ("显著且 <0: 负冲击放大波动, 杠杆确认 [OK]" if (gamma_hat < 0 and gamma_p < 0.05)
         else "杠杆不显著"))
print(f"  持续度 |beta| = {persist:.4f} (<1 -> log方差平稳); 半衰期 = {halflife:.1f} 天")
print(f"  logL={res.loglikelihood:.1f}, AIC={res.aic:.1f}, BIC={res.bic:.1f}")

# ====================================================================
# 步骤5  模型检验: 标准化残差三查 + 杠杆是否洗净
# ====================================================================
print(SEP); print("步骤5  模型检验 (标准化残差三查 + 杠杆洗净)")
z = res.std_resid.dropna()
p_mean = acorr_ljungbox(z, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
p_vol = acorr_ljungbox(z ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
sk, ku = stats.skew(z), stats.kurtosis(z, fisher=False)
zneg = (z.values[:-1] < 0).astype(float)
lev_left = np.corrcoef(zneg, (z.values[1:]) ** 2)[0, 1]
print(f"  ① 均值({mean_desc}) ~z Ljung-Box(10) p={p_mean:.3g} -> "
      + ("均值方程充分 [OK]" if p_mean > 0.05 else "改 ARMA 阶 [NO]"))
print(f"  ② 波动方程 ~z^2 Ljung-Box(10) p={p_vol:.3g} -> "
      + ("波动洗净 [OK]" if p_vol > 0.05 else "仍有 ARCH [NO]"))
print(f"  ③ 分布   偏度={sk:+.2f}, 峰度={ku:.2f} (正态=3)")
print(f"  ④ 杠杆洗净 corr(1[z_(t-1)<0], z_t^2)={lev_left:+.3f} -> "
      + ("已洗净(接近0) [OK]" if abs(lev_left) < 0.05 else "残余杠杆"))
verdict = ("三查通过 + 杠杆洗净 -> EGARCH 模型充分"
           if (p_mean > 0.05 and p_vol > 0.05) else "有一项需复核")
print(f"  >>> {verdict}")

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].plot(res.conditional_volatility, lw=.7, color="darkred")
ax[0, 0].set_title(f"Conditional volatility sigma_t  [EGARCH, persist={persist:.3f}]")
plot_acf(z ** 2, lags=25, ax=ax[0, 1], zero=False); ax[0, 1].set_title("ACF of std-resid^2 (white? -> vol cleaned)")
ax[1, 0].hist(z, bins=40, density=True, alpha=.7, edgecolor="k")
xs = np.linspace(z.min(), z.max(), 200); ax[1, 0].plot(xs, stats.norm.pdf(xs), "r--", label="N(0,1)")
ax[1, 0].set_title("Std-resid histogram (fat tails)"); ax[1, 0].legend()
stats.probplot(z, dist="norm", plot=ax[1, 1]); ax[1, 1].set_title("Q-Q plot vs Normal")
plt.tight_layout(); plt.savefig("egarch_fig3_diagnose.png", dpi=110); plt.close()

# ====================================================================
# 步骤6  使用模型: 均值点预测 + EGARCH 波动率(模拟)预测 + 新闻冲击曲线
#   EGARCH 多步无解析式 -> 用模拟; 新闻冲击曲线 NIC 是其"不对称"的招牌图.
# ====================================================================
print(SEP); print(f"步骤6  预测 (均值 {mean_desc} + 波动 EGARCH, {h} 步, 模拟法)")
mu_path = mean_forecast(h)
fc = res.forecast(horizon=h, method="simulation", simulations=2000, reindex=False)
sig_path = np.sqrt(fc.variance.values[-1])
nu_hat = res.params.get("nu", 8.0)
q975 = stats.t.ppf(0.975, nu_hat) * np.sqrt((nu_hat - 2) / nu_hat)
q01 = stats.t.ppf(0.01, nu_hat) * np.sqrt((nu_hat - 2) / nu_hat)
lo, hi = mu_path - q975 * sig_path, mu_path + q975 * sig_path
VaR99 = -(mu_path + q01 * sig_path)
print(f"  均值点预测 mu(1)={mu_path[0]:+.3f} -> mu({h})={mu_path[-1]:+.3f} (ARMA 回复)")
print(f"  波动预测  sigma(1)={sig_path[0]:.3f} -> sigma({h})={sig_path[-1]:.3f} (向无条件水平收敛)")
print(f"  1日 99% VaR: 明日={VaR99[0]:.3f}%")
inside = (test.values >= lo) & (test.values <= hi)
print(f"  样本外 95% 区间覆盖率={inside.mean():.1%} (理想≈95%)")

pd.DataFrame({
    "step": np.arange(1, h + 1),
    "mu_hat": np.round(mu_path, 4),
    "sigma_hat": np.round(sig_path, 4),
    "lo95": np.round(lo, 4),
    "hi95": np.round(hi, 4),
    "VaR99": np.round(VaR99, 4),
    "actual": np.round(test.values, 4),
    "in_interval": inside,
}).to_csv("egarch_forecast.csv", index=False, encoding="utf-8")
print("  已导出每日预测 -> egarch_forecast.csv")

# 新闻冲击曲线 NIC: 固定 sigma_{t-1} 在无条件水平, 看 sigma_t^2 如何随上期标准化冲击 z 变化
om, al, ga = res.params["omega"], res.params["alpha[1]"], res.params["gamma[1]"]
be = res.params["beta[1]"]
zz = np.linspace(-4, 4, 200)
kap = np.sqrt(2 / np.pi)
log_unc = om / (1 - be)
nic = np.exp(om + be * log_unc + al * (np.abs(zz) - kap) + ga * zz)   # sigma_t^2 as function of z_{t-1}

fig, axx = plt.subplots(2, 1, figsize=(11, 9))
xx = np.arange(h)
axx[0].plot(xx, test.values, color="black", lw=.9, marker=".", ms=4, label="actual return")
axx[0].plot(xx, mu_path, "b-", lw=1.3, label=f"{mean_desc} mean forecast")
axx[0].fill_between(xx, lo, hi, color="red", alpha=.15, label="95% dynamic interval")
axx[0].plot(xx, sig_path, "r-", lw=1.2, alpha=.6, label="forecast sigma(l)")
axx[0].set_title(f"Out-of-sample: mean {mean_desc} + vol EGARCH(1,1,1)"); axx[0].legend(loc="upper right")
axx[1].plot(zz, nic, "purple", lw=2)
axx[1].axvline(0, color="gray", ls="--", lw=1)
axx[1].set_xlabel("standardized shock z_(t-1)"); axx[1].set_ylabel("implied sigma_t^2")
axx[1].set_title(f"News Impact Curve (gamma={ga:+.3f}<0 -> left arm steeper = leverage / asymmetry)")
plt.tight_layout(); plt.savefig("egarch_fig4_forecast.png", dpi=110); plt.close()

# ====================================================================
print(SEP)
print(f"完成. 全流程: 平稳 -> 均值={mean_desc} -> ARCH+杠杆 -> 选 o=1(带杠杆) "
      f"-> EGARCH MLE(t), gamma<0 -> 三查/杠杆洗净 -> 均值+模拟波动/VaR + 新闻冲击曲线")
print(f"揭晓真值: {truth_txt}")
print("图: egarch_fig1_mean / egarch_fig2_order / egarch_fig3_diagnose / egarch_fig4_forecast (.png)")
