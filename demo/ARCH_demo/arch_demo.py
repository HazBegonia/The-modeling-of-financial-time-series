# -*- coding: utf-8 -*-
"""
ARCH/GARCH 波动率建模 —— 一条龙完整流程 (严格对照 6 步法, 均值自动选 常数 / ARMA)

  第0步 准备数据    : 价格 -> 对数收益率 r_t (平稳序列, 建模入口)
  第1步 均值方程    : 查 r_t 自相关(Ljung-Box) -> 自动分支:
                      无自相关 -> mu_t 取常数 ; 有自相关 -> BIC 网格定阶并拟合 ARMA
                      两条路都得到残差/冲击 a_t = r_t - mu_t
  第2步 ARCH 效应   : 对 a_t^2 查波动聚集(Ljung-Box Q(m) + Engle LM) -> 要不要建波动率模型
  第3步 定阶 m      : 主用 a_t^2 的 PACF 截尾 + AIC/BIC 网格
  第4步 估计参数    : eps_t ~ Student-t(厚尾), MLE 估波动率方程 -> alpha0..alpha_m
  第5步 模型检验    : 标准化残差 ~a_t 三查(均值/波动/分布), 不过关 -> 迭代(ARCH 不够换 GARCH)
  第6步 使用模型    : 点预测 mu_{t+1}(常数/ARMA回复) + 波动率递推 sigma^2(l) -> 动态区间 + VaR

  开关 MEAN_KIND 控制造哪种数据:
    "const" -> 常数均值 + GARCH(1,1)-t (第1步自动走常数分支)
    "arma"  -> ARMA(1,1) 均值 + GARCH(1,1)-t (第1步自动走 ARMA 分支)
  无论造哪种, 第1步都靠 Ljung-Box 自动判别, 不靠开关作弊.

  工具说明: Python 的 arch 包均值只支持 AR(无 MA), 故 ARMA 走"两步法":
    statsmodels 建 ARMA 取残差 a_t -> arch 对 a_t 建波动率(mean='Zero').
    常数均值分支同样先去均值再对 a_t 建模, 两条路下游完全统一.
"""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")               # Windows 控制台强制 UTF-8
warnings.filterwarnings("ignore")                      # 屏蔽 ARIMA 网格中个别阶的收敛警告
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                                  # 不弹窗, 直接存图
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model

SEP = "=" * 70
MEAN_KIND = "arma"          # <<< 改这里: "const" 或 "arma"

# ====================================================================
# 步骤0  准备数据: 造"价格" -> 转对数收益率 r_t (平稳)
#   波动真值统一为 GARCH(1,1)-t; 均值真值按 MEAN_KIND 取 常数 或 ARMA(1,1).
#   单位用"百分比收益"(x100), 量纲适中, 是 arch 包推荐做法.
# ====================================================================
omega, alpha_t, beta_t, nu = 0.05, 0.10, 0.86, 7.0     # GARCH(1,1)-t: alpha+beta=0.96 高持续, df=7 厚尾
n, burn = 1500, 1000

def sim_garch(rng):                                    # 造 GARCH(1,1)-t 冲击序列 a_t
    e = rng.standard_t(nu, n + burn) * np.sqrt((nu - 2) / nu)   # 标准化 t 新息(单位方差)
    a = np.zeros(n + burn); s2 = np.zeros(n + burn)
    s2[0] = omega / (1 - alpha_t - beta_t); a[0] = np.sqrt(s2[0]) * e[0]
    for t in range(1, n + burn):
        s2[t] = omega + alpha_t * a[t - 1] ** 2 + beta_t * s2[t - 1]
        a[t] = np.sqrt(s2[t]) * e[t]
    return a

if MEAN_KIND == "arma":
    rng = np.random.default_rng(2024)
    phi, theta, mu_true = 0.6, 0.3, 0.05               # ARMA(1,1) 真值
    a = sim_garch(rng); x = np.zeros(n + burn); x[0] = a[0]
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + a[t] + theta * a[t - 1]    # ARMA(1,1) 递推
    ret = mu_true + x[burn:]
    truth_txt = f"均值 ARMA(1,1) phi={phi}, theta={theta}, mu={mu_true}"
else:
    rng = np.random.default_rng(202)
    mu_true = 0.03                                      # 常数均值真值
    ret = mu_true + sim_garch(rng)[burn:]
    truth_txt = f"均值 常数 mu={mu_true}"
ret = pd.Series(ret).reset_index(drop=True)

price = pd.Series(np.r_[100.0, 100.0 * np.exp(np.cumsum(ret.values) / 100.0)])  # 收益->价格
r = 100.0 * np.log(price / price.shift(1)).dropna().reset_index(drop=True)      # 价格->对数收益 r_t

h = 60                                                  # 留最后 60 天做样本外预测
train, test = r.iloc[:-h], r.iloc[-h:]
pd.DataFrame({"t": np.arange(len(r)), "logret_pct": r.values,
              "split": ["train"] * len(train) + ["test"] * len(test)}).to_csv(
    "arch_returns.csv", index=False, encoding="utf-8")

print(SEP); print(f"步骤0  准备数据 (MEAN_KIND='{MEAN_KIND}'): 价格 -> 对数收益率 r_t")
print(f"  真值: {truth_txt}; 波动 GARCH(1,1)-t omega={omega}, alpha={alpha_t}, beta={beta_t}, df={nu}")
print(f"  样本 n={len(r)} (train={len(train)}, test={len(test)}), 单位=百分比")
print(f"  平稳性 ADF p={adfuller(train)[1]:.3g} -> 平稳, 可建模 [OK]")
print("  已导出 -> arch_returns.csv")

# ====================================================================
# 步骤1  均值方程 (管"中心"): r_t 有没有自相关? 自动分支 常数 / ARMA
#   都产出残差/冲击 a_t = r_t - mu_t, 以及 mean_fc(h) 用于第6步均值预测.
# ====================================================================
print(SEP); print("步骤1  均值方程 (定 mu_t, 得冲击 a_t)")
lb_mean = acorr_ljungbox(train, lags=[10], return_df=True)["lb_pvalue"].iloc[0]

if lb_mean < 0.05:
    # ---- ARMA 分支: BIC 网格 p,q∈0..3 定阶 -> 拟合 -> 残差 ----
    print(f"  r_t 的 Ljung-Box(10) p={lb_mean:.3g} -> 有自相关, 走 ARMA 分支")
    best_ic, best_order, grid = np.inf, (1, 0), {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                # 个别阶不收敛属正常, 静默
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
    a_t = arma.resid                                   # 残差 = 冲击
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
        tv = {"ar.L1": locals().get('phi'), "ma.L1": locals().get('theta'),
              "const": locals().get('mu_true')}.get(nm)
        print(f"     {nm:<8}={v:+.3f}" + (f"   | 真值 {tv}" if tv is not None else ""))
    def mean_forecast(steps):                          # ARMA 点预测: 逐步回复到无条件均值
        return arma.get_forecast(steps=steps).predicted_mean.values
    mu_hat = arma.params.get("const", train.mean())
else:
    # ---- 常数均值分支 ----
    print(f"  r_t 的 Ljung-Box(10) p={lb_mean:.3g} -> 无显著自相关, mu_t 取常数 [OK]")
    mu_hat = train.mean()
    a_t = train - mu_hat                               # 冲击 = 残差
    arma, mean_desc = None, "Const"
    def mean_forecast(steps):                          # 常数点预测: 各步相同
        return np.full(steps, mu_hat)
print(f"  均值模型={mean_desc}; 冲击 a_t = r_t - mu_t (下面研究它的波动)")

fig, ax = plt.subplots(3, 1, figsize=(10, 9))
ax[0].plot(train.values, lw=.6, color="steelblue")
ax[0].set_title(f"Log returns r_t  [mean: {mean_desc}]")
plot_acf(train, lags=25, ax=ax[1], zero=False); ax[1].set_title("ACF of r_t")
plot_pacf(train, lags=25, ax=ax[2], method="ywm", zero=False); ax[2].set_title("PACF of r_t")
plt.tight_layout(); plt.savefig("arch_fig1_mean.png", dpi=110); plt.close()

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
# 步骤3  定阶 m: a_t^2 的 PACF 截尾 + AIC/BIC 网格 (对 a_t 建模, mean='Zero')
# ====================================================================
print(SEP); print("步骤3  定阶 m (ARCH 阶数)")
MAXLAG = 12
pac = pacf(a2, nlags=MAXLAG, method="ywm"); band = 1.96 / np.sqrt(len(a2))
sig = [k for k in range(1, MAXLAG + 1) if abs(pac[k]) > band]
pacf_m = max(sig) if sig else 1
rows = {}
for m in range(1, MAXLAG + 1):
    rr = arch_model(a_t, mean="Zero", vol="ARCH", p=m, dist="t").fit(disp="off")
    rows[m] = (rr.aic, rr.bic)
aic_m = min(rows, key=lambda k: rows[k][0]); bic_m = min(rows, key=lambda k: rows[k][1])
m_hat = bic_m
print(f"  [3a] a_t^2 的 PACF 截尾: 最后显著 lag = {pacf_m} (band=±{band:.3f})")
print(f"  [3b] AIC/BIC 网格 -> AIC 选 m={aic_m}, BIC 选 m={bic_m}")
print(f"  投票: PACF={pacf_m} | AIC={aic_m} | BIC={bic_m}  >>> 最终定阶 m = {m_hat} (以BIC为准)")

fig, axx = plt.subplots(1, 2, figsize=(12, 4.5))
plot_pacf(a2, lags=MAXLAG, ax=axx[0], method="ywm", zero=False)
axx[0].set_title(f"PACF of a_t^2 (cuts off -> ARCH order ~{pacf_m})")
ks = list(rows.keys())
axx[1].plot(ks, [rows[k][0] for k in ks], "o-", label="AIC")
axx[1].plot(ks, [rows[k][1] for k in ks], "s-", label="BIC")
axx[1].axvline(m_hat, color="C1", ls=":", label=f"chosen m={m_hat}")
axx[1].set_xlabel("ARCH order m"); axx[1].set_title("Order selection"); axx[1].legend()
plt.tight_layout(); plt.savefig("arch_fig2_order.png", dpi=110); plt.close()

# ====================================================================
# 步骤4  估计参数: ARCH(m) on a_t, Student-t, MLE
# ====================================================================
print(SEP); print(f"步骤4  估计波动率参数 (ARCH({m_hat}) on a_t, Student-t)")
res = arch_model(a_t, mean="Zero", vol="ARCH", p=m_hat, dist="t").fit(disp="off")
print(f"  omega(=alpha0), alpha1..alpha{m_hat}, nu 已估出; "
      f"logL={res.loglikelihood:.1f}, AIC={res.aic:.1f}, BIC={res.bic:.1f}")

# ====================================================================
# 步骤5  模型检验: 标准化残差三查, 不过关 -> 升级 GARCH(1,1)
# ====================================================================
def diagnose(vres, name):
    z = vres.std_resid.dropna()
    p_mean = acorr_ljungbox(z, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    p_vol = acorr_ljungbox(z ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    sk, ku = stats.skew(z), stats.kurtosis(z, fisher=False)
    print(f"  [{name}] 标准化残差三查:")
    print(f"     ① 均值({mean_desc}够不够) ~a_t Ljung-Box(10) p={p_mean:.3g} -> "
          + ("均值方程充分 [OK]" if p_mean > 0.05 else "均值阶数不够, 改均值方程 [NO]"))
    print(f"     ② 波动方程 ~a_t^2 Ljung-Box(10) p={p_vol:.3g} -> "
          + ("波动洗净 [OK]" if p_vol > 0.05 else "仍有 ARCH, 加大 m / 换 GARCH [NO]"))
    print(f"     ③ 分布 偏度={sk:+.2f}, 峰度={ku:.2f} (正态=3)")
    return p_mean, p_vol

print(SEP); print("步骤5  模型检验 (标准化残差三查)")
pm1, pv1 = diagnose(res, f"ARCH({m_hat})")
best, best_name = res, f"ARCH({m_hat})"
if pv1 < 0.05:
    print("  -> 波动检验不过关, 迭代: 升级到 GARCH(1,1)-t 重估 ...")
    res2 = arch_model(a_t, mean="Zero", vol="Garch", p=1, q=1, dist="t").fit(disp="off")
    pm2, pv2 = diagnose(res2, "GARCH(1,1)")
    print(f"  对比 BIC: ARCH({m_hat})={res.bic:.1f} vs GARCH(1,1)={res2.bic:.1f}")
    if pv2 > pv1 and res2.bic < res.bic:
        best, best_name = res2, "GARCH(1,1)"
    print(f"  >>> 采用 {best_name}")
else:
    print(f"  >>> {best_name} 三查通过, 模型充分")

zf = best.std_resid.dropna()
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].plot(best.conditional_volatility, lw=.7, color="darkred")
ax[0, 0].set_title(f"Conditional volatility sigma_t  [{best_name}]")
plot_acf(zf ** 2, lags=25, ax=ax[0, 1], zero=False); ax[0, 1].set_title("ACF of std-resid^2 (white?)")
ax[1, 0].hist(zf, bins=40, density=True, alpha=.7, edgecolor="k")
xs = np.linspace(zf.min(), zf.max(), 200); ax[1, 0].plot(xs, stats.norm.pdf(xs), "r--", label="N(0,1)")
ax[1, 0].set_title("Std-resid histogram (fat tails)"); ax[1, 0].legend()
stats.probplot(zf, dist="norm", plot=ax[1, 1]); ax[1, 1].set_title("Q-Q plot vs Normal")
plt.tight_layout(); plt.savefig("arch_fig3_diagnose.png", dpi=110); plt.close()

# ====================================================================
# 步骤6  使用模型: 均值点预测(常数/ARMA回复) + 波动率递推 -> 动态区间 + VaR
#   说明: 1步区间精确; 多步用 均值路径 ± q*sigma(l) 近似(ARMA 多步还应叠加 psi 权重方差, 此处教学简化).
# ====================================================================
print(SEP); print(f"步骤6  预测 (均值 {mean_desc} + 波动 {best_name}, {h} 步)")
mu_path = mean_forecast(h)
sig_path = np.sqrt(best.forecast(horizon=h, reindex=False).variance.values[-1])
uncond = np.sqrt(best.params["omega"] / (1 - sum(best.params.filter(like="alpha"))
                                          - sum(best.params.filter(like="beta"))))
nu_hat = best.params.get("nu", 8.0)
q975 = stats.t.ppf(0.975, nu_hat) * np.sqrt((nu_hat - 2) / nu_hat)
q01 = stats.t.ppf(0.01, nu_hat) * np.sqrt((nu_hat - 2) / nu_hat)
lo, hi = mu_path - q975 * sig_path, mu_path + q975 * sig_path
VaR99 = -(mu_path + q01 * sig_path)
print(f"  均值点预测 mu_(t+1)={mu_path[0]:+.3f} -> mu_(t+{h})={mu_path[-1]:+.3f} "
      + ("(ARMA 逐步回复)" if arma is not None else "(常数, 各步相同)"))
print(f"  波动预测  sigma(1)={sig_path[0]:.3f} -> sigma({h})={sig_path[-1]:.3f} (向无条件水平 {uncond:.3f} 收敛)")
print(f"  1日 99% VaR: 明日={VaR99[0]:.3f}%  (区间随市场松紧变宽变窄)")
inside = (test.values >= lo) & (test.values <= hi)
print(f"  样本外 95% 区间覆盖率={inside.mean():.1%} (理想≈95%)")

# 导出每日预测: 中心 mu_hat / 波动 sigma_hat / 95%区间上下界 / VaR / 实际值 / 是否落在区间内
pd.DataFrame({
    "step": np.arange(1, h + 1),               # 预测第几步(未来第几天)
    "mu_hat": np.round(mu_path, 4),            # 条件均值(中心)
    "sigma_hat": np.round(sig_path, 4),        # 条件波动(决定区间宽度)
    "lo95": np.round(lo, 4),                   # 95% 区间下界
    "hi95": np.round(hi, 4),                   # 95% 区间上界
    "VaR99": np.round(VaR99, 4),               # 1日 99% VaR(正数=潜在损失%)
    "actual": np.round(test.values, 4),        # 实际收益(事后真实值, 用于对照)
    "in_interval": inside,                     # 实际值是否落在 95% 区间内
}).to_csv("arch_forecast.csv", index=False, encoding="utf-8")
print("  已导出每日预测(中心/波动/区间/VaR/实际) -> arch_forecast.csv")

fig, axx = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
xx = np.arange(h)
axx[0].plot(xx, test.values, color="black", lw=.9, marker=".", ms=4, label="actual return")
axx[0].plot(xx, mu_path, "b-", lw=1.3, label=f"{mean_desc} mean forecast")
axx[0].fill_between(xx, lo, hi, color="red", alpha=.15, label="95% dynamic interval")
axx[0].set_title(f"Out-of-sample: mean {mean_desc} + vol {best_name}"); axx[0].legend(loc="upper right")
axx[1].plot(xx, sig_path, "r-", lw=1.5, label="forecast volatility sigma(l)")
axx[1].plot(xx, VaR99, color="purple", lw=1.2, label="1-day 99% VaR (%)")
axx[1].set_xlabel("forecast horizon (days)"); axx[1].set_title("Volatility & VaR forecast")
axx[1].legend(loc="upper right")
plt.tight_layout(); plt.savefig("arch_fig4_forecast.png", dpi=110); plt.close()

# ====================================================================
print(SEP)
print(f"完成. 全流程: 平稳 -> 均值={mean_desc} -> ARCH效应 -> 定阶 m={m_hat} "
      f"-> MLE(t) -> 检验/迭代 -> {best_name} -> 均值+动态波动/VaR 预测")
print(f"揭晓真值: {truth_txt} + GARCH(1,1)-t (alpha={alpha_t}, beta={beta_t})")
print("图: arch_fig1_mean / arch_fig2_order / arch_fig3_diagnose / arch_fig4_forecast (.png)")
