# -*- coding: utf-8 -*-
"""
IGARCH 波动率建模 —— 一条龙完整流程 (均值 ARMA; 波动先定阶 GARCH(m,s), 再按"持续度≈1"追加单位根约束成 IGARCH)

  IGARCH = 求和 GARCH = 单位根 GARCH (Integrated GARCH), 是 GARCH 的"边界特例":
    GARCH(1,1):   sigma_t^2 = omega + alpha*a_{t-1}^2 + beta*sigma_{t-1}^2,   持续度 alpha+beta < 1
    IGARCH(1,1):  alpha + beta = 1  (方差有"单位根")
                  写成书里的形式: sigma_t^2 = omega + beta*sigma_{t-1}^2 + (1-beta)*a_{t-1}^2
                  -> alpha1 = 1-beta 被 beta 锁死, 比 GARCH(1,1) 少一个自由参数
                  -> 持续度恰为 1: 冲击"永不消散"(半衰期=无穷); 无有限无条件方差 omega/(1-alpha-beta)=omega/0
                  -> 多步波动"不回复": sigma^2(l) = sigma^2(1) + (l-1)*omega   (式 3.22, 线性外推, 永不收敛)
    著名特例: omega=0 即 RiskMetrics/EWMA:  sigma_t^2 = (1-beta)*a_{t-1}^2 + beta*sigma_{t-1}^2 (第7章算 VaR 用).

  ★ 一个必须分清的点 (本 demo 的主线, 也是它和 GARCH demo 唯一的结构差别):
    "阶 (m,s)" 和 "单位根约束(那个 I)" 是两回事 ——
      · 阶 (m,s) 是"定"出来的: a^2 的 PACF 定 ARCH 阶量级(与 ARCH 同) + 低阶 (m,s) 网格 AIC/BIC(与 GARCH 同);
      · "I"(alpha+beta=1) 不是默认 (1,1), 而是"定好阶、估完 GARCH 后, 看 Σα̂+Σβ̂ 是否≈1"再追加的约束.
    所以本 demo 绝不写死 (1,1): 先老实定阶 -> 估无约束 GARCH -> 发现持续度≈1 -> 才施加单位根约束重估成 IGARCH.

  六步法 (下游一律用"定出来的阶", 不写死):
    第0步 准备数据  : 价格 -> 对数收益率 r_t (平稳序列, 建模入口)
    第1步 均值方程  : r_t 有自相关 -> BIC 网格定阶并拟合 ARMA -> 冲击 a_t = r_t - mu_t
    第2步 ARCH 效应 : 对 a_t^2 查波动聚集(Ljung-Box Q(m) + Engle LM) -> 要不要建波动率模型
    第3步 定阶 (m,s): Schwert 上限 + a_t^2 的 PACF(定 ARCH 阶量级) + 低阶 (m,s) AIC/BIC 网格 (与 ARCH/GARCH 同)
    第4步 估计参数  : (4a) 无约束 GARCH(m,s) MLE 拿 Σα̂+Σβ̂ ->
                      (4b) 若 ≈1 判定 IGARCH -> (4c) 施加单位根约束, 自写"标准化 t"似然重估 IGARCH(1,1) (少一参)
    第5步 模型检验  : 约束模型的标准化残差三查 + 持续度=1 -> 严格平稳但非协方差平稳
    第6步 使用模型  : ARMA 点预测 + IGARCH 波动"线性外推不回复" sigma^2(l)=sigma^2(1)+(l-1)*omega -> 区间 + VaR

  工具说明:
    · Python 的 arch 包均值只支持 AR(无 MA), 故 ARMA 走"两步法": statsmodels 建 ARMA 取残差 a_t -> 对 a_t 建波动.
    · arch 包没有现成的"带 alpha+beta=1 约束"的 IGARCH, 故第4c步用 scipy 自写"标准化 t"的条件似然做约束 MLE,
      递推式直接令 alpha1=1-beta (正是书里 IGARCH(1,1) 的写法), 忠实复刻"少一个自由参数"的约束估计.
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
from scipy.special import gammaln
from scipy.optimize import minimize
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model

SEP = "=" * 70


# ---- IGARCH(1,1)-t 约束 MLE 的两个零件 (arch 包无现成 IGARCH, 自写标准化 t 条件似然) ----
def igarch11_filter(a, omega, beta, s2_0):
    """IGARCH(1,1) 方差递推: sigma2_t = omega + beta*sigma2_{t-1} + (1-beta)*a_{t-1}^2  (alpha1=1-beta 锁死)."""
    T = len(a); s2 = np.empty(T); s2[0] = s2_0
    for t in range(1, T):
        s2[t] = omega + beta * s2[t - 1] + (1.0 - beta) * a[t - 1] ** 2
    return s2


def neg_loglik_igarch(params, a, s2_0):
    """标准化 Student-t(单位方差) 下 IGARCH(1,1) 的负对数似然; 参数 = (omega, beta, nu)."""
    omega, beta, nu = params
    if omega <= 1e-10 or not (1e-4 < beta < 0.9999) or nu <= 2.05:
        return 1e12                                    # 越界罚一个大数, 保证优化留在可行域内
    s2 = igarch11_filter(a, omega, beta, s2_0)
    z2 = a ** 2 / s2
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    ll = c - 0.5 * (nu + 1) * np.log1p(z2 / (nu - 2)) - 0.5 * np.log(s2)   # log[ f(a_t/sigma_t)/sigma_t ]
    return -ll.sum()


# ====================================================================
# 步骤0  准备数据: 造"价格" -> 转对数收益率 r_t (平稳)
#   均值真值 = ARMA(1,1); 波动真值 = IGARCH(1,1)-t (alpha+beta=1.0, 恰好积分/单位根).
#   单位用"百分比收益"(x100), 量纲适中, 是 arch 包推荐做法.
# ====================================================================
omega, alpha_t, beta_t, nu = 0.02, 0.06, 0.94, 7.0     # IGARCH(1,1)-t: alpha+beta=1.0 (恰好 integrated!)
phi, theta, mu_true = 0.5, 0.3, 0.05                   # ARMA(1,1) 均值真值
n, burn = 2500, 1000
persist_true = alpha_t + beta_t
truth_txt = (f"均值 ARMA(1,1) phi={phi}, theta={theta}, mu={mu_true}; "
             f"波动 IGARCH(1,1)-t omega={omega}, alpha={alpha_t}, beta={beta_t}, df={nu} "
             f"(持续度 alpha+beta={persist_true:.2f} = 1 -> 积分型)")

rng = np.random.default_rng(11)


def sim_igarch(rng):                                   # 造 IGARCH(1,1)-t 冲击序列 a_t
    e = rng.standard_t(nu, n + burn) * np.sqrt((nu - 2) / nu)   # 标准化 t 新息(单位方差)
    a = np.zeros(n + burn); s2 = np.zeros(n + burn)
    s2[0] = 1.0; a[0] = np.sqrt(s2[0]) * e[0]           # 积分型无有限无条件方差, 初值给定 1.0
    for t in range(1, n + burn):
        s2[t] = omega + alpha_t * a[t - 1] ** 2 + beta_t * s2[t - 1]   # alpha+beta=1: 方差走"带漂移随机游走"
        a[t] = np.sqrt(s2[t]) * e[t]
    return a


a_full = sim_igarch(rng)                               # 先造带波动聚集的冲击
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
    "igarch_returns.csv", index=False, encoding="utf-8")

print(SEP); print("步骤0  准备数据: 价格 -> 对数收益率 r_t")
print(f"  真值: {truth_txt}")
print(f"  样本 n={len(r)} (train={len(train)}, test={len(test)}), 单位=百分比")
print(f"  平稳性 ADF p={adfuller(train)[1]:.3g} -> 平稳, 可建模 [OK]")
print("  已导出 -> igarch_returns.csv")

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
plot_pacf(train, lags=25, ax=ax[2], method="ols", zero=False); ax[2].set_title("PACF of r_t")
plt.tight_layout(); plt.savefig("igarch_fig1_mean.png", dpi=110); plt.close()

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
# 步骤3  定阶 (m,s): Schwert 上限 + a_t^2 的 PACF(定 ARCH 阶量级) + 低阶 (m,s) AIC/BIC 网格
#   上限规则与 ARCH 完全一致: MAXLAG=floor(12*(T/100)^0.25), 只依赖样本量 T, 不偷看真值.
#   a^2 的 PACF 截尾只对"纯 ARCH"干净; 一旦有 beta 项, a^2 成 ARMA 形 -> ACF/PACF 双拖尾 -> 交给网格.
#   注意: 这里定的是"阶", 还不是 IGARCH; 单位根约束是第4步估完 GARCH 后才追加的.
# ====================================================================
print(SEP); print("步骤3  定阶 (m,s) (m=ARCH阶/数 a^2 滞后, s=GARCH阶/数 sigma^2 滞后)")
MAXLAG = int(12 * (len(a2) / 100) ** 0.25)             # Schwert 规则: 与 ARCH demo 同一把尺, 只看 T
print(f"  搜索上限 MAXLAG = {MAXLAG}  (Schwert: floor(12*(T/100)^0.25), T={len(a2)}; 与 ARCH 定阶同规则)")

# 3a: a^2 的 PACF 截尾 -> ARCH 阶 m 的量级 (§3.4.3, 和 ARCH 定阶同一套口诀)
pac = pacf(a2, nlags=MAXLAG, method="ols"); band = 1.96 / np.sqrt(len(a2))
sig = [k for k in range(1, MAXLAG + 1) if abs(pac[k]) > band]
pacf_m = max(sig) if sig else 1
print(f"  [3a] a_t^2 的 PACF 截尾: 最后显著 lag = {pacf_m} (band=±{band:.3f}); "
      f"持续度≈1 时 a^2 近积分, PACF 常拖尾不干净 -> 仅作 ARCH 阶量级参考")

# 3b: 低阶 (m,s) 二维 AIC/BIC 网格 (GARCH 高阶 MLE 难收敛/实证极少胜低阶, 诚实截到 3)
GRIDMAX = min(MAXLAG, 3)                                # 网格诚实上限 (显式标注): 与 GARCH demo 同
PMAX = QMAX = GRIDMAX
ggrid_bic, ggrid_aic = {}, {}
best_gic, best_gorder = np.inf, (1, 1)
for p in range(1, PMAX + 1):                           # p=ARCH阶 m (>=1); q=0 即退化为纯 ARCH
    for q in range(0, QMAX + 1):                       # q=GARCH阶 s
        try:
            rr = arch_model(a_t, mean="Zero", vol="Garch", p=p, q=q, dist="t").fit(disp="off")
            ggrid_bic[(p, q)] = rr.bic; ggrid_aic[(p, q)] = rr.aic
            if rr.bic < best_gic:
                best_gic, best_gorder = rr.bic, (p, q)
        except Exception:
            ggrid_bic[(p, q)] = np.nan; ggrid_aic[(p, q)] = np.nan
p_hat, q_hat = best_gorder
aic_order = min(ggrid_aic, key=lambda k: ggrid_aic[k])
print(f"  [3b] 低阶网格 (Schwert 上限 {MAXLAG}, GARCH 高阶难收敛 -> 网格诚实截到 {GRIDMAX}):")
print("       BIC 网格 (行=ARCH阶 m, 列=GARCH阶 s; s=0 即纯 ARCH(m); [*]=最优):")
print("         " + "".join(f"s={q:<8}" for q in range(QMAX + 1)))
for p in range(1, PMAX + 1):
    cells = ""
    for q in range(QMAX + 1):
        v = ggrid_bic.get((p, q), np.nan)
        star = "*" if (p, q) == best_gorder else " "
        cells += (f"{v:8.1f}{star} " if np.isfinite(v) else "   --     ")
    print(f"     m={p} {cells}")
print(f"  投票: PACF(ARCH阶)~{pacf_m} | AIC 选 (m,s)={aic_order} | BIC 选 (m,s)={best_gorder}"
      f"  >>> 最终定阶 (m,s)={best_gorder} (以BIC为准)")
if p_hat == GRIDMAX or q_hat == GRIDMAX:
    print(f"  ⚠ 定阶贴到网格上限 {GRIDMAX} (可能被截断), 建议放大 GRIDMAX 复核")

fig, axx = plt.subplots(1, 2, figsize=(12, 4.5))
plot_acf(a2, lags=25, ax=axx[0], zero=False)
axx[0].set_title("ACF of a_t^2 (very slow decay -> persistence ~ 1, IGARCH hint)")
plot_pacf(a2, lags=MAXLAG, ax=axx[1], method="ols", zero=False)
axx[1].set_title(f"PACF of a_t^2 (ARCH-order read ~{pacf_m}; tails off once beta present)")
plt.tight_layout(); plt.savefig("igarch_fig2_order.png", dpi=110); plt.close()

# ====================================================================
# 步骤4  估计参数: 先无约束 GARCH(m,s) 拿持续度 -> 若≈1 追加单位根约束 -> 约束 MLE 重估 IGARCH(1,1)
# ====================================================================
print(SEP); print(f"步骤4  估计参数 (先无约束 GARCH{best_gorder}, 再按持续度≈1 追加单位根约束成 IGARCH)")

# --- 4a 无约束 GARCH(m,s): 拿到持续度 Σα̂+Σβ̂ (这一步还是普通 GARCH, 不是 IGARCH) ---
g = arch_model(a_t, mean="Zero", vol="Garch", p=p_hat, q=q_hat, dist="t").fit(disp="off")
alpha_hat = sum(g.params.filter(like="alpha"))
beta_hat = sum(g.params.filter(like="beta"))
omega_hat = g.params["omega"]; nu_hat = g.params.get("nu", 8.0)
persist = alpha_hat + beta_hat
print(f"  [4a] 无约束 GARCH{best_gorder}: omega={omega_hat:.4f}, alpha(Σ)={alpha_hat:.4f}, "
      f"beta(Σ)={beta_hat:.4f}, nu={nu_hat:.2f}")
print(f"       -> 持续度 Σα̂+Σβ̂ = {persist:.4f} (真值 {persist_true:.2f})")

# --- 4b 判定: 持续度≈1 才追加"I", 这不是默认 (1,1) ---
is_igarch = persist >= 0.99
print(f"  [4b] 判定: 持续度 {persist:.4f} "
      + (">= 0.99 ≈ 1 -> 施加单位根约束 alpha+beta=1, 改用 IGARCH [IGARCH]" if is_igarch
         else "< 0.99 -> 留在普通 GARCH, 不必上 IGARCH"))
print(f"       ★ 这个'I'是估完 GARCH 后按 Σα̂+Σβ̂≈1 才追加的约束, 不是一开始就默认 (1,1)")

# --- 4c 约束 MLE: 令 alpha1=1-beta, 自写标准化 t 似然重估 IGARCH(1,1) (比 GARCH(1,1) 少一个自由参数) ---
a_arr = np.asarray(a_t, dtype=float)
s2_0 = float(np.var(a_arr))                            # 书里建议: 用 a_t 样本方差作 sigma^2 初值
x0 = [max(omega_hat, 1e-4), float(np.clip(beta_hat / persist, 0.5, 0.99)), max(nu_hat, 4.0)]
opt = minimize(neg_loglik_igarch, x0, args=(a_arr, s2_0),
               method="L-BFGS-B", bounds=[(1e-8, None), (1e-4, 0.9999), (2.05, 200)])
omega_i, beta_i, nu_i = opt.x
alpha_i = 1.0 - beta_i                                 # 单位根约束: alpha1 = 1 - beta1 (被锁死)
s2_fit = igarch11_filter(a_arr, omega_i, beta_i, s2_0)
sigma_fit = np.sqrt(s2_fit)
loglik_i = -opt.fun
k_i = 3                                                # 自由参数 = omega, beta, nu (alpha1 不算)
aic_i = 2 * k_i - 2 * loglik_i
bic_i = k_i * np.log(len(a_arr)) - 2 * loglik_i
print(f"  [4c] 约束 IGARCH(1,1)-t (自写标准化 t 似然, alpha1=1-beta 锁死, 只估 omega/beta/nu 三参):")
print(f"       omega        ={omega_i:+.4f}   | 真值 {omega}")
print(f"       beta1        ={beta_i:+.4f}   | 真值 {beta_t}")
print(f"       alpha1=1-beta={alpha_i:+.4f}   | 真值 {alpha_t}   (被 beta 锁死, 非自由参数)")
print(f"       nu           ={nu_i:+.2f}   | 真值 {nu}")
print(f"       -> 持续度 alpha1+beta1 = {alpha_i + beta_i:.4f} (恰好=1, 单位根)")
print(f"       logL={loglik_i:.1f}, AIC={aic_i:.1f}, BIC={bic_i:.1f}  (对比无约束 GARCH BIC={g.bic:.1f}; 少一参)")
print(f"  无条件波动率 = 无穷 (IGARCH 无有限无条件方差!);  波动冲击半衰期 = 无穷 (冲击永不消散)")
print(f"  特例: 若 omega=0, 这就退化成 RiskMetrics/EWMA (日频常直接固定 beta=0.94, 连优化都免了)")

# ====================================================================
# 步骤5  模型检验: 约束 IGARCH 的标准化残差三查 + 持续度判定
# ====================================================================
print(SEP); print("步骤5  模型检验 (约束 IGARCH 的标准化残差三查 + 持续度判定)")
z = a_arr / sigma_fit                                  # 标准化残差 z_t = a_t / sigma_t
p_mean = acorr_ljungbox(z, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
p_vol = acorr_ljungbox(z ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
sk, ku = stats.skew(z), stats.kurtosis(z, fisher=False)
print(f"  ① 均值({mean_desc}够不够) ~a_t Ljung-Box(10) p={p_mean:.3g} -> "
      + ("均值方程充分 [OK]" if p_mean > 0.05 else "均值阶数不够, 改 ARMA [NO]"))
print(f"  ② 波动方程 ~a_t^2 Ljung-Box(10) p={p_vol:.3g} -> "
      + ("波动洗净, 模型充分 [OK]" if p_vol > 0.05 else "仍有 ARCH, 升阶/换模型 [NO]"))
print(f"  ③ 分布   偏度={sk:+.2f}, 峰度={ku:.2f} (正态=3; t 新息应仍略厚尾)")
print(f"  ④ 持续度  alpha+beta={alpha_i + beta_i:.4f} -> "
      + ("=1, 这是 IGARCH (积分型, 严格平稳但非协方差平稳)" if is_igarch else "<1, 协方差平稳的普通 GARCH"))
verdict = ("三查通过 + 持续度=1 -> 确认 IGARCH, 预测应'不回复'"
           if (p_mean > 0.05 and p_vol > 0.05 and is_igarch) else "有一项需复核")
print(f"  >>> {verdict}")

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].plot(sigma_fit, lw=.7, color="darkred")
ax[0, 0].set_title(f"Conditional volatility sigma_t  [IGARCH(1,1), persist={alpha_i + beta_i:.3f}]")
plot_acf(z ** 2, lags=25, ax=ax[0, 1], zero=False); ax[0, 1].set_title("ACF of std-resid^2 (white? -> vol cleaned)")
ax[1, 0].hist(z, bins=40, density=True, alpha=.7, edgecolor="k")
xs = np.linspace(z.min(), z.max(), 200); ax[1, 0].plot(xs, stats.norm.pdf(xs), "r--", label="N(0,1)")
ax[1, 0].set_title("Std-resid histogram (fat tails)"); ax[1, 0].legend()
stats.probplot(z, dist="norm", plot=ax[1, 1]); ax[1, 1].set_title("Q-Q plot vs Normal")
plt.tight_layout(); plt.savefig("igarch_fig3_diagnose.png", dpi=110); plt.close()

# ====================================================================
# 步骤6  使用模型: ARMA 均值点预测 + IGARCH 波动率"线性外推不回复"
#   IGARCH 多步波动: sigma^2(l) = omega + (alpha+beta)*sigma^2(l-1), 因 alpha+beta=1
#     -> sigma^2(l) = sigma^2(1) + (l-1)*omega  (式 3.22): 线性外推, 永不收敛 (对比 GARCH 几何收敛).
# ====================================================================
print(SEP); print(f"步骤6  预测 (均值 {mean_desc} + 波动 IGARCH 线性外推不回复, {h} 步)")
mu_path = mean_forecast(h)
a_T = a_arr[-1]; s2_T = s2_fit[-1]                      # 样本末端: 最后的冲击与条件方差
var1 = omega_i + beta_i * s2_T + (1.0 - beta_i) * a_T ** 2      # sigma^2(1)
var_path = var1 + np.arange(h) * omega_i                        # 式(3.22): sigma^2(l)=sigma^2(1)+(l-1)*omega
sig_path = np.sqrt(var_path)
q975 = stats.t.ppf(0.975, nu_i) * np.sqrt((nu_i - 2) / nu_i)
q01 = stats.t.ppf(0.01, nu_i) * np.sqrt((nu_i - 2) / nu_i)
lo, hi = mu_path - q975 * sig_path, mu_path + q975 * sig_path
VaR99 = -(mu_path + q01 * sig_path)
print(f"  均值点预测 mu(1)={mu_path[0]:+.3f} -> mu({h})={mu_path[-1]:+.3f} (ARMA 逐步回复到长期均值)")
print(f"  波动预测  sigma(1)={sig_path[0]:.3f} -> sigma({h})={sig_path[-1]:.3f} "
      f"(方差每步 +omega={omega_i:.3f}, sigma 线性外推)")
print(f"  ★ 关键: 波动'不回复'到任何固定水平 -> 持续度=1, 冲击永久 (对比 GARCH 向无条件水平几何收敛)")
print(f"  1日 99% VaR: 明日={VaR99[0]:.3f}%")
inside = (test.values >= lo) & (test.values <= hi)
print(f"  样本外 95% 区间覆盖率={inside.mean():.1%} (理想≈95%)")

pd.DataFrame({
    "step": np.arange(1, h + 1),               # 预测第几步(未来第几天)
    "mu_hat": np.round(mu_path, 4),            # 条件均值(中心, ARMA 给)
    "sigma_hat": np.round(sig_path, 4),        # 条件波动(区间宽度, IGARCH 给; 线性外推不回复)
    "lo95": np.round(lo, 4),                   # 95% 区间下界
    "hi95": np.round(hi, 4),                   # 95% 区间上界
    "VaR99": np.round(VaR99, 4),               # 1日 99% VaR(正数=潜在损失%)
    "actual": np.round(test.values, 4),        # 实际收益(事后真实值, 用于对照)
    "in_interval": inside,                     # 实际值是否落在 95% 区间内
}).to_csv("igarch_forecast.csv", index=False, encoding="utf-8")
print("  已导出每日预测(中心/波动/区间/VaR/实际) -> igarch_forecast.csv")

fig, axx = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
xx = np.arange(h)
axx[0].plot(xx, test.values, color="black", lw=.9, marker=".", ms=4, label="actual return")
axx[0].plot(xx, mu_path, "b-", lw=1.3, label=f"{mean_desc} mean forecast")
axx[0].fill_between(xx, lo, hi, color="red", alpha=.15, label="95% dynamic interval")
axx[0].set_title(f"Out-of-sample: mean {mean_desc} + vol IGARCH(1,1) (persist=1)")
axx[0].legend(loc="upper right")
axx[1].plot(xx, sig_path, "r-", lw=1.5, label="forecast volatility sigma(l)  (does NOT mean-revert)")
axx[1].plot(xx, VaR99, color="purple", lw=1.2, label="1-day 99% VaR (%)")
axx[1].set_xlabel("forecast horizon (days)")
axx[1].set_title("IGARCH: volatility forecast keeps drifting (no unconditional level)")
axx[1].legend(loc="upper right")
plt.tight_layout(); plt.savefig("igarch_fig4_forecast.png", dpi=110); plt.close()

# ====================================================================
print(SEP)
print(f"完成. 全流程: 平稳 -> 均值={mean_desc} -> ARCH效应 -> 定阶 (m,s)={best_gorder} "
      f"-> 无约束GARCH(持续度={persist:.3f}≈1) -> 追加单位根约束 -> IGARCH(1,1) 约束MLE "
      f"-> 三查 -> 均值+'不回复'波动/VaR 预测")
print(f"揭晓真值: {truth_txt}")
print("图: igarch_fig1_mean / igarch_fig2_order / igarch_fig3_diagnose / igarch_fig4_forecast (.png)")
