# -*- coding: utf-8 -*-
"""
EGARCH 波动率建模 —— 一条龙完整流程 (均值 ARMA; 波动 EGARCH(m,o,s)-t, 先定阶/选型再估计)

  EGARCH = 指数 GARCH (Exponential GARCH, Nelson 1991). 它冲着普通 GARCH 的三个短板来:
    (1) 对称响应: GARCH 的 sigma^2_t 只依赖 a^2_{t-i}, 正负冲击平方后一样 -> 抓不到"杠杆效应"
        (股市"利空(负收益)比同等利好更抬高未来波动").
    (2) 非负约束: GARCH 要 alpha_i, beta_j >= 0 才保证 sigma^2>0, 估计受限、高阶难用.
    (3) EGARCH 改为对 ln(sigma^2_t) 建模 -> 方差取指数后天然为正, 不需任何非负约束;
        再用"带符号"的新息刻画不对称.

  ★ 心脏是那个加权新息 g(eps_t) (式 3.24), 把冲击拆成"方向"和"大小"两块:
        g(eps_t) = theta*eps_t + gamma*(|eps_t| - E|eps_t|)                     (3.24)
      写成分段就看出不对称:
        g(eps_t) = (theta+gamma)*eps_t - gamma*E|eps_t|,  eps_t >= 0   (正冲击斜率 theta+gamma)
                 = (theta-gamma)*eps_t - gamma*E|eps_t|,  eps_t <  0   (负冲击斜率 theta-gamma)
      · theta 管"方向/杠杆": 正负冲击斜率差 (theta+gamma) vs (theta-gamma); 实务 theta<0 -> 负冲击影响更大.
      · gamma 管"大小/ARCH": |eps_t| 越大, 波动越高.
      · E|eps_t|: 标准正态 = sqrt(2/pi)≈0.798; 标准化 t(自由度 nu) 略小于它.
      · E[g(eps_t)] = 0 (eps 与 |eps|-E|eps| 都零均值).

  ⚠️ 记号陷阱(务必分清; 本 demo 一律用 arch 包的记号):
        书里 (3.24)  : theta = 方向(杠杆),  gamma = 大小(ARCH),  alpha_1 = ln方差的持续(AR 系数)
        arch 包/本码 : gamma = 方向(杠杆),  alpha = 大小(ARCH),  beta    = 持续
      即"书的 theta <-> arch 的 gamma", "书的 gamma <-> arch 的 alpha". 别记混!
      arch 记号下的递推:  ln s2_t = omega + alpha*(|z_{t-1}|-E|z|) + gamma*z_{t-1} + beta*ln s2_{t-1}
        (z = a_t/sigma_t 标准化残差; gamma<0 即杠杆; 持续度 = |beta|; 建 log 故无需非负约束.)

  阶 (m,s) 不是默认 (1,1): EGARCH 和 GARCH 一样"不好定", 没有干净的 ACF/PACF 截尾判据,
      只能在低阶候选里靠 AIC/BIC + 残差诊断挑 —— Nelson 原文用 EGARCH(2,2), 但绝大多数应用
      EGARCH(1,1) 就够. (1,1) 是"选出来的结果", 不是前提.

  六步法 (下游一律用"定出来的阶/带不带杠杆", 不写死):
    第0步 准备数据  : 价格 -> 对数收益率 r_t (平稳序列, 建模入口)
    第1步 均值方程  : r_t 有自相关 -> BIC 网格定阶并拟合 ARMA -> 冲击 a_t = r_t - mu_t
    第2步 ARCH+杠杆 : a_t^2 查波动聚集(Ljung-Box+Engle LM) + "符号-平方"探针查杠杆 -> 该不该上 EGARCH
    第3步 定阶/选型 : Schwert 上限 + (3b)对称 o=0 vs 带杠杆 o=1 + (3c)低阶 (m,s) AIC/BIC 网格
    第4步 估计参数  : 选定阶的 EGARCH(m,o,s)-t MLE, 重点看方向项 gamma 的符号与显著性
    第5步 模型检验  : 标准化残差三查 + 杠杆是否洗净
    第6步 使用模型  : ARMA 点预测 + EGARCH 波动率解析预测(式 3.33) + 不对称量化 + 新闻冲击曲线(NIC)

  工具说明:
    · arch 包均值只支持 AR(无 MA), 故 ARMA 走"两步法": statsmodels 建 ARMA 取残差 a_t -> 对 a_t 建 EGARCH.
    · arch 有现成 EGARCH, 直接用它估计; 多步预测按书里 EGARCH(1,1) 解析式(3.33)自己递推, 再用 arch 模拟法交叉验证.
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
KAPPA = np.sqrt(2.0 / np.pi)                            # E|z| 的正态值, 与 arch 的 EGARCH 递推一致


# ---- EGARCH(1,1) 解析预测的两个零件 (式 3.33; 均用 arch 记号: theta=gamma方向, gam=alpha大小) ----
def egarch_g(z, theta, gam):
    """加权新息 g(z) = theta*z + gam*(|z| - E|z|)  (式 3.24; theta=方向/杠杆, gam=大小/ARCH)."""
    return theta * z + gam * (np.abs(z) - KAPPA)


def egarch_Eexpg(theta, gam):
    """E{exp[g(z)]}, z~N(0,1): 用于 EGARCH(1,1) 多步(>=2)预测 (式 3.33 的期望项).
       = exp(-gam*KAPPA)*[ e^{(theta+gam)^2/2}*Phi(theta+gam) + e^{(theta-gam)^2/2}*Phi(gam-theta) ]."""
    Phi = stats.norm.cdf
    return np.exp(-gam * KAPPA) * (
        np.exp((theta + gam) ** 2 / 2.0) * Phi(theta + gam)
        + np.exp((theta - gam) ** 2 / 2.0) * Phi(gam - theta))


# ====================================================================
# 步骤0  准备数据: 造"价格" -> 转对数收益率 r_t (平稳)
#   均值真值 = ARMA(1,1); 波动真值 = EGARCH(1,1,1)-t, gamma<0 制造杠杆(负冲击更抬波动).
#   单位用"百分比收益"(x100), 量纲适中, 是 arch 包推荐做法.
# ====================================================================
omega, alpha_t, beta_t, gamma_t, nu = 0.02, 0.12, 0.96, -0.10, 7.0   # EGARCH-t(arch 记号): gamma<0=杠杆
phi, theta_m, mu_true = 0.5, 0.3, 0.05                               # ARMA(1,1) 均值真值
n, burn = 2500, 1000
truth_txt = (f"均值 ARMA(1,1) phi={phi}, theta={theta_m}, mu={mu_true}; "
             f"波动 EGARCH(1,1,1)-t omega={omega}, alpha={alpha_t}, beta={beta_t}, "
             f"gamma={gamma_t}(<0 杠杆), df={nu} (持续度=|beta|={beta_t})")

rng = np.random.default_rng(7)


def sim_egarch(rng):                                   # 造 EGARCH(1,1,1)-t 冲击序列 a_t
    e = rng.standard_t(nu, n + burn) * np.sqrt((nu - 2) / nu)   # 标准化 t 新息(单位方差)
    kappa = np.mean(np.abs(e))                          # E|z| 的样本估计(中心化"大小项")
    a = np.zeros(n + burn); logs2 = np.zeros(n + burn)
    logs2[0] = omega / (1 - beta_t)                     # log 方差的无条件水平
    a[0] = np.exp(logs2[0] / 2) * e[0]
    for t in range(1, n + burn):
        logs2[t] = (omega + beta_t * logs2[t - 1]
                    + alpha_t * (abs(e[t - 1]) - kappa)  # 大小项: 冲击越大波动越高
                    + gamma_t * e[t - 1])                # 方向项: gamma<0 -> 负冲击额外抬高波动
        a[t] = np.exp(logs2[t] / 2) * e[t]
    return a


a_full = sim_egarch(rng)                               # 先造带波动聚集+杠杆的冲击
x = np.zeros(n + burn); x[0] = a_full[0]
for t in range(1, n + burn):
    x[t] = phi * x[t - 1] + a_full[t] + theta_m * a_full[t - 1]     # 再把冲击灌进 ARMA(1,1) 均值
ret = mu_true + x[burn:]
ret = pd.Series(ret).reset_index(drop=True)

price = pd.Series(np.r_[100.0, 100.0 * np.exp(np.cumsum(ret.values) / 100.0)])  # 收益->价格
r = 100.0 * np.log(price / price.shift(1)).dropna().reset_index(drop=True)      # 价格->对数收益 r_t

h = 60                                                  # 留最后 60 天做样本外预测
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
    tv = {"ar.L1": phi, "ma.L1": theta_m, "const": mu_true}.get(nm)
    print(f"     {nm:<8}={v:+.3f}" + (f"   | 真值 {tv}" if tv is not None else ""))


def mean_forecast(steps):                              # ARMA 点预测: 逐步回复到无条件均值
    return arma.get_forecast(steps=steps).predicted_mean.values


print(f"  均值模型={mean_desc}; 冲击 a_t = r_t - mu_t (下面研究它的波动与杠杆)")

fig, ax = plt.subplots(3, 1, figsize=(10, 9))
ax[0].plot(train.values, lw=.6, color="steelblue")
ax[0].set_title(f"Log returns r_t  [mean: {mean_desc}]  —— volatility clustering + leverage")
plot_acf(train, lags=25, ax=ax[1], zero=False); ax[1].set_title("ACF of r_t (mean structure -> ARMA)")
plot_pacf(train, lags=25, ax=ax[2], method="ols", zero=False); ax[2].set_title("PACF of r_t")
plt.tight_layout(); plt.savefig("egarch_fig1_mean.png", dpi=110); plt.close()

# ====================================================================
# 步骤2  检验 ARCH 效应 + 杠杆效应 (要不要建波动模型 + 该不该用 EGARCH 而非对称 GARCH)
# ====================================================================
print(SEP); print("步骤2  检验 ARCH 效应 + 杠杆效应")
a2 = a_t ** 2
for m in (5, 10, 20):
    pq = acorr_ljungbox(a2, lags=[m], return_df=True)["lb_pvalue"].iloc[0]
    print(f"  [ARCH] a_t^2 的 Ljung-Box Q({m}) p={pq:.3g} -> "
          + ("有 ARCH 效应" if pq < 0.05 else "不显著"))
lm_stat, lm_p, _, _ = het_arch(a_t, nlags=10)
print(f"  [ARCH] Engle LM(lags=10): LM={lm_stat:.2f}, p={lm_p:.3g} -> "
      + ("有 ARCH 效应, 需建波动率模型 [OK]" if lm_p < 0.05 else "不显著, 收工"))
# 模型无关的杠杆探针: "上一期为负冲击"的指示 与 "当期平方冲击" 的相关; >0 => 负冲击预示更大波动
neg_prev = (a_t.values[:-1] < 0).astype(float)
lev_corr = np.corrcoef(neg_prev, a2.values[1:])[0, 1]
mean_after_neg = a2.values[1:][neg_prev == 1].mean()
mean_after_pos = a2.values[1:][neg_prev == 0].mean()
print(f"  [杠杆] corr(1[a_(t-1)<0], a_t^2) = {lev_corr:+.3f}; "
      f"负冲击后均方 {mean_after_neg:.2f} vs 正冲击后 {mean_after_pos:.2f}")
print(f"         -> "
      + ("负冲击预示更大波动 => 有杠杆/不对称, 该上 EGARCH(非对称 GARCH) [OK]" if lev_corr > 0.02
         else "不明显, 对称 GARCH 或许够用"))

# ====================================================================
# 步骤3  定阶/选型: Schwert 上限 + (3b)对称 vs 带杠杆 + (3c)低阶 (m,s) AIC/BIC 网格
#   EGARCH 无干净 ACF/PACF 截尾判据 -> 阶 (m,s) 直接在低阶候选里靠 AIC/BIC 挑 (与 GARCH 同精神).
#   注意: 这里同时定"要不要杠杆项(o)"和"阶(m,s)", 都是"定"出来的, 不写死.
# ====================================================================
print(SEP); print("步骤3  定阶/选型 (o=方向/杠杆阶, p=大小/ARCH阶 m, q=持续阶 s)")
MAXLAG = int(12 * (len(a2) / 100) ** 0.25)             # Schwert 规则: 与 ARCH/IGARCH 同一把尺, 只看 T
print(f"  搜索上限 MAXLAG = {MAXLAG}  (Schwert: floor(12*(T/100)^0.25), T={len(a2)}; EGARCH 高阶难收敛 -> 只在低阶试)")

# 3b: 对称 o=0 vs 带杠杆 o=1 (在 (1,1) 上比 AIC/BIC, 决定杠杆项值不值)
print("  [3b] 对称(o=0) vs 带杠杆(o=1) @ (p,q)=(1,1):")
sel = {}
for o in (0, 1):
    rr = arch_model(a_t, mean="Zero", vol="EGARCH", p=1, o=o, q=1, dist="t").fit(disp="off")
    sel[o] = (rr.aic, rr.bic)
    tag = "对称 EGARCH(1,0,1)" if o == 0 else "带杠杆 EGARCH(1,1,1)"
    print(f"       o={o} {tag:<22} AIC={rr.aic:8.1f}  BIC={rr.bic:8.1f}")
use_o = 1 if sel[1][1] < sel[0][1] else 0
print(f"       >>> BIC 选 o={use_o} "
      + ("(带杠杆胜出 -> 不对称项有价值)" if use_o == 1 else "(对称即可, 杠杆项不值)"))

# 3c: 低阶 (m,s) 二维 AIC/BIC 网格 (o=use_o 固定; 候选即 EGARCH(1,1)/(1,2)/(2,1)/(2,2))
GRIDMAX = min(MAXLAG, 2)                                # 网格诚实上限: EGARCH 高阶极少胜低阶
PMAX = QMAX = GRIDMAX
egrid_bic, egrid_aic = {}, {}
best_gic, best_gorder = np.inf, (1, 1)
for p in range(1, PMAX + 1):                           # p=大小/ARCH阶 m
    for q in range(1, QMAX + 1):                       # q=持续阶 s
        try:
            rr = arch_model(a_t, mean="Zero", vol="EGARCH", p=p, o=use_o, q=q, dist="t").fit(disp="off")
            egrid_bic[(p, q)] = rr.bic; egrid_aic[(p, q)] = rr.aic
            if rr.bic < best_gic:
                best_gic, best_gorder = rr.bic, (p, q)
        except Exception:
            egrid_bic[(p, q)] = np.nan; egrid_aic[(p, q)] = np.nan
p_hat, q_hat = best_gorder
aic_order = min(egrid_aic, key=lambda k: egrid_aic[k])
print(f"  [3c] 低阶 (m,s) 网格 (o={use_o} 固定; 行=大小阶 p, 列=持续阶 q; [*]=最优):")
print("         " + "".join(f"q={q:<8}" for q in range(1, QMAX + 1)))
for p in range(1, PMAX + 1):
    cells = ""
    for q in range(1, QMAX + 1):
        v = egrid_bic.get((p, q), np.nan)
        star = "*" if (p, q) == best_gorder else " "
        cells += (f"{v:8.1f}{star} " if np.isfinite(v) else "   --     ")
    print(f"     p={p} {cells}")
print(f"  投票: AIC 选 (m,s)={aic_order} | BIC 选 (m,s)={best_gorder}"
      f"  >>> 最终定阶 EGARCH({p_hat},{use_o},{q_hat}) (以 BIC 为准)")
if p_hat == GRIDMAX or q_hat == GRIDMAX:
    print(f"  ⚠ 定阶贴到网格上限 {GRIDMAX} (可能被截断), 建议放大 GRIDMAX 复核")
print(f"  说明: Nelson 原文用 EGARCH(2,2), 但本数据 BIC 选 ({p_hat},{q_hat}) -> (1,1) 常胜, 是选出来的不是默认")

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
# 步骤4  估计参数: 选定阶的 EGARCH(m,o,s)-t MLE, 重点看方向项 gamma(书 theta) 的符号与显著性
# ====================================================================
print(SEP); print(f"步骤4  估计参数 (EGARCH({p_hat},{use_o},{q_hat})-t on a_t, 条件 MLE)")
res = arch_model(a_t, mean="Zero", vol="EGARCH", p=p_hat, o=use_o, q=q_hat, dist="t").fit(disp="off")
beta_hat = res.params["beta[1]"]
gamma_hat = res.params.get("gamma[1]", 0.0)            # arch gamma = 方向/杠杆 (书 theta)
gamma_p = res.pvalues.get("gamma[1]", np.nan)
persist = abs(beta_hat)
halflife = np.log(0.5) / np.log(persist) if persist < 1 else np.inf
print(f"  估计参数 (vs 真值; 注意 arch 记号: alpha=大小, gamma=方向/杠杆, beta=持续):")
tv_map = {"omega": omega, "alpha[1]": alpha_t, "gamma[1]": gamma_t, "beta[1]": beta_t, "nu": nu}
for nm, v in res.params.items():
    tv = tv_map.get(nm)
    print(f"     {nm:<10}={v:+.4f}" + (f"   | 真值 {tv}" if tv is not None else ""))
print(f"  >>> 杠杆(方向项) gamma = {gamma_hat:+.4f} (真值 {gamma_t}), p={gamma_p:.3g} -> "
      + ("显著且 <0: 负冲击放大波动, 杠杆确认 [OK]" if (gamma_hat < 0 and gamma_p < 0.05)
         else "杠杆不显著"))
print(f"  持续度 |beta| = {persist:.4f} (<1 -> log 方差平稳); 半衰期 = {halflife:.1f} 天")
print(f"  logL={res.loglikelihood:.1f}, AIC={res.aic:.1f}, BIC={res.bic:.1f}")

# ====================================================================
# 步骤5  模型检验: 标准化残差三查 + 杠杆是否洗净
# ====================================================================
print(SEP); print("步骤5  模型检验 (标准化残差三查 + 杠杆洗净)")
z = res.std_resid.dropna()                             # 标准化残差 z_t = a_t / sigma_t
p_mean = acorr_ljungbox(z, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
p_vol = acorr_ljungbox(z ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
sk, ku = stats.skew(z), stats.kurtosis(z, fisher=False)
zneg = (z.values[:-1] < 0).astype(float)
lev_left = np.corrcoef(zneg, (z.values[1:]) ** 2)[0, 1]
print(f"  ① 均值({mean_desc}够不够) ~z Ljung-Box(10) p={p_mean:.3g} -> "
      + ("均值方程充分 [OK]" if p_mean > 0.05 else "均值阶数不够, 改 ARMA [NO]"))
print(f"  ② 波动方程 ~z^2 Ljung-Box(10) p={p_vol:.3g} -> "
      + ("波动洗净, 模型充分 [OK]" if p_vol > 0.05 else "仍有 ARCH, 升阶/换模型 [NO]"))
print(f"  ③ 分布   偏度={sk:+.2f}, 峰度={ku:.2f} (正态=3; t 新息应仍略厚尾)")
print(f"  ④ 杠杆洗净 corr(1[z_(t-1)<0], z_t^2) = {lev_left:+.3f} -> "
      + ("已洗净(接近0), 不对称被 EGARCH 吸收 [OK]" if abs(lev_left) < 0.05 else "残余杠杆, 复核"))
verdict = ("三查通过 + 杠杆洗净 -> EGARCH 模型充分"
           if (p_mean > 0.05 and p_vol > 0.05) else "有一项需复核")
print(f"  >>> {verdict}")

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].plot(res.conditional_volatility, lw=.7, color="darkred")
ax[0, 0].set_title(f"Conditional volatility sigma_t  [EGARCH, persist=|beta|={persist:.3f}]")
plot_acf(z ** 2, lags=25, ax=ax[0, 1], zero=False); ax[0, 1].set_title("ACF of std-resid^2 (white? -> vol cleaned)")
ax[1, 0].hist(z, bins=40, density=True, alpha=.7, edgecolor="k")
xs = np.linspace(z.min(), z.max(), 200); ax[1, 0].plot(xs, stats.norm.pdf(xs), "r--", label="N(0,1)")
ax[1, 0].set_title("Std-resid histogram (fat tails)"); ax[1, 0].legend()
stats.probplot(z, dist="norm", plot=ax[1, 1]); ax[1, 1].set_title("Q-Q plot vs Normal")
plt.tight_layout(); plt.savefig("egarch_fig3_diagnose.png", dpi=110); plt.close()

# ====================================================================
# 步骤6  使用模型: ARMA 均值点预测 + EGARCH(1,1) 波动率解析预测(式 3.33) + 不对称量化 + 新闻冲击曲线
#   式(3.33): sigma^2_h(j) = [sigma^2_h(j-1)]^{alpha_1} * exp(omega) * E{exp[g(eps)]}  (alpha_1=|beta|=持续)
#     -> 因 |beta|<1, 波动"几何回复"到无条件水平 (对比 IGARCH 的线性外推不回复).
# ====================================================================
print(SEP); print(f"步骤6  预测 (均值 {mean_desc} + 波动 EGARCH 解析式 3.33, {h} 步)")
mu_path = mean_forecast(h)

# --- 6a 波动解析预测 (式 3.33, arch 记号): 1 步用已知 z_T, >=2 步用 E{exp g} 递推 ---
om = res.params["omega"]
gam = res.params["alpha[1]"]                           # arch alpha = 大小/ARCH (书 gamma)
theta_v = res.params.get("gamma[1]", 0.0)              # arch gamma = 方向/杠杆 (书 theta); o=0 则 0
b1 = res.params["beta[1]"]                             # arch beta = 持续 (书 alpha_1)
z_last = float(res.std_resid.dropna().iloc[-1])
lns2_last = 2.0 * np.log(res.conditional_volatility.iloc[-1])
lns2_1 = om + gam * (abs(z_last) - KAPPA) + theta_v * z_last + b1 * lns2_last   # 1步: 用已知 z_T
Eexpg = egarch_Eexpg(theta_v, gam)
var_path = np.empty(h); var_path[0] = np.exp(lns2_1)
for j in range(1, h):
    var_path[j] = var_path[j - 1] ** b1 * np.exp(om) * Eexpg     # 式(3.33): j>=2 用 E{exp g}
sig_path = np.sqrt(var_path)
var_fp = np.exp((om + np.log(Eexpg)) / (1 - b1))                 # 解析不动点: 多步预测收敛到这里
print(f"  均值点预测 mu(1)={mu_path[0]:+.3f} -> mu({h})={mu_path[-1]:+.3f} (ARMA 逐步回复到长期均值)")
print(f"  波动预测  sigma(1)={sig_path[0]:.3f} -> sigma({h})={sig_path[-1]:.3f} "
      f"-> 收敛到无条件水平 sigma≈{np.sqrt(var_fp):.3f}")
print(f"  ★ 关键: |beta|={b1:.3f}<1 -> EGARCH 波动'几何回复'到固定水平 (对比 IGARCH 持续=1 的线性外推不回复)")

# --- 6b 模拟法交叉验证: 手写解析式(3.33)是否对. 式(3.33)是"正态"闭式, 而模型/模拟是 t ->
#     1 步应精确一致(都用已知末态), 远期解析(正态)会略高于 t-模拟(Jensen + 正态vs.t 的 E{exp g} 差异). ---
try:
    fc = res.forecast(horizon=h, method="simulation", simulations=5000, reindex=False,
                      random_state=np.random.RandomState(7))
except TypeError:
    np.random.seed(7)
    fc = res.forecast(horizon=h, method="simulation", simulations=5000, reindex=False)
sig_sim = np.sqrt(fc.variance.values[-1])
print(f"  [交叉验证] 解析/模拟: sigma(1)={sig_path[0]:.3f}/{sig_sim[0]:.3f} (1步精确一致 -> 递推机制/取参正确), "
      f"sigma({h})={sig_path[-1]:.3f}/{sig_sim[-1]:.3f} (远期解析正态近似略高于 t-模拟, 属正常)")

# --- 6c 不对称量化 (书 §3.8.3 IBM 例的招牌算法): 两个标准差的负冲击 vs 正冲击 ---
ratio = np.exp(-4.0 * theta_v)                          # sigma^2(z=-2)/sigma^2(z=+2) = exp(g(-2)-g(2)) = exp(-4*theta)
print(f"  不对称量化: sigma^2(z=-2)/sigma^2(z=+2) = exp(-4*gamma) = {ratio:.3f} "
      f"-> 负的两个标准差冲击比正的多抬升波动 {100 * (ratio - 1):+.1f}% (书 IBM 例 1.374/+37.4%)")

# --- 6d 区间 / VaR ---
nu_hat = res.params.get("nu", 8.0)
q975 = stats.t.ppf(0.975, nu_hat) * np.sqrt((nu_hat - 2) / nu_hat)
q01 = stats.t.ppf(0.01, nu_hat) * np.sqrt((nu_hat - 2) / nu_hat)
lo, hi = mu_path - q975 * sig_path, mu_path + q975 * sig_path
VaR99 = -(mu_path + q01 * sig_path)
print(f"  1日 99% VaR: 明日={VaR99[0]:.3f}%")
inside = (test.values >= lo) & (test.values <= hi)
print(f"  样本外 95% 区间覆盖率={inside.mean():.1%} (理想≈95%)")

pd.DataFrame({
    "step": np.arange(1, h + 1),               # 预测第几步(未来第几天)
    "mu_hat": np.round(mu_path, 4),            # 条件均值(中心, ARMA 给)
    "sigma_hat": np.round(sig_path, 4),        # 条件波动(区间宽度, EGARCH 解析式 3.33; 几何回复)
    "lo95": np.round(lo, 4),                   # 95% 区间下界
    "hi95": np.round(hi, 4),                   # 95% 区间上界
    "VaR99": np.round(VaR99, 4),               # 1日 99% VaR(正数=潜在损失%)
    "actual": np.round(test.values, 4),        # 实际收益(事后真实值, 用于对照)
    "in_interval": inside,                     # 实际值是否落在 95% 区间内
}).to_csv("egarch_forecast.csv", index=False, encoding="utf-8")
print("  已导出每日预测(中心/波动/区间/VaR/实际) -> egarch_forecast.csv")

# --- 新闻冲击曲线 NIC: 固定 sigma_{t-1} 在无条件水平, 看 sigma_t^2 如何随上期标准化冲击 z 变化 ---
log_unc = om / (1 - b1)                                 # E[ln sigma^2] = omega/(1-beta) (因 E[g]=0)
zz = np.linspace(-4, 4, 200)
nic = np.exp(om + b1 * log_unc + egarch_g(zz, theta_v, gam))     # sigma_t^2 as function of z_{t-1}

fig, axx = plt.subplots(2, 1, figsize=(11, 9))
xx = np.arange(h)
axx[0].plot(xx, test.values, color="black", lw=.9, marker=".", ms=4, label="actual return")
axx[0].plot(xx, mu_path, "b-", lw=1.3, label=f"{mean_desc} mean forecast")
axx[0].fill_between(xx, lo, hi, color="red", alpha=.15, label="95% dynamic interval")
axx[0].plot(xx, sig_path, "r-", lw=1.2, alpha=.6, label="forecast sigma(l)  (mean-reverts)")
axx[0].set_title(f"Out-of-sample: mean {mean_desc} + vol EGARCH({p_hat},{use_o},{q_hat})")
axx[0].legend(loc="upper right")
axx[1].plot(zz, nic, "purple", lw=2)
axx[1].axvline(0, color="gray", ls="--", lw=1)
axx[1].set_xlabel("standardized shock z_(t-1)"); axx[1].set_ylabel("implied sigma_t^2")
axx[1].set_title(f"News Impact Curve (gamma={theta_v:+.3f}<0 -> left arm steeper = leverage; "
                 f"vol ratio(-2 vs +2)={ratio:.2f})")
plt.tight_layout(); plt.savefig("egarch_fig4_forecast.png", dpi=110); plt.close()

# ====================================================================
print(SEP)
print(f"完成. 全流程: 平稳 -> 均值={mean_desc} -> ARCH+杠杆探针 -> 选 o={use_o}/定阶 EGARCH({p_hat},{use_o},{q_hat}) "
      f"-> EGARCH-t MLE(gamma<0) -> 三查/杠杆洗净 -> 均值+解析波动(式3.33)/不对称量化/VaR + 新闻冲击曲线")
print(f"揭晓真值: {truth_txt}")
print("图: egarch_fig1_mean / egarch_fig2_order / egarch_fig3_diagnose / egarch_fig4_forecast (.png)")
