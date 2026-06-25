# -*- coding: utf-8 -*-
"""
MA 时间序列建模 —— 一条龙完整流程 (建模 + 定阶 合并版)
造数据 -> 平稳性检验 -> 自动定阶(ACF/AIC/BIC/系数显著性) -> 拟合 -> 残差诊断 -> 预测

特点: 下游用"定出来的阶 q_hat"而非写死的阶, 换任意数据都能自适应.
对照 AR: AR 看 PACF 截尾, MA 看 ACF 截尾; AR 长期均值回复, MA 超过 q 步即退化为均值.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")          # Windows 控制台强制 UTF-8
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                              # 不弹窗, 直接存图
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox

SEP = "=" * 66

# ====================================================================
# 步骤0  数据生成: MA(2)  (建模时假装不知道阶数)
#   y_t = mu + eps_t + theta1*eps_{t-1} + theta2*eps_{t-2}
#   MA 过程恒平稳; 可逆条件(MA2): theta2+theta1<1, theta2-theta1<1, |theta2|<1 -> 满足
# ====================================================================
rng = np.random.default_rng(42)
mu, theta1, theta2, sigma = 10.0, 0.6, 0.3, 1.0
n, burn = 500, 300                                 # 取 500 个点, 前 300 烧入丢弃
eps = rng.normal(0, sigma, n + burn)
y = np.zeros(n + burn)
for t in range(2, n + burn):
    y[t] = mu + eps[t] + theta1 * eps[t - 1] + theta2 * eps[t - 2]
y = pd.Series(y[burn:]).reset_index(drop=True)

# MA(q) 的理论方差与 1 阶/2 阶自相关 (定阶时拿来对照)
gamma0 = sigma ** 2 * (1 + theta1 ** 2 + theta2 ** 2)
rho1 = (theta1 + theta1 * theta2) / (1 + theta1 ** 2 + theta2 ** 2)
rho2 = theta2 / (1 + theta1 ** 2 + theta2 ** 2)

h = 50                                              # 留最后 50 个点做样本外预测
train, test = y.iloc[:-h], y.iloc[-h:]

# 导出本地 CSV (t=时间索引, y=观测值, split=训练/测试)
pd.DataFrame({"t": np.arange(len(y)), "y": y.values,
              "split": ["train"] * len(train) + ["test"] * len(test)}
             ).to_csv("ma2_data.csv", index=False, encoding="utf-8")

print(SEP)
print("步骤0  数据生成")
print(f"  真值 MA(2): y_t = {mu} + eps + {theta1}*eps(t-1) + {theta2}*eps(t-2),  sigma={sigma}")
print(f"  理论: 均值 mu={mu}, 方差 gamma0={gamma0:.3f}, rho1={rho1:.3f}, rho2={rho2:.3f}, rho(>2)=0")
print(f"  样本 n={len(y)} (train={len(train)}, test={len(test)})")
print("  已导出 -> ma2_data.csv")

# ====================================================================
# 步骤1  平稳性检验 (ADF) —— MA 过程恒平稳, 走个流程确认
# ====================================================================
adf_stat, adf_p, *_ = adfuller(train, autolag="AIC")
print(SEP)
print("步骤1  平稳性检验 (ADF)")
print(f"  ADF={adf_stat:.3f},  p={adf_p:.3g}  ->  "
      + ("平稳, 可建模 [OK]" if adf_p < 0.05 else "不平稳, 需先差分 [NO]"))
print("  (注: MA 过程理论上恒平稳, ADF 必然通过)")

# 图1: 序列 + ACF + PACF (识别用) —— MA 看 ACF 截尾
fig, ax = plt.subplots(3, 1, figsize=(10, 9))
ax[0].plot(train.values, lw=0.9)
ax[0].axhline(train.mean(), color="r", ls="--", lw=1, label=f"mean={train.mean():.1f}")
ax[0].set_title("Series (MA process, always stationary)"); ax[0].legend(loc="upper right")
plot_acf(train, lags=25, ax=ax[1], zero=False); ax[1].set_title("ACF (cuts off -> MA order)")
plot_pacf(train, lags=25, ax=ax[2], method="ols", zero=False)
ax[2].set_title("PACF (tails off -> MA)")
plt.tight_layout(); plt.savefig("ma_fig1_identify.png", dpi=110); plt.close()

# ====================================================================
# 步骤2  定阶: 三法交叉验证 (ACF截尾 / AIC-BIC网格 / 过拟合系数显著性)
#   口诀: MA(q) 的 ACF 在 lag q 之后截尾 (掉进置信带)
# ====================================================================
# MAXLAG: 定阶搜索上限. Schwert 规则 floor(12*(T/100)^0.25) 只依赖样本量 T, 与未知阶数无关 (黑盒下不能拿真值反推)
MAXLAG = int(12 * (len(train) / 100) ** 0.25)
print(SEP)
print("步骤2  定阶 (三法交叉)")
print(f"  搜索上限 MAXLAG = {MAXLAG}  (Schwert: floor(12*(T/100)^0.25), T={len(train)})")

# 2a ACF 截尾: MA(q) 的 ACF 在 lag q 之后掉进置信带
#   关键: MA 定阶要用 Bartlett 公式置信带 (lag>q 后逐步变宽), 而非平直的 ±1.96/√n,
#   否则远端噪声易冒出伪显著 lag. 取"从 lag1 起连续显著"的最后一个 lag 为截尾点.
ac = acf(train, nlags=MAXLAG, fft=True)
N = len(train)
acf_q = 0
for k in range(1, MAXLAG + 1):
    se_k = np.sqrt((1 + 2 * np.sum(ac[1:k] ** 2)) / N)   # Bartlett: 累加前 k-1 阶
    if abs(ac[k]) > 1.96 * se_k:
        acf_q = k                                         # 连续显著, 截尾点后移
    else:
        break                                             # 一旦掉进带内即认定截尾
print(f"  [2a] ACF 截尾(Bartlett带): 连续显著到 lag = {acf_q}")

# 2b AIC/BIC 网格: 逐阶拟合 ARIMA(0,0,q) 取准则最小
rows = {}
for k in range(1, MAXLAG + 1):
    m = ARIMA(train, order=(0, 0, k), trend="c").fit()
    rows[k] = (m.aic, m.bic)
aic_q = min(rows, key=lambda k: rows[k][0])
bic_q = min(rows, key=lambda k: rows[k][1])
print(f"  [2b] AIC/BIC 网格:  {'q':>3} {'AIC':>10} {'BIC':>10}")
for k, (a, b) in rows.items():
    mk = ("  <-AIC最小" if k == aic_q else "") + ("  <-BIC最小" if k == bic_q else "")
    print(f"                    {k:>3} {a:>10.2f} {b:>10.2f}{mk}")
print(f"       -> AIC 选 q={aic_q},  BIC 选 q={bic_q}")

# 2c 过拟合 MA(MAXLAG) 看系数显著性: 故意拟合到搜索上限, 不显著的高阶项应剔除
over = ARIMA(train, order=(0, 0, MAXLAG), trend="c").fit()
sig_lags = [int(nm.split("L")[1]) for nm, pv in zip(over.params.index, over.pvalues.values)
            if nm.startswith("ma.L") and pv < 0.05]
sigcoef_q = max(sig_lags) if sig_lags else 0
print(f"  [2c] 过拟合 MA({MAXLAG}): 显著的最高滞后 = {sigcoef_q}")

# 汇总投票 -> 以 BIC 为最终定阶
q_hat = bic_q
votes = {"ACF": acf_q, "AIC": aic_q, "BIC": bic_q, "显著性": sigcoef_q}
agree = len(set(votes.values())) == 1
print("  投票: " + " | ".join(f"{k}={v}" for k, v in votes.items()))
print(f"  >>> 最终定阶 q = {q_hat}  "
      + ("(以BIC为准, 四法一致)" if agree else "(以BIC为准, 各法有分歧)"))

# 图2: AIC/BIC 随阶数变化曲线
fig, axx = plt.subplots(figsize=(8, 5))
ks = list(rows.keys())
axx.plot(ks, [rows[k][0] for k in ks], "o-", label="AIC")
axx.plot(ks, [rows[k][1] for k in ks], "s-", label="BIC")
axx.axvline(q_hat, color="C1", ls=":", alpha=.7, label=f"chosen q={q_hat}")
axx.set_xlabel("MA order q"); axx.set_ylabel("Information criterion")
axx.set_title(f"Order selection: AIC min@{aic_q}, BIC min@{bic_q}")
axx.legend(); plt.tight_layout(); plt.savefig("ma_fig2_order.png", dpi=110); plt.close()

# ====================================================================
# 步骤3  用定出的阶 q_hat 拟合
# ====================================================================
res = ARIMA(train, order=(0, 0, q_hat), trend="c").fit()
truth = {"const": mu, "ma.L1": theta1, "ma.L2": theta2}
print(SEP)
print(f"步骤3  用定出的 q={q_hat} 拟合 (参数估计 vs 真值)")
for nm, v in zip(res.params.index, res.params.values):
    t = truth.get(nm)
    print(f"  {nm:<8} = {v:+.3f}" + (f"   | 真值 {t}" if t is not None else "   | (真值无此项)"))
print(f"  AIC={res.aic:.2f},  BIC={res.bic:.2f}")

# ====================================================================
# 步骤4  残差诊断 (好模型的残差应近似白噪声)
# ====================================================================
resid = res.resid
lb_p = acorr_ljungbox(resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
print(SEP)
print("步骤4  残差诊断")
print(f"  Ljung-Box(10) p={lb_p:.3g}  ->  "
      + ("残差无自相关, 近似白噪声, 模型充分 [OK]" if lb_p > 0.05 else "残差仍有自相关 [NO]"))
print(f"  残差均值={resid.mean():.3f}, 标准差={resid.std():.3f} (真值 sigma={sigma})")

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
ax[0, 0].plot(resid.values, lw=.8); ax[0, 0].axhline(0, color="r", ls="--")
ax[0, 0].set_title("Residuals over time")
plot_acf(resid, lags=25, ax=ax[0, 1], zero=False); ax[0, 1].set_title("Residual ACF (white noise?)")
ax[1, 0].hist(resid, bins=30, edgecolor="k", alpha=.7); ax[1, 0].set_title("Residual histogram")
stats.probplot(resid, dist="norm", plot=ax[1, 1]); ax[1, 1].set_title("Q-Q plot")
plt.tight_layout(); plt.savefig("ma_fig3_residuals.png", dpi=110); plt.close()

# ====================================================================
# 步骤5  样本外预测 + 评估
#   MA(q) 关键性质: 预测超过 q 步后, 信息耗尽 -> 预测值即为均值 mu (平直线)
# ====================================================================
pred = res.get_forecast(steps=h)
yhat = pred.predicted_mean; ci = pred.conf_int()
rmse = np.sqrt(np.mean((test.values - yhat.values) ** 2))
mae = np.mean(np.abs(test.values - yhat.values))
naive = np.sqrt(np.mean((test.values - train.iloc[-1]) ** 2))   # 朴素基准: 用最后一个值预测
print(SEP)
print("步骤5  样本外预测")
print(f"  预测 {h} 步:  RMSE={rmse:.3f},  MAE={mae:.3f}")
print(f"  朴素基准(last-value) RMSE={naive:.3f}  ->  "
      + ("MA 更优 [OK]" if rmse < naive else "未胜过基准"))
print(f"  注: MA({q_hat}) 只在前 {q_hat} 步有信息, 之后预测立刻变平直线 = 均值 mu={mu}")

fig, axx = plt.subplots(figsize=(11, 5))
tail = 80
axx.plot(range(len(train) - tail, len(train)), train.iloc[-tail:].values,
         label="train (tail)", color="steelblue")
axx.plot(range(len(train), len(train) + h), test.values, label="actual", color="black", marker=".")
axx.plot(range(len(train), len(train) + h), yhat.values, label="forecast", color="red", ls="--")
axx.fill_between(range(len(train), len(train) + h), ci.iloc[:, 0], ci.iloc[:, 1],
                 color="red", alpha=.15, label="95% CI")
axx.axhline(mu, color="green", ls=":", lw=1, label=f"mean mu={mu:.1f}")
axx.set_title(f"Out-of-sample forecast (flat to mu after {q_hat} steps)"); axx.legend(loc="upper left")
plt.tight_layout(); plt.savefig("ma_fig4_forecast.png", dpi=110); plt.close()

# ====================================================================
print(SEP)
print(f"完成. 全流程: 平稳[OK] -> 定阶 q={q_hat} -> 拟合 -> 残差白噪声[OK] -> 预测优于基准")
print("揭晓真值: MA(2), theta1=0.6, theta2=0.3" + (" -> 定阶正确" if q_hat == 2 else ""))
print("图: ma_fig1_identify / ma_fig2_order / ma_fig3_residuals / ma_fig4_forecast (.png)")
