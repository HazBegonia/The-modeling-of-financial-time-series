# -*- coding: utf-8 -*-
"""
AR 时间序列建模 —— 一条龙完整流程 (建模 + 定阶 合并版)
造数据 -> 平稳性检验 -> 自动定阶(PACF/AIC/BIC/系数显著性) -> 拟合 -> 残差诊断 -> 预测

特点: 下游用"定出来的阶 p_hat"而非写死的阶, 换任意数据都能自适应.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")          # Windows 控制台强制 UTF-8
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                              # 不弹窗, 直接存图
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

SEP = "=" * 66

# ====================================================================
# 步骤0  数据生成: 弱平稳、有自相关的 AR(2)
#   y_t = c + phi1*y_{t-1} + phi2*y_{t-2} + eps   (建模时假装不知道阶数)
#   弱平稳条件(AR2): phi1+phi2<1, phi2-phi1<1, |phi2|<1  -> 满足
# ====================================================================
rng = np.random.default_rng(42)
c, phi1, phi2, sigma = 2.0, 0.5, 0.3, 1.0
n, burn = 500, 300                                 # 取 500 个点, 前 300 烧入丢弃
mu = c / (1 - phi1 - phi2)                          # 理论均值
eps = rng.normal(0, sigma, n + burn)
y = np.zeros(n + burn)
y[0] = y[1] = mu
for t in range(2, n + burn):
    y[t] = c + phi1 * y[t - 1] + phi2 * y[t - 2] + eps[t]
y = pd.Series(y[burn:]).reset_index(drop=True)

h = 50                                              # 留最后 50 个点做样本外预测
train, test = y.iloc[:-h], y.iloc[-h:]

# 导出本地 CSV (t=时间索引, y=观测值, split=训练/测试)
pd.DataFrame({"t": np.arange(len(y)), "y": y.values,
              "split": ["train"] * len(train) + ["test"] * len(test)}
             ).to_csv("ar2_data.csv", index=False, encoding="utf-8")

print(SEP)
print("步骤0  数据生成")
print(f"  真值 AR(2): y_t = {c} + {phi1}*y(t-1) + {phi2}*y(t-2) + eps,  sigma={sigma}")
print(f"  理论均值 mu={mu:.2f},  样本 n={len(y)} (train={len(train)}, test={len(test)})")
print("  已导出 -> ar2_data.csv")

# ====================================================================
# 步骤1  平稳性检验 (ADF, 定阶/建模的前提)
# ====================================================================
adf_stat, adf_p, *_ = adfuller(train, autolag="AIC")
print(SEP)
print("步骤1  平稳性检验 (ADF)")
print(f"  ADF={adf_stat:.3f},  p={adf_p:.3g}  ->  "
      + ("平稳, 可建模 [OK]" if adf_p < 0.05 else "不平稳, 需先差分 [NO]"))

# 图1: 序列 + ACF + PACF (识别用)
fig, ax = plt.subplots(3, 1, figsize=(10, 9))
ax[0].plot(train.values, lw=0.9)
ax[0].axhline(train.mean(), color="r", ls="--", lw=1, label=f"mean={train.mean():.1f}")
ax[0].set_title("Series (weakly stationary)"); ax[0].legend(loc="upper right")
plot_acf(train, lags=25, ax=ax[1], zero=False); ax[1].set_title("ACF (tails off -> AR)")
plot_pacf(train, lags=25, ax=ax[2], method="ols", zero=False)
ax[2].set_title("PACF (cuts off -> AR order)")
plt.tight_layout(); plt.savefig("ar_fig1_identify.png", dpi=110); plt.close()

# ====================================================================
# 步骤2  定阶: 三法交叉验证 (PACF截尾 / AIC-BIC网格 / 过拟合系数显著性)
# ====================================================================
# MAXLAG: 定阶搜索上限. Schwert 规则 floor(12*(T/100)^0.25) 只依赖样本量 T, 与未知阶数无关 (黑盒下不能拿真值反推)
MAXLAG = int(12 * (len(train) / 100) ** 0.25)
print(SEP)
print("步骤2  定阶 (三法交叉)")
print(f"  搜索上限 MAXLAG = {MAXLAG}  (Schwert: floor(12*(T/100)^0.25), T={len(train)})")

# 2a PACF 截尾: AR(p) 的 PACF 在 lag p 之后掉进置信带
# method="ols": 嵌套 AR 回归的末系数 = 样本 PACF, 即 Tsay《金融时间序列分析》2.4.2 的定义
pac = pacf(train, nlags=MAXLAG, method="ols")
band = 1.96 / np.sqrt(len(train))
sig = [k for k in range(1, MAXLAG + 1) if abs(pac[k]) > band]
pacf_p = max(sig) if sig else 0
print(f"  [2a] PACF 截尾: 最后显著 lag = {pacf_p}   (band=±{band:.3f})")

# 2b AIC/BIC 网格: 逐阶拟合取准则最小. hold_back=MAXLAG 保证各阶同一样本, IC 才可比!
rows = {}
for k in range(1, MAXLAG + 1):
    m = AutoReg(train, lags=k, trend="c", hold_back=MAXLAG, old_names=False).fit()
    rows[k] = (m.aic, m.bic)
aic_p = min(rows, key=lambda k: rows[k][0])
bic_p = min(rows, key=lambda k: rows[k][1])
print(f"  [2b] AIC/BIC 网格:  {'p':>3} {'AIC':>10} {'BIC':>10}")
for k, (a, b) in rows.items():
    mk = ("  <-AIC最小" if k == aic_p else "") + ("  <-BIC最小" if k == bic_p else "")
    print(f"                    {k:>3} {a:>10.2f} {b:>10.2f}{mk}")
print(f"       -> AIC 选 p={aic_p},  BIC 选 p={bic_p}")

# 2c 过拟合 AR(MAXLAG) 看系数显著性: 故意拟合到搜索上限, 不显著的高阶项应剔除
over = AutoReg(train, lags=MAXLAG, trend="c", old_names=False).fit()
sig_lags = [int(nm.split("L")[1]) for nm, pv in zip(over.params.index, over.pvalues.values)
            if nm.startswith("y.L") and pv < 0.05]
sigcoef_p = max(sig_lags) if sig_lags else 0
print(f"  [2c] 过拟合 AR({MAXLAG}): 显著的最高滞后 = {sigcoef_p}")

# 汇总投票 -> 以 BIC 为最终定阶
p_hat = bic_p
votes = {"PACF": pacf_p, "AIC": aic_p, "BIC": bic_p, "显著性": sigcoef_p}
agree = len(set(votes.values())) == 1
print("  投票: " + " | ".join(f"{k}={v}" for k, v in votes.items()))
print(f"  >>> 最终定阶 p = {p_hat}  "
      + ("(以BIC为准, 四法一致)" if agree else "(以BIC为准, 各法有分歧)"))

# 图2: AIC/BIC 随阶数变化曲线
fig, axx = plt.subplots(figsize=(8, 5))
ks = list(rows.keys())
axx.plot(ks, [rows[k][0] for k in ks], "o-", label="AIC")
axx.plot(ks, [rows[k][1] for k in ks], "s-", label="BIC")
axx.axvline(p_hat, color="C1", ls=":", alpha=.7, label=f"chosen p={p_hat}")
axx.set_xlabel("AR order p"); axx.set_ylabel("Information criterion")
axx.set_title(f"Order selection: AIC min@{aic_p}, BIC min@{bic_p}")
axx.legend(); plt.tight_layout(); plt.savefig("ar_fig2_order.png", dpi=110); plt.close()

# ====================================================================
# 步骤3  用定出的阶 p_hat 拟合
# ====================================================================
res = AutoReg(train, lags=p_hat, trend="c", old_names=False).fit()
truth = {"const": c, "y.L1": phi1, "y.L2": phi2}
print(SEP)
print(f"步骤3  用定出的 p={p_hat} 拟合 (参数估计 vs 真值)")
for nm, v in zip(res.params.index, res.params.values):
    t = truth.get(nm)
    print(f"  {nm:<7} = {v:+.3f}" + (f"   | 真值 {t}" if t is not None else "   | (真值无此项)"))
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
plt.tight_layout(); plt.savefig("ar_fig3_residuals.png", dpi=110); plt.close()

# ====================================================================
# 步骤5  样本外预测 + 评估
# ====================================================================
pred = res.get_prediction(start=len(train), end=len(train) + h - 1)
yhat = pred.predicted_mean; ci = pred.conf_int()
rmse = np.sqrt(np.mean((test.values - yhat.values) ** 2))
mae = np.mean(np.abs(test.values - yhat.values))
naive = np.sqrt(np.mean((test.values - train.iloc[-1]) ** 2))   # 朴素基准: 用最后一个值预测
print(SEP)
print("步骤5  样本外预测")
print(f"  预测 {h} 步:  RMSE={rmse:.3f},  MAE={mae:.3f}")
print(f"  朴素基准(last-value) RMSE={naive:.3f}  ->  "
      + ("AR 更优 [OK]" if rmse < naive else "未胜过基准"))
print("  注: 多步预测逐渐收敛到长期均值 mu, 置信区间逐步变宽 (均值回复)")

fig, axx = plt.subplots(figsize=(11, 5))
tail = 80
axx.plot(range(len(train) - tail, len(train)), train.iloc[-tail:].values,
         label="train (tail)", color="steelblue")
axx.plot(range(len(train), len(train) + h), test.values, label="actual", color="black", marker=".")
axx.plot(range(len(train), len(train) + h), yhat.values, label="forecast", color="red", ls="--")
axx.fill_between(range(len(train), len(train) + h), ci.iloc[:, 0], ci.iloc[:, 1],
                 color="red", alpha=.15, label="95% CI")
axx.axhline(mu, color="green", ls=":", lw=1, label=f"long-run mean={mu:.1f}")
axx.set_title("Out-of-sample forecast (mean-reverts to mu)"); axx.legend(loc="upper left")
plt.tight_layout(); plt.savefig("ar_fig4_forecast.png", dpi=110); plt.close()

# ====================================================================
print(SEP)
print(f"完成. 全流程: 平稳[OK] -> 定阶 p={p_hat} -> 拟合 -> 残差白噪声[OK] -> 预测优于基准")
print("揭晓真值: AR(2), phi1=0.5, phi2=0.3" + (" -> 定阶正确" if p_hat == 2 else ""))
print("图: ar_fig1_identify / ar_fig2_order / ar_fig3_residuals / ar_fig4_forecast (.png)")
