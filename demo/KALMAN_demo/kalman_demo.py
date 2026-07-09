# -*- coding: utf-8 -*-
"""
卡尔曼滤波 / 状态空间模型 —— 一条龙完整流程 (局部水平模型 local level model)
造数据 -> 写成状态空间 -> 扩散初始化 -> 前向滤波(预测-更新) -> 预测误差分解算似然
     -> MLE 估参 -> 定参重跑+后向平滑 -> 样本外预测 -> 标准化残差诊断

模型 (Tsay《金融时间序列分析》式 11.1-11.2 局部水平模型):
    观测方程:  y_t     = mu_t + e_t,      e_t  ~ N(0, sig2_e)   <- 含噪观测
    状态方程:  mu_{t+1}= mu_t + eta_t,    eta_t~ N(0, sig2_eta) <- 状态随机游动(不可观测)

要从含噪 y 里递归还原不可观测的状态 mu; 未知参数 theta = (sig2_e, sig2_eta) 由 MLE 估计.
特点: 下游用"估出来的参数"而非写死真值; 滤波/平滑/预测/似然/诊断全是同一套递推的排列组合.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")          # Windows 控制台强制 UTF-8
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                              # 不弹窗, 直接存图
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

SEP = "=" * 66

# ====================================================================
# 步骤0  数据生成: 局部水平模型 (状态=随机游动, 观测=状态+测量噪声)
#   建模时假装只看得到 y, 看不到真实状态 mu
# ====================================================================
rng = np.random.default_rng(42)
sig_e, sig_eta = 0.5, 0.15                          # 真值: 测量噪声 / 状态新息 标准差
n, mu0 = 400, 5.0                                   # 400 个点, 初始状态锚定在 5.0
eta = rng.normal(0, sig_eta, n)
e   = rng.normal(0, sig_e,   n)
mu  = np.zeros(n)                                   # 不可观测的真实状态
mu[0] = mu0
for t in range(1, n):
    mu[t] = mu[t - 1] + eta[t]                      # 状态随机游动
y = mu + e                                          # 含噪观测
y = pd.Series(y)

h = 40                                              # 留最后 40 个点做样本外预测
train, test = y.iloc[:-h], y.iloc[-h:]
mu_train, mu_test = mu[:-h], mu[-h:]

# 信噪比 q = sig2_eta / sig2_e: 状态动得快 vs 测量有多脏
q_true = sig_eta ** 2 / sig_e ** 2

# 导出本地 CSV (t=时间, y=含噪观测, mu_true=真实状态, split=训练/测试)
pd.DataFrame({"t": np.arange(n), "y": y.values, "mu_true": mu,
              "split": ["train"] * len(train) + ["test"] * len(test)}
             ).to_csv("kalman_data.csv", index=False, encoding="utf-8")

print(SEP)
print("步骤0  数据生成 (局部水平模型)")
print(f"  真值: sig_e={sig_e} (测量噪声), sig_eta={sig_eta} (状态新息)")
print(f"  信噪比 q = sig2_eta/sig2_e = {q_true:.4f}  (q 小=状态平滑/观测脏)")
print(f"  样本 n={n} (train={len(train)}, test={len(test)}),  真实状态 mu 已藏起")
print("  已导出 -> kalman_data.csv")

# ====================================================================
# 步骤1  建模: 写成状态空间形式
#   observed='local level' 即 y_t=mu_t+e_t, mu_{t+1}=mu_t+eta_t
#   系统矩阵 T=1, Z=1, R=1, H=sig2_e, Q=sig2_eta; 未知参数 theta=(sig2_e, sig2_eta)
# ====================================================================
print(SEP)
print("步骤1  写成状态空间 (系统矩阵 T=1,Z=1,R=1; H=sig2_e, Q=sig2_eta)")
model = UnobservedComponents(train, level="local level")
print("  状态向量 s_t = [mu_t] (1 维), 观测 y_t (1 维)")
print("  未知参数 theta = (sig2_e, sig2_eta), 待 MLE 估计")

# ====================================================================
# 步骤2  初始化: 状态是随机游动(非平稳) -> 扩散初始化 (Sigma_1|0 -> inf)
#   statsmodels 对含单位根的状态默认自动采用 diffuse 初始化, 等价于用首个观测定锚
# ====================================================================
print(SEP)
print("步骤2  初始化 (扩散 diffuse: 对初始状态一无所知, Sigma_1|0 -> inf)")
print(f"  初始化方式: {model.ssm.initialization.initialization_type}")

# ====================================================================
# 步骤3+4+5  MLE: 外层对 theta 优化, 每次 propose 就跑一遍前向卡尔曼滤波,
#            由预测误差分解 lnL = -0.5*sum[ln V_t + v_t^2/V_t] 算似然, 反馈优化器
# ====================================================================
res = model.fit(disp=False)                         # 内部: 滤波(步骤3)+似然(步骤4)包进优化(步骤5)
est_e   = np.sqrt(res.params["sigma2.irregular"])   # 估出的 sig_e
est_eta = np.sqrt(res.params["sigma2.level"])       # 估出的 sig_eta
q_hat   = res.params["sigma2.level"] / res.params["sigma2.irregular"]
print(SEP)
print("步骤3-5  前向滤波(预测-更新) + 预测误差分解算似然 + MLE 外层优化")
print(f"  收敛后 lnL = {res.llf:.2f},  迭代调用滤波 {res.mle_retvals.get('iterations', 'n/a')} 轮")
print(f"  参数估计 vs 真值:")
print(f"    sig_e   = {est_e:.3f}   | 真值 {sig_e}")
print(f"    sig_eta = {est_eta:.3f}   | 真值 {sig_eta}")
print(f"    信噪比 q = {q_hat:.4f}   | 真值 {q_true:.4f}")

# ----- 滤波产物: 卡尔曼增益 K_t 与一步预测方差 Sigma_{t|t-1} 的收敛(稳态) -----
# 系统矩阵不随时间变 -> Sigma_{t|t-1} 收敛到常数(解 Riccati 方程) -> K_t, V_t 也变常量
pred_state_cov = res.filter_results.predicted_state_cov[0, 0, :]   # Sigma_{t|t-1}
K_gain = pred_state_cov / (pred_state_cov + res.params["sigma2.irregular"])  # K_t=Sig/(Sig+H)
print(f"  稳态: Sigma_(t|t-1) 收敛到 {pred_state_cov[-1]:.4f}, 卡尔曼增益 K 收敛到 {K_gain[-1]:.4f}")

# ====================================================================
# 步骤6  定参重跑: 前向滤波估计 vs 后向平滑估计 (平滑用全样本, 一定更确定)
# ====================================================================
filt = res.filtered_state[0]                        # mu_{t|t}   (只用当前及以前)
smth = res.smoothed_state[0]                         # mu_{t|T}   (用全部样本回看)
filt_rmse = np.sqrt(np.mean((filt - mu_train) ** 2))
smth_rmse = np.sqrt(np.mean((smth - mu_train) ** 2))
print(SEP)
print("步骤6  后向平滑 (还原不可观测状态: 滤波 vs 平滑, 对比真实 mu)")
print(f"  滤波 mu_(t|t) 还原真实状态 RMSE = {filt_rmse:.3f}")
print(f"  平滑 mu_(t|T) 还原真实状态 RMSE = {smth_rmse:.3f}  "
      + ("(平滑更准, 用了未来信息 [OK])" if smth_rmse < filt_rmse else ""))

# ====================================================================
# 步骤7  预测: 把未来观测当"缺失值", 只做预测步/跳过更新步, 向前递推 h 步
# ====================================================================
fc = res.get_forecast(steps=h)
yhat = fc.predicted_mean
ci = fc.conf_int(alpha=0.05)
rmse = np.sqrt(np.mean((test.values - yhat.values) ** 2))
mae = np.mean(np.abs(test.values - yhat.values))
naive = np.sqrt(np.mean((test.values - train.iloc[-1]) ** 2))   # 朴素基准: 最后一个观测
print(SEP)
print("步骤7  样本外预测 (未来=缺失值, 只走预测步)")
print(f"  预测 {h} 步:  RMSE={rmse:.3f},  MAE={mae:.3f}")
print(f"  朴素基准(last-value) RMSE={naive:.3f}  ->  "
      + ("Kalman 更优 [OK]" if rmse < naive else "未胜过基准 (随机游动预测本就接近 last-value)"))
print("  注: 局部水平模型的最优多步预测是一条水平线(=最后的滤波状态), CI 随步数变宽")

# ====================================================================
# 步骤8  诊断: 标准化单步预测残差 v_t/sqrt(V_t) 应为 iid N(0,1)
# ====================================================================
std_resid = res.standardized_forecasts_error[0]     # v_t / sqrt(V_t)
std_resid = std_resid[~np.isnan(std_resid)]
lb_p = acorr_ljungbox(std_resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
jb_stat, jb_p = stats.jarque_bera(std_resid)
print(SEP)
print("步骤8  诊断 (标准化残差 v_t/sqrt(V_t) 应为 iid N(0,1))")
print(f"  Ljung-Box(10) p={lb_p:.3g}  ->  "
      + ("无残余自相关, 模型充分 [OK]" if lb_p > 0.05 else "仍有自相关 [NO]"))
print(f"  Jarque-Bera   p={jb_p:.3g}  ->  "
      + ("近正态 [OK]" if jb_p > 0.05 else "偏离正态 [NO]"))
print(f"  标准化残差 均值={std_resid.mean():.3f}, 标准差={std_resid.std():.3f} (应≈0,1)")

# ====================================================================
# 图1: 含噪观测 vs 不可观测的真实状态
# ====================================================================
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(train.values, lw=.7, color="silver", label="y (noisy observation)")
ax.plot(mu_train, lw=1.6, color="black", label="true state mu (hidden)")
ax.set_title("Local level model: noisy y vs hidden state mu")
ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig("kalman_fig1_series.png", dpi=110); plt.close()

# ====================================================================
# 图2: 滤波估计 vs 平滑估计 (都对比真实状态)
# ====================================================================
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(mu_train, lw=1.6, color="black", label="true state mu")
ax.plot(filt, lw=1.0, color="tab:red", ls="--", label=f"filtered mu_t|t (RMSE={filt_rmse:.3f})")
ax.plot(smth, lw=1.2, color="tab:blue", label=f"smoothed mu_t|T (RMSE={smth_rmse:.3f})")
ax.set_title("Filtering (uses past only) vs Smoothing (uses all data)")
ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig("kalman_fig2_filter_smooth.png", dpi=110); plt.close()

# ====================================================================
# 图3: 卡尔曼增益 / 一步预测方差 收敛到稳态 (Riccati)
# ====================================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].plot(pred_state_cov, color="tab:purple")
ax[0].set_title("Predicted state var Sigma_t|t-1 -> steady state")
ax[0].set_xlabel("t"); ax[0].axhline(pred_state_cov[-1], color="gray", ls=":", lw=1)
ax[1].plot(K_gain, color="tab:green")
ax[1].set_title("Kalman gain K_t -> constant (steady state)")
ax[1].set_xlabel("t"); ax[1].axhline(K_gain[-1], color="gray", ls=":", lw=1)
plt.tight_layout(); plt.savefig("kalman_fig3_steady.png", dpi=110); plt.close()

# ====================================================================
# 图4: 标准化残差四联诊断
# ====================================================================
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
ax[0, 0].plot(std_resid, lw=.8); ax[0, 0].axhline(0, color="r", ls="--")
ax[0, 0].set_title("Standardized residuals v_t/sqrt(V_t)")
plot_acf(std_resid, lags=25, ax=ax[0, 1], zero=False); ax[0, 1].set_title("Residual ACF (white noise?)")
ax[1, 0].hist(std_resid, bins=30, edgecolor="k", alpha=.7); ax[1, 0].set_title("Residual histogram")
stats.probplot(std_resid, dist="norm", plot=ax[1, 1]); ax[1, 1].set_title("Q-Q plot")
plt.tight_layout(); plt.savefig("kalman_fig4_diagnose.png", dpi=110); plt.close()

# ====================================================================
# 图5: 样本外预测 + 95% CI (未来当缺失值滑行, 水平预测)
# ====================================================================
fig, ax = plt.subplots(figsize=(11, 5))
tail = 80
ax.plot(range(len(train) - tail, len(train)), train.iloc[-tail:].values,
        label="train (tail)", color="steelblue", lw=.8)
ax.plot(range(len(train), len(train) + h), test.values, label="actual", color="black", marker=".")
ax.plot(range(len(train), len(train) + h), yhat.values, label="forecast", color="red", ls="--")
ax.fill_between(range(len(train), len(train) + h), ci.iloc[:, 0], ci.iloc[:, 1],
                color="red", alpha=.15, label="95% CI")
ax.set_title("Out-of-sample forecast (future = missing values -> flat prediction)")
ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig("kalman_fig5_forecast.png", dpi=110); plt.close()

# 导出预测结果
pd.DataFrame({"t": np.arange(len(train), len(train) + h),
              "actual": test.values, "forecast": yhat.values,
              "lo95": ci.iloc[:, 0].values, "hi95": ci.iloc[:, 1].values}
             ).to_csv("kalman_forecast.csv", index=False, encoding="utf-8")

# ====================================================================
print(SEP)
print(f"完成. 全流程: 状态空间 -> 扩散初始化 -> 滤波+似然 -> MLE -> 平滑 -> 预测 -> 残差诊断")
print(f"揭晓真值: sig_e={sig_e}, sig_eta={sig_eta}  (估: {est_e:.3f}, {est_eta:.3f})")
print("图: kalman_fig1_series / fig2_filter_smooth / fig3_steady / fig4_diagnose / fig5_forecast (.png)")
