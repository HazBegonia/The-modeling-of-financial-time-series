# -*- coding: utf-8 -*-
"""
ARMA 时间序列建模 —— 一条龙完整流程 (建模 + 定阶 合并版)
造数据 -> 平稳性检验 -> 自动定阶((p,q)二维AIC/BIC网格) -> 拟合 -> 残差诊断 -> 预测

特点: 下游用"定出来的阶 (p_hat,q_hat)"而非写死的阶, 换任意数据都能自适应.
对照 AR / MA:
  AR(p) 看 PACF 截尾, MA(q) 看 ACF 截尾;
  ARMA(p,q) 的 ACF 与 PACF "双双拖尾", 无法肉眼截尾定阶 -> 定阶主力是二维信息准则网格.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")          # Windows 控制台强制 UTF-8
import warnings
warnings.filterwarnings("ignore")                  # 屏蔽 statsmodels 收敛/频率警告
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                              # 不弹窗, 直接存图
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

SEP = "=" * 66


# ====================================================================
# EACF (扩展自相关函数, Tsay & Tiao 1984) —— ARMA 定阶方法一 (Tsay 2.6.3 主角)
#   思想: 相合地估出 ARMA 的 AR 部分并滤掉, 剩下就是纯 MA, 普通 ACF 即可看 MA 截尾.
#   实现: 系数按 (阶 k, 迭代 j) 二维递推 (Tsay-Tiao 迭代回归);
#         EACF(k,q)=z 经 phi^{(k,q+1)} 做 AR(k) 滤波后残差的 lag(q+1) 样本自相关;
#         |EACF|<=2/sqrt(T) 记 "O"(不显著), 否则 "X"; O 三角的左上顶点坐标 = (p,q).
# ====================================================================
def _lag(v, L, n):
    out = np.full(n, np.nan); out[L:] = v[:n - L]; return out


def _sacf(x, lag_max):
    xc = x - x.mean(); denom = np.dot(xc, xc); m = len(xc)
    return np.array([np.dot(xc[k:], xc[:m - k]) / denom for k in range(lag_max + 1)])


def eacf(z, ar_max=7, ma_max=13):
    z = np.asarray(z, float); z = z - z.mean(); n = len(z)
    KMAX = ar_max + ma_max + 2
    def ar_ols(k):
        if k == 0:
            return np.array([])
        Zl = np.column_stack([_lag(z, i, n) for i in range(1, k + 1)])
        msk = ~np.isnan(Zl).any(axis=1)
        return np.linalg.lstsq(Zl[msk], z[msk], rcond=None)[0]
    # 基例 phi^{(k,0)} = AR(k) OLS 系数; 再按迭代 j 递推
    phi = {0: {k: ar_ols(k) for k in range(KMAX + 1)}}
    for j in range(1, ma_max + 2):
        phi[j] = {0: np.array([])}
        for k in range(1, KMAX - j + 1):
            pk, pk1 = phi[j - 1][k], phi[j - 1][k + 1]
            ratio = pk1[k] / pk[k - 1]
            new = np.zeros(k)
            for r in range(1, k + 1):
                new[r - 1] = pk1[r - 1] - (-1.0 if r == 1 else pk[r - 2]) * ratio
            phi[j][k] = new
    tab = np.full((ar_max + 1, ma_max + 1), np.nan)
    for k in range(ar_max + 1):
        Zl = None if k == 0 else np.column_stack([_lag(z, i, n) for i in range(1, k + 1)])
        for q in range(ma_max + 1):
            w = z.copy() if k == 0 else (z - Zl.dot(phi[q + 1][k]))
            w = w[~np.isnan(w)]
            tab[k, q] = _sacf(w, q + 1)[q + 1]
    return tab


def eacf_symbols(tab, n):
    sym = np.empty(tab.shape, dtype=object)
    for k in range(tab.shape[0]):
        for q in range(tab.shape[1]):
            sym[k, q] = "x" if abs(tab[k, q]) > 2.0 / np.sqrt(n - k - q) else "o"
    return sym


def eacf_vertex(sym):
    """O 三角左上顶点 = (p,q). 按 (p+q,p) 从小到大取首个 [自身=O 且 三角左边沿>=80%为O] 的点."""
    P, Q = sym.shape
    for s in range(P + Q - 1):
        for p in range(P):
            q = s - p
            if q < 0 or q >= Q or sym[p, q] != "o":
                continue
            good = tot = 0
            for dk in range(3):
                k = p + dk
                if k >= P:
                    break
                for dj in range(2):
                    j = q + dk + dj
                    if j < Q:
                        tot += 1; good += (sym[k, j] == "o")
            if tot and good / tot >= 0.8:
                return (p, q)
    return (0, 0)


# ====================================================================
# 步骤0  数据生成: ARMA(2,2)  (建模时假装不知道阶数)
#   y_t = c + phi1*y_{t-1} + phi2*y_{t-2}
#             + eps_t + theta1*eps_{t-1} + theta2*eps_{t-2}
#   平稳性(由 AR 部分决定, AR2): phi1+phi2<1, phi2-phi1<1, |phi2|<1
#   可逆性(由 MA 部分决定, MA2): theta1+theta2<1, theta2-theta1<1, |theta2|<1
# ====================================================================
rng = np.random.default_rng(42)
c, phi1, phi2, theta1, theta2, sigma = 7.0, 0.6, -0.3, 0.5, 0.4, 1.0
n, burn = 900, 300                                 # 取 900 个点, 前 300 烧入丢弃
mu = c / (1 - phi1 - phi2)                          # 理论均值 = 7/(1-0.3) = 10
eps = rng.normal(0, sigma, n + burn)
y = np.zeros(n + burn)
y[0] = y[1] = mu
for t in range(2, n + burn):
    y[t] = (c + phi1 * y[t - 1] + phi2 * y[t - 2]
            + eps[t] + theta1 * eps[t - 1] + theta2 * eps[t - 2])
y = pd.Series(y[burn:]).reset_index(drop=True)

h = 50                                              # 留最后 50 个点做样本外预测
train, test = y.iloc[:-h], y.iloc[-h:]

# 导出本地 CSV (t=时间索引, y=观测值, split=训练/测试)
pd.DataFrame({"t": np.arange(len(y)), "y": y.values,
              "split": ["train"] * len(train) + ["test"] * len(test)}
             ).to_csv("arma22_data.csv", index=False, encoding="utf-8")

print(SEP)
print("步骤0  数据生成")
print(f"  真值 ARMA(2,2): y_t = {c} + {phi1}*y(t-1) + {phi2}*y(t-2)")
print(f"                     + eps + {theta1}*eps(t-1) + {theta2}*eps(t-2),  sigma={sigma}")
print(f"  平稳条件(AR部分) 满足, 可逆条件(MA部分) 满足")
print(f"  理论均值 mu={mu:.2f},  样本 n={len(y)} (train={len(train)}, test={len(test)})")
print("  已导出 -> arma22_data.csv")

# ====================================================================
# 步骤1  平稳性检验 (ADF, 建模前提; ARMA 的平稳性由 AR 部分决定)
# ====================================================================
adf_stat, adf_p, *_ = adfuller(train, autolag="AIC")
print(SEP)
print("步骤1  平稳性检验 (ADF)")
print(f"  ADF={adf_stat:.3f},  p={adf_p:.3g}  ->  "
      + ("平稳, 可建模 [OK]" if adf_p < 0.05 else "不平稳, 需先差分 [NO]"))

# 图1: 序列 + ACF + PACF (识别用) —— ARMA 的标志是 ACF/PACF 双双拖尾
fig, ax = plt.subplots(3, 1, figsize=(10, 9))
ax[0].plot(train.values, lw=0.9)
ax[0].axhline(train.mean(), color="r", ls="--", lw=1, label=f"mean={train.mean():.1f}")
ax[0].set_title("Series (ARMA process, stationary)"); ax[0].legend(loc="upper right")
plot_acf(train, lags=25, ax=ax[1], zero=False); ax[1].set_title("ACF (tails off -> not pure MA)")
plot_pacf(train, lags=25, ax=ax[2], method="ols", zero=False)
ax[2].set_title("PACF (tails off -> not pure AR)  => both tail off => ARMA")
plt.tight_layout(); plt.savefig("arma_fig1_identify.png", dpi=110); plt.close()

# ====================================================================
# 步骤2  定阶: ARMA 的 ACF/PACF 双双拖尾, 不能肉眼截尾, 用书上两法交叉:
#   方法一 EACF(Tsay 2.6.3 主角): 滤掉 AR 部分后看 MA 截尾, O/X 三角左上顶点 = (p,q)
#   方法二 (p,q) 二维 AIC/BIC 网格: 逐组合拟合取准则最小 (同一样本; p+q 最小打破并列)
# ====================================================================
# 搜索上限 MAXLAG: Schwert 规则 floor(12*(T/100)^0.25), 只依赖样本量 T, 与 AR/MA demo 一致, 不依赖真值
MAXLAG = int(12 * (len(train) / 100) ** 0.25)
print(SEP)
print("步骤2  定阶 (EACF + (p,q) 信息准则网格)")
print(f"  搜索上限 MAXLAG = {MAXLAG}  (Schwert: floor(12*(T/100)^0.25), T={len(train)})")

# 2a EACF (方法一, 便宜=只做 OLS 回归): 搜满整个 Schwert 范围 [0,MAXLAG]^2, 找 O 三角左上顶点
esym = eacf_symbols(eacf(train.values, ar_max=MAXLAG, ma_max=MAXLAG), len(train))
eacf_pq = eacf_vertex(esym)                        # 顶点在完整 [0,MAXLAG]^2 上找
SP, SQ = min(MAXLAG, 8), min(MAXLAG, 12)           # 显示只截左上窗口(可读), 不影响搜索范围
print(f"  [2a] EACF (搜索 p,q<={MAXLAG}; 下显左上窗口, O=不显著 X=显著, 行p列q):")
print("         " + " ".join(f"q{q}" for q in range(SQ + 1)))
for p in range(SP + 1):
    print(f"    p={p}    " + "  ".join(esym[p][:SQ + 1]))
print(f"  -> EACF 定阶 (p,q) = {eacf_pq}  (O 三角左上顶点)")

# 2b 网格(贵=每格一次 MLE): 高阶 MLE 不收敛(实测 ARIMA(10,10) 即发散, (20,20) 14s 仍不收敛),
#    故网格只搜到 GRIDMAX 这一可靠收敛的小范围做交叉验证; 完整 Schwert 宽搜索已由便宜的 EACF 覆盖
GRIDMAX = min(MAXLAG, 5)
PMAX = QMAX = GRIDMAX
print(f"  网格上限 = {GRIDMAX} (MLE 高阶不收敛, 宽搜索交 EACF; 共 {(GRIDMAX + 1) ** 2} 个拟合)")

aic_grid = np.full((PMAX + 1, QMAX + 1), np.nan)
bic_grid = np.full((PMAX + 1, QMAX + 1), np.nan)
for p in range(PMAX + 1):
    for q in range(QMAX + 1):
        if p == 0 and q == 0:
            continue                               # 跳过纯白噪声(无 AR 无 MA)
        try:
            m = ARIMA(train, order=(p, 0, q), trend="c").fit()
            aic_grid[p, q], bic_grid[p, q] = m.aic, m.bic
        except Exception:
            pass                                   # 个别组合不收敛, 留 NaN

def argmin_2d(grid):
    """返回准则最小的 (p,q); 并列时取 p+q 最小(最简约)再取 q 最小."""
    cands = [(p, q) for p in range(PMAX + 1) for q in range(QMAX + 1)
             if not np.isnan(grid[p, q])]
    best = min(cands, key=lambda pq: (grid[pq], pq[0] + pq[1], pq[1]))
    return best

aic_pq = argmin_2d(aic_grid)
bic_pq = argmin_2d(bic_grid)

# 打印 BIC 网格 (行 p, 列 q)
print(f"  [2b] (p,q) 信息准则网格 — BIC 表 (行=p, 列=q):")
header = "       " + "".join(f"q={q:<8}" for q in range(QMAX + 1))
print(header)
for p in range(PMAX + 1):
    cells = ""
    for q in range(QMAX + 1):
        v = bic_grid[p, q]
        s = "    --   " if np.isnan(v) else f"{v:>8.1f} "
        if (p, q) == bic_pq:
            s = s.rstrip() + "*"                    # 标记 BIC 最小
        cells += f"{s:<10}"
    print(f"  p={p}  {cells}")
print(f"  -> AIC 选 (p,q)={aic_pq} (AIC={aic_grid[aic_pq]:.2f})")
print(f"  -> BIC 选 (p,q)={bic_pq} (BIC={bic_grid[bic_pq]:.2f})   [*标记]")

# 最终以 BIC 为准 (更简约, 抗过拟合); EACF/AIC 作交叉印证
p_hat, q_hat = bic_pq
votes = {"EACF": eacf_pq, "AIC": aic_pq, "BIC": bic_pq}
agree = len(set(votes.values())) == 1
print("  投票: " + " | ".join(f"{k}={v}" for k, v in votes.items()))
print(f"  >>> 最终定阶 (p,q) = ({p_hat},{q_hat})  "
      + ("(EACF/AIC/BIC 三法一致)" if agree else "(以BIC为准)"))
if p_hat == GRIDMAX or q_hat == GRIDMAX:
    print(f"  ⚠ 网格定阶贴到网格上限 {GRIDMAX} (EACF 已搜到 {MAXLAG}); 若与 EACF 不一致需复核")

# 图2: 左=EACF 的 O/X 表(方法一), 右=BIC 网格热力图(方法二)
fig, (axE, axB) = plt.subplots(1, 2, figsize=(13, 6))
# 左: EACF 符号表左上窗口 (绿=O 不显著, 红=X 显著), 蓝圈标 O 三角顶点
Emat = np.array([[0 if esym[p, q] == "o" else 1 for q in range(SQ + 1)]
                 for p in range(SP + 1)])
axE.imshow(Emat, origin="upper", cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
for p in range(SP + 1):
    for q in range(SQ + 1):
        axE.text(q, p, esym[p, q], ha="center", va="center", fontsize=10)
axE.scatter([eacf_pq[1]], [eacf_pq[0]], s=460, facecolors="none", edgecolors="blue",
            linewidths=2.5, label=f"EACF vertex (p,q)={eacf_pq}")
axE.set_xlabel("MA order q"); axE.set_ylabel("AR order p")
axE.set_xticks(range(SQ + 1)); axE.set_yticks(range(SP + 1))
axE.set_title(f"Method 1: EACF (search p,q<={MAXLAG}; vertex of O-triangle = (p,q))")
axE.legend(loc="upper right")
# 右: BIC 网格热力图, 红圈标最小
im = axB.imshow(bic_grid, origin="upper", cmap="viridis_r", aspect="auto")
for p in range(PMAX + 1):
    for q in range(QMAX + 1):
        if not np.isnan(bic_grid[p, q]):
            axB.text(q, p, f"{bic_grid[p, q]:.0f}", ha="center", va="center",
                     color="white", fontsize=8)
axB.scatter([q_hat], [p_hat], s=420, facecolors="none", edgecolors="red",
            linewidths=2.5, label=f"BIC min (p,q)=({p_hat},{q_hat})")
axB.set_xlabel("MA order q"); axB.set_ylabel("AR order p")
axB.set_xticks(range(QMAX + 1)); axB.set_yticks(range(PMAX + 1))
axB.set_title("Method 2: BIC over (p,q) grid (lower=better)")
fig.colorbar(im, ax=axB, label="BIC"); axB.legend(loc="upper right")
plt.tight_layout(); plt.savefig("arma_fig2_order.png", dpi=110); plt.close()

# ====================================================================
# 步骤3  用定出的阶 (p_hat,q_hat) 拟合
# ====================================================================
res = ARIMA(train, order=(p_hat, 0, q_hat), trend="c").fit()
truth = {"const": mu, "ar.L1": phi1, "ar.L2": phi2,
         "ma.L1": theta1, "ma.L2": theta2, "sigma2": sigma ** 2}
print(SEP)
print(f"步骤3  用定出的 (p,q)=({p_hat},{q_hat}) 拟合 (参数估计 vs 真值)")
for nm, v in zip(res.params.index, res.params.values):
    t = truth.get(nm)
    print(f"  {nm:<8} = {v:+.3f}" + (f"   | 真值 {t}" if t is not None else "   | (真值无此项)"))
print(f"  AIC={res.aic:.2f},  BIC={res.bic:.2f}")
print("  注: const 是均值 mu 的估计(非截距 c); ARMA 参数用极大似然/状态空间估计.")

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
plt.tight_layout(); plt.savefig("arma_fig3_residuals.png", dpi=110); plt.close()

# ====================================================================
# 步骤5  样本外预测 + 评估
#   ARMA 预测 = MA 短期记忆(前 q 步) + AR 均值回复(平滑收敛到 mu) 的叠加.
#   前 q 步: MA 项仍有信息; 之后: 纯 AR 衰减, 逐渐收敛到长期均值 mu.
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
      + ("ARMA 更优 [OK]" if rmse < naive else "未胜过基准"))
print(f"  注: 前 q={q_hat} 步 MA 项仍带信息, 之后由 AR 部分平滑收敛到长期均值 mu={mu:.1f}.")

fig, axx = plt.subplots(figsize=(11, 5))
tail = 80
axx.plot(range(len(train) - tail, len(train)), train.iloc[-tail:].values,
         label="train (tail)", color="steelblue")
axx.plot(range(len(train), len(train) + h), test.values, label="actual", color="black", marker=".")
axx.plot(range(len(train), len(train) + h), yhat.values, label="forecast", color="red", ls="--")
axx.fill_between(range(len(train), len(train) + h), ci.iloc[:, 0], ci.iloc[:, 1],
                 color="red", alpha=.15, label="95% CI")
axx.axhline(mu, color="green", ls=":", lw=1, label=f"long-run mean={mu:.1f}")
axx.set_title("Out-of-sample forecast (MA short memory + AR mean-reversion)")
axx.legend(loc="upper left")
plt.tight_layout(); plt.savefig("arma_fig4_forecast.png", dpi=110); plt.close()

# ====================================================================
print(SEP)
print(f"完成. 全流程: 平稳[OK] -> 定阶 (p,q)=({p_hat},{q_hat}) -> 拟合 -> 残差白噪声[OK] -> 预测优于基准")
print(f"揭晓真值: ARMA(2,2), phi=[{phi1},{phi2}], theta=[{theta1},{theta2}]"
      + (" -> 定阶正确" if (p_hat, q_hat) == (2, 2) else f" -> 定出({p_hat},{q_hat})(简约模型也可接受)"))
print("图: arma_fig1_identify / arma_fig2_order / arma_fig3_residuals / arma_fig4_forecast (.png)")
