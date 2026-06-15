# IGARCH 波动率建模 —— 学习总结

> 配套代码：[igarch_demo.py](igarch_demo.py)（一条龙：均值 ARMA + 波动 IGARCH(1,1)-t）
> 数据/产物：`igarch_returns.csv`（输入收益）、`igarch_forecast.csv`（预测输出）、`igarch_fig1~4_*.png`
> 结构对照 [GARCH.md](../GARCH_demo/GARCH.md)、[EGARCH.md](../EGARCH_demo/EGARCH.md)、[TGARCH.md](../TGARCH_demo/TGARCH.md)。

---

## 0. 一句话理解 IGARCH

IGARCH（Integrated GARCH）是 GARCH 的**边界特例**：把持续度 $\alpha+\beta$ 顶到**恰好等于 1**。

$$\text{GARCH}(1,1):\ \sigma_t^2=\omega+\alpha a_{t-1}^2+\beta\sigma_{t-1}^2,\quad \alpha+\beta<1$$
$$\text{IGARCH}(1,1):\ \alpha+\beta=1\ \Rightarrow\ \text{方差有"单位根"}$$

一旦 $\alpha+\beta=1$，三件事同时发生：

1. **冲击永不消散**：半衰期 $\ln0.5/\ln(\alpha+\beta)=\ln0.5/\ln1=\infty$。
2. **无有限无条件方差**：$\dfrac{\omega}{1-\alpha-\beta}=\dfrac{\omega}{0}=\infty$。
3. **多步波动不回复**：$\sigma_t^2(\ell)=\sigma_t^2(1)+(\ell-1)\omega$，线性外推、永不收敛。

> 一句话对照：**GARCH 的波动"有记性但会忘"，IGARCH 的波动"过目不忘"。** 著名特例：$\omega=0$ 时即 RiskMetrics/EWMA。

---

## 1. 核心概念速查

| 概念 | 含义 |
|---|---|
| 持续度 $\alpha+\beta$ | IGARCH 的灵魂：**恰为 1**（GARCH 是 <1） |
| 单位根 / 积分型 | 方差过程像"带漂移的随机游走"，不回复 |
| 半衰期 | $\to\infty$（冲击永久） |
| 无条件方差 | $\to\infty$（不存在有限的长期水平） |
| 严格平稳 vs 协方差平稳 | IGARCH **严格平稳但非协方差平稳**（方差矩不存在） |

**关键认知**：实务里只要 $\hat\alpha+\hat\beta\ge0.99$，就该警惕已落入 IGARCH 领域——**别再套"均值回复"的波动预测**。

---

## 2. 数据要求

与 GARCH 相同（平稳、已去 ARMA 均值、有 ARCH 效应、足够长、厚尾用 t 兜），唯一区别在**结论侧**：持续度估出来 ≈1。本 demo 真值持续度**恰为 1.00**（$\alpha=0.06,\beta=0.94$）。

---

## 3. 六步建模流程（本次真实数字）

### 第0步 准备数据
- 真值：均值 **ARMA(1,1)**（$\phi=0.5,\theta=0.3,\mu=0.05$）+ 波动 **IGARCH(1,1)-t**（$\omega=0.02,\alpha=0.06,\beta=0.94,\nu=7$，**持续度 = 1.00**）。
- 样本 $n=2500$（train=2440, test=60）；ADF p≈2e−21 → 平稳 ✅。

### 第1步 均值方程（ARMA）
$r_t$ Ljung-Box(10) p≈1e−240 → 有自相关。BIC 网格定回 **ARMA(1,1)**（$\hat\phi=0.408,\hat\theta=0.387$）；得冲击 $a_t$。

### 第2步 检验 ARCH 效应
Engle LM(10)=446.3，p≈1e−89 → **有 ARCH 效应，需建波动率模型** ✅。

### 第3步 定阶 (p,q)
$a_t^2$ 的 ACF **极缓慢衰减**（持续度≈1 的指纹，见 fig2）；AIC、BIC 一致选 **(1,1)** 形。

### 第4步 估计参数（看持续度是否顶到 1）

| 参数 | 估计 | 真值 |
|---|---|---|
| $\omega$ | 0.0235 | 0.02 |
| $\alpha_1$ | 0.0569 | 0.06 |
| $\beta_1$ | 0.9388 | 0.94 |
| $\nu$ | 8.35 | 7.0 |

- **持续度 $\hat\alpha+\hat\beta=0.9957$**（真值 1.00）→ **≥0.99，落入 IGARCH 领域** 🔴。
- 半衰期 **162 天**（对比普通 GARCH 约 60 天）——波动记忆极久。

> ⚠️ 工程坑：`arch` 包对 GARCH 强制 $\alpha+\beta<1$ 的平稳性约束，所以即便真值=1，估计也只会顶到 0.996 这种"贴着 1 的边界值"，无法精确取到 1。**判据看"是否≥0.99"，而非"是否=1"。**

### 第5步 模型检验
① $z$ Ljung-Box p=0.358 ✅　② $z^2$ Ljung-Box p=0.938 ✅　③ 偏度−0.07/峰度4.35（厚尾，t 合理）　④ 持续度 0.9957≈1 → **确认 IGARCH**。

### 第6步 使用模型（预测）
- 均值：ARMA 逐步回复（mu(1)=−1.561 → mu(60)=0.064）。
- **波动：sigma(1)=2.022 → sigma(60)=2.099，几乎不动**——这是 IGARCH 的招牌（见 fig4 下图：波动预测线近乎平直，不向任何固定水平收敛）。
- 1日 99% VaR=6.62%；样本外 95% 区间覆盖率 78.3%（60 点小样本）。

---

## 4. 与 GARCH / EGARCH / TGARCH 的对照

| 维度 | GARCH(1,1) | **IGARCH(1,1)** | EGARCH | TGARCH |
|---|---|---|---|---|
| 持续度 $\alpha+\beta$ | <1 | **=1** | $|\beta|$<1 | $\alpha+\beta+\gamma/2$<1 |
| 半衰期 | 有限 | **∞** | 有限 | 有限 |
| 无条件方差 | 有限 | **∞（不存在）** | 有限 | 有限 |
| 多步波动预测 | 几何收敛 | **线性外推、不回复** | 收敛 | 收敛 |
| 抓什么 | 一般聚集 | **极高持续/永久冲击** | 杠杆 | 杠杆 |

---

## 5. 易混点澄清

1. **IGARCH ≠ 方差爆炸**：$\alpha+\beta=1$ 时方差过程是"随机游走+漂移"，严格平稳，只是没有有限的二阶矩。
2. **持续度=1 别再做均值回复预测**：套用 GARCH 的"向无条件水平收敛"会严重低估远期波动。
3. **arch 包估不到精确的 1**：约束所致，看 ≥0.99 即判 IGARCH。
4. **omega=0 的 IGARCH 就是 EWMA/RiskMetrics**：业界最常用的"指数加权移动方差"其实是 IGARCH 的特例。

---

## 6. 文件清单与运行

```
igarch_demo.py            # 主程序（全流程一个文件）
requirements.txt          # 依赖：numpy/scipy/pandas/matplotlib/statsmodels/arch
igarch_returns.csv        # 步骤0导出的收益（t, logret_pct, split）
igarch_forecast.csv       # 步骤6导出的每日预测（mu/sigma/区间/VaR/actual）
igarch_fig1_mean.png      # r_t 序列 + ACF/PACF
igarch_fig2_order.png     # a_t² 的 ACF/PACF（极缓慢衰减 = 持续度≈1）
igarch_fig3_diagnose.png  # 条件波动 + 标准化残差四联诊断
igarch_fig4_forecast.png  # 样本外：ARMA 中心 + 波动"不回复"曲线/VaR
```

运行：`pip install -r requirements.txt && python igarch_demo.py`

---

## 7. 一句话总结

> **IGARCH = GARCH 把持续度 $\alpha+\beta$ 顶到 1 的边界特例 = 波动冲击永久、无有限无条件方差、多步预测线性外推不回复。** 流程与 GARCH 完全一致，差别只在第4步读出"持续度≈1"并据此改用"不回复"的波动外推。本 demo 从一个真值持续度=1 的序列出发，估出 $\hat\alpha+\hat\beta=0.996$、半衰期 162 天，波动预测近乎平直——IGARCH 的教科书式画像。
