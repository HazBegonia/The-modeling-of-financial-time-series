# TGARCH 波动率建模 —— 学习总结

> 配套代码：[tgarch_demo.py](tgarch_demo.py)（一条龙：均值 ARMA + 波动 TGARCH/GJR(1,1,1)-t）
> 数据/产物：`tgarch_returns.csv`、`tgarch_forecast.csv`、`tgarch_fig1~4_*.png`
> 结构对照 [GARCH.md](../GARCH_demo/GARCH.md)、[IGARCH.md](../IGARCH_demo/IGARCH.md)、[EGARCH.md](../EGARCH_demo/EGARCH.md)。

---

## 0. 一句话理解 TGARCH

TGARCH（Threshold GARCH / GJR-GARCH, Glosten-Jagannathan-Runkle 1993）是抓**杠杆效应**的另一条路：在普通 GARCH 上加一个**门限开关**，下跌时给冲击多算一份波动。

$$\sigma_t^2=\omega+\big(\alpha+\gamma\,\mathbb 1[a_{t-1}<0]\big)a_{t-1}^2+\beta\sigma_{t-1}^2$$

门限 $\mathbb 1[a_{t-1}<0]$ 是"上期是否下跌"的开关：

- 上涨（$a\ge0$）：冲击系数 $=\alpha$；
- 下跌（$a<0$）：冲击系数 $=\alpha+\gamma$。

**$\gamma>0$ → 利空多加一份波动 = 杠杆**。持续度 $=\alpha+\beta+\gamma/2$（对称分布下 $E[\mathbb 1]=0.5$），<1 即协方差平稳。

> 一句话对照：EGARCH 用连续的 $\gamma z$ 让曲线"光滑地歪"，TGARCH 用门限 $\mathbb 1[a<0]$ 直接"折一刀"。

---

## 1. 核心概念速查

| 概念 | 含义 |
|---|---|
| 门限 $\mathbb 1[a_{t-1}<0]$ | TGARCH 的灵魂：上期下跌才触发额外波动 |
| 杠杆参数 $\gamma$ | $\gamma>0$ 即杠杆（与 EGARCH 的 $\gamma<0$ 相反！） |
| 上涨/下跌系数 | $\alpha$ vs $\alpha+\gamma$，后者更大 |
| 持续度 | $\alpha+\beta+\gamma/2$ |
| 新闻冲击曲线 NIC | **带折点的抛物线**，0 处有拐角、左半支更陡 |

---

## 2. 数据要求

与 EGARCH 相同：GARCH 那套 + 额外查杠杆。本 demo 真值 $\gamma=0.08>0$。

---

## 3. 六步建模流程（本次真实数字）

### 第0步 准备数据
- 真值：均值 **ARMA(1,1)**（$\phi=0.5,\theta=0.3,\mu=0.05$）+ 波动 **TGARCH(1,1,1)-t**（$\omega=0.05,\alpha=0.04,\beta=0.90,\gamma=0.08,\nu=7$，**持续度 $\alpha+\beta+\gamma/2=0.98$**）。
- 样本 $n=2500$（train=2440, test=60）；ADF p≈6e−16 → 平稳 ✅。

### 第1步 均值方程（ARMA）
BIC 网格定回 **ARMA(1,1)**（$\hat\phi=0.493,\hat\theta=0.350$，贴真值）；得冲击 $a_t$。

### 第2步 检验 ARCH 效应 + 杠杆
- ARCH：Engle LM(10)=454.2，p≈3e−91 ✅。
- 杠杆探针 $\text{corr}(\mathbb 1[a_{t-1}<0],a_t^2)=+0.002$ → **此粗探针偏弱、没报出**；但下面的 o=0/o=1 对比和 $\gamma$ 显著性会抓出来。
> 教训：**模型无关的"符号-平方相关"是个弱探测器，可能漏报杠杆；正式判据应看 o=0 vs o=1 的信息准则与 $\gamma$ 的显著性。**

### 第3步 选型（对称 GARCH o=0 vs 门限 TGARCH o=1）

| 模型 | AIC | BIC |
|---|---|---|
| 对称 GARCH(1,1) | 8629.1 | 8652.3 |
| **门限 TGARCH(1,1,1)** | **8620.2** | **8649.2** ✅ |

**BIC 选 o=1** → 门限项有价值。

### 第4步 估计参数（看 $\gamma$ 的符号与显著性）

| 参数 | 估计 | 真值 |
|---|---|---|
| $\omega$ | 0.0363 | 0.05 |
| $\alpha_1$ | 0.0415 | 0.04 |
| **$\gamma_1$** | **+0.0612** | **+0.08** |
| $\beta_1$ | 0.9144 | 0.90 |
| $\nu$ | 6.87 | 7.0 |

- **杠杆 $\hat\gamma=+0.061$，p=0.0022 → 显著且为正，门限杠杆确认** ✅。
- **上涨时系数 $\alpha=0.042$；下跌时 $\alpha+\gamma=0.103$**——同样大小的冲击，下跌带来约 2.5 倍的方差增量。
- 持续度 $0.986$（<1，平稳）；无条件波动率 1.638；半衰期 50.9 天。

### 第5步 模型检验
① $z$ Ljung-Box p=0.685 ✅　② $z^2$ Ljung-Box p=0.075 ✅　③ 偏度−0.40/峰度9.40（厚尾，t 合理）　④ **杠杆洗净** $\text{corr}=-0.010≈0$ ✅　⑤ 持续度0.986<1 ✅。

### 第6步 使用模型（预测 + 新闻冲击曲线）
- 均值 ARMA 回复（mu(1)=0.370 → mu(60)=0.061）；波动模拟法预测 sigma(1)=1.130 → sigma(60)=1.419，向无条件 1.638 收敛。
- 1日 99% VaR=2.50%；样本外 95% 覆盖率 **93.3%**。
- **新闻冲击曲线（fig4 下图）**：抛物线在 0 处**有折点**、**左半支（负冲击）更陡**——与 EGARCH 的光滑曲线形成鲜明对照。

> TGARCH 含门限，多步波动同样用 `method='simulation'` 模拟法预测。

---

## 4. TGARCH vs EGARCH（两条抓杠杆的路）

| 维度 | TGARCH/GJR | EGARCH |
|---|---|---|
| 建模对象 | $\sigma^2$ | $\ln\sigma^2$ |
| 不对称方式 | 门限 $\mathbb 1[a<0]$（折点） | $\gamma z$（光滑） |
| 杠杆符号 | **$\gamma>0$** | $\gamma<0$ |
| 新闻冲击曲线 | 带折点抛物线 | 光滑不对称 |
| 正定约束 | 需要 | 不需要 |
| 持续度 | $\alpha+\beta+\gamma/2$ | $|\beta|$ |

直觉：EGARCH 把曲线"光滑地歪"，TGARCH 在 0 处"折一刀"。本 demo 两者样本外覆盖率都是 93.3%，实务里按 AIC/BIC 二选一。

---

## 5. 易混点澄清

1. **$\gamma$ 符号：TGARCH 是 $\gamma>0$，EGARCH 是 $\gamma<0$**——同样表"杠杆"，符号相反，最容易记混。
2. **持续度多了 $\gamma/2$**：别只算 $\alpha+\beta$，门限项贡献 $\gamma/2$。
3. **弱探针可能漏报杠杆**：第2步的"符号-平方相关"不显著不代表没杠杆，要靠 o=0/o=1 信息准则和 $\gamma$ 的 t 检验兜底（本 demo 正是如此）。
4. **TGARCH≈GJR**：本 demo 用 `power=2`（对方差建模）即 GJR；`power=1`（对标准差建模）是 Zakoian 原版 TGARCH，思想相同。

---

## 6. 文件清单与运行

```
tgarch_demo.py            # 主程序
requirements.txt          # 依赖
tgarch_returns.csv        # 步骤0收益（t, logret_pct, split）
tgarch_forecast.csv       # 步骤6每日预测（mu/sigma/区间/VaR/actual）
tgarch_fig1_mean.png      # r_t 序列 + ACF/PACF
tgarch_fig2_order.png     # a_t² 的 ACF + 杠杆散点
tgarch_fig3_diagnose.png  # 条件波动 + 标准化残差四联诊断
tgarch_fig4_forecast.png  # 样本外预测 + 新闻冲击曲线（带折点招牌图）
```

运行：`pip install -r requirements.txt && python tgarch_demo.py`

---

## 7. 一句话总结

> **TGARCH/GJR = 在 GARCH 上加门限 $\mathbb 1[a_{t-1}<0]$，下跌时冲击系数从 $\alpha$ 跳到 $\alpha+\gamma$，用 $\gamma>0$ 刻画杠杆。** 流程与 GARCH 一致，关键在第3步用 o=0/o=1 选型、第4步看 $\gamma$ 显著为正。本 demo 从真值 $\gamma=0.08$ 的序列出发，估出 $\hat\gamma=+0.061$（p=0.002）、下跌系数 0.103 vs 上涨 0.042，新闻冲击曲线带折点且左臂更陡——门限杠杆的教科书式画像。
