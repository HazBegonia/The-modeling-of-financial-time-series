# EGARCH 波动率建模 —— 学习总结

> 配套代码：[egarch_demo.py](egarch_demo.py)（一条龙：均值 ARMA + 波动 EGARCH(1,1,1)-t）
> 数据/产物：`egarch_returns.csv`、`egarch_forecast.csv`、`egarch_fig1~4_*.png`
> 结构对照 [GARCH.md](../GARCH_demo/GARCH.md)、[IGARCH.md](../IGARCH_demo/IGARCH.md)、[TGARCH.md](../TGARCH_demo/TGARCH.md)。

---

## 0. 一句话理解 EGARCH

普通 GARCH 用 $a_{t-1}^2$，**正负冲击一视同仁**，抓不到股市的**杠杆效应**——利空（下跌）往往比同等利好（上涨）更抬高未来波动。EGARCH（Exponential GARCH, Nelson 1991）对 **log 方差**建模，并把冲击拆成"大小"和"方向"两块：

$$\ln\sigma_t^2=\omega+\beta\ln\sigma_{t-1}^2+\underbrace{\alpha(|z_{t-1}|-E|z|)}_{\text{大小项: 聚集}}+\underbrace{\gamma z_{t-1}}_{\text{方向项: 杠杆}}$$

其中 $z_t=a_t/\sigma_t$ 是标准化残差。**$\gamma<0$ → 负冲击（$z<0$）额外抬高 $\ln\sigma^2$ = 杠杆**；$\gamma=0$ 退回对称。

EGARCH 的两大好处：
1. **天然抓不对称**（方向项 $\gamma z$）；
2. **无需正定约束**——建模 $\ln\sigma^2$，取指数后必正，参数可自由取值（GARCH 必须 $\omega,\alpha,\beta\ge0$）。

> 一句话对照：**GARCH 只看冲击"多大"，EGARCH 还看冲击"是涨是跌"。**

---

## 1. 核心概念速查

| 概念 | 含义 |
|---|---|
| 杠杆效应 leverage | 负冲击比正冲击更抬高未来波动（股市常见） |
| 方向项 $\gamma z_{t-1}$ | EGARCH 的灵魂；$\gamma<0$ 即杠杆 |
| 大小项 $\alpha(|z|-E|z|)$ | 管波动聚集（冲击越大波动越高） |
| 持续度 $=|\beta|$ | log 方差的 AR 系数，<1 即平稳 |
| 新闻冲击曲线 NIC | $\sigma_t^2$ 随上期冲击的曲线，**不对称 = 杠杆的招牌图** |

---

## 2. 数据要求

GARCH 那套照旧（平稳、已去 ARMA 均值、有 ARCH 效应、足够长、厚尾 t），**额外查"有无杠杆"**：负冲击是否预示更大波动。本 demo 真值 $\gamma=-0.10$。

---

## 3. 六步建模流程（本次真实数字）

### 第0步 准备数据
- 真值：均值 **ARMA(1,1)**（$\phi=0.5,\theta=0.3,\mu=0.05$）+ 波动 **EGARCH(1,1,1)-t**（$\omega=0.02,\alpha=0.12,\beta=0.96,\gamma=-0.10,\nu=7$）。
- 样本 $n=2500$（train=2440, test=60）；ADF p≈2e−28 → 平稳 ✅。

### 第1步 均值方程（ARMA）
BIC 网格定回 **ARMA(1,1)**（$\hat\phi=0.491,\hat\theta=0.334$，贴真值）；得冲击 $a_t$。

### 第2步 检验 ARCH 效应 + 杠杆
- ARCH：Engle LM(10)=201.2，p≈9e−38 ✅。
- **杠杆探针**：$\text{corr}(\mathbb 1[a_{t-1}<0],\,a_t^2)=+0.024>0$ → 负冲击预示更大波动，**该上 EGARCH** ✅。

### 第3步 选型（对称 o=0 vs 带杠杆 o=1）

| 模型 | AIC | BIC |
|---|---|---|
| 对称 EGARCH(1,0,1) | 8040.8 | 8064.0 |
| **带杠杆 EGARCH(1,1,1)** | **8018.9** | **8047.9** ✅ |

**BIC 选 o=1** → 不对称项有价值（类比 GARCH 里"q=0 纯 ARCH 更差"的对照逻辑）。

### 第4步 估计参数（看 $\gamma$ 的符号与显著性）

| 参数 | 估计 | 真值 |
|---|---|---|
| $\omega$ | 0.0227 | 0.02 |
| $\alpha_1$ | 0.1322 | 0.12 |
| **$\gamma_1$** | **−0.0711** | **−0.10** |
| $\beta_1$ | 0.9628 | 0.96 |
| $\nu$ | 7.46 | 7.0 |

- **杠杆 $\hat\gamma=-0.071$，p≈3e−05 → 显著且为负，杠杆确认** ✅。
- 持续度 $|\beta|=0.963$（<1，平稳）；半衰期 18.3 天。

### 第5步 模型检验
① $z$ Ljung-Box p=0.831 ✅　② $z^2$ Ljung-Box p=0.53 ✅　③ 偏度−0.03/峰度4.16　④ **杠杆洗净** $\text{corr}=+0.005≈0$ ✅。

### 第6步 使用模型（预测 + 新闻冲击曲线）
- 均值 ARMA 回复（mu(1)=−0.519 → mu(60)=0.003）；波动模拟法预测 sigma(1)=1.224 → sigma(60)=1.320 收敛。
- 1日 99% VaR=3.61%；样本外 95% 覆盖率 **93.3%**（很准）。
- **新闻冲击曲线（fig4 下图）**：光滑的不对称 V，**左臂（负冲击）明显更陡**——EGARCH 杠杆的招牌画像。

> EGARCH 多步波动**无解析式**，故用 `method='simulation'` 模拟法预测。

---

## 4. EGARCH vs TGARCH（两条抓杠杆的路）

| 维度 | **EGARCH** | TGARCH/GJR |
|---|---|---|
| 建模对象 | $\ln\sigma^2$ | $\sigma^2$ |
| 不对称方式 | $\gamma z$ 连续 | 门限 $\mathbb 1[a<0]$ 折一刀 |
| 杠杆符号 | $\gamma<0$ | $\gamma>0$ |
| 新闻冲击曲线 | **光滑**不对称 | **带折点**抛物线 |
| 正定约束 | 不需要 | 需要 |
| 持续度 | $|\beta|$ | $\alpha+\beta+\gamma/2$ |

两者都比对称 GARCH 多一个方向参数 $\gamma$，谁更优按 AIC/BIC 选。

---

## 5. 易混点澄清

1. **$\gamma$ 的符号约定**：EGARCH 里 $\gamma<0$ 才是杠杆（与 TGARCH 的 $\gamma>0$ 相反，别记混）。
2. **持续度看 $|\beta|$，不是 $\alpha+\beta$**：EGARCH 的持续度由 log 方差的 AR 系数 $\beta$ 决定。
3. **EGARCH 预测要模拟**：没有 GARCH 那种解析递推，多步预测用模拟/自助法。
4. **不对称 ≠ 偏度**：杠杆讲的是"方差对冲击方向不对称"，不是收益分布本身偏斜。

---

## 6. 文件清单与运行

```
egarch_demo.py            # 主程序
requirements.txt          # 依赖
egarch_returns.csv        # 步骤0收益（t, logret_pct, split）
egarch_forecast.csv       # 步骤6每日预测（mu/sigma/区间/VaR/actual）
egarch_fig1_mean.png      # r_t 序列 + ACF/PACF
egarch_fig2_order.png     # a_t² 的 ACF + 杠杆散点
egarch_fig3_diagnose.png  # 条件波动 + 标准化残差四联诊断
egarch_fig4_forecast.png  # 样本外预测 + 新闻冲击曲线（不对称招牌图）
```

运行：`pip install -r requirements.txt && python egarch_demo.py`

---

## 7. 一句话总结

> **EGARCH = 对 ln(σ²) 建模、把冲击拆成"大小项 + 方向项 γz"，用 $\gamma<0$ 连续地刻画杠杆效应，且免去正定约束。** 流程与 GARCH 一致，关键多两步：第2步查杠杆、第4步看 $\gamma$ 显著为负。本 demo 从真值 $\gamma=-0.10$ 的序列出发，o=1 带杠杆按 BIC 胜出、估出 $\hat\gamma=-0.071$（p≈3e−5），新闻冲击曲线左臂更陡——杠杆的教科书式画像。
