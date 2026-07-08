# IGARCH 模型学习总结

> 配套 demo：[igarch_demo.py](igarch_demo.py) —— 一条龙完整流程（造价格 → 对数收益 → 均值方程 ARMA → ARCH 效应检验 → 定阶 (m,s) → 无约束 GARCH 估计 → 按持续度≈1 追加单位根约束、约束 IGARCH MLE → 标准化残差三查 → “不回复”波动/VaR 预测）。
> 本文把 IGARCH 模型的核心理论和 demo 的每一步对应起来，并附上本次真实运行的数字。
> 对照 Tsay《金融时间序列分析》第 3 章：§3.6 求和 GARCH（第 121–122 页），并用到 §3.3 建模四步法、§3.2 ARCH 检验、§3.4.3 定阶、§3.5 GARCH 表示与预测。

---

## 1. IGARCH 模型是什么

**IGARCH（Integrated GARCH，求和 / 单位根 GARCH）** 是 GARCH 的**边界特例**：把持续度 $\alpha+\beta$ 顶到**恰好等于 1**。

$$\text{GARCH}(1,1):\ \sigma_t^2=\omega+\alpha\, a_{t-1}^2+\beta\,\sigma_{t-1}^2,\qquad \alpha+\beta<1$$

$$\text{IGARCH}(1,1):\ \alpha+\beta=1\ \Longrightarrow\ \sigma_t^2=\omega+\beta\,\sigma_{t-1}^2+(1-\beta)\,a_{t-1}^2$$

- **核心思想**：把波动的持续度顶到 1，冲击不再随时间衰减——GARCH 的波动“有记性但会忘”，IGARCH 的波动“过目不忘”。
- **来历**（式 3.15）：令 $\eta_t=a_t^2-\sigma_t^2$（一个鞅差白噪声），GARCH(1,1) 可写成平方扰动的 ARMA(1,1)：$a_t^2=\alpha_0+(\alpha_1+\beta_1)a_{t-1}^2+\eta_t-\beta_1\eta_{t-1}$。它的 AR 多项式 $1-(\alpha_1+\beta_1)B$ 在 $\alpha_1+\beta_1=1$ 时**有一个单位根**——这就是 IGARCH（“平方扰动服从单位根 ARIMA”的 GARCH）。
- 本 demo 真值：波动 **IGARCH(1,1)-t**，$\omega=0.02,\alpha=0.06,\beta=0.94$（**持续度恰 = 1.00**），$\nu=7$（厚尾）；均值 **ARMA(1,1)** $\phi=0.5,\theta=0.3,\mu=0.05$。建模时“假装不知道”，最后揭晓验证。

### 一旦 $\alpha+\beta=1$，三件事同时发生

1. **冲击永不消散**：半衰期 $\ln0.5/\ln(\alpha+\beta)=\ln0.5/\ln1=\infty$。
2. **无有限无条件方差**：$\dfrac{\omega}{1-\alpha-\beta}=\dfrac{\omega}{0}=\infty$（$a_t$ 严格平稳但**非协方差平稳**，二阶矩不存在，Nelson 1990 证 $\sigma_t^2$ 是鞅）。
3. **多步波动不回复**（式 3.22）：$\sigma_t^2(\ell)=\sigma_t^2(1)+(\ell-1)\omega$，线性外推、永不收敛。

> **著名特例 $\omega=0$ 就是 RiskMetrics / EWMA**：反复迭代 $\sigma_t^2=(1-\beta)a_{t-1}^2+\beta\sigma_{t-1}^2$ 得 $\sigma_t^2=(1-\beta)\sum_{i\ge1}\beta^{i-1}a_{t-i}^2$，正是贴现因子 $\beta$ 的指数加权移动方差（日频常固定 $\beta=0.94$，连优化都免了）。业界算 VaR（第 7 章）用的就是它。

### 阶 $(m,s)$ 与单位根约束是两回事（本模型的关键）

一个完整的 IGARCH$(m,s)$ 是

$$r_t=\mu_t+a_t\ (\text{均值 ARMA}(p,q)),\quad a_t=\sigma_t\varepsilon_t,\quad \sigma_t^2=\alpha_0+\sum_{i=1}^m\alpha_i a_{t-i}^2+\sum_{j=1}^s\beta_j\sigma_{t-j}^2,\quad \text{s.t. }\sum(\alpha_i+\beta_i)=1$$

- **阶 $(m,s)$ 是“定”出来的**：$m$（ARCH 阶，数 $a^2$ 的滞后）靠 $a_t^2$ 的 PACF 量级 + 低阶 AIC/BIC 网格；$s$（GARCH 阶，数 $\sigma^2$ 的滞后）书里明说“不好定”、只在低阶里试。**这一层和普通 GARCH 完全一样。**
- **“I”（$\sum\alpha+\sum\beta=1$ 的单位根约束）不是定阶的产物，更不是默认 $(1,1)$**：它是在**定好阶、估完 GARCH 之后**，看 $\hat\alpha+\hat\beta$ 是否 $\approx1$，再决定加不加的约束。

所以本 demo 绝不写死 $(1,1)$：**先老实定阶 → 估无约束 GARCH → 发现持续度≈1 → 才施加单位根约束重估成 IGARCH**。

---

## 2. 完整建模流程（第 0~6 步）

demo 把波动率建模的标准流程串成 7 步，**下游一律用“定出来的阶 `best_gorder`、估出的持续度 `persist`、约束后重估的 IGARCH”，而不是写死**，所以换任意数据都能自适应：

| 步骤 | 做什么 | 对应代码 | 产物 |
|---|---|---|---|
| 0 | 造价格 → 对数收益 $r_t$ + 划 train/test | [igarch_demo.py:76-122](igarch_demo.py#L76-L122) | `igarch_returns.csv` |
| 1 | 均值方程：Ljung-Box → BIC 网格定 ARMA → 冲击 $a_t$ | [igarch_demo.py:124-177](igarch_demo.py#L124-L177) | 图1 |
| 2 | 检验 ARCH 效应（要不要建波动模型） | [igarch_demo.py:179-190](igarch_demo.py#L179-L190) | — |
| 3 | 定阶 $(m,s)$：Schwert 上限 + $a_t^2$ PACF + 低阶 AIC/BIC 网格 | [igarch_demo.py:192-245](igarch_demo.py#L192-L245) | 图2 |
| 4 | 估计：4a 无约束 GARCH → 4b 判持续度≈1 → 4c 约束 IGARCH MLE | [igarch_demo.py:247-291](igarch_demo.py#L247-L291) | — |
| 5 | 约束模型标准化残差三查 + 持续度=1 判定 | [igarch_demo.py:293-320](igarch_demo.py#L293-L320) | 图3 |
| 6 | 均值 + “不回复”波动 预测 → 区间 / VaR | [igarch_demo.py:322-369](igarch_demo.py#L322-L369) | 图4 |

---

## 3. 各步骤详解 + 本次运行结果

### 步骤 0 · 准备数据
用已知参数递推造 2500 个 IGARCH(1,1)-t 冲击，前 1000 个为**烧入期（burn-in）丢弃**，叠加 ARMA(1,1) 均值后得对数收益率（单位 ×100 = 百分比，量纲适中、`arch` 包推荐做法）。留最后 `h=60` 天做**样本外测试**，其余 2440 天做训练。建模时“假装不知道”真值，最后再揭晓验证。

本次：**ADF p = 1.76e−21 ≪ 0.05 → 平稳，可建模 ✅**（$n=2500$，train=2440，test=60）。

### 步骤 1 · 均值方程（ARMA）
$r_t$ 有自相关就先用 ARMA 榨干线性依赖，得冲击 $a_t=r_t-\mu_t$（ARMA 的 ACF/PACF 都拖尾、读不出阶，故用 **BIC 网格** $p,q\in0..3$）。

本次：**$r_t$ 的 Ljung-Box(10) p = 1.06e−240 → 有自相关**。BIC 网格选 **ARMA(1,1)**（BIC=11363.2）：

```
       q=0       q=1       q=2       q=3
 p=0    --      11538.1   11394.7   11368.1
 p=1  11510.4   11363.2*  11370.9   11375.0     <- (1,1) 全局最小
 p=2  11377.1   11370.9   11376.7   11383.7
 p=3  11376.2   11375.3   11382.9   11380.5
```

拟合参数 vs 真值：$\hat\phi=0.408$（真 0.5）、$\hat\theta=0.387$（真 0.3）、const $=0.064$（真 0.05）。

配图 [igarch_fig1_mean.png](igarch_fig1_mean.png)：$r_t$ 序列 + ACF + PACF。

> **两步法**：`arch` 包均值只支持 AR、不支持 MA，含 MA 的 ARMA 走两步——statsmodels 拟合 ARMA 取残差 $a_t$，再对 $a_t$ 建波动（`mean="Zero"`）。完整联合估计可用 R 的 `rugarch`。

### 步骤 2 · 检验 ARCH 效应
对 $a_t^2$ 两法互证：Ljung-Box $Q(m)$ + Engle LM（辅助回归 $a_t^2$ 对滞后 $a_{t-i}^2$，$LM=nR^2\sim\chi^2_m$）。

本次：$a_t^2$ 的 **Q(5)=1.9e−141、Q(10)=6.1e−245、Q(20)≈0**；Engle **LM=446.31，p=1.3e−89 ≪ 0.05** → **确有 ARCH 效应，需建波动率模型 ✅**。

### 步骤 3 · 定阶 $(m,s)$（本步精华，上限与 ARCH 同规则）
先定**搜索上限**：Schwert $\text{MAXLAG}=\lfloor 12\,(T/100)^{1/4}\rfloor$——**只依赖样本量 $T$、与 AR/MA/ARCH 同一把尺**（本例 $T=2440\Rightarrow$ **MAXLAG=26**，[igarch_demo.py:199](igarch_demo.py#L199)）。

- **3a · $a_t^2$ 的 PACF 量级**（§3.4.3，和 ARCH 定阶同一套口诀）：令 $\eta_t=a_t^2-\sigma_t^2$，纯 ARCH 时 $a_t^2$ 服从 AR($m$)、PACF 在 $m$ 阶后截尾。本次：**最后显著 lag = 25**（band=±0.040）——**故意这么大**：持续度≈1 时 $a_t^2$ 近积分、PACF 拖尾并不干净（一旦有 $\beta$ 项 $a_t^2$ 就成 ARMA 形），所以它只作 ARCH 阶**量级**参考，真正定阶交给网格。
- **3b · 低阶 $(m,s)$ 网格**：GARCH 高阶 MLE 难收敛、实证极少胜低阶，故在 Schwert 上限基础上**诚实截到 3**（[igarch_demo.py:210](igarch_demo.py#L210)）。对 $m\in1..3,\ s\in0..3$ 各估一遍取 BIC 最小（$s=0$ 那列即纯 ARCH）：

```
       s=0       s=1       s=2       s=3
 m=1  10628.3   10154.1*  10161.9   10169.7     <- (1,1) 全局最小
 m=2  10459.2   10161.0   10167.1   10174.5
 m=3  10404.5   10168.8   10174.9   10182.3
```

**投票**：`PACF(量级)~25 | AIC=(1,1) | BIC=(1,1)` → **最终定阶 $(m,s)=(1,1)$（以 BIC 为准）**。注意 $s=0$（纯 ARCH）整列都远差于含 $\beta$ 的 $(1,1)$——**一个 $\beta$ 项就抓住了高持续波动**。

配图 [igarch_fig2_order.png](igarch_fig2_order.png)：左 = $a_t^2$ 的 ACF（极缓慢衰减 → 持续度≈1 的指纹），右 = $a_t^2$ 的 PACF（量级 ~25，含 $\beta$ 后拖尾）。

### 步骤 4 · 估计参数（先无约束 → 判持续度 → 再约束：IGARCH 的核心）

**4a · 无约束 GARCH(1,1)**（[igarch_demo.py:253](igarch_demo.py#L253)，此刻还是普通 GARCH）：

| 参数 | 估计 | 真值 |
|---|---|---|
| $\omega$ | 0.0235 | 0.02 |
| $\alpha_1$ | 0.0569 | 0.06 |
| $\beta_1$ | 0.9388 | 0.94 |
| $\nu$ | 8.35 | 7.0 |

→ **持续度 $\hat\alpha+\hat\beta=0.9957$**（真值 1.00）。

**4b · 判定**（[igarch_demo.py:263](igarch_demo.py#L263)）：$0.9957\ge0.99\approx1$ → **施加单位根约束 $\alpha+\beta=1$，改用 IGARCH**。这个“I”是估完 GARCH 后按 $\hat\alpha+\hat\beta\approx1$ 才追加的约束，**不是一开始就默认 $(1,1)$**。

**4c · 约束 IGARCH(1,1)-t MLE**（[igarch_demo.py:273](igarch_demo.py#L273)）：`arch` 包没有现成的带 $\alpha+\beta=1$ 约束的 IGARCH，故自写“标准化 Student-t”条件似然（零件 [igarch_demo.py:56](igarch_demo.py#L56)、[igarch_demo.py:64](igarch_demo.py#L64)），递推里直接令 $\alpha_1=1-\beta$（书里 IGARCH(1,1) 的写法），用 scipy 优化。**只估 $\omega,\beta,\nu$ 三个自由参数**（比 GARCH(1,1) 少一个）：

| 参数 | 约束估计 | 真值 | 说明 |
|---|---|---|---|
| $\omega$ | 0.0176 | 0.02 | |
| $\beta_1$ | 0.9381 | 0.94 | 自由参数 |
| $\alpha_1=1-\beta$ | 0.0619 | 0.06 | **被 $\beta$ 锁死，非自由参数** |
| $\nu$ | 7.85 | 7.0 | |
| **持续度** | **1.0000** | 1.00 | 恰好 = 1，单位根 |

- **logL=−5062.3，AIC=10130.5，BIC=10147.9**——比无约束 GARCH 的 BIC=10154.1 **更低**（少一个参数、拟合几乎不损），且约束估计（$\nu=7.85$）反而比无约束（$\nu=8.35$）**更接近真值**：这正是“真值本就 = 1 时，加对约束能提升估计”的教科书演示。
- 无条件波动率 = **∞**（IGARCH 无有限无条件方差）；半衰期 = **∞**（冲击永不消散）。

> ⚠️ 工程细节：`arch` 包的 GARCH 会把持续度估到“贴着 1 的边界值”（本次 0.9957），**判据看“是否 ≥0.99”而非“是否精确 =1”**；真正的 $=1$ 由 4c 的约束 MLE 强制实现。

### 步骤 5 · 模型检验（约束模型的标准化残差三查）
用 4c 的 $\sigma_t$ 算**标准化残差** $\tilde a_t=a_t/\sigma_t$——模型对的话它应是 i.i.d. 标准分布。三查 + 持续度判定：

本次：**① $\tilde a_t$ Ljung-Box(10) p=0.35 ✅（均值充分）；② $\tilde a_t^2$ Ljung-Box(10) p=0.964 ✅（波动洗净）；③ 偏度 −0.09、峰度 4.37**（厚尾，t 兜住）；**④ 持续度 = 1.0000 → 确认 IGARCH（严格平稳但非协方差平稳）** → 三查全过。

配图 [igarch_fig3_diagnose.png](igarch_fig3_diagnose.png)：条件波动 $\sigma_t$（看聚集）、$\tilde a_t^2$ 的 ACF（白噪声）、直方图、Q-Q（厚尾）。

### 步骤 6 · 使用模型（预测：IGARCH 的招牌“不回复”）
对未参与训练的最后 60 天，给出**均值 + “不回复”波动**预测：
- **均值**：ARMA 逐步回复（$\mu(1)=-1.561\to\mu(60)=+0.064$）。
- **波动（式 3.22）**：$\sigma^2(\ell)=\sigma^2(1)+(\ell-1)\omega$——方差每步 $+\omega=0.018$，$\sigma(1)=2.064\to\sigma(60)=2.302$ **线性外推、不向任何固定水平收敛**（对比 GARCH 向无条件方差几何收敛）。
- 1 日 99% VaR 明日 = **6.75%**；样本外 95% 区间覆盖率 **78.3%**（60 点小样本，且测试期头几天恰是 −8.2/−6.7 的大负异常，拉低覆盖）。

> 见 [igarch_forecast.csv](igarch_forecast.csv)：每行 = 一天的 `mu_hat`(中心) + `sigma_hat`(宽度) + `lo95/hi95`(区间) + `VaR99`，`actual` 是实际值、`in_interval` 标是否命中；`sigma_hat` 逐日微增就是“不回复”的直接体现。

配图 [igarch_fig4_forecast.png](igarch_fig4_forecast.png)：上 = 实际 vs ARMA 中心 + 95% 动态区间；下 = 波动 $\sigma(\ell)$（缓缓上漂、不回复）+ 99% VaR。

---

## 4. 阶 $(m,s)$ vs 单位根“I”：书中实例把这条链走了一遍

Tsay §3.5.1 例 3.3（S&P 500 月超额收益率）是最好的示范——**先定阶、后追加约束**：

| 步骤 | 做法 | 结果 |
|---|---|---|
| 均值定阶 | 看 $r_t$ 的 ACF（滞后 1、3 有相关） | MA(3) 或 AR(3) |
| ARCH 检验 | $r_t^2$ 的 PACF 显强相关 | 有显著 ARCH 效应 |
| 波动定阶 | 只在低阶里试 | GARCH(1,1) |
| 联合估计 | AR(3)-GARCH(1,1) | $\sigma_t^2=0.000084+0.1213a_{t-1}^2+0.8523\sigma_{t-1}^2$ |
| **判单位根** | $\hat\alpha_1+\hat\beta_1=0.1213+0.8523=\mathbf{0.9736}\approx1$ | **→ 改用 IGARCH(1,1)** |

书里最终的 IGARCH(1,1)：$\sigma_t^2=0.000119+0.8059\sigma_{t-1}^2+0.1941a_{t-1}^2$（注意 $0.8059+0.1941=1$，正是 $\beta+(1-\beta)$）。**参数和普通 GARCH(1,1) 很接近，唯一本质区别是无条件方差没有定义。**

> **警示（书中原话）**：这种“波动持续性”很可能是波动率水平发生**结构性移动（level shift）**造成的假象，“真正原因值得仔细研究”——实务上别盲目把 $\hat\alpha+\hat\beta\approx1$ 当成真持续，先排查结构突变。

---

## 5. 关键概念速查（IGARCH ↔ GARCH 对照）

| 维度 | GARCH(1,1) | **IGARCH(1,1)** | EGARCH | TGARCH |
|---|---|---|---|---|
| 持续度 $\alpha+\beta$ | <1 | **=1** | $\lvert\beta\rvert$<1 | <1 |
| 半衰期 | 有限 | **∞** | 有限 | 有限 |
| 无条件方差 | 有限 | **∞（不存在）** | 有限 | 有限 |
| 多步波动预测 | 几何收敛到 $V=\omega/(1-\alpha-\beta)$ | **线性外推、不回复** | 收敛 | 收敛 |
| 自由参数（(1,1)） | $\omega,\alpha,\beta,\nu$（4） | **$\omega,\beta,\nu$（3，$\alpha=1-\beta$）** | 更多 | 更多 |
| 抓什么 | 一般聚集 | **极高持续 / 永久冲击** | 杠杆 | 杠杆 |

- **阶 vs “I” 分两层**：$(m,s)$ 靠 $a^2$ 的 PACF + 低阶 AIC/BIC 定；“I” 是估完看 $\hat\alpha+\hat\beta\approx1$ 才追加的约束——**不是默认 $(1,1)$**。
- **约束估计少一参**：$\alpha_1=1-\beta$ 被锁死，IGARCH(1,1) 只有 3 个自由参数；真值恰 =1 时，加约束还能提升估计（本次 BIC 反更低）。
- **判据看 ≥0.99 而非 =1**：无约束 GARCH 只会顶到“贴着 1”的边界值，精确的 1 由约束 MLE 实现。
- **方差版“反均值回复”**：AR/GARCH 预测都回复（到 $\mu$ / 到 $V$），IGARCH 波动预测**不回复**、按 $\sigma^2(\ell)=\sigma^2(1)+(\ell-1)\omega$ 线性外推。
- **$\omega=0$ 即 EWMA/RiskMetrics**：业界最常用的指数加权移动方差就是 IGARCH 的特例。
- **持续=1 别做均值回复预测**：套 GARCH 的“向 $V$ 收敛”会严重低估远期波动。

---

## 6. 文件清单与运行

```
igarch_demo.py            # 主程序（全流程一个文件）
requirements.txt          # 依赖：numpy / scipy / pandas / matplotlib / statsmodels / arch
igarch_returns.csv        # 步骤0导出的对数收益（列：t, logret_pct, split）
igarch_forecast.csv       # 步骤6导出的每日预测（中心/波动/区间/VaR/实际/是否命中）
igarch_fig1_mean.png      # r_t 序列 + ACF/PACF
igarch_fig2_order.png     # a_t² 的 ACF（极缓慢衰减=持续度≈1）+ PACF（ARCH 阶量级）
igarch_fig3_diagnose.png  # 条件波动 + 标准化残差四联诊断
igarch_fig4_forecast.png  # 样本外：ARMA 中心 + 波动“不回复”曲线/VaR
```

运行：

```bash
pip install -r requirements.txt
python igarch_demo.py
```

---

## 7. 一句话总结

> **IGARCH = GARCH 把持续度 $\alpha+\beta$ 顶到 1 的边界特例 = 波动冲击永久、无有限无条件方差、多步预测线性外推不回复。** 完整流程是：
> **建均值 ARMA → 查 ARCH 效应 → 定阶 $(m,s)$（Schwert 上限 + $a^2$ PACF + 低阶 AIC/BIC）→ 估无约束 GARCH → 发现 $\hat\alpha+\hat\beta\approx1$ → 追加单位根约束、约束 MLE 重估 IGARCH(1,1)（少一参）→ 标准化残差三查 → 用式(3.22) 做“不回复”波动预测（$\omega=0$ 即 RiskMetrics/EWMA）。**
> 本 demo 从真值持续度=1 的序列出发，定阶定回 $(1,1)$、无约束估出 0.9957、约束后恰为 1.0000、BIC 反更低、波动预测缓缓上漂——一次教科书式的闭环，也把“阶不是默认 $(1,1)$、I 是后加的约束”这件事讲透。

### 易错点提醒
1. **把 IGARCH 的阶默认写死 $(1,1)$**——阶要像 GARCH 一样定（$a^2$ PACF + 低阶 AIC/BIC），$(1,1)$ 是本数据定出来的结果，不是前提。
2. **把“定阶”和“单位根”混为一谈**——阶 $(m,s)$ 先定；“I”（$\alpha+\beta=1$）是估完 GARCH 后按 $\hat\alpha+\hat\beta\approx1$ 追加的约束。
3. **搜索上限拍脑袋写死**——上限用 Schwert（随样本量走），与 AR/MA/ARCH 同一把尺，并自检有没有贴到上限被截断。
4. **看到持续度≈1 仍做均值回复预测**——IGARCH 波动不回复，套 GARCH 的收敛公式会低估远期波动与 VaR。
5. **把 IGARCH 当成“方差爆炸”**——$\alpha+\beta=1$ 时方差过程是“随机游走+漂移”，严格平稳、只是无有限二阶矩。
6. **盲信 $\hat\alpha+\hat\beta\approx1$ 是真持续**——它常是波动水平结构性移动（level shift）的假象，先排查结构突变。
