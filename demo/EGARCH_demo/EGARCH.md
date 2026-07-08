# EGARCH 模型学习总结

> 配套 demo：[egarch_demo.py](egarch_demo.py) —— 一条龙完整流程（造价格 → 对数收益 → 均值方程 ARMA → ARCH 效应 + 杠杆探针 → 定阶/选型 (o,m,s) → EGARCH-t 估计 → 标准化残差三查 + 杠杆洗净 → 解析波动预测(式 3.33) + 不对称量化 + 新闻冲击曲线）。
> 本文把 EGARCH 模型的核心理论和 demo 的每一步对应起来，并附上本次真实运行的数字。
> 对照 Tsay《金融时间序列分析》第 3 章：§3.8 指数 GARCH（第 123–129 页），并用到 §3.3 建模四步法、§3.2 ARCH 检验、§3.8.4 预测、§3.9 门限 GARCH。

---

## 1. EGARCH 模型是什么

**EGARCH（Exponential GARCH，指数 GARCH，Nelson 1991）** 冲着普通 GARCH 的**三个短板**来：

1. **对称响应**：GARCH 的 $\sigma_t^2$ 只依赖 $a_{t-i}^2$，正负冲击平方后一样 → 抓不到**杠杆效应**（股市“利空（负收益）比同等利好更抬高未来波动”）。
2. **非负约束**：GARCH 要 $\alpha_i,\beta_j\ge0$ 才保证 $\sigma^2>0$，估计受限、高阶难用。
3. EGARCH 改为对 $\ln\sigma_t^2$ 建模 → 取指数后**天然为正、无需任何非负约束**；再用**带符号**的新息刻画不对称。

**心脏是那个加权新息 $g(\varepsilon_t)$（式 3.24）**，把冲击拆成“方向”和“大小”两块：

$$g(\varepsilon_t)=\underbrace{\theta\,\varepsilon_t}_{\text{方向/杠杆}}+\underbrace{\gamma\,(|\varepsilon_t|-E|\varepsilon_t|)}_{\text{大小/ARCH}}$$

写成分段就看出**不对称**——正负冲击斜率不同：

$$g(\varepsilon_t)=\begin{cases}(\theta+\gamma)\,\varepsilon_t-\gamma E|\varepsilon_t|,&\varepsilon_t\ge0\\[2pt](\theta-\gamma)\,\varepsilon_t-\gamma E|\varepsilon_t|,&\varepsilon_t<0\end{cases}$$

- $\theta$ 管**方向/杠杆**：正负冲击斜率差 $(\theta+\gamma)$ vs $(\theta-\gamma)$；实务假定 $\theta<0$ → 负冲击影响更大。
- $\gamma$ 管**大小/ARCH**：$|\varepsilon_t|$ 越大波动越高。
- $E|\varepsilon_t|$：标准正态 $=\sqrt{2/\pi}\approx0.798$；标准化 $t$ 略小于它。且 $E[g(\varepsilon_t)]=0$。

一般 **EGARCH$(m,s)$**（式 3.25，$B$ 为后移算子，两多项式根都在单位圆外）：

$$a_t=\sigma_t\varepsilon_t,\qquad \ln\sigma_t^2=\alpha_0+\frac{1+\beta_1B+\cdots+\beta_{s-1}B^{s-1}}{1-\alpha_1B-\cdots-\alpha_mB^m}\,g(\varepsilon_{t-1})$$

- 建模 $\ln\sigma^2$ → 无条件均值 $=\alpha_0$，且无需正定约束；
- EGARCH(1,1) 分子退化成 1，即式 (3.26)：$(1-\alpha B)\ln\sigma_t^2=(1-\alpha)\alpha_0+g(\varepsilon_{t-1})$。

### ⚠️ 记号陷阱（务必分清，本 demo 一律用 `arch` 包记号）

$\theta,\gamma$ 在书里和软件里**恰好对调**，这是 EGARCH 最容易记混的地方：

| 角色 | 书 (3.24) | `arch` 包 / 本 demo |
|---|---|---|
| **大小 / ARCH**（波动聚集） | $\gamma$ | $\alpha$（`alpha`） |
| **方向 / 杠杆** | $\theta$ | $\gamma$（`gamma`） |
| **持续**（$\ln\sigma^2$ 的 AR 系数） | $\alpha_1$ | $\beta$（`beta`） |

`arch` 记号下的递推（本 demo 全程用它）：

$$\ln\sigma_t^2=\omega+\underbrace{\alpha\,(|z_{t-1}|-E|z|)}_{\text{大小}}+\underbrace{\gamma\,z_{t-1}}_{\text{方向/杠杆}}+\underbrace{\beta\,\ln\sigma_{t-1}^2}_{\text{持续}},\qquad z_t=\frac{a_t}{\sigma_t}$$

**$\gamma<0$ 即杠杆**；持续度 $=|\beta|$；$\gamma=0$ 退回对称。（另一常见写法 §3.8.1 式 3.28 里 $s$ 数“ARCH 项”、$m$ 数“$\ln\sigma^2$ 项”，下标位置又和 (3.25) 相反——定阶前务必先看清用的是谁的记号。）

### 阶 $(m,s)$ 不是默认 $(1,1)$

EGARCH 和 GARCH 一样**“阶不好定”**——**没有干净的 ACF/PACF 截尾判据**，只能在低阶候选（EGARCH(1,1)、(2,1)、(2,2) …）里靠 **AIC/BIC + 残差诊断**挑。**Nelson 原文用了 EGARCH(2,2)**，但绝大多数应用 EGARCH(1,1) 就够——**$(1,1)$ 是“选出来的结果”，不是前提**。本 demo 因此绝不写死：**先比对称/带杠杆（$o$）、再在低阶网格里定 $(m,s)$**。

本 demo 真值：波动 **EGARCH(1,1,1)-t**，$\omega=0.02,\alpha=0.12,\beta=0.96,\gamma=-0.10$（$<0$ 杠杆），$\nu=7$（厚尾）；均值 **ARMA(1,1)** $\phi=0.5,\theta=0.3,\mu=0.05$。建模时“假装不知道”，最后揭晓验证。

---

## 2. 完整建模流程（第 0~6 步）

demo 把波动率建模串成 7 步，**下游一律用“定出来的 $o$ 与阶 $(m,s)$、估出的参数”，而不是写死**，换任意数据都能自适应：

| 步骤 | 做什么 | 对应代码 | 产物 |
|---|---|---|---|
| 0 | 造价格 → 对数收益 $r_t$ + 划 train/test | [egarch_demo.py:81-130](egarch_demo.py#L81-L130) | `egarch_returns.csv` |
| 1 | 均值方程：Ljung-Box → BIC 网格定 ARMA → 冲击 $a_t$ | [egarch_demo.py:132-184](egarch_demo.py#L132-L184) | 图1 |
| 2 | 检验 ARCH 效应 + **杠杆探针**（该不该上 EGARCH） | [egarch_demo.py:187-208](egarch_demo.py#L187-L208) | — |
| 3 | 定阶/选型：Schwert 上限 + (3b) $o$=0/1 + (3c) 低阶 $(m,s)$ 网格 | [egarch_demo.py:210-269](egarch_demo.py#L210-L269) | 图2 |
| 4 | 估计 EGARCH$(m,o,s)$-t，看方向项 $\gamma$ 的符号与显著性 | [egarch_demo.py:272-291](egarch_demo.py#L272-L291) | — |
| 5 | 标准化残差三查 + 杠杆是否洗净 | [egarch_demo.py:293-321](egarch_demo.py#L293-L321) | 图3 |
| 6 | 均值 + **解析波动预测(式 3.33)** + 不对称量化 + NIC | [egarch_demo.py:324-407](egarch_demo.py#L324-L407) | 图4、`egarch_forecast.csv` |

---

## 3. 各步骤详解 + 本次运行结果

### 步骤 0 · 准备数据
用已知参数递推造 2500 个 EGARCH(1,1,1)-t 冲击（$\gamma=-0.10$ 制造杠杆），前 1000 个为**烧入期（burn-in）丢弃**，叠加 ARMA(1,1) 均值后得对数收益率（单位 ×100 = 百分比）。留最后 `h=60` 天做**样本外测试**，其余 2440 天训练。

本次：**ADF p = 2.01e−28 ≪ 0.05 → 平稳，可建模 ✅**（$n=2500$，train=2440，test=60）。

### 步骤 1 · 均值方程（ARMA）
$r_t$ 有自相关就先用 ARMA 榨干线性依赖，得冲击 $a_t=r_t-\mu_t$（ARMA 的 ACF/PACF 都拖尾、读不出阶，故用 **BIC 网格** $p,q\in0..3$）。

本次：**$r_t$ 的 Ljung-Box(10) p ≈ 0 → 有自相关**。BIC 网格选 **ARMA(1,1)**（BIC=8386.0）：

```
       q=0       q=1       q=2       q=3
 p=0    --      8655.5    8461.3    8416.6
 p=1  8486.5    8386.0*   8387.7    8395.4     <- (1,1) 全局最小
 p=2  8411.4    8388.4    8395.1    8403.2
 p=3  8393.3    8395.4    8402.4    8410.3
```

拟合参数 vs 真值：$\hat\phi=0.491$（真 0.5）、$\hat\theta=0.334$（真 0.3）、const $=0.003$（真 0.05，均值近 0 属正常）。配图 [egarch_fig1_mean.png](egarch_fig1_mean.png)。

> **两步法**：`arch` 包均值只支持 AR、不支持 MA，含 MA 的 ARMA 走两步——statsmodels 拟合 ARMA 取残差 $a_t$，再对 $a_t$ 建 EGARCH（`mean="Zero"`）。完整联合估计可用 R 的 `rugarch`。若要用波动解释风险溢价，可在均值里加 GARCH-M 项（见 `GARCH-M_demo`）。

### 步骤 2 · 检验 ARCH 效应 + 杠杆
两件事：**有没有 ARCH 效应**（要不要建波动模型）、**有没有杠杆**（该用 EGARCH 还是对称 GARCH）。

本次：$a_t^2$ 的 **Q(5)=1.7e−49、Q(10)=1.4e−80、Q(20)=2.8e−97**；Engle **LM=201.21，p=9.0e−38 ≪ 0.05** → **确有 ARCH 效应 ✅**。杠杆探针 $\text{corr}(\mathbb 1[a_{t-1}<0],\,a_t^2)=+0.024>0$（负冲击后均方 **1.89** vs 正冲击后 **1.71**）→ **负冲击预示更大波动，该上 EGARCH ✅**。

### 步骤 3 · 定阶/选型（本步精华：$o$ 与 $(m,s)$ 都“定”出来）
先定**搜索上限**：Schwert $\text{MAXLAG}=\lfloor12(T/100)^{1/4}\rfloor$——**只依赖样本量、与 AR/MA/ARCH/IGARCH 同一把尺**（$T=2440\Rightarrow$ **MAXLAG=26**，[egarch_demo.py:215](egarch_demo.py#L215)）。EGARCH 无 PACF 截尾判据、高阶又难收敛，故直接在低阶试。

- **3b · 对称 $o=0$ vs 带杠杆 $o=1$**（@ (1,1)，[egarch_demo.py:226](egarch_demo.py#L226)）：

  | 模型 | AIC | BIC |
  |---|---|---|
  | 对称 EGARCH(1,0,1) | 8040.8 | 8064.0 |
  | **带杠杆 EGARCH(1,1,1)** | **8018.9** | **8047.9** ✅ |

  **BIC 选 $o=1$** → 不对称项有价值（类比 GARCH 里“$q=0$ 纯 ARCH 更差”的对照逻辑）。

- **3c · 低阶 $(m,s)$ 网格**（$o=1$ 固定，[egarch_demo.py:230](egarch_demo.py#L230)）：

  ```
         q=1       q=2
   p=1  8047.9*   8055.4      <- (1,1) 全局最小
   p=2  8055.7    8063.2
  ```

  **投票**：`AIC=(1,1) | BIC=(1,1)` → **最终定阶 EGARCH(1,1,1)**。Nelson 用 (2,2)，本数据 (1,1) 胜——**是选出来的，不是默认**。配图 [egarch_fig2_order.png](egarch_fig2_order.png)：左 = $a_t^2$ 的 ACF（波动聚集），右 = **杠杆散点**（左半负冲击整体更高）。

### 步骤 4 · 估计参数（看方向项 $\gamma$ 的符号与显著性）
EGARCH(1,1,1)-t 条件 MLE（[egarch_demo.py:275](egarch_demo.py#L275)）。**注意 `arch` 记号：$\alpha$=大小、$\gamma$=方向/杠杆、$\beta$=持续**：

| 参数 | 估计 | 真值 |
|---|---|---|
| $\omega$ | 0.0227 | 0.02 |
| $\alpha_1$（大小） | 0.1322 | 0.12 |
| **$\gamma_1$（方向/杠杆）** | **−0.0711** | **−0.10** |
| $\beta_1$（持续） | 0.9628 | 0.96 |
| $\nu$ | 7.46 | 7.0 |

- **杠杆 $\hat\gamma=-0.0711$，p=2.87e−05 → 显著且为负，杠杆确认 ✅**。
- 持续度 $|\beta|=0.9628$（$<1$ → $\ln\sigma^2$ 平稳）；半衰期 **18.3 天**。
- logL=−4004.5，AIC=8018.9，BIC=8047.9。

### 步骤 5 · 模型检验（标准化残差三查 + 杠杆洗净）
用 $\sigma_t$ 算**标准化残差** $z_t=a_t/\sigma_t$——模型对的话它应是 i.i.d. 标准分布，**且杠杆被吸收干净**。

本次：**① $z$ Ljung-Box(10) p=0.831 ✅（均值充分）；② $z^2$ Ljung-Box(10) p=0.53 ✅（波动洗净）；③ 偏度 −0.03、峰度 4.16**（厚尾，t 兜住）；**④ 杠杆洗净 $\text{corr}(\mathbb 1[z_{t-1}<0],z_t^2)=+0.005\approx0$ ✅** → 全过。配图 [egarch_fig3_diagnose.png](egarch_fig3_diagnose.png)。

### 步骤 6 · 使用模型（预测：EGARCH 的解析式 3.33 + 招牌 NIC）
对未参与训练的最后 60 天，给出**均值 + 解析波动**预测：

- **均值**：ARMA 逐步回复（$\mu(1)=-0.519\to\mu(60)=+0.003$）。
- **波动（式 3.33）**：$\sigma_h^2(j)=\big[\sigma_h^2(j-1)\big]^{\alpha_1}\exp(\omega)\,E\{e^{g(\varepsilon)}\}$，1 步用已知末态 $z_T$，$\ge2$ 步用 $E\{e^{g(\varepsilon)}\}$ 递推（[egarch_demo.py:342](egarch_demo.py#L342)，零件 [egarch_demo.py:66](egarch_demo.py#L66)、[egarch_demo.py:71](egarch_demo.py#L71)）。$\sigma(1)=1.224\to\sigma(60)=1.444$，**几何回复**到无条件水平 $\sigma\approx1.473$（$|\beta|<1$ → **对比 IGARCH 持续=1 的线性外推不回复**）。
- **交叉验证**：解析 vs `arch` 模拟法，$\sigma(1)=1.224/1.224$（1 步精确一致 → 递推机制/取参正确）；$\sigma(60)=1.444/1.33$（远期解析是**正态**闭式、模型是 $t$，故解析略高，属正常）。
- **不对称量化**（[egarch_demo.py:363](egarch_demo.py#L363)）：$\dfrac{\sigma_t^2(z=-2)}{\sigma_t^2(z=+2)}=e^{-4\gamma}=1.329$ → **负的两个标准差冲击比正的多抬升波动 +32.9%**（书 IBM 例为 1.374 / +37.4%）。
- 1 日 99% VaR 明日 = **3.61%**；样本外 95% 区间覆盖率 **93.3%**（很准）。

配图 [egarch_fig4_forecast.png](egarch_fig4_forecast.png)：上 = 实际 vs ARMA 中心 + 95% 动态区间 + 缓缓回复的 $\sigma(\ell)$；下 = **新闻冲击曲线（NIC）**——一条光滑的不对称 V，**左臂（负冲击）明显更陡**，正是 EGARCH 杠杆的招牌画像。

---

## 4. 阶 $(m,s)$ 与“要不要杠杆”：书中实例

Tsay §3.8 的两个实例，正好一个把“阶不是默认”、一个把“杠杆量化”讲透：

**① Nelson (1991) 价值加权 CRSP 月超额收益，6408 obs** —— 阶不总是 (1,1)：拟合 **AR(1)-EGARCH(2,2)-M**（还带非交易天数项），杠杆 $\theta=-0.118$，风险溢价 $c=-3.361$（但不显著），$\nu=1.576$。这就是 Nelson 用 (2,2) 的原始出处——**阶要试出来**。

**② IBM 月对数收益，864 obs** —— 干净的 EGARCH(1,1)（式 3.30–3.31）：

$$r_t=0.0105+0.092\,r_{t-1}+a_t,\qquad g(\varepsilon_{t-1})=\underbrace{-0.0795}_{\theta<0\,\text{杠杆}}\varepsilon_{t-1}+\underbrace{0.2647}_{\gamma\,\text{大小}}\big(|\varepsilon_{t-1}|-\sqrt{2/\pi}\big),\quad \text{持续}=0.856$$

**不对称量化**（招牌算法）：两个标准差的负冲击 vs 正冲击

$$\frac{\sigma_t^2(\varepsilon=-2)}{\sigma_t^2(\varepsilon=+2)}=\frac{\exp[(\theta-\gamma)\cdot(-2)]}{\exp[(\theta+\gamma)\cdot 2]}=\frac{e^{0.6884}}{e^{0.3704}}=e^{0.318}=1.374$$

→ **负扰动比正扰动多抬升波动 37.4%**。（本 demo 用同一算法得 +32.9%，见步骤 6。）扩样本到 2003（936 obs）用 GED 拟合 (3.28 形式)：$\ln\sigma_t^2=-0.557+0.220\,(|a_{t-1}|-0.264\,a_{t-1})/\sigma_{t-1}+0.929\ln\sigma_{t-1}^2$，杠杆 LEV$=-0.264$（$t=-2.094$，5% 显著为负）。

> **预测（§3.8.4）**：书里 IBM（原点 $t=864$）的多步波动 $\hat\sigma^2(1)=6.05\text{e−}3\to\hat\sigma^2(2)=5.82\text{e−}3\to\cdots\to$ 收敛到样本方差 $4.37\text{e−}3$——**几何回复**，正是本 demo 步骤 6 的同一招式。

---

## 5. 关键概念速查（EGARCH ↔ GARCH / IGARCH / TGARCH 对照）

| 维度 | GARCH(1,1) | IGARCH(1,1) | **EGARCH** | TGARCH/GJR |
|---|---|---|---|---|
| 建模对象 | $\sigma^2$ | $\sigma^2$ | **$\ln\sigma^2$** | $\sigma^2$ |
| 抓什么 | 一般聚集 | 极高持续 | **杠杆/不对称** | 杠杆/不对称 |
| 不对称方式 | 无 | 无 | **$\gamma z$ 连续（带符号）** | 门限 $\mathbb 1[a<0]$ 折一刀 |
| 杠杆符号 | — | — | **$\gamma<0$** | $\gamma>0$ |
| 正定约束 | 需要 | 需要 | **不需要**（建 $\ln\sigma^2$） | 需要 |
| 持续度 | $\alpha+\beta$ | $=1$ | **$|\beta|$** | $\alpha+\beta+\gamma/2$ |
| 新闻冲击曲线 | 对称抛物线 | 对称 | **光滑不对称 V** | 带折点抛物线 |
| 多步波动预测 | 几何收敛 | 线性外推不回复 | **几何收敛（式 3.33）** | 几何收敛 |

- **$\theta/\gamma$ 记号对调**：书 $\theta$=方向、$\gamma$=大小；`arch` 里 $\gamma$=方向、$\alpha$=大小。**别记混**（见 §1 表）。
- **持续度看 $|\beta|$，不是 $\alpha+\beta$**：EGARCH 的持续由 $\ln\sigma^2$ 的 AR 系数 $\beta$ 决定。
- **阶不是默认 $(1,1)$**：EGARCH 无 ACF/PACF 截尾判据，$(m,s)$ 和“要不要杠杆($o$)”都靠低阶 AIC/BIC + 残差诊断挑。
- **无需正定约束**：建 $\ln\sigma^2$ 取指数后必正，参数可自由取值（GARCH 必须 $\omega,\alpha,\beta\ge0$）。
- **不对称 ≠ 偏度**：杠杆讲的是“方差对冲击方向不对称”，不是收益分布本身偏斜。
- **EGARCH 会回复**：$|\beta|<1$ 时多步波动几何收敛到无条件水平，和 IGARCH 的“不回复”正相反。

---

## 6. 文件清单与运行

```
egarch_demo.py            # 主程序（全流程一个文件）
requirements.txt          # 依赖：numpy / scipy / pandas / matplotlib / statsmodels / arch
egarch_returns.csv        # 步骤0导出的对数收益（列：t, logret_pct, split）
egarch_forecast.csv       # 步骤6导出的每日预测（中心/波动/区间/VaR/实际/是否命中）
egarch_fig1_mean.png      # r_t 序列 + ACF/PACF
egarch_fig2_order.png     # a_t² 的 ACF（波动聚集）+ 杠杆散点（负冲击更高）
egarch_fig3_diagnose.png  # 条件波动 + 标准化残差四联诊断
egarch_fig4_forecast.png  # 样本外：ARMA 中心 + 回复波动 + 新闻冲击曲线（不对称招牌图）
```

运行：

```bash
pip install -r requirements.txt
python egarch_demo.py
```

---

## 7. 一句话总结

> **EGARCH = 对 $\ln\sigma^2$ 建模、把冲击拆成“大小项 $\alpha(|z|-E|z|)$ + 方向项 $\gamma z$”，用 $\gamma<0$ 连续刻画杠杆效应，且免去正定约束。** 完整流程是：
> **建均值 ARMA → 查 ARCH 效应 + 杠杆探针 → 定阶/选型（Schwert 上限 + 对称/带杠杆 $o$ + 低阶 $(m,s)$ AIC/BIC）→ EGARCH-t MLE（看 $\gamma$ 显著为负）→ 标准化残差三查 + 杠杆洗净 → 用式(3.33) 做几何回复的波动预测 + 不对称量化 + 新闻冲击曲线。**
> 本 demo 从真值 $\gamma=-0.10$ 的序列出发，$o=1$ 带杠杆按 BIC 胜出、定阶定回 (1,1)、估出 $\hat\gamma=-0.071$（p≈3e−5）、负冲击多抬波动 +32.9%、NIC 左臂更陡——一次教科书式的杠杆闭环，也把“阶不是默认 $(1,1)$、$\theta/\gamma$ 记号别对调”这两件事讲透。

### 易错点提醒
1. **把 $\theta$/$\gamma$ 记号搞反**——书里 $\theta$=方向/杠杆、$\gamma$=大小；`arch` 里 $\gamma$=方向、$\alpha$=大小，恰好对调。读参数、写公式前先确认用谁的记号。
2. **把 EGARCH 的阶默认写死 $(1,1)$**——EGARCH 无 ACF/PACF 截尾判据，$(m,s)$ 与“要不要杠杆”都要靠低阶 AIC/BIC 挑；Nelson 原文就是 (2,2)。
3. **持续度错用 $\alpha+\beta$**——EGARCH 的持续是 $|\beta|$（$\ln\sigma^2$ 的 AR 系数），不是 $\alpha+\beta$。
4. **杠杆符号记反**——EGARCH 里 $\gamma<0$ 才是杠杆（与 TGARCH 的 $\gamma>0$ 相反）。
5. **EGARCH 预测硬套 GARCH 的解析递推**——EGARCH 的多步预测走式(3.33) 的 $E\{e^{g(\varepsilon)}\}$ 递推；$t$ 新息时正态闭式只是近似，远期可用模拟法校准。
6. **把不对称当成收益偏度**——杠杆是“方差对冲击方向不对称”，与收益分布是否偏斜是两回事。
