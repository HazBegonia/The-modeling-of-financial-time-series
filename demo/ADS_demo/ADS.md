# ADS 分解模型学习总结

> 配套 demo：[ads_demo.py](ads_demo.py) —— 一条龙完整流程（造逐笔数据 → 拆 A/D/S → 条件分解 → 三块建模 → 极大似然 → 诊断解读 → 预测下一笔 ΔP 的完整分布）。
> 对应 Tsay《金融时间序列分析》第 5 章「高频（逐笔）数据」的**价格变化分解模型**（式 5.21–5.28）。本文把理论与 demo 每一步对应起来，并附本次真实运行的数字。

---

## 1. ADS 分解模型是什么

**ADS（Activity–Direction–Size，分解模型）** 解决的问题是：逐笔（tick）成交的价格变化 $\Delta P_i = P_i - P_{i-1}$ 是个「难建模的离散变量」——它**带大量 0**（很多笔不动）、**有正有负**、还**只能跳整数格**（1 格、2 格…），普通的连续模型（AR/ARMA）根本套不上。

ADS 的核心思路：**把一笔价格变化拆成三个独立的小问题**，每个小问题各配一个简单模型：

$$\Delta P_i = \underbrace{A_i}_{\text{动不动}}\cdot\underbrace{D_i}_{\text{往哪动}}\cdot\underbrace{S_i}_{\text{动多大}}$$

| 块 | 含义 | 取值 | 触发条件 |
|---|---|---|---|
| $A_i$ | **Activity** 动不动 | $0$(不动) / $1$(动) | 每笔都有 |
| $D_i$ | **Direction** 往哪动 | $+1$(涨) / $-1$(跌) | 仅当 $A_i=1$ |
| $S_i$ | **Size** 动多大（跳几格） | $1,2,3,\dots$ | 仅当 $A_i=1$ |

一笔价格变化 $\Leftrightarrow$ 一组 $(A_i, D_i, S_i)$。例如 $\Delta P_i=+2 \Leftrightarrow (1,+1,2)$；$\Delta P_i=0 \Leftrightarrow (0,-,-)$；$\Delta P_i=-1 \Leftrightarrow (1,-1,1)$。

- 本 demo 真值（解释变量取自历史：`prevA/prevD/prevS` = 上一笔的 A/D/S）：
  - ① 活动 $A$：$\text{logit}(p) = 0.30 + 0.60\cdot\text{prevA} + 0.05\cdot\text{prevS}$（`prevA` 系数为正 → **活跃度有聚集性**）
  - ② 方向 $D$：$\text{logit}(\delta) = 0.0 + (-0.80)\cdot\text{prevD}$（`prevD` 系数为负 → **反转效应/买卖价差跳动**）
  - ③ 大小 $S$：$S-1\sim\text{Geom}(\lambda_j)$，$\text{logit}(\lambda_U)=0.90-0.12\cdot\text{prevS}$、$\text{logit}(\lambda_D)=0.60-0.10\cdot\text{prevS}$（**上/下两套**，size 分布按方向不同）

---

## 2. 完整建模流程（七步法）

demo 按用户给的 0–7 步流程组织，每步对应 5.21–5.28 的某个公式，**下游一律用「估出来的参数」而非写死真值**：

| 步骤 | 做什么 | 对应公式 | 对应代码 | 产物 |
|---|---|---|---|---|
| 0 | 数据生成（用真值逐笔造 ΔP）+ 划分 train/test | 5.21–5.23 | [ads_demo.py:62-136](ads_demo.py#L62-L136) | `ads_data.csv`、图1 |
| 1 | 把每笔 ΔP 拆成 (A,D,S) | 5.21–5.23 | [ads_demo.py:138-149](ads_demo.py#L138-L149) | — |
| 2 | 条件分解：联合 → 三连乘 | 5.24 | [ads_demo.py:150-160](ads_demo.py#L150-L160) | — |
| 3 | 三块各配一个模型并拟合 | 5.25–5.28 | [ads_demo.py:162-190](ads_demo.py#L162-L190) | — |
| 4 | 写似然 + 极大似然估计（vs 真值） | — | [ads_demo.py:192-209](ads_demo.py#L192-L209) | — |
| 5 | 诊断与解读（反转/不对称/残差） | — | [ads_demo.py:211-292](ads_demo.py#L211-L292) | 图2、图3 |
| 6 | 预测：下一笔 ΔP 的完整概率分布 + 三档评估 | 5.24 | [ads_demo.py:294-396](ads_demo.py#L294-L396) | `ads_forecast.csv`、图4 |

---

## 3. 各步骤详解 + 本次运行结果

### 步骤 0 · 数据生成（5.21–5.23）
用已知 ADS 真值逐笔递推 4000 笔（前 600 笔烧入丢弃），每笔依次抽 ①是否动 → ②若动则方向 → ③若动则大小，再合成 $\Delta P_i=A_i D_i S_i$，并累加还原一条 tick 价格路径。留最后 `h=800` 笔做样本外。
- 本次：活动率 $P(A{=}1)=66.8\%$，动时上涨占比 $50.8\%$，$\Delta P$ 范围 $[-9, 8]$。
- 配图 [ads_fig1_decompose.png](ads_fig1_decompose.png)：价格路径 + 逐笔 ΔP（红涨/蓝跌/灰不动）+ ΔP 的经验离散分布（**0 处一根尖峰**——这正是 AR/ARMA 处理不了的形态）。

### 步骤 1 · 拆分（5.21–5.23）
把前几笔 ΔP 翻译成 $(A,D,S)$ 三元组，演示「怪数据 → 三个规整小变量」。如 $\Delta P=-3\Rightarrow(1,-1,3)$。

### 步骤 2 · 条件分解（5.24）
一个难的联合分布拆成三个独立简单模型的连乘：
$$P(A_i,D_i,S_i\mid F_{i-1}) = \underbrace{P(A_i\mid F_{i-1})}_{\text{①动不动}}\cdot\underbrace{P(D_i\mid A_i,F_{i-1})}_{\text{②往哪动}}\cdot\underbrace{P(S_i\mid D_i,A_i,F_{i-1})}_{\text{③动多大}}$$
**关键红利**：三块可用**不同子样本**分开估——① 用全部交易（本次 train 3199 笔）；② 只用「动了」的（train 2144 笔）；③ 动了再按方向分（上行 1356 / 下行 1315）。

### 步骤 3 · 三块建模（5.25–5.28）
| 块 | 模型 | 公式 | 工具 |
|---|---|---|---|
| ① $A$ | logit | $\ln\frac{p}{1-p}=x\beta$ (5.25) | `statsmodels.Logit`（全样本） |
| ② $D$ | logit | $\ln\frac{\delta}{1-\delta}=\cdots$ (5.26) | `statsmodels.Logit`（A=1 子样本） |
| ③ $S$ | 几何分布 | $S{-}1\sim\text{Geom}(\lambda_j),\ \ln\frac{\lambda_j}{1-\lambda_j}=\cdots$ (5.27–5.28) | **手写极大似然**（scipy BFGS） |

- 为什么 ② 让 $\delta$ 依赖**上一笔方向**：抓 §5.2/5.3 的**买卖价差跳动**（涨完倾向跌 → 负相关）。
- 为什么 ③ 用**几何分布**：天然给出「1 格最多、2 格少、3 格更少」的递减形状；$j$ 标记方向，允许上/下 size 分布不同。
- 为什么 ③ 手写：`statsmodels` 无现成「几何回归」GLM 族，故直接写对数似然 $\sum[\ln\lambda+(S{-}1)\ln(1-\lambda)]$ 让 scipy 极大化，标准误取 `hess_inv`。

### 步骤 4 · 似然 + 极大似然（参数 vs 真值）
似然按每笔「只把发生了的部分」乘进去：
$$L=\prod_i\Big[(1-p_i)^{1-A_i}\cdot\big(p_i\cdot P(D_i)\cdot P(S_i)\big)^{A_i}\Big],\quad P(S_i{=}k)=\lambda_{j,i}(1-\lambda_{j,i})^{k-1}$$
取对数后三块可加 → **分开极大化即得联合 MLE**（条件分解的直接好处）。本次估计 vs 真值：

| 参数 | 估计 | 真值 | | 参数 | 估计 | 真值 |
|---|---|---|---|---|---|---|
| A:常数 | +0.259 | +0.30 | | D:prevD | **−0.779** | **−0.80** |
| A:prevA | +0.613 | +0.60 | | S_U:常数 | +0.972 | +0.90 |
| A:prevS | +0.055 | +0.05 | | S_U:prevS | −0.137 | −0.12 |
| D:常数 | +0.024 | +0.00 | | S_D:常数 | +0.600 | +0.60 |
| | | | | S_D:prevS | −0.112 | −0.10 |

全部贴近真值。联合对数似然 $\log L = -1988.5 -1377.0 -968.0 -1143.6 = -5477.0$。

### 步骤 5 · 诊断与解读
- **[5a] 系数显著性**：`A:prevA` $z=+5.48$（活动聚集性显著）；`D:prevD` $z=-14.23,\ p\approx6\text{e-}46$（**系数为负、极显著 → 反转效应坐实**）。
- **[5b] 反转效应**：$P(\text{涨}\mid\text{上一笔涨})=32.0\% < P(\text{涨}\mid\text{上一笔跌})=69.1\%$ → 涨完倾向跌、跌完倾向涨（买卖价差跳动的负相关）。
- **[5c] size 方向不对称**：估计均值上行 $\approx1.38$ 格、下行 $\approx1.55$ 格 → 用两套 $\lambda$ 才抓得住。
- **[5d] 残差**：A 模型 Pearson 残差 Ljung-Box(10) $p=0.458>0.05$ → 无剩余自相关，活动模型充分。
- 配图 [ads_fig2_models.png](ads_fig2_models.png)（三块「数据 vs 模型」：活动聚集 / 方向反转下斜 / size 几何递减）、[ads_fig3_diagnostics.png](ads_fig3_diagnostics.png)（A、D 校准曲线 + size 拟合残差）。

### 步骤 6 · 预测：下一笔 ΔP 的**完整概率分布**
给定 $F_{i-1}$，依次组合三块得到下一笔取每个 tick 值的概率（而非一个点）：
$$P(\Delta P{=}0)=1-p,\quad P(\Delta P{=}{+}k)=p\,\delta\,\lambda_U(1-\lambda_U)^{k-1},\quad P(\Delta P{=}{-}k)=p\,(1-\delta)\,\lambda_D(1-\lambda_D)^{k-1}$$
本次预测原点（上一笔 $(A,D,S)=(1,-1,1)$）：$\hat p=71.7\%,\ P(\text{涨}\mid\text{动})=69.1\%$，得到分布（$\Delta P{=}{+}1$ 概率最高 34.5%，0 次之 28.3%；因上一笔为跌、反转使分布**偏向上涨**），期望 $E[\Delta P]=+0.35$。

**样本外三档评估**（test，one-step）：

| 档 | 指标 | 结果 |
|---|---|---|
| [活动] | A 命中率 | 65.9%（基准恒猜多数类 65.9%；活动信号弱、与基准持平） |
| [方向] | D 命中率 | **64.3%**（基准 52.0%；反转使方向显著可预测） |
| [整体] | 点预测 RMSE | **1.418** < 朴素(预测 0) 1.460 ✅ |
| [分布] | 预测对数似然/笔 | 模型 **−1.714** > 无条件基准 −1.763 ✅ |

- 配图 [ads_fig4_forecast.png](ads_fig4_forecast.png)：左 = 下一笔 ΔP 的预测分布柱图；右 = test 经验分布 vs 模型平均预测分布（验证整条分布吻合）。

---

## 4. 关键概念速查

| 维度 | 说明 |
|---|---|
| 适用对象 | 逐笔（tick）离散价格变化 $\Delta P$：带 0、有正负、跳整数格 |
| 核心动作 | $\Delta P = A\cdot D\cdot S$，把一个难问题拆成三个独立小问题 |
| 三块模型 | A：logit ｜ D：logit（依赖上一笔方向）｜ S：几何分布（按方向两套） |
| 条件分解 | $P(A,D,S)=P(A)P(D\mid A)P(S\mid D,A)$ → 三块可分开估 |
| 估计方法 | 极大似然；因分解，三个子模型各自极大化即得联合 MLE |
| 反转效应 | D 模型 `prevD` 系数为负 = 涨完倾向跌（买卖价差跳动） |
| 输出 | **下一笔 ΔP 的完整概率分布**，信息远多于一个点预测 |
| 与 AR/ARMA 的区别 | AR/ARMA 建模连续条件均值；ADS 建模**离散变化的整条概率分布** |

---

## 5. 文件清单与运行

```
ads_demo.py             # 主程序（全流程一个文件）
requirements.txt        # 依赖：numpy / scipy / pandas / matplotlib / statsmodels
ads_data.csv            # 步骤0导出（列：i, price, dP, A, D, S, prevA, prevD, prevS, split）
ads_forecast.csv        # 步骤6导出（下一笔 ΔP 的预测分布：dP_value, prob）
ads_fig1_decompose.png  # 价格路径 + 逐笔 ΔP + ΔP 离散分布
ads_fig2_models.png     # 三块「数据 vs 模型」：活动 / 方向反转 / size 几何
ads_fig3_diagnostics.png# A、D 校准 + size 拟合残差
ads_fig4_forecast.png   # 下一笔 ΔP 预测分布 + test 经验 vs 模型分布
```

运行：

```bash
pip install -r requirements.txt
python ads_demo.py
```

---

## 6. 一句话总结

> **ADS 模型 = 把「难建模的离散价格变化 ΔP」拆成「动不动 A / 往哪动 D / 动多大 S」三个独立小问题（logit、logit、几何分布），条件概率连乘成似然，极大似然估计，最后组合出下一笔 ΔP 的完整概率分布。**
> 本 demo 从一个已知 ADS 真值的逐笔序列出发，三块分开极大似然把九个参数都估回真值附近，方向模型的 `prevD` 负系数坐实了买卖价差跳动的反转效应，最终给出的不是一个点、而是下一笔每个 tick 取值的整条概率分布——样本外方向命中、点预测 RMSE、预测对数似然三档均胜过基准。

### 易错点提醒
1. **想用 AR/ARMA 直接套 ΔP**——离散、带 0、跳格的数据连续模型抓不住，必须先分解。
2. **忘了 D、S 只在 A=1 时存在**——似然里 $A_i=0$ 的笔只贡献 $(1-p_i)$，别把方向/大小项也乘进去。
3. **几何分布参数化搞反**——本 demo 用 numpy 约定 $P(S{=}k)=\lambda(1-\lambda)^{k-1}$（$k\ge1$，均值 $1/\lambda$），$\lambda$ 越大 size 越小。
4. **方向不依赖历史**——若 D 模型不放 `prevD`，就抓不住反转效应，方向退化成抛硬币。
5. **size 不分方向**——上/下分布常不对称，应按方向 $j$ 用两套 $\lambda$。
6. **只要点预测**——ADS 的价值在**整条分布**（每个 tick 值的概率），只取期望会丢掉大量信息。
