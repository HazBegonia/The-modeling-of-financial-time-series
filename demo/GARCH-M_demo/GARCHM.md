# GARCH-M 模型学习总结

> 配套 demo：[garchm_demo.py](garchm_demo.py) —— 一条龙完整流程（造价格 → 对数收益 → 简约均值 ARMA → ARCH 效应 → 波动定阶 (m,s) → 选风险溢价形式(表 3-2) → **联合 MLE** 估 ARMA+GARCH+c → 标准化残差三查 + 风险溢价 LR 检验 → 先波动后均值的动态预测 + VaR）。
> 本文把 GARCH-M 模型的核心理论和 demo 的每一步对应起来，并附上本次真实运行的数字。
> 对照 Tsay《金融时间序列分析》第 3 章：§3.6 GARCH 求和与 §3.7 GARCH-in-Mean（式 3.23、表 3-2），并用到 §3.3 建模四步法、§3.2 ARCH 检验、§3.5 GARCH 定阶与预测。结构对照 [GARCH.md](../GARCH_demo/GARCH.md)、[EGARCH.md](../EGARCH_demo/EGARCH.md)、[IGARCH.md](../IGARCH_demo/IGARCH.md)。

---

## 1. GARCH-M 模型是什么

金融里有个核心直觉——**「高风险要高回报」**（风险-收益权衡）：市场越动荡（波动越大），投资者要求的**预期收益**也越高。普通 GARCH 只让波动随时间变，**均值里却没有波动**；**GARCH-M（GARCH-in-Mean）** 把「今天的条件波动」直接塞进均值方程（"M" = in the mean），让"高波动 → 高预期收益"这件事能被估出来。

**模型（式 3.23）**：

$$r_t=\mu+\underbrace{c\,\sigma_t^2}_{\text{风险溢价}}+a_t,\qquad a_t=\sigma_t\varepsilon_t$$
$$\sigma_t^2=\alpha_0+\sum_{i=1}^{m}\alpha_i a_{t-i}^2+\sum_{j=1}^{s}\beta_j\sigma_{t-j}^2\quad(\text{就是普通 GARCH})$$

- $c$ 叫**风险溢价参数**（risk premium）：$c>0$ ⇒ 收益率与其波动**正相关**（Merton ICAPM 直觉）；$c=0$ 退回普通 GARCH。
- 波动方程与普通 GARCH 完全一样，约束 $\alpha_0>0,\alpha_i\ge0,\beta_j\ge0,\sum(\alpha_i+\beta_j)<1$。

**风险溢价项 $c\cdot g(\sigma_t)$ 有三种常见形式**（书**表 3-2**，本 demo 步骤 4 用 AIC/BIC **三选一**）：

| $g(\sigma_t)$ | 含义 | S-Plus 命令 |
|---|---|---|
| $\sigma_t^2$ | 用条件**方差** | `var.in.mean` |
| $\sigma_t$ | 用条件**标准差** | `sd.in.mean` |
| $\ln\sigma_t^2$ | 用**对数**条件方差 | `logvar.in.mean` |

即文献里也有 $r_t=\mu+c\,\sigma_t+a_t$ 和 $r_t=\mu+c\ln\sigma_t^2+a_t$。本 demo 真值用 $\sigma_t^2$（`var.in.mean`）。

### ⚠️ 一条必须记住的理论性质：风险溢价会给 $r_t$ 制造序列相关

> **（3.23）的 GARCH-M 蕴涵 $r_t$ 存在序列相关，而这种序列相关是由波动率过程 $\{\sigma_t^2\}$ 的序列相关导致的。风险溢价的存在，是历史收益率具有序列相关性的「另一个原因」。**

这句话对**建模流程有直接影响**：观测收益率 $r_t$ 的自相关，**可能不是来自 ARMA 动态，而是被风险溢价项 $c\,\sigma_t^2$ 制造出来的**（$\sigma_t^2$ 高度持续 → 灌进一段缓慢自相关、抬高 AR 根）。所以定均值阶时**不能盲目对着 $r_t$ 的 ACF 硬套 ARMA**，否则会把本该由风险溢价解释的相关性错当成 ARMA。稳妥做法：**先设简约均值**（常数或低阶 AR），把明显的线性相关留到步骤 5 和风险溢价项**一起联合判断**。

### 和普通 GARCH 的三点区别（一句话记忆）

1. **均值里多一项 $c\,g(\sigma_t)$** —— 波动进均值。
2. **必须"联合估计"** —— $\sigma_t^2$ 同时进均值和方差，两方程耦合，**两步法**（先 ARMA 取残差、再对残差建 GARCH）会有偏；`arch` 包不支持 in-mean，故本 demo **手写联合 MLE**（numpy/scipy；R 的 `rugarch` 有 `archm` 选项）。
3. **会给 $r_t$ 制造序列相关** —— 定均值阶时别把风险溢价造出来的相关误当成 ARMA（见上）。

### 阶 $(m,s)$ 不是默认 $(1,1)$

波动方程和普通 GARCH 一样**「阶不好定」**：$m$（ARCH 阶）借 $a_t^2$ 的 **PACF 量级**，$s$（GARCH 阶）书里明说不好定、**只在低阶候选里**靠 **AIC/BIC + 残差诊断**挑。$(1,1)$ 是「选出来的结果」，不是前提——本 demo 因此在低阶网格里**定**出 $(m,s)$，不写死。

**本 demo 真值**：均值 **ARMA(1,1)**（$\phi=0.5,\theta=-0.3,c_{\text{const}}=0$）**+ 风险溢价 $c=0.2$**（in-mean=$\sigma_t^2$）；波动 **GARCH(1,1)-t**（$\omega=0.05,\alpha=0.08,\beta=0.90,\nu=7$，持续度 0.98，无条件方差 2.50）。建模时"假装不知道"，最后揭晓验证。

---

## 2. 完整建模流程（第 0~7 步）

demo 把标准流程串成 **8 步**（对应书里 7 步流程 + 第 0 步造数据），**下游一律用"定出来的阶、选出的风险溢价形式、联合估出的参数"，而不是写死**，换任意数据都能自适应：

| 步骤 | 做什么 | 对应代码 | 产物 |
|---|---|---|---|
| 0 | 造价格 → 对数收益 $r_t$ + 划 train/test | [garchm_demo.py:76-120](garchm_demo.py#L76-L120) | `garchm_returns.csv` |
| 1 | 均值**初步设定**：Ljung-Box → 简约 ARMA 初定阶 → 暂得冲击 $a_t$ | [garchm_demo.py:122-168](garchm_demo.py#L122-L168) | 图1 |
| 2 | 检验 ARCH 效应（有 ARCH 才谈得上 GARCH-M） | [garchm_demo.py:170-182](garchm_demo.py#L170-L182) | — |
| 3 | 波动定阶 $(m,s)$：$a_t^2$ PACF + 低阶二维 AIC/BIC 网格 | [garchm_demo.py:184-220](garchm_demo.py#L184-L220) | 图2 |
| 4 | **选风险溢价形式**（表 3-2 三选一）+ 定新息分布 | [garchm_demo.py:311-334](garchm_demo.py#L311-L334) | — |
| 5 | **联合 MLE** 一次估 ARMA+GARCH+$c$（看风险溢价 $c$） | [garchm_demo.py:336-387](garchm_demo.py#L336-L387) | — |
| 6 | 三查 + 平稳 + 风险溢价 LR 检验（也是定阶裁判） | [garchm_demo.py:389-452](garchm_demo.py#L389-L452) | 图3、图5 |
| 7 | **先波动后均值**预测（均值随波动起伏）→ 区间/VaR | [garchm_demo.py:454-531](garchm_demo.py#L454-L531) | 图4、`garchm_forecast.csv` |

> 与普通 GARCH 的最大流程差异：**第 5 步不再"两步走"**（先 ARMA 取残差、再对残差建 GARCH），而是**联合 MLE 一次估**——因为 in-mean 把 $\sigma_t^2$ 喂回均值，两方程耦合（手写联合 MLE 在 [garchm_demo.py:243](garchm_demo.py#L243) 的 `garchm_filter` + [garchm_demo.py:286](garchm_demo.py#L286) 的 `fit_mle`）。

---

## 3. 各步骤详解 + 本次运行结果

### 步骤 0 · 准备数据
用已知参数递推造 2500 个 ARMA(1,1)-GARCH(1,1)-M-t 收益（风险溢价 $c=0.2$、in-mean=$\sigma_t^2$），前 1500 个为**烧入期（burn-in）丢弃**，单位 ×100 = 百分比。留最后 `h=60` 天做**样本外测试**，其余 2440 天训练。

本次：**ADF p ≈ 0 ≪ 0.05 → 平稳，可建模 ✅**（$n=2500$，train=2440，test=60）。

### 步骤 1 · 均值方程"初步设定"（简约起步）
$r_t$ 有自相关就先用 ARMA 榨干线性依赖（ACF/PACF 都拖尾、读不出阶，故用 **BIC 网格** $p,q\in0..3$）。**但注意**：这里只**初步**定阶、保持简约——$r_t$ 的自相关**一部分来自 in-mean**，真正的阶/参数留到步骤 5 联合估 + 步骤 6 诊断复核。

本次：**$r_t$ 的 Ljung-Box(10) p ≈ 8e−122 → 有自相关**（部分来自 in-mean）。BIC 网格初定 **ARMA(1,1)**（BIC=9049.1）：

```
       q=0       q=1       q=2       q=3
 p=0    --      9206.0    9114.0    9078.4
 p=1  9136.1   9049.1*   9053.5    9060.4     <- (1,1) 全局最小
 p=2  9057.1   9053.3    9061.1    9061.0
 p=3  9053.5   9056.6    9068.6    9073.1
```

> ⚠️ 识别坑：in-mean 的 $c\,\sigma_t^2$ 高度持续，会**给 $r_t$ 灌进缓慢自相关、抬高 AR 根**。demo 选了 $\theta<0$（与 AR 反号、ACF 形态分明）的真值，让 BIC 能干净定回 (1,1)；同号小 MA 容易被并进 AR、定成 (1,0)。配图 [garchm_fig1_mean.png](garchm_fig1_mean.png)。

### 步骤 2 · 检验 ARCH 效应（有 ARCH 才谈得上 GARCH-M）
波动率得先"会动"（$\sigma_t^2$ 可预测），放进均值才有信息。

本次：$a_t^2$ 的 **Q(5)=4.4e−36、Q(10)=5.7e−67、Q(20)=3.3e−93**；Engle **LM=184.55，p=2.7e−34 ≪ 0.05** → **确有 ARCH 效应，值得上 GARCH-M ✅**。

### 步骤 3 · 波动方程定阶 $(m,s)$（不默认 $(1,1)$）
$a_t^2$ 的 ACF/PACF 双双拖尾（图2，ACF 缓慢衰减=高持续指纹；PACF 量级给 $m$ 的直觉），上 $(m,s)$ 二维网格（$m\ge1$；$s=0$ 即纯 ARCH）：

```
       q=0(纯ARCH) q=1       q=2
  p=1   8902.3    8772.6*   8780.4     <- (1,1) 全局最小
  p=2   8869.9    8779.8    8786.1
```

**AIC、BIC 一致选 $(1,1)$**；含 $\beta$ 的 GARCH(1,1) 因高持续完胜纯 ARCH 列——**是选出来的，不是默认**。配图 [garchm_fig2_order.png](garchm_fig2_order.png)。

### 步骤 4 · 选风险溢价形式（表 3-2 三选一）+ 新息分布（本步精华）
in-mean 用 $\sigma_t^2$ / $\sigma_t$ / $\ln\sigma_t^2$ **各估一遍**、比 AIC/BIC 挑（[garchm_demo.py:315-323](garchm_demo.py#L315-L323)，定选阶段用快速拟合省时）：

| 形式 | $g(\sigma_t)$ | AIC | BIC | $\hat c$ |
|---|---|---|---|---|
| **`var.in.mean`** | $\sigma_t^2$ | **8678.2** | **8724.6** ✅ | +0.166 |
| `sd.in.mean` | $\sigma_t$ | 8683.8 | 8730.2 | +0.527 |
| `logvar.in.mean` | $\ln\sigma_t^2$ | 8691.1 | 8737.5 | +0.381 |

**BIC 选 `var.in.mean`（$g=\sigma_t^2$）→ 与真值一致 ✅**（三种形式高度相关、经验上难分辨，但真值形式这里胜出）。新息分布：冲击标准化**峰度=4.59（正态=3）→ 厚尾，取 Student-t**（正态会低估尾部；更尖峰可试 GED）。

### 步骤 5 · 联合估计（手写 MLE：一次估 ARMA + GARCH + $c$）
参数向量 $[c,\phi,\theta,\lambda,\omega,\alpha,\beta,\nu]$ 一次性联合 MLE（数值 Hessian 给标准误/t 值，[garchm_demo.py:341](garchm_demo.py#L341)）：

| 参数 | 估计 | t 值 | 真值 |
|---|---|---|---|
| $c$（常数） | −0.013 | −0.26 | 0.0 |
| $\phi_1$ | +0.631 | 13.01 | 0.5 |
| $\theta_1$ | −0.427 | −7.60 | −0.3 |
| **$\lambda$（风险溢价 $c$）** | **+0.166** | **5.28** | **0.2** |
| $\omega$ | +0.063 | 3.40 | 0.05 |
| $\alpha_1$ | +0.073 | 6.58 | 0.08 |
| $\beta_1$ | +0.902 | 63.85 | 0.90 |
| $\nu$ | 9.11 | 5.64 | 7.0 |

- **风险溢价 $\lambda=0.166$（真值 0.2），t=5.28 ≫ 1.96 → 显著正，高波动确实索取更高预期收益 ✅**；
- 持续度 $\alpha+\beta=0.975$（真值 0.98）< 1 → 平稳；无条件方差 2.48（真值 2.50）；半衰期 ≈27 天；
- $\phi,\theta$ 略偏高（0.63/−0.43 vs 0.5/−0.3）：in-mean 的持续溢价被 AR 根吸收了一部分——**GARCH-M 固有的识别耦合**，不影响 $\lambda$ 与波动参数的回收；
- logL=−4331.1，AIC=8678.2，BIC=8724.6。

### 步骤 6 · 模型检验（三查 + 平稳 + 风险溢价检验，也是定阶裁判）

| 查什么 | 工具 | 本次 | 结论 |
|---|---|---|---|
| ① 均值方程 | $z_t$ Ljung-Box(10) | p=0.636 | 充分 ✅ |
| ② 波动方程 | $z_t^2$ Ljung-Box(10) | p=0.139 | 波动洗净 ✅ |
| ③ 分布 | 偏度/峰度/QQ | 偏度 −0.09, 峰度 4.06 | 仍略厚尾，t 合理 |
| ④ 平稳性 | $\alpha+\beta$ | 0.975 < 1 | 方差过程平稳 ✅ |
| ⑤ **风险溢价** | **LR：$c=0$ vs GARCH-M** | LR=70.75, **p≈4e−17** | **拒绝 $c=0$，溢价显著 ✅** |

第⑤项是 GARCH-M 专属：重估一个**受限模型**（$c=0$，即普通 ARMA-GARCH，[garchm_demo.py:406](garchm_demo.py#L406)），$LR=2[\ell_{\text{M}}-\ell_{0}]\sim\chi^2(1)$。$\log L$ 从 −4366.5 升到 −4331.1，**LR=70.75、p≈4e−17 → in-mean 项确实值得加**。配图 [garchm_fig3_diagnose.png](garchm_fig3_diagnose.png)（条件波动 + 标准化残差四联诊断）、[garchm_fig5_riskpremium.png](garchm_fig5_riskpremium.png)（左=$r_t$ 随 $\sigma_t^2$ 升高而抬升的分箱均值+拟合溢价线；右=高波动时段实际收益的滚动均值也更高）。

### 步骤 7 · 使用模型（预测：先波动、后均值）
GARCH-M 的招牌：**均值预测会"随波动一起均值回复"**，因为 $\mu(\ell)=c+\lambda\,g(\sigma^2(\ell))+\text{ARMA}$。⚠️ **顺序要紧**：均值预测依赖波动预测 $\sigma^2(\ell)$，所以**必须先算波动、再代回均值**（[garchm_demo.py:464](garchm_demo.py#L464) 的循环即先 `s2f[l]` 后 `muf[l]`）。

- **波动率预测（先）**：$\sigma^2(\ell)=\omega+(\alpha+\beta)\sigma^2(\ell-1)$ 几何收敛到无条件水平（$\sigma(1)=1.16\to\sigma(60)=1.49\to 1.57$）；
- **均值点预测（后）**：$\mu(1)=+0.353\to\mu(60)=+0.964$——**不是回到一个固定常数**，而是跟着 $\sigma^2(\ell)$ 一起爬升；其中风险溢价贡献 $\lambda\sigma^2(\ell)$ 从 +0.224 升到 +0.371（向 $\lambda\cdot$无条件方差 +0.412 收敛）；
- **产出**：动态区间 $\mu(\ell)\pm q_t\sigma(\ell)$、VaR（明日 1日99% VaR=2.53%）。样本外 95% 区间覆盖率 **96.7%**（理想≈95%）。

> 见 [garchm_forecast.csv](garchm_forecast.csv)：`mu_hat`(含溢价的中心) / `premium`(其中 $\lambda g(\sigma^2(\ell))$) / `mu_no_premium`(去掉溢价的纯 ARMA 漂移对照) / `sigma_hat` / `lo95,hi95` / `VaR99` / `actual`。**`mu_hat` 与 `mu_no_premium` 的差就是风险溢价对预测的贡献——它随波动放大**（图4 [garchm_fig4_forecast.png](garchm_fig4_forecast.png)：蓝实线高于绿虚线，且差距随波动上行而拉大）。

---

## 4. 书中实例：标普 500 月超额收益 GARCH(1,1)-M

Tsay §3.7 的实例正好把"$c$ 要验显著性、不显著就别硬上"讲透。数据：**S&P 500 月超额收益率 1926.1–1991.12，高斯新息**。拟合结果：

$$r_t=\underset{(0.0023)}{0.0055}+\underset{(0.818)}{1.09}\,\sigma_t^2+a_t,\qquad
\sigma_t^2=\underset{(2.51\text{e−}5)}{8.76\text{e−}5}+\underset{(0.0205)}{0.123}\,a_{t-1}^2+\underset{(0.0196)}{0.849}\,\sigma_{t-1}^2$$

（括号内为标准误；S-Plus 命令 `sp.fit = garch(sp ~ 1 + var.in.mean, ~ garch(1,1))`。）

- 风险溢价 $\hat c=1.09$ **为正**，但 $t=1.09/0.818\approx1.33<1.96$ → **在 5% 水平下不显著**——**这批数据里"高波动 → 高收益"的证据不强**。这正是"$c$ 不显著就退回普通 GARCH"的现实注脚（与本 demo 造数据 $\lambda$ 显著、恰成对照）。
- 波动持续度 $0.123+0.849=0.972$，仍是高持续 GARCH(1,1)。
- 风险溢价的思想也可套到别的模型：如 §3.8 Nelson 的 **AR(1)-EGARCH(2,2)-M**（均值里就带 $c\,\sigma_t^2$，见 [EGARCH.md](../EGARCH_demo/EGARCH.md)）。

---

## 5. 关键概念速查（GARCH ↔ GARCH-M ↔ EGARCH 对照）

| 维度 | GARCH(p,q) | **GARCH-M** | EGARCH |
|---|---|---|---|
| 解决什么 | 波动聚集（高持续） | **风险溢价（波动→收益）** | 杠杆效应（涨跌不对称） |
| 动了哪个方程 | 波动方程 | **均值方程（+$c\,g(\sigma_t)$）** | 波动方程（对 $\ln\sigma^2$） |
| 关键新参数 | $\alpha,\beta$ | **$c$（风险溢价）** | $\gamma$（杠杆） |
| 怎么验它值不值 | $a_t^2$ 有 ARCH 效应 | **$c$ 的 t / LR 检验** | $\gamma$ 显著 / AIC 比对称 |
| 估计方式 | 两步法（ARMA→残差→GARCH） | **联合 MLE（耦合，不能分步）** | 两步法 |
| 对 $r_t$ 自相关 | 只由 ARMA 给 | **ARMA + $\sigma_t^2$ 都贡献** | 只由 ARMA 给 |
| 预测特征 | 波动几何收敛；均值回固定常数 | **均值随波动一起回复** | 波动用 NIC 非对称收敛 |

- **in-mean 用 $\sigma_t^2$ / $\sigma_t$ / $\ln\sigma_t^2$**（表 3-2 三选一）：换形式只改均值那一项，$c$ 的量纲与解释随之变——本 demo 步骤 4 用 AIC/BIC 选。
- **持续度看 $\alpha+\beta$**：GARCH-M 的波动方程就是普通 GARCH，持续度还是 $\alpha+\beta$。
- **$r_t$ 的自相关不全是 ARMA**：持续的 $c\,\sigma_t^2$ 也贡献一段缓慢自相关（GARCH-M 独有）。
- **均值预测"随波动起伏"**：普通 ARMA 多步预测回固定常数；GARCH-M 的 $\mu(\ell)$ 跟着 $\sigma^2(\ell)$ 一起均值回复。

---

## 6. 文件清单与运行

```
garchm_demo.py             # 主程序（全流程一个文件，含手写联合 MLE）
requirements.txt           # 依赖：numpy / scipy / pandas / matplotlib / statsmodels / arch
garchm_returns.csv         # 步骤0导出的收益（列：t, logret_pct, split）
garchm_forecast.csv        # 步骤7每日预测（mu/premium/mu_no_premium/sigma/区间/VaR/actual）
garchm_fig1_mean.png       # r_t 序列 + ACF/PACF（均值结构，部分来自 in-mean）
garchm_fig2_order.png      # a_t² 的 ACF/PACF（双拖尾、缓慢衰减=高持续）
garchm_fig3_diagnose.png   # 条件波动 + 标准化残差四联诊断
garchm_fig4_forecast.png   # 样本外：含/不含溢价的均值对照 + 动态区间；波动/溢价/VaR 回复
garchm_fig5_riskpremium.png# 风险溢价招牌图：r_t 随 σ² 抬升 + 高波动时段平均收益更高
```

运行：

```bash
pip install -r requirements.txt
python garchm_demo.py
```

---

## 7. 一句话总结

> **GARCH-M = 在 GARCH 上把条件波动塞进均值方程（$+c\,g(\sigma_t)$）= 用一个参数 $c$ 刻画"高风险高回报"的风险溢价；$c=0$ 即退回 GARCH。** 完整流程是：
> **先验平稳（ADF）→ 简约设均值（注意 in-mean 会给 $r_t$ 造自相关、抬高 AR 根）→ 查 $a_t^2$ 有无聚集 → $(m,s)$ 网格定波动阶 → 表 3-2 三选一定风险溢价形式 → 联合 MLE 一次估 ARMA+GARCH+$c$ → 三查/平稳 + 风险溢价 LR/t 检验 → 先波动后均值，产出"随波动起伏的中心 + 动态波动"的区间与 VaR。**
> 本 demo 从一个真值为 ARMA(1,1)+GARCH(1,1)-t+风险溢价 $c=0.2$（`var.in.mean`）的序列出发，BIC 干净定回 ARMA(1,1)/GARCH(1,1)、**步骤 4 三选一选回 `var.in.mean`**、**联合 MLE 估出 $\lambda=0.166$（t=5.28）、LR=70.75（p≈4e−17）确认风险溢价显著存在**，波动参数（$\alpha=0.073,\beta=0.902$）接近真值、三查全过，预测给出一条会"随波动一起爬升"的均值与动态波动带——GARCH 加一个 in-mean 项后的一次教科书式风险溢价闭环。而书里标普 500 的 $\hat c=1.09$（t≈1.33）**不显著**，恰好从反面提醒：$c$ 得验，验不过就老实用 GARCH。

### 易错点提醒
1. **把 GARCH-M 当 GARCH 两步估**——in-mean 耦合两方程，必须联合 MLE，分步会让 $c$ 有偏（`arch` 包不支持 in-mean，本 demo 手写 MLE）。
2. **不验 $c$ 显著性就上 GARCH-M**——先用 LR/t；不显著（如书里标普 500 例）就退回 GARCH。
3. **以为 $r_t$ 的自相关全是 ARMA**——持续的 $c\,\sigma_t^2$ 也贡献自相关、干扰定阶、抬高 $\phi$；故均值要"简约起步"。
4. **以为多步均值预测会回到固定常数**——GARCH-M 的 $\mu(\ell)$ 随 $\sigma^2(\ell)$ 一起均值回复；且预测要"先波动、后均值"。
5. **in-mean 形式混用**（$\sigma_t^2$ vs $\sigma_t$ vs $\ln\sigma_t^2$）——$c$ 的量纲与解释不同，报告时要讲清（本 demo 用 AIC/BIC 选出 `var.in.mean`）。
6. **把 GARCH-M 的阶默认写死 $(1,1)$**——波动阶要靠 $a_t^2$ PACF + 低阶 AIC/BIC 网格定；样本还比 GARCH 多一个 $c$ 更吃样本，日频建议 ≥1500。
