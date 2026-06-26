# GARCH 模型学习总结

> 配套 demo：[garch_demo.py](garch_demo.py) —— 一条龙完整流程（造价格 → 对数收益 → ARMA 均值 → ARCH 效应检验 → (p,q) 网格定阶 → MLE 估计 → 标准化残差三查 + 平稳性 → 动态区间/VaR 预测）。
> 本文把 GARCH 的核心理论和 demo 的每一步对应起来，并附上本次真实运行的数字。

---

## 1. GARCH 模型是什么

金融收益率有个顽固现象：**波动会扎堆**——大涨大跌之后往往还跟着大波动，平静之后往往还是平静（**波动率聚集 volatility clustering**）。ARCH 用「过去若干个冲击的平方」解释今天的波动，但真实市场波动**记忆很长**（高持续），纯 ARCH 要堆很高的阶才抓得住。**GARCH 在 ARCH 上多加一项 $\beta\sigma_{t-1}^2$**——把「昨天估出来的方差」也接进来：

$$
\text{ARCH}(m):\ \sigma_t^2=\alpha_0+\sum_{i=1}^m\alpha_i a_{t-i}^2 \qquad
\text{GARCH}(p,q):\ \sigma_t^2=\omega+\sum_{i=1}^p\alpha_i a_{t-i}^2+\sum_{j=1}^q\beta_j\sigma_{t-j}^2
$$

- **GARCH(1,1) = 无穷阶 ARCH**：把递推展开 $\sigma_t^2=\frac{\omega}{1-\beta}+\alpha\sum_{k\ge0}\beta^k a_{t-1-k}^2$，用 **3 个参数**就刻画了高持续的长记忆波动，所以实务里 **GARCH(1,1) 几乎是默认起点**，远比高阶 ARCH 常用。
- **核心分工**：和 ARCH 一样，**GARCH 不碰均值**（一阶矩交给常数/ARMA），只给**方差（二阶矩）随时间的变化**建模，研究对象是去均值后的冲击 $a_t=r_t-\mu_t$。
- **关键认知**：$a_t$ **不相关但不独立**——线性层面（$a_t$）白噪声，平方层面（$a_t^2$）有结构；GARCH 靠 $\beta$ 项抓住它的**长记忆**。
- **三个灵魂读数**：**持续度** $\alpha+\beta$（越近 1 记忆越久）、**半衰期** $\ln0.5/\ln(\alpha+\beta)$、**无条件方差** $\omega/(1-\alpha-\beta)$（要求 $\alpha+\beta<1$）。
- 本 demo **真值**：均值 **ARMA(1,1)** $\phi=0.5,\theta=0.3,\mu=0.05$；波动 **GARCH(1,1)-t** $\omega=0.04,\alpha=0.09,\beta=0.90,\nu=7$（**持续度 0.99，极高持续**）。建模时"假装不知道"，最后揭晓验证。

### 建模前提（喂进去之前先验收）

| 要求 | 怎么查 | 不满足怎么办 |
|---|---|---|
| **平稳** | ADF（`adfuller`，[garch_demo.py:83](garch_demo.py#L83)） | 价格 → 对数收益率 $r_t$ |
| **均值已建模** | $r_t$ 的 Ljung-Box | 有自相关 → 先上 **ARMA**，再对残差建波动 |
| **有 ARCH 效应** | $a_t^2$ 的 Q(m) / Engle LM | 不显著 → 常数方差即可，不用 GARCH |
| **持续度 < 1** | 估完看 $\alpha+\beta$ | ≥1 = IGARCH/方差爆炸，分段或换模型 |
| **厚尾用分布兜住 + 样本足够** | 峰度/QQ；计数 | 新息选 t/偏t/GED；GARCH 多估一个 $\beta$ 更吃样本（日频建议 ≥1000） |

> 一句话：喂进 GARCH 的应是——**平稳、已去（ARMA）均值结构、且残差确有波动聚集的（百分比）对数收益率**。

---

## 2. 完整建模流程（第 0~6 步）

demo 把波动率建模串成 7 步，**下游一律用"定出来的阶 `best_gorder`、自动判出的均值 `mean_desc`"，而不是写死**，换数据自适应：

| 步骤 | 做什么 | 对应代码 | 产物 |
|---|---|---|---|
| 0 | 造价格 → 对数收益 $r_t$ + 划 train/test | [garch_demo.py:42-84](garch_demo.py#L42-L84) | `garch_returns.csv` |
| 1 | ARMA 均值（BIC 网格）→ 得冲击 $a_t$ | [garch_demo.py:87-136](garch_demo.py#L87-L136) | 图1 |
| 2 | 检验 ARCH 效应（要不要建波动模型） | [garch_demo.py:139-149](garch_demo.py#L139-L149) | — |
| 3 | 定阶 $(p,q)$（二维 AIC/BIC 网格） | [garch_demo.py:152-194](garch_demo.py#L152-L194) | 图2 |
| 4 | MLE 估参 + 报持续度/半衰期 | [garch_demo.py:197-208](garch_demo.py#L197-L208) | — |
| 5 | 标准化残差三查 + 平稳性 | [garch_demo.py:217-243](garch_demo.py#L217-L243) | 图3 |
| 6 | 均值 + 动态波动 预测 → 区间/VaR | [garch_demo.py:246-288](garch_demo.py#L246-L288) | 图4 |

---

## 3. 各步骤详解 + 本次运行结果

### 步骤 0 · 准备数据
用已知参数递推造 3000 个 GARCH(1,1)-t 冲击、灌进 ARMA(1,1) 均值，前 1000 个为**烧入期（burn-in）丢弃**，得 2000 个对数收益率（单位 ×100 = 百分比）。留最后 `h=60` 个做样本外测试，其余 1940 个训练。

本次：**ADF p ≈ 0 ≪ 0.05 → 平稳，可建模 ✅**（样本 $n=2000$，train=1940，test=60）。

### 步骤 1 · 均值方程（管"中心"，用 ARMA）
先对 $r_t$ 做 **Ljung-Box(10)**；有自相关就用 **ARMA** 建 $\mu_t$。ARMA 的 ACF/PACF 双双拖尾、读不出阶，故用 **BIC 网格**（$p,q\in0..3$）定阶（[garch_demo.py:94-108](garch_demo.py#L94-L108)）。

本次：**$r_t$ 的 Ljung-Box(10) p = 2.2e−229 ≪ 0.05 → 有自相关 → 走 ARMA 分支**。BIC 网格选出 **ARMA(1,1)**（BIC=7338.7）：

```
       q=0       q=1       q=2       q=3
 p=0    --      7556.8    7359.7    7348.2
 p=1   7447.7   7338.7*   7342.5    7347.7      <- (1,1) 全局最小
 p=2   7344.3   7343.7    7349.2    7351.0
 p=3   7341.5   7345.6    7352.0    7358.5
```

拟合参数 vs 真值（$\hat\phi=0.477$≈0.5、$\hat\theta=0.345$≈0.3，接近）：

| 参数 | 估计值 | 真值 |
|---|---|---|
| const | −0.073 | 0.05 |
| ar.L1 ($\phi$) | +0.477 | 0.5 |
| ma.L1 ($\theta$) | +0.345 | 0.3 |

得冲击 $a_t=r_t-\hat\mu_t$。配图 [garch_fig1_mean.png](garch_fig1_mean.png)：$r_t$ 序列（肉眼可见波动聚集）+ ACF + PACF。

> **⚠️ 工程坑**：**强 GARCH 异方差会让高斯-BIC 偏向多估 ARMA 阶**（似然在同方差假设下算）。demo 选了能干净定回 (1,1) 的随机种子；实务中可对 $a_t$ 再核对第 5 步均值检验①是否过关来兜底。
> **工具说明（两步法）**：`arch` 包均值只支持 AR、不支持 MA，所以含 MA 的 ARMA 走"两步法"——statsmodels 拟合 ARMA 取残差 $a_t$ → `arch` 对 $a_t$ 建 GARCH（`mean="Zero"`）。完整联合估计可用 R 的 `rugarch`。

### 步骤 2 · 检验 ARCH 效应（要不要建波动率模型）
对 $a_t^2$ 查波动聚集，**两法互证**（[garch_demo.py:142-149](garch_demo.py#L142-L149)）：
- **方法1**：$a_t^2$ 的 **Ljung-Box $Q(m)$**——$a_t$ 不相关但 $a_t^2$ 若相关，就是波动聚集的直接证据；
- **方法2**：**Engle LM 检验**——辅助回归 $a_t^2$ 对滞后 $a_{t-i}^2$，$LM=nR^2\sim\chi^2_m$。

本次：$a_t^2$ 的 Ljung-Box **Q(5)=1.48e−100、Q(10)=9.26e−135、Q(20)=3.27e−196**；Engle **LM=301.26，p=8.43e−59**（全 ≪ 0.05）→ **确有 ARCH 效应，需建波动率模型 ✅**。

### 步骤 3 · 定阶 (p,q)（本步精华：二维网格）

> **为什么 GARCH 用二维网格、而不是 PACF 截尾**：和 ARCH 的「$a_t^2$ 服从 AR($m$)」类似，令 $\eta_t=a_t^2-\sigma_t^2$（鞅差白噪声），代入 GARCH 定义可得
> $$a_t^2=\omega+\sum_{i=1}^{\max(p,q)}(\alpha_i+\beta_i)a_{t-i}^2+\eta_t-\sum_{j=1}^q\beta_j\eta_{t-j}$$
> ——**$a_t^2$ 服从 ARMA($\max(p,q),\,q$)**（$q$=$\beta$ 阶=MA 部分阶）！加了 $\beta$ 后多出 MA 部分，于是 **ACF 和 PACF 双双拖尾**（配图2 的 ACF 缓慢衰减正是"高持续"的指纹），读不出阶。所以和 ARMA 一样上 **(p,q) 二维 AIC/BIC 网格**（$q=0$ 那列即退化为纯 ARCH）。

同 ARCH/ARMA，先给 **Schwert 参考上限** $\lfloor 12(T/100)^{1/4}\rfloor$——只依赖样本量（本例 $T=1940\Rightarrow$ **25**）。但 **GARCH 高阶 MLE 既难收敛、实证上也极少胜过低阶**（GARCH(1,1) 是铁律默认），所以二维网格**诚实截到 `GRIDMAX=3` 并显式标注**（[garch_demo.py:157-160](garch_demo.py#L157-L160)，共 12 个拟合），不是偷偷缩小、也不靠"凑个小数"；若最优贴到上限会自检告警（[garch_demo.py:186](garch_demo.py#L186)）。

本次（行=$\alpha$ 阶 p，列=$\beta$ 阶 q）：

```
       q=0       q=1       q=2       q=3
 p=1   6847.8   6647.4*   6655.0    6662.6      <- (1,1) 最小
 p=2   6821.5   6654.7    6661.3    6668.9
 p=3   6784.7   6661.0    6667.3    6674.9
```

**AIC、BIC 一致选 (1,1)**，没贴到上限 3（自检未触发）。

> **看点（$\beta$ 项的价值）**：**q=0 那一整列就是纯 ARCH**——BIC 随 p 一路下降（6847.8→6821.5→6784.7，说明高持续逼着纯 ARCH 不停加阶），可**始终远逊于 GARCH(1,1)=6647.4**。含 $\beta$ 的 GARCH 用**一个参数**干掉了高阶 ARCH，这就是"GARCH(1,1) 几乎是默认起点"的实证。

配图 [garch_fig2_order.png](garch_fig2_order.png)：左 = $a_t^2$ 的 ACF（缓慢衰减→高持续），右 = PACF（也拖尾→只能上网格）。

### 步骤 4 · 估计参数（MLE + 持续度/半衰期）
**怎么估的（为什么不是 OLS）**：波动方程里 $\sigma_t^2$ 不可观测、且逐期递推（GARCH 还自引用 $\sigma_{t-1}^2$），没法 OLS。做法是写**似然**：设 $a_t=\sigma_t\varepsilon_t$、$\varepsilon_t\sim$ 标准化 Student-t($\nu$)，给定参数递推出每个 $\sigma_t^2$、算密度连乘得 $L$，**MLE 数值最大化 $\ln L$**（[garch_demo.py:200](garch_demo.py#L200)）。用 t 不用正态，因金融残差厚尾。

本次估计 vs 真值：

| 参数 | 估计值 | 真值 |
|---|---|---|
| $\omega$ | 0.0269 | 0.04 |
| $\alpha_1$ | 0.0686 | 0.09 |
| $\beta_1$ | 0.9200 | 0.90 |
| $\nu$ | 6.06 | 7.0 |

- **持续度 $\alpha+\beta=0.9886$**（真值 0.99）< 1 → 协方差平稳 ✅；
- **无条件波动率 $\sqrt{\omega/(1-\alpha-\beta)}=1.535$**（波动长期回复到的水平）；
- **半衰期 $=\ln0.5/\ln0.9886\approx 60$ 天**——持续度越接近 1，半衰期越长、波动记忆越久（$\alpha+\beta\to1$ 时半衰期 $\to\infty$，即 IGARCH）。这正是纯 ARCH 难抓、必须 GARCH 的原因。
- logL=−3308.6，AIC=6625.2，BIC=6647.4。

### 步骤 5 · 模型检验（标准化残差三查 + 平稳性）
算 $\tilde a_t=a_t/\sigma_t$——模型对的话应是 i.i.d. 标准分布。**三查 + 一查平稳**（[garch_demo.py:215-223](garch_demo.py#L215-L223)）：

| 查什么 | 工具 | 本次 | 结论 |
|---|---|---|---|
| ① 均值方程 | $\tilde a_t$ Ljung-Box(10) | p=1.00 | 充分 ✅（不过关 → 改 ARMA 阶） |
| ② 波动方程 | $\tilde a_t^2$ Ljung-Box(10) | p=0.699 | 波动洗净 ✅（不过关 → 升阶/换模型） |
| ③ 分布 | 偏度/峰度/QQ | 偏度 −0.46，峰度 8.66 | 仍厚尾，t 合理 |
| ④ 平稳性 | $\alpha+\beta$ | 0.9886 < 1 | 方差过程协方差平稳 ✅ |

**四项全过 → GARCH(1,1) 模型充分**。配图 [garch_fig3_diagnose.png](garch_fig3_diagnose.png)：条件波动 $\sigma_t$（围绕无条件水平起伏）、$\tilde a_t^2$ 的 ACF（落带内=波动洗净）、直方图、Q-Q（显厚尾）。

### 步骤 6 · 使用模型（预测）
- **点预测（中心）**：$\mu_{t+\ell}$ 由 ARMA 给 → 逐步**均值回复**到长期均值；
- **波动率预测**：递推 $\sigma_t^2(\ell)=\omega+(\alpha+\beta)\sigma_t^2(\ell-1)$，每步把方差按 $(\alpha+\beta)$ 向**无条件方差** $V=\omega/(1-\alpha-\beta)$ 拉近——**几何收敛**，速度由持续度决定。**这是方差版的均值回复**：ARMA 点预测收敛到 $\mu$，GARCH 波动预测收敛到 $V$；持续度 0.99 极高 → 收敛很慢（半衰期 60 天）。
- **产出**：动态预测区间 $\mu\pm q\cdot\sigma(\ell)$、**VaR**（分布尾分位）。

本次：均值 $\mu(1)=-0.670\to\mu(60)=-0.073$（**ARMA 逐步回复**）；波动 $\sigma(1)=0.834\to\sigma(60)=1.230$（**向无条件水平 1.535 几何收敛**）；1 日 99% VaR 明日 = **2.808%**；**样本外 95% 区间覆盖率 = 86.7%**（60 点小样本，理想≈95%）。

> 见 [garch_forecast.csv](garch_forecast.csv)：`sigma_hat` 起点低、逐日向无条件水平爬升——**这条"波动均值回复曲线"是 GARCH 区别于固定方差/纯 ARCH 的招牌**。配图 [garch_fig4_forecast.png](garch_fig4_forecast.png)：上 = 实际 vs ARMA 中心 + 95% 动态区间；下 = 波动 $\sigma(\ell)$ 向无条件水平爬升 + 99% VaR。

---

## 4. 三模型对照（ARMA ↔ ARCH ↔ GARCH）

| 维度 | ARMA(p,q) | ARCH(m) | **GARCH(p,q)** |
|---|---|---|---|
| 建模对象 | 均值（一阶矩） | 方差（二阶矩） | 方差（二阶矩） |
| 方程右边 | 过去值 + 过去误差 | 过去冲击平方 | **过去冲击平方 + 过去方差** |
| 抓什么 | 序列惯性/滞后冲击 | 波动聚集 | **长记忆波动聚集（高持续）** |
| $a_t^2$ 的结构 | — | AR($m$)：PACF 截尾 | **ARMA：ACF/PACF 双拖尾** |
| 定阶 | (p,q) AIC/BIC 网格 | $a_t^2$ PACF 截尾 + 1D 网格 | **(p,q) 二维 AIC/BIC 网格** |
| 关键读数 | $\mu=c/(1-\sum\phi)$ | 各 $\alpha_i$ | **持续度 $\alpha+\beta$、半衰期** |
| 多步预测 | 均值回复到 $\mu$ | 波动几步即近无条件 | **波动几何收敛到 $V$（慢，因高持续）** |
| 简约性 | 小 p+q 替代高阶 AR/MA | 高持续需很高 m | **GARCH(1,1) 即可替代高阶 ARCH** |

本 demo = **ARMA（均值）+ GARCH（波动）合体**：ARMA 管中心，GARCH 管波动的宽度。

---

## 5. 关键概念速查

- **持续度 $\alpha+\beta$ 是 GARCH 的灵魂**：决定半衰期、决定波动预测向无条件水平收敛的快慢；越近 1 越"黏"，=1 即 IGARCH（冲击永不消散）。
- **$a_t^2$ 服从 ARMA**：所以 GARCH 定阶 ACF/PACF 都拖尾、读不出阶，只能上 (p,q) 二维网格（ARCH 是 AR、PACF 截尾，这是两者定阶法不同的根源）。
- **均值 OLS、波动 MLE**：$\sigma_t^2$ 不可观测且递推，只能写似然 MLE；分布用厚尾 t。
- **不相关 ≠ 独立**：$a_t$ 线性白噪声、$a_t^2$ 有结构——GARCH 存在的理由。
- **方差版均值回复**：多步波动预测几何收敛到 $V=\omega/(1-\alpha-\beta)$。
- **ARCH vs GARCH**：高持续波动用低阶 ARCH 抓不住，GARCH 加一个 $\beta$ 项就够、更简约（本 demo 纯 ARCH 列完败即实证）。
- **每步门槛都是一个 p 值/准则**：ADF<0.05、$r_t$ LB<0.05 上 ARMA、$a_t^2$ Q/LM<0.05 有 ARCH、$\tilde a_t$/$\tilde a_t^2$ LB>0.05 洗净、$\alpha+\beta<1$ 平稳。

---

## 6. 文件清单与运行

```
garch_demo.py            # 主程序（全流程一个文件）
requirements.txt         # 依赖：numpy / scipy / pandas / matplotlib / statsmodels / arch
garch_returns.csv        # 步骤0导出的对数收益（列：t, logret_pct, split）
garch_forecast.csv       # 步骤6导出的每日预测（中心/波动/区间/VaR/实际/是否命中）
garch_fig1_mean.png      # r_t 序列 + ACF/PACF（均值结构 → ARMA）
garch_fig2_order.png     # a_t² 的 ACF/PACF（双拖尾、缓慢衰减=高持续）
garch_fig3_diagnose.png  # 条件波动 + 标准化残差四联诊断
garch_fig4_forecast.png  # 样本外：ARMA 中心 + GARCH 动态区间 + 波动回复/VaR
```

运行：

```bash
pip install -r requirements.txt
python garch_demo.py
```

---

## 7. 一句话总结

> **GARCH = 在 ARCH 上加一个 $\beta\sigma_{t-1}^2$（把昨天的方差接进来）= 用 3 个参数刻画高持续的长记忆波动；均值交给 ARMA，波动交给 GARCH。** 完整流程是：
> **先验平稳（ADF）→ ARMA 建均值得冲击 $a_t$ → 查 $a_t^2$ 有无聚集（ARCH 效应）→ (p,q) 网格定阶（GARCH(1,1) 常胜出）→ t 分布 MLE、读持续度/半衰期 → 标准化残差三查 + 平稳性 → 用 中心+动态波动 产出区间与 VaR。**
> 本 demo 从真值为 ARMA(1,1)+GARCH(1,1)-t、持续度 0.99 的序列出发，BIC 把均值定回 (1,1)、把波动定回 GARCH(1,1)，参数（$\alpha=0.069,\beta=0.92$）接近真值，三查全过、半衰期约 60 天印证高持续，预测给出会"均值回复"的动态波动带——ARMA 与 GARCH 合体的一次教科书式闭环。

### 易错点提醒
1. **不去均值就直接建波动**——GARCH 研究的是冲击 $a_t=r_t-\mu_t$，均值有自相关要先上 ARMA。
2. **强 GARCH 异方差会扰乱 ARMA 定阶**——高斯-BIC 偏向多估阶，需以第 5 步均值检验①兜底。
3. **只看 $\alpha$ 不看 $\alpha+\beta$**——持续度才是核心，决定记忆长短与收敛快慢。
4. **持续度 ≥ 1 还硬套 GARCH**——已是 IGARCH/方差不收敛，应分段或换模型。
5. **样本太少就估 GARCH**——多一个 $\beta$ 更吃样本，日频建议 ≥1000。
6. **搜索上限拍脑袋写死**——上限应随样本量走（Schwert），高阶 MLE 不收敛要诚实截断 + 标注，并自检有没有贴上限。
7. **②不过关不迭代**——残差平方仍有结构 = 该升阶或换 EGARCH/GJR 等；别停在第一个模型。
</content>
