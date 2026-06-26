# ARCH / GARCH 模型学习总结

> 配套 demo：[arch_demo.py](arch_demo.py) —— 一条龙完整流程（造价格 → 对数收益 → 均值方程<自动选 常数/ARMA> → ARCH 效应检验 → 定阶 → MLE 估计 → 标准化残差三查 → 动态区间/VaR 预测）。
> 本文把 ARCH/GARCH 的核心理论和 demo 的每一步对应起来，并附上本次真实运行的数字。
> 主线用默认开关 `MEAN_KIND="arma"`；文末第 4 节给出 `"const"` 分支的对照（含 ARCH→GARCH 迭代）。

---

## 1. ARCH / GARCH 模型是什么

金融收益率有个顽固现象：**波动会扎堆**——大涨大跌之后往往还跟着大波动，平静之后往往还是平静（**波动率聚集 volatility clustering**）。普通模型假设方差恒定（同方差），抓不住这个。**ARCH/GARCH 就是给"方差"本身建了一个随时间变化的模型**：用过去冲击的大小来预测今天的波动有多大。

$$
\text{ARCH}(m):\quad \sigma_t^2=\alpha_0+\alpha_1 a_{t-1}^2+\dots+\alpha_m a_{t-m}^2
$$

$$
\text{GARCH}(1,1):\quad \sigma_t^2=\omega+\alpha\,a_{t-1}^2+\beta\,\sigma_{t-1}^2
$$

- **核心分工**：一阶矩（均值/中心 $\mu_t$）交给常数或 ARMA；**ARCH 专管二阶矩（方差/波动 $\sigma_t^2$）随时间的变化**，研究对象是去掉均值后的冲击 $a_t=r_t-\mu_t$。
- **关键认知**：$a_t$ **不相关但不独立**——线性层面（$a_t$）是白噪声，平方层面（$a_t^2$）有结构。ARCH 抓的就是藏在方差里的这种"不独立"。
- **GARCH 为什么更常用**：多了 $\beta\sigma_{t-1}^2$（把"昨天的方差"也纳进来），等价于无穷阶 ARCH，能用很少参数刻画**长记忆/高持续**的波动，所以实务里 GARCH(1,1) 比高阶 ARCH 更常用。
- 本 demo **波动真值**：**GARCH(1,1)-t**，$\omega=0.05,\alpha=0.10,\beta=0.86$（$\alpha+\beta=0.96$，**高持续**），新息自由度 $\nu=7$（厚尾）；**均值真值**（默认分支）：**ARMA(1,1)** $\phi=0.6,\theta=0.3,\mu=0.05$。建模时"假装不知道"这些，最后揭晓验证。

### 建模前提

| 要求 | 怎么查 | 不满足怎么办 |
|---|---|---|
| **平稳** | ADF（`adfuller`，[arch_demo.py:86](arch_demo.py#L86)） | 价格 → 对数收益率 $r_t$ |
| **均值已建模** | $r_t$ 的 Ljung-Box | 有自相关 → 先上 ARMA，再对残差建波动 |
| **有 ARCH 效应** | $a_t^2$ 的 Q(m) / Engle LM | 不显著 → 常数方差即可，不用 ARCH |
| **厚尾用分布兜住** | 峰度 / QQ | 新息选 t / 偏t / GED |
| **样本足够 + 量纲适中** | 计数 / 看数值 | 太短不收敛；价格 ×100 转百分比 |

> 一句话：喂进 ARCH 的应是——**平稳、已去均值结构、且残差确有波动聚集的（百分比）对数收益率**。

---

## 2. 完整建模流程（第 0~6 步）

demo 把波动率建模的标准流程串成 7 步，**下游一律用"自动判别出的均值类型 `mean_desc`、定出来的阶 `m_hat`、检验后选中的模型 `best_name`"，而不是写死**，所以换任意数据都能自适应：

| 步骤 | 做什么 | 对应代码 | 产物 |
|---|---|---|---|
| 0 | 造价格 → 对数收益 $r_t$ + 划 train/test | [arch_demo.py:42-87](arch_demo.py#L42-L87) | `arch_returns.csv` |
| 1 | 均值方程：Ljung-Box 自动选 常数/ARMA → 得冲击 $a_t$ | [arch_demo.py:89-150](arch_demo.py#L89-L150) | 图1 |
| 2 | 检验 ARCH 效应（要不要建波动模型） | [arch_demo.py:152-163](arch_demo.py#L152-L163) | — |
| 3 | 定阶 $m$（PACF 截尾 + AIC/BIC 网格） | [arch_demo.py:165-194](arch_demo.py#L165-L194) | 图2 |
| 4 | MLE 估参（ARCH($m$) on $a_t$，Student-t） | [arch_demo.py:196-200](arch_demo.py#L196-L200) | — |
| 5 | 标准化残差三查 → 不过关就迭代（ARCH→GARCH） | [arch_demo.py:204-243](arch_demo.py#L204-L243) | 图3 |
| 6 | 均值 + 动态波动 预测 → 区间 / VaR | [arch_demo.py:245-289](arch_demo.py#L245-L289) | 图4 |

---

## 3. 各步骤详解 + 本次运行结果

### 步骤 0 · 准备数据
用已知参数递推造 2500 个 GARCH(1,1)-t 冲击，前 1000 个为**烧入期（burn-in）丢弃**（消除初始值影响、进入稳态），叠加均值结构后得 1500 个对数收益率（单位 ×100 = 百分比，量纲适中、`arch` 包推荐做法）。留最后 `h=60` 个做**样本外测试**，其余 1440 个做训练。

本次：**ADF p = 3.67e−29 ≪ 0.05 → 平稳，可建模 ✅**（样本 $n=1500$，train=1440，test=60）。

### 步骤 1 · 均值方程（管"中心"）
**方法**：先对 $r_t$ 做 **Ljung-Box(10)** 看有没有自相关，**自动分支**（不靠开关作弊，[arch_demo.py:94-96](arch_demo.py#L94-L96)）：
- **无自相关** → $\mu_t$ 取**常数**（金融日收益常见）；
- **有自相关** → 用 **ARMA** 建 $\mu_t$。ARMA 的 ACF/PACF 都拖尾、读不出阶，所以用 **BIC 网格搜索**（遍历 $p,q\in0..3$ 取 BIC 最小）定阶。

两条路都得到冲击 $a_t=r_t-\mu_t$，下面专门研究它的波动。

本次（arma 分支）：**$r_t$ 的 Ljung-Box(10) p = 2.66e−272 ≪ 0.05 → 有自相关 → 走 ARMA 分支**。BIC 网格选出 **ARMA(1,1)**（BIC=4238.0）：

```
       q=0       q=1       q=2       q=3
 p=0    --      4554.8    4344.0    4282.9
 p=1   4271.3   4238.0*   4245.3    4252.1     <- (1,1) 全局最小
 p=2   4239.9   4245.3    4252.5    4259.2
 p=3   4245.8   4252.4    4259.4    4260.7
```

拟合参数 vs 真值（都很接近 → 均值结构抓对了）：

| 参数 | 估计值 | 真值 |
|---|---|---|
| const | +0.082 | 0.05 |
| ar.L1 ($\phi$) | +0.633 | 0.6 |
| ma.L1 ($\theta$) | +0.228 | 0.3 |

配图 [arch_fig1_mean.png](arch_fig1_mean.png)：$r_t$ 序列 + ACF + PACF。

> **工具说明（两步法）**：Python 的 `arch` 包均值只支持 AR、不支持 MA，所以含 MA 的 ARMA 走"两步法"——先用 statsmodels 拟合 ARMA 取残差 $a_t$，再用 `arch` 对 $a_t$ 建波动（`mean="Zero"`）。常数分支也是先去均值再对 $a_t$ 建模，**两条路下游完全统一**。完整联合估计可用 R 的 `rugarch`。

### 步骤 2 · 检验 ARCH 效应（要不要建波动率模型）
对 $a_t^2$ 查波动聚集，**两法互证**（[arch_demo.py:155-163](arch_demo.py#L155-L163)）：
- **方法1**：$a_t^2$ 的 **Ljung-Box $Q(m)$**——$a_t$ 不相关但 $a_t^2$ 若相关，就是波动聚集的直接证据；
- **方法2**：**Engle LM 检验**——辅助回归 $a_t^2$ 对滞后 $a_{t-1}^2,\dots,a_{t-m}^2$，统计量 $LM=n R^2\sim\chi^2_m$，$R^2$ 大说明过去的平方冲击能解释今天 → 有 ARCH 效应。

本次：$a_t^2$ 的 Ljung-Box **Q(5)=5.15e−19、Q(10)=1.07e−26、Q(20)=2.97e−48**（全 ≪ 0.05）；Engle **LM=92.52，p=1.7e−15 ≪ 0.05** → **确有 ARCH 效应，需建波动率模型 ✅**。（若这里都不显著，说明方差恒定，直接收工、不必 ARCH。）

### 步骤 3 · 定阶 m（本步精华）
**主用 $a_t^2$ 的 PACF 截尾**——这背后有个漂亮的理论依据：

> **为什么 ARCH 的阶 = $a_t^2$ 这个 AR 的阶**：把 ARCH 定义 $\sigma_t^2=\alpha_0+\sum_{i=1}^m\alpha_i a_{t-i}^2$ 里令 $\eta_t=a_t^2-\sigma_t^2$（这是个**鞅差白噪声**，因为 $E[a_t^2\mid\mathcal F_{t-1}]=\sigma_t^2$），代入即得
> $$a_t^2=\alpha_0+\sum_{i=1}^m\alpha_i\,a_{t-i}^2+\eta_t$$
> ——**$a_t^2$ 正好服从一个 AR($m$)**！于是"定 ARCH 阶 $m$" = "定 $a_t^2$ 这个 AR 的阶" = 看 $a_t^2$ 的 **PACF 在第 $m$ 阶后截尾**（和 AR 定阶口诀一模一样）。

同 AR/MA/ARMA，先定**搜索上限 `MAXLAG`**：用 **Schwert 规则** $\text{MAXLAG}=\lfloor 12\,(T/100)^{1/4}\rfloor$——**只依赖样本量 $T$、不偷看真值**（本例 $T=1440\Rightarrow$ **MAXLAG=23**，[arch_demo.py:169](arch_demo.py#L169)），PACF 与 AIC/BIC 网格都在 $[1,23]$ 内找。

- **3a · PACF 截尾**：$a_t^2$ 的 PACF 在第几阶后掉进 $\pm1.96/\sqrt n$ 带，那阶就是 $m$。代码 `method="ols"`（末系数法，[arch_demo.py:171](arch_demo.py#L171)）。本次：**最后显著 lag = 15**（band=±0.052）。
- **3b · AIC/BIC 网格**：对 $m=1..23$ 各拟合 ARCH($m$)，选准则最小的阶（公式 $\text{AIC}=-2\ln L+2k$、$\text{BIC}=-2\ln L+k\ln n$，BIC 罚得更重、更简约）。本次：**AIC 选 m=11，BIC 选 m=8**。

**投票汇总**：`PACF=15 | AIC=11 | BIC=8` → **最终定阶 m = 8（以 BIC 为准）**。三法没贴到上限 23（自检 [arch_demo.py:183](arch_demo.py#L183) 未触发），说明 23 足够宽。

> **注意**：波动定阶天生比均值定阶模糊——$a_t^2$ 噪声大、PACF 抖，AIC 又偏向大阶，所以三法分歧（15/11/8）比 AR 的"四法一致"大得多。这正是**以 BIC 取最简约、再交给第 5 步残差检验兜底**的原因（阶不够第 5 步会暴露）。

配图 [arch_fig2_order.png](arch_fig2_order.png)：左 = $a_t^2$ 的 PACF（截尾），右 = AIC/BIC 随阶曲线（选中 m=8）。

### 步骤 4 · 估计参数（MLE）
**怎么估的（为什么不是 OLS）**：均值方程能 OLS，是因为回归元（$y$ 的滞后）可观测；但波动方程里 $\sigma_t^2$ **不可观测、且逐期递推**（GARCH 还自引用 $\sigma_{t-1}^2$），没法直接最小二乘。做法是写**似然**：设 $a_t=\sigma_t\varepsilon_t$、$\varepsilon_t\sim$ 标准化 Student-t($\nu$)，给定参数就能递推出每个 $\sigma_t^2$、算出每个 $a_t$ 的密度，连乘得似然 $L(\alpha_0,\dots,\alpha_m,\nu)$，**MLE 数值最大化 $\ln L$**（[arch_demo.py:200](arch_demo.py#L200)）。

用 **Student-t** 而非正态，是因为金融残差厚尾（本次标准化残差峰度 4.70 > 正态的 3），t 的自由度 $\nu$ 会自动拟合尾巴肥瘦。

本次：ARCH(8) on $a_t$，**logL=−1982.2，AIC=3984.4，BIC=4037.1**（$\omega(=\alpha_0),\alpha_1..\alpha_8,\nu$ 全部估出）。

### 步骤 5 · 模型检验（标准化残差三查）
算**标准化残差** $\tilde a_t=a_t/\sigma_t$——模型对的话它应是 i.i.d. 标准分布。**三查**（[arch_demo.py:207-218](arch_demo.py#L207-L218)）：

| 查什么 | 工具 | 不过关 → |
|---|---|---|
| ① 均值方程够不够 | $\tilde a_t$ 的 Ljung-Box | 改 ARMA 阶数 |
| ② 波动方程够不够 | $\tilde a_t^2$ 的 Ljung-Box | **加大 $m$ / 换 GARCH** |
| ③ 分布假设 | 偏度、峰度、QQ | 换 t / 偏t / GED |

本次（ARCH(8)）：**① p=0.744 ✅（均值充分）；② p=0.845 ✅（波动洗净）；③ 偏度+0.09、峰度 4.70**（厚尾，已由 t 兜住）→ **三查全过，ARCH(8) 采用，无需迭代**。

> **迭代机制**：若 ② 不过（残差平方仍有结构），代码会自动**升级到 GARCH(1,1) 重估并对比 BIC**（[arch_demo.py:222-227](arch_demo.py#L222-L227)）。本次 arma 分支直接通过没触发；**`const` 分支会触发这条迭代**——见第 4 节。

配图 [arch_fig3_diagnose.png](arch_fig3_diagnose.png)：条件波动 $\sigma_t$（看波动聚集）、$\tilde a_t^2$ 的 ACF（是否白噪声）、$\tilde a_t$ 直方图（厚尾 vs N(0,1)）、Q-Q 图。

### 步骤 6 · 使用模型（预测）
对未参与训练的最后 60 天，给出**均值 + 动态波动**预测：
- **点预测（中心）**：$\mu_{t+\ell}$（常数 = 平线；ARMA = 逐步均值回复）；
- **波动率预测**：递推 $\sigma_t^2(\ell)$。GARCH(1,1) 为 $\sigma_t^2(\ell)=\omega+(\alpha+\beta)\sigma_t^2(\ell-1)$，每多一步把方差按 $(\alpha+\beta)$ 向**无条件方差** $V=\omega/(1-\alpha-\beta)$ 拉近一点（$\alpha+\beta<1$ 才收敛）。**这是方差版的均值回复**：AR 的点预测收敛到 $\mu$，GARCH 的波动预测收敛到 $V$；$\alpha+\beta$ 越近 1，波动"长记忆"越强、回归越慢。
- **产出**：动态预测区间 $\mu\pm q\cdot\sigma(\ell)$（随市场松紧变宽变窄）、**VaR**（分布尾分位）。

本次：均值 $\mu_{t+1}=-1.228\to\mu_{t+60}=+0.082$（**ARMA 逐步回复**到无条件均值）；波动 $\sigma(1)=0.761\to\sigma(60)=1.125$（**向无条件水平 1.131 收敛**）；1 日 99% VaR 明日 = **3.165%**；**样本外 95% 区间覆盖率 = 88.3%**（理想≈95%）。

> 见 [arch_forecast.csv](arch_forecast.csv)：每行 = 一天的 `mu_hat`(中心) + `sigma_hat`(宽度) + `lo95/hi95`(区间) + `VaR99`，`actual` 是实际值、`in_interval` 标是否落在区间内。`sigma_hat` 逐日变化 → 区间动态伸缩，这是固定方差模型做不到的。

配图 [arch_fig4_forecast.png](arch_fig4_forecast.png)：上 = 实际 vs 均值预测 + 95% 动态区间；下 = 波动 $\sigma(\ell)$ + 99% VaR。

---

## 4. 两条均值分支对照（同一份 demo，开关 `MEAN_KIND`）

demo 的**招牌特性**：均值类型不写死，靠步骤 1 的 Ljung-Box **自动判别**。两套真值数据各跑一遍，全流程数字如下（均用 Schwert MAXLAG=23）：

| | `MEAN_KIND="arma"`（默认/主线） | `MEAN_KIND="const"` |
|---|---|---|
| 均值真值 | ARMA(1,1) $\phi=.6,\theta=.3$ | 常数 $\mu=.03$ |
| 步1 Ljung-Box(10) | p=2.7e−272 → **有自相关** | p=0.899 → **无自相关** |
| 步1 选出均值 | BIC → ARMA(1,1) | 常数 |
| 步2 ARCH 效应 | Engle LM=92.5（有） | Engle LM=92.9（有） |
| 步3 定阶 m（PACF/AIC/BIC） | 15 / 11 / **8** | 17 / 16 / **2** |
| 步5 三查 | ARCH(8) ①✅②✅③ → **直接通过** | ARCH(2) ①✅**②✗(p=0.024)** → 迭代 |
| 步5 迭代结果 | 无需迭代 | **升级 GARCH(1,1)**：②p=0.968✅，BIC 4078.9 < 4127.7 → 采用 |
| 步5 采用 | **ARCH(8)** | **GARCH(1,1)** |
| 步6 点预测 | ARMA 回复曲线 | 常数平线 +0.016 |
| 步6 覆盖率 | 88.3% | 93.3% |

> **`const` 分支正是 ARCH/GARCH 最经典的一课**：BIC 选了简约的 **ARCH(2)**，但 ② 检验不过（残差平方 p=0.0237，仍有 ARCH）→ 自动**升级 GARCH(1,1)**，三查全过且 BIC 更低（4078.9 vs 4127.7）→ 采用。**一个 $\beta$ 项就抓住了高阶 ARCH 抓不住的高持续波动**——这就是 GARCH 比高阶 ARCH 更受欢迎的实证。

---

## 5. 关键概念速查

- **不相关 ≠ 独立**：$a_t$ 线性是白噪声，$a_t^2$ 有结构——这正是 ARCH 存在的理由，也是步骤 2 的检验对象。
- **$a_t^2$ 服从 AR($m$)**：所以看 $a_t^2$ 的 PACF 截尾定阶，和 AR 定阶同一套口诀。
- **均值 OLS、波动 MLE**：$y$ 的滞后可观测 → OLS；$\sigma_t^2$ 不可观测且递推 → 只能写似然 MLE。
- **AIC vs BIC**：都越小越好；BIC 罚更重、更简约。波动定阶分歧大，**以 BIC 为准、残差检验兜底**。
- **方差版均值回复**：GARCH 多步波动预测收敛到无条件方差 $V=\omega/(1-\alpha-\beta)$；$\alpha+\beta$ 越近 1 越"长记忆"。
- **ARCH vs GARCH**：高持续波动用低阶 ARCH 抓不住，GARCH(1,1) 加一个 $\beta$ 项就够、更简约。
- **预测是"区间"、数据是"值"**：模型每天给 中心 $\mu_t$ + 宽度 $\sigma_t$ = 一条概率带；实际收益是带里落下的一个点。
- **每步的"门槛"都是一个 p 值或信息准则**：ADF<0.05 平稳、$r_t$ LB<0.05 上 ARMA、$a_t^2$ Q/LM<0.05 有 ARCH、$\tilde a_t$/$\tilde a_t^2$ LB>0.05 洗净。不达标就回上一步。

---

## 6. 文件清单与运行

```
arch_demo.py             # 主程序（全流程一个文件，开关 MEAN_KIND 选 const/arma）
requirements.txt         # 依赖：numpy / scipy / pandas / matplotlib / statsmodels / arch
arch_returns.csv         # 步骤0导出的对数收益（列：t, logret_pct, split）
arch_forecast.csv        # 步骤6导出的每日预测（中心/波动/区间/VaR/实际/是否命中）
arch_fig1_mean.png       # r_t 序列 + ACF + PACF
arch_fig2_order.png      # a_t² 的 PACF 截尾 + AIC/BIC 定阶曲线
arch_fig3_diagnose.png   # 条件波动 + 标准化残差²ACF + 直方图 + Q-Q
arch_fig4_forecast.png   # 样本外 均值+95%动态区间 + 波动/VaR
```

运行：

```bash
pip install -r requirements.txt
python arch_demo.py            # 改顶部 MEAN_KIND 切换 const / arma
```

---

## 7. 一句话总结

> **ARCH/GARCH = 给"波动"本身建模。** 完整流程是：
> **先验平稳（ADF）→ 管住中心（Ljung-Box 自动选 常数/ARMA）→ 看残差波动会不会扎堆（ARCH 效应）→ 定阶（$a_t^2$ 的 PACF/BIC）、t 分布、MLE → 标准化残差三查、不行就迭代（ARCH 不够换 GARCH）→ 用 中心+动态波动 产出预测区间与 VaR。**
> 本 demo 从真值为「ARMA(1,1) 均值 + GARCH(1,1)-t 波动」的序列出发，自动判出 ARMA(1,1)、定阶 ARCH(8)、三查通过、预测出逐日伸缩的动态区间——一次教科书式的闭环；切到 `const` 分支还能看到 ARCH(2)→GARCH(1,1) 的经典升级。

### 易错点提醒
1. **不去均值就直接建波动**——ARCH 研究的是冲击 $a_t=r_t-\mu_t$，均值有自相关要先上 ARMA。
2. **不检验 ARCH 效应就硬上**——$a_t^2$ 不显著相关时方差恒定，建 ARCH 是过拟合。
3. **定阶只信 PACF 一种**——波动定阶噪声大，PACF/AIC/BIC 会打架，以 BIC 取简约 + 残差检验兜底。
4. **搜索上限拍脑袋写死**——上限应随样本量走（Schwert），并自检有没有贴到上限被截断。
5. **波动方程也想 OLS**——$\sigma_t^2$ 不可观测，只能写似然 MLE；分布别忘了用厚尾 t。
6. **看到 ② 不过不迭代**——残差平方仍有结构 = 阶不够或该换 GARCH，别停在第一个模型。
7. **用正态算 VaR**——金融厚尾，正态会低估尾部风险；demo 用 t 分位数。
</content>
</invoke>
