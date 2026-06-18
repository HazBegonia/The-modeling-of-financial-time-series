# GARCH-M 风险溢价建模 —— 学习总结

> 配套代码：[garchm_demo.py](garchm_demo.py)（一条龙：均值 ARMA + in-mean 项 + 波动 GARCH(p,q)-t，**联合 MLE**）
> 数据/产物：`garchm_returns.csv`（输入收益）、`garchm_forecast.csv`（预测输出）、`garchm_fig1~5_*.png`
> 结构对照 [GARCH.md](../GARCH_demo/GARCH.md)、[EGARCH.md](../EGARCH_demo/EGARCH.md)，便于「GARCH / GARCH-M / EGARCH」三者对比记忆。

---

## 0. 一句话理解 GARCH-M

普通 GARCH 把均值和波动**分家**：均值由 ARMA 管，波动由 GARCH 管，互不干涉。但金融里有个核心直觉——**「高风险要高回报」**：市场越动荡（波动越大），投资者要求的**预期收益**也越高。要把这条"风险→收益"的链路写进模型，就得让**条件方差进入均值方程**：

$$\text{GARCH}:\ r_t=\mu_t^{\text{(ARMA)}}+a_t \qquad
\text{GARCH-M}:\ r_t=\underbrace{\mu_t^{\text{(ARMA)}}}_{中心}+\underbrace{\lambda\,\sigma_t^2}_{风险溢价}+a_t$$

多出来的 $\lambda\sigma_t^2$ 叫 **in-mean 项 / 风险溢价项**，$\lambda$ 是**风险溢价系数**（risk premium）：

- $\lambda>0$：波动越大 → 预期收益越高（风险与收益正相关，Merton ICAPM 直觉）；
- $\lambda=0$：退回普通 GARCH（波动不影响中心）。

> Tsay《金融时间序列分析》§3.6 的写法即 $r_t=\mu+c\,\sigma_t^2+a_t$（in-mean 用**方差** $\sigma_t^2$；也可用 $\sigma_t$ 或 $\ln\sigma_t^2$）。本 demo 用 $\sigma_t^2$、均值真值是 **ARMA(1,1)**。

**一个副作用（也是识别难点）**：因为 $\sigma_t^2$ 本身高度自相关，$\lambda\sigma_t^2$ 会给 $r_t$ **带进一段缓慢、持续的自相关**，看起来很像一个高 AR 根——所以 GARCH-M 里 $r_t$ 的自相关并不全是 ARMA 的功劳。

---

## 1. 核心概念速查

| 概念 | 含义 |
|---|---|
| 对数收益率 $r_t=\ln(P_t/P_{t-1})$ | 把不平稳的价格转成平稳序列，建模入口 |
| 冲击 / 残差 $a_t=r_t-\mu_t$ | 去掉条件均值后剩下的"意外"，$a_t=\sigma_t\varepsilon_t$ |
| 条件方差 $\sigma_t^2$ | $t-1$ 信息下今天波动多大（随时间变） |
| **in-mean 项** $\lambda\sigma_t^2$ | **GARCH-M 的灵魂**：把波动塞进均值，刻画"风险溢价" |
| **风险溢价系数** $\lambda$ | $>0$ 即高风险高回报；显著性=该不该上 GARCH-M |
| 持续度 $\alpha+\beta$ | 波动记忆长短（越接近 1 越"黏"） |
| 无条件方差 $\dfrac{\omega}{1-\alpha-\beta}$ | 波动长期回复到的水平 |
| 标准化残差 $z_t=a_t/\sigma_t$ | 模型对的话应是 i.i.d. 标准 t，用于检验 |

**关键认知**：均值方程不再是常数中心。$\sigma_t^2$ 既出现在波动方程、又出现在均值方程——**两个方程被 $\lambda$ 耦合**。因 $\sigma_t^2$ 由 $t-1$ 信息算出（先于 $a_t$），均值方程仍无同期内生，递推与似然都良定义。

---

## 2. 数据要求（喂进去之前先验收）

| 要求 | 怎么查 | 不满足怎么办 |
|---|---|---|
| **平稳** | ADF / KPSS | 价格 → 对数收益率 |
| **均值已建模** | $r_t$ 的 Ljung-Box | 有自相关 → 先上 ARMA（部分来自 in-mean） |
| **有 ARCH 效应** | $a_t^2$ 的 Q(m) / Engle LM | 不显著 → 连 GARCH 都不用，更别说 GARCH-M |
| **样本足够**（日频≥1500） | 计数 | in-mean 多一个 $\lambda$，更吃样本 |
| **量纲适中** | 看数值大小 | ×100 转百分比 |
| **厚尾用分布兜住** | 峰度 / QQ | 选 t / 偏t / GED |
| **持续度 < 1** | 估完看 $\alpha+\beta$ | ≥1 → IGARCH/方差爆炸，分段或换模型 |
| **风险溢价确实存在** | $\lambda$ 的 t / LR 检验 | 不显著 → 退回普通 GARCH（in-mean 项可省） |

> 一句话：**GARCH-M 适用于——已确认有波动聚集、且有理由相信"波动会推高预期收益"的（百分比）对数收益率**；若 $\lambda$ 不显著，就老老实实用 GARCH。

---

## 3. 六步建模流程（对照 demo，本次真实数字）

```
价格 ─→ 对数收益 r_t ─→ ARMA 定阶 ─→ 暂得冲击 a_t ─→ 查 a_t² 有无聚集
                         │(BIC网格)                  │(Ljung-Box/Engle LM)
                         ↓                            ↓有ARCH效应
   预测 ←── 三查/平稳/风险溢价LR ←── 联合MLE一次估 ←── (p,q)二维AIC/BIC网格
(μ随波动起伏+动态区间/VaR)  (lambda 显著?)  ARMA+GARCH+lambda
```

> 与普通 GARCH 的最大流程差异：**第 4 步不再"两步走"**（先 ARMA 取残差、再对残差建 GARCH），而是**联合 MLE 一次估**——因为 in-mean 把 $\sigma_t^2$ 喂回均值，两方程耦合，分步会有偏。`arch` 包不支持 in-mean，故本 demo **手写联合 MLE**（numpy/scipy）。

### 第0步 准备数据
价格 → 对数收益率 $r_t$，ADF 验平稳，单位 ×100。
- 真值：均值 **ARMA(1,1)**（$\phi=0.5,\theta=-0.3,c=0$）**+ 风险溢价 $\lambda=0.2$**（in-mean=$\sigma_t^2$）；波动 **GARCH(1,1)-t**（$\omega=0.05,\alpha=0.08,\beta=0.90,\nu=7$，持续度 0.98）。
- 样本 $n=2500$（train=2440, test=60）；**ADF p≈0 → 平稳 ✅**。

### 第1步 均值方程（管"中心"，用 ARMA）
$r_t$ 的 Ljung-Box(10) **p≈8e−122 ≪ 0.05 → 有自相关**（一部分来自 in-mean）。BIC 网格（$p,q\in0..3$）：

```
       q=0       q=1       q=2       q=3
  p=0    --     9206.0    9114.0    9078.4
  p=1  9136.1  *9049.1*   9053.5    9060.4
  p=2  9057.1   9053.3    9061.1    9061.0
  p=3  9053.5   9056.6    9068.6    9073.1
```

**BIC 最小 @ (1,1)** → 暂取冲击 $a_t$ 供第 2/3 步用；**in-mean 风险溢价项留到第 4 步联合估**（这步只定 ARMA 的阶）。
> ⚠️ 识别坑：in-mean 的 $\lambda\sigma_t^2$ 高度持续，会**给 $r_t$ 灌进缓慢自相关、抬高 AR 根**。demo 选了 $\theta<0$（与 AR 反号、ACF 形态分明）的真值，让 BIC 能干净定回 (1,1)；同号小 MA 容易被并进 AR、定成 (1,0)。

### 第2步 检验 ARCH 效应（要不要建波动率模型）
- **方法1** $a_t^2$ Ljung-Box：$Q(5)$ p≈4e−36、$Q(10)$ p≈6e−67、$Q(20)$ p≈3e−93 → **全部有 ARCH 效应**。
- **方法2** Engle LM(10)：LM=184.6, **p≈3e−34 → 有 ARCH 效应，需建波动率模型 ✅**。

### 第3步 定阶 (p,q)（GARCH 波动阶，二维网格）
$a_t^2$ 的 ACF/PACF 双双拖尾（图2，ACF 缓慢衰减=高持续指纹），上 (p,q) 二维网格（$p\ge1$；$q=0$ 即纯 ARCH）：

```
       q=0(纯ARCH) q=1       q=2
  p=1   8902.3    *8772.6*   8780.4
  p=2   8869.9     8779.8    8786.1
```

**AIC、BIC 一致选 (1,1)**；含 $\beta$ 的 GARCH(1,1) 因高持续完胜纯 ARCH 列。

### 第4步 联合估计（手写 MLE：一次估 ARMA + GARCH + λ）
新息取 **Student-t**，参数向量 $[c,\phi,\theta,\lambda,\omega,\alpha,\beta,\nu]$ 一次性 MLE（数值 Hessian 给标准误/t 值）：

| 参数 | 估计 | t 值 | 真值 |
|---|---|---|---|
| $c$ | −0.013 | −0.26 | 0.0 |
| $\phi_1$ | +0.631 | 13.0 | 0.5 |
| $\theta_1$ | −0.427 | −7.6 | −0.3 |
| **$\lambda$（风险溢价）** | **+0.166** | **5.28** | **0.2** |
| $\omega$ | 0.063 | 3.4 | 0.05 |
| $\alpha_1$ | 0.073 | 6.6 | 0.08 |
| $\beta_1$ | 0.902 | 63.9 | 0.90 |
| $\nu$ | 9.11 | 5.6 | 7.0 |

- **风险溢价 $\lambda=0.166$（真值 0.2），t=5.28 ≫1.96 → 显著正，高波动确实索取更高预期收益 ✅**；
- 持续度 $\alpha+\beta=0.975$（真值 0.98）< 1 → 平稳；无条件方差 2.48（真值 2.50）；半衰期 ≈27 天；
- $\phi,\theta$ 略偏高（0.63/−0.43 vs 0.5/−0.3）：in-mean 的持续溢价被 AR 根吸收了一部分——这是 GARCH-M 固有的识别耦合，不影响 $\lambda$ 与波动参数的回收。

### 第5步 模型检验（三查 + 平稳 + 风险溢价检验）

| 查什么 | 工具 | 本次 | 结论 |
|---|---|---|---|
| ① 均值方程 | $z_t$ Ljung-Box(10) | p=0.636 | 充分 ✅ |
| ② 波动方程 | $z_t^2$ Ljung-Box(10) | p=0.139 | 波动洗净 ✅ |
| ③ 分布 | 偏度/峰度/QQ | 偏度−0.09, 峰度 4.06 | 仍略厚尾，t 合理 |
| ④ 平稳性 | $\alpha+\beta$ | 0.975 < 1 | 方差过程平稳 ✅ |
| ⑤ **风险溢价** | **LR：$\lambda=0$ vs GARCH-M** | LR=70.8, **p≈4e−17** | **拒绝 $\lambda=0$，溢价显著 ✅** |

第⑤项是 GARCH-M 专属：重估一个**受限模型**（$\lambda=0$，即普通 ARMA-GARCH），$LR=2[\ell_{\text{M}}-\ell_{0}]\sim\chi^2(1)$。$\log L$ 从 −4366.5 升到 −4331.1，**LR=70.8、p≈4e−17 → in-mean 项确实值得加**。（图5：左=$r_t$ 随 $\sigma_t^2$ 升高而抬升的分箱均值+拟合溢价线；右=高波动时段实际收益的滚动均值也更高。）

### 第6步 使用模型（预测）
GARCH-M 的招牌：**均值预测会"随波动一起均值回复"**，因为 $\mu(\ell)=c+\lambda\,\sigma^2(\ell)+\text{ARMA}$。

- **波动率预测**：$\sigma^2(\ell)=\omega+(\alpha+\beta)\sigma^2(\ell-1)$ 几何收敛到无条件水平（$\sigma(1)=1.16\to\sigma(60)=1.49\to 1.57$）；
- **均值点预测**：$\mu(1)=0.353\to\mu(60)=0.964$——**不是回到一个固定常数**，而是跟着 $\sigma^2(\ell)$ 一起爬升；其中风险溢价贡献 $\lambda\sigma^2(\ell)$ 从 0.224 升到 0.371（向 $\lambda\cdot$无条件方差 0.412 收敛）；
- **产出**：动态区间 $\mu(\ell)\pm q_t\sigma(\ell)$、VaR（明日 1日99% VaR=2.53%）。样本外 95% 区间覆盖率 **96.7%**（理想≈95%）。

> 见 [garchm_forecast.csv](garchm_forecast.csv)：`mu_hat`(含溢价的中心) / `premium`(其中 $\lambda\sigma^2(\ell)$) / `mu_no_premium`(去掉溢价的纯 ARMA 漂移对照) / `sigma_hat` / `lo95,hi95` / `VaR99` / `actual`。**`mu_hat` 与 `mu_no_premium` 的差就是风险溢价对预测的贡献——它随波动放大，这正是 GARCH-M 区别于 GARCH 的招牌**（图4：蓝实线高于绿虚线，且差距随波动上行而拉大）。

---

## 4. 判据一览（每步用什么 p 值/准则决策）

| 步骤 | 判据 | 通过条件 |
|---|---|---|
| 平稳 | ADF p | < 0.05 |
| 均值要不要 ARMA | $r_t$ Ljung-Box p | < 0.05 → 上 ARMA |
| ARMA 定阶 | BIC 网格 | 取最小 |
| 有无 ARCH 效应 | $a_t^2$ Q(m) / Engle LM p | < 0.05 → 有 |
| GARCH 定阶 (p,q) | (p,q) 二维 AIC/BIC | 准则最小（常落 (1,1)） |
| 均值检验 | $z_t$ Ljung-Box p | > 0.05 → 充分 |
| 波动检验 | $z_t^2$ Ljung-Box p | > 0.05 → 洗净 |
| 平稳性 | $\alpha+\beta$ | < 1 |
| **风险溢价** | **$\lambda$ 的 t / LR(χ²(1)) p** | **t>1.96 / p<0.05 → 显著，该上 GARCH-M** |

---

## 5. 易混点澄清

1. **GARCH vs GARCH-M**：GARCH 均值波动分家；GARCH-M 多一个 $\lambda\sigma_t^2$ 把波动塞进均值=刻画"风险溢价"。$\lambda=0$ 就退回 GARCH。
2. **为什么必须联合估**：in-mean 把 $\sigma_t^2$ 喂回均值，两方程耦合，**不能再"先 ARMA 取残差再建 GARCH"**（那样 $\lambda$ 估不出/有偏）。`arch` 包不支持，故手写 MLE；R 的 `rugarch` 有 `archm` 选项。
3. **in-mean 用 $\sigma_t^2$ 还是 $\sigma_t$？** 本 demo 与 Tsay 用方差 $\sigma_t^2$；也常见用标准差 $\sigma_t$（rugarch 默认）或 $\ln\sigma_t^2$。换形式只改均值那一项，$\lambda$ 的解释随之变。
4. **$r_t$ 的自相关不全是 ARMA**：持续的 $\lambda\sigma_t^2$ 会贡献一段缓慢自相关、抬高 AR 根——这会干扰 ARMA 定阶，也让 $\phi$ 略偏高。
5. **GARCH-M 的均值预测会"随波动起伏"**：普通 ARMA 多步预测回到固定常数；GARCH-M 的 $\mu(\ell)$ 跟着 $\sigma^2(\ell)$ 一起均值回复——这是它最直观的预测特征。
6. **$\lambda$ 不显著就别硬上**：先用 LR/t 验风险溢价；不显著说明数据不支持"波动推高收益"，老实用 GARCH。

---

## 6. 三模型对照（GARCH ↔ GARCH-M ↔ EGARCH）

| 维度 | GARCH(p,q) | **GARCH-M** | EGARCH |
|---|---|---|---|
| 解决什么 | 波动聚集（高持续） | **风险溢价（波动→收益）** | 杠杆效应（涨跌不对称） |
| 动了哪个方程 | 波动方程 | **均值方程（+$\lambda\sigma_t^2$）** | 波动方程（对 $\ln\sigma^2$） |
| 关键新参数 | $\alpha,\beta$ | **$\lambda$（风险溢价）** | $\gamma$（杠杆） |
| 怎么验它值不值 | $a_t^2$ 有 ARCH 效应 | **$\lambda$ 的 t / LR 检验** | $\gamma$ 显著 / AIC 比对称 |
| 估计方式 | 两步法（ARMA→残差→GARCH） | **联合 MLE（耦合，不能分步）** | 两步法 |
| 预测特征 | 波动几何收敛；均值回固定常数 | **均值随波动一起回复** | 波动用 NIC 非对称收敛 |

本 demo = **ARMA（均值）+ in-mean 风险溢价 + GARCH（波动）合体**：均值的中心**会随波动水涨船高**。

---

## 7. 文件清单与运行

```
garchm_demo.py             # 主程序（全流程一个文件，含手写联合 MLE）
requirements.txt           # 依赖：numpy/scipy/pandas/matplotlib/statsmodels/arch
garchm_returns.csv         # 步骤0导出的收益（列：t, logret_pct, split）
garchm_forecast.csv        # 步骤6每日预测（mu/premium/mu_no_premium/sigma/区间/VaR/actual）
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

## 8. 一句话总结

> **GARCH-M = 在 GARCH 上把条件方差塞进均值方程（$+\lambda\sigma_t^2$）= 用一个参数 $\lambda$ 刻画"高风险高回报"的风险溢价；$\lambda=0$ 即退回 GARCH。** 完整流程是：
> **先验平稳（ADF）→ ARMA 定阶（注意 in-mean 会抬高 AR 根）→ 查 $a_t^2$ 有无聚集 → (p,q) 网格定波动阶 → 联合 MLE 一次估 ARMA+GARCH+$\lambda$ → 三查/平稳 + 风险溢价 LR/t 检验 → 用"随波动起伏的中心 + 动态波动"产出区间与 VaR。**
> 本 demo 从一个真值为 ARMA(1,1)+GARCH(1,1)-t+风险溢价 $\lambda=0.2$ 的序列出发，BIC 定回 (1,1)/(1,1)，**联合 MLE 估出 $\lambda=0.166$（t=5.28）、LR=70.8（p≈4e−17）确认风险溢价显著存在**，波动参数（$\alpha=0.073,\beta=0.902$）接近真值、三查全过，预测给出一条会"随波动一起爬升"的均值与动态波动带——GARCH 加一个 in-mean 项后的一次教科书式风险溢价闭环。

### 易错点提醒
1. **把 GARCH-M 当 GARCH 两步估**——in-mean 耦合两方程，必须联合 MLE，分步会让 $\lambda$ 有偏。
2. **不验 $\lambda$ 显著性就上 GARCH-M**——先用 LR/t；不显著就退回 GARCH。
3. **以为 $r_t$ 的自相关全是 ARMA**——持续的 $\lambda\sigma_t^2$ 也贡献自相关、干扰定阶、抬高 $\phi$。
4. **以为多步均值预测会回到固定常数**——GARCH-M 的 $\mu(\ell)$ 随 $\sigma^2(\ell)$ 一起均值回复。
5. **in-mean 形式混用**（$\sigma_t^2$ vs $\sigma_t$ vs $\ln\sigma_t^2$）——$\lambda$ 的量纲与解释不同，报告时要讲清。
6. **样本太少**——比 GARCH 多一个 $\lambda$ 更吃样本，日频建议 ≥1500。
