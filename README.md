# The-modeling-of-financial-time-series

## 说明

本仓库为金融时间序列分析模型，由 FF 创作。

推荐使用编程语言：R 语言。本仓库使用 Python 编程语言。所需要的库每个 `demo` 里均有 `requirements.txt`。

理想学习顺序：`AR` --> `MA` --> `ARMA` --> `ARCH` --> `GARCH` --> `IGARCH` --> `GARCH-M` --> `EARCH` --> `TARCH` --> `TAR` --> 中间一堆模型（之后补） --> `VAR` --> `VMA` --> `VARMA`

仓库里有对应的 `demo`，提供代码和具体说明。之后会不定期更新新模型，如有疑问可联系。希望点个小星星(*^_^*)。

## 安装指南

1. 前置要求
请确保你的 Python 版本 >= 3.10（请根据你的实际版本修改）。在终端/CMD中检查：

```bash
python --version
```
2. 进入项目根目录
请使用终端（Mac/Linux）或命令提示符/PowerShell（Windows）进入包含 requirements.txt 的文件夹：

```bash
cd 你的项目文件夹路径
```
3. （强烈推荐）创建虚拟环境
这一步是为了隔离项目依赖，避免污染全局 Python 环境。

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```
Mac / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```
（激活后，命令行前面会出现 (venv) 字样）

4. 升级 pip（防止因 pip 版本过旧导致安装失败）
```bash
python -m pip install --upgrade pip
```
5. 安装依赖（核心命令）
```bash
pip install -r requirements.txt
```
