# 第三方来源与许可范围

[English](THIRD_PARTY_NOTICES.md) · 简体中文

TH06NC-Neural-Pilot 的项目代码采用 MIT，版权所有者为xialongxl。第三方组件和原作权利不因此转移给本项目。

## 直接派生的代码

`th06ai/pausable_ppo.py` 的 rollout 控制流程参考并改写了 Stable-Baselines3 `OnPolicyAlgorithm.collect_rollouts`。当前验证依赖为 Stable-Baselines3 2.9.2，文件头已有来源说明。保留上游版权及 MIT 正文于 [LICENSES/stable-baselines3-MIT.txt](LICENSES/stable-baselines3-MIT.txt)，其中上游版权署名不得替换为项目作者。

后续贡献如复制、改写其他代码，需增加对应来源、版本、涉及文件和许可文本；不能仅因为原仓库公开可读就将其视为本项目 MIT 原创代码。

## 主要依赖

以下为本次本地依赖元数据与锁文件所示的摘要，不是包含全部传递依赖的最终合规清单：

| 组件 | 主要用途 | 本地元数据许可摘要 |
| --- | --- | --- |
| Stable-Baselines3 / Gymnasium | PPO 与环境接口 | MIT |
| React / React DOM | 控制台界面 | MIT |
| FastAPI / Pydantic | 本机 API 与请求校验 | MIT |
| Uvicorn / httpx2 | 服务与测试支持 | BSD-3-Clause |
| NumPy | 数值计算 | BSD 系列及分发包附带的第三方许可 |
| PyTorch CUDA wheel | 网络训练与推理 | 多个许可证组合，不能简化成“全是 MIT” |
| Lucide React | 图标 | ISC |
| TypeScript | 前端类型及构建 | Apache-2.0 |
| Vite / Vitest | 前端构建／测试 | MIT |
| Rust / PyO3 / NVIDIA NVML 相关组件 | 原生扩展／显卡读数 | 以实际版本上游文件及分发条款为准 |

源码通过安装过程获取依赖，不把 `.venv`、`node_modules`、Cargo 缓存或 CUDA wheel 整包重新分发。若未来发布携带依赖的二进制程序，应针对那个实际产物重新收集全部通知及许可，不把此摘要当作已完成二进制分发审查。

## 原作与研究参考

东方 Project 原作者为 **ZUN／上海アリス幻樂団（Team Shanghai Alice）**，见[东方 Project 官方介绍](https://touhou-project.news/touhou_project/)。《东方红魔乡：新典》的开发署名为 **上海アリス幻樂団、上海アリスReprise（上海爱丽丝幻回奏）**，发行方为 **Alliance Arts**，见 [Steam 官方商店页](https://store.steampowered.com/app/4659620/)；ZUN、JYUNYA 等成员署名及游戏版权标注见[《新典》官方网站](https://touhou-project.news/shanghai-alice-reprise/cn/)。

东方 Project、《东方红魔乡：新典》及其游戏内容属于相应权利人；本仓库的 MIT 许可不授予其游戏、角色、美术、音乐或商标的权利，也不表示获得官方授权或背书。参见[免责声明](DISCLAIMER.zh-CN.md)。

游戏结构研究曾参考 [TH06NC 开发笔记](https://github.com/vittorioromeo/th12_hfr/blob/main/docs/games/TH06NC_DEVNOTES.md)。本项目不包含该项目游戏补丁；实际接入受本版 EXE 哈希与代码特征验证约束。参考思路不等同于可以任意复制上游代码或资源。

许可证核对入口包括本机 Python `*.dist-info/licenses`、npm 包的 LICENSE、Cargo.lock 对应源包，以及上游正式仓库。任何完整通知的遗漏应在 Issue 中说明，维护者据实际涉及代码处理。
