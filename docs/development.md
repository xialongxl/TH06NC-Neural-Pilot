# 开发、安装、配置与本机部署

以下命令均从项目根目录执行，另有说明除外。它们是操作指南，不代表阅读文档会获得接管游戏的授权。实机命令会清楚标注是否使用 GPU、按键和学习更新。

## 1. 环境基线

| 项目 | 要求或验证范围 |
| --- | --- |
| 系统 | Windows 10/11 x64；`win32.py` 在非 Windows 上拒绝实机边界 |
| Python | 安装脚本及现有原生 ABI 固定 CPython 3.10 x64；`requires-python >=3.10` 不等于其他小版本已实测 |
| GPU | 当前实机路径使用 NVIDIA NVML；CUDA wheel 为 `torch-2.14.1+cu130-cp310-cp310-win_amd64.whl` |
| 驱动 | 需支持该 wheel；不在此猜测未经验证的最低驱动版本 |
| Python 包管理 | 安装脚本要求 `uv`；不自动修改全局 Python 环境 |
| 前端开发 | Node.js/npm；构建脚本提示 Node 22.12+ 或 24 LTS；已构建运行无需 Node 服务 |
| 原生构建 | 可选 Rust + MSVC Build Tools；PyO3 固定 0.30.1，使用 Cargo.lock |
| 游戏 | 用户自行购买并安装本版 Steam 游戏，具体身份见 README |

`device=cpu` 是显式的网络设备选择，不是无 NVIDIA 环境的完整运行模式：普通实机会话仍经过 NVML 独立保护。离线 CPU 单元测试则使用夹具，不要求接入真实游戏。

## 2. 安装依赖与首次配置

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
notepad .\config.json
powershell -ExecutionPolicy Bypass -File .\build-ui.ps1
.\启动AI.cmd
```

`ExecutionPolicy Bypass` 仅用于这次 PowerShell 进程，不要求全局修改执行策略。先检查脚本来源，再运行。

`install.ps1`：

1. 切到 `$PSScriptRoot`，将 uv 缓存设为本项目 `.cache/uv`。
2. 无 `.venv` 时创建 Python 3.10 环境，已有环境不删除重建。
3. 使用 `tools/fetch_torch_cuda.py` 从官方 PyTorch 索引找到准确 wheel、取得 SHA-256，分片下载、检查响应大小、拼接并校验。
4. 安装 wheel 及脚本固定版本的数值、训练、Web、测试依赖。
5. 没有 `config.json` 才从 `config.example.json` 创建，不覆盖现有本机路径、模型或校准。

安装脚本不自动构建 UI、安装系统 Rust、启动训练或替用户确认校准。`build-ui.ps1` 运行 npm ci 与正式构建；公开源码默认不含 dist，需要首次构建。

CUDA wheel 下载体积较大，缓存会同时出现分片和拼接文件。存在已验证文件时复用；下载校验失败不能跳过校验安装。不要为“省几分钟”换用来源不明 wheel。

### 依赖版本与复现

- npm 使用 `frontend/package-lock.json`，通常用 `npm ci`，不要随手整体升级。
- Cargo 使用 `native/Cargo.lock`，构建脚本设置独立 CARGO_HOME/TARGET_DIR。
- Python 当前是安装脚本固定直接依赖 + pyproject 声明，尚未完整锁定所有传递依赖。安装成功后应记录实际包版本；不能称为跨时间完全可复现环境。
- 原生扩展对 CPython 小版本和 Windows 架构敏感。换解释器要重新构建，不能简单改 `.pyd` 文件名。

## 3. 配置与环境变量

### 配置权威来源

`config.json` 由 `load_config(path=None)` 读取，默认位置是源码推导的项目根目录，不依赖作者目录或当前 shell 的工作目录。入口脚本也通过自己的位置定位项目。

真实游戏 EXE 是本机资源，需在每台机器的 `game_exe` 指定。它是允许保留的本机绝对路径，**不会提交到公开 Git**。运行时没有作者磁盘路径的隐藏 fallback。

| 字段 | 默认／范围 | 说明 |
| --- | --- | --- |
| `game_exe` | 必填字符串 | 实际游戏 EXE 完整路径，不是游戏文件夹 |
| `gpu_index` | 0 | NVML 设备索引，后续按 UUID 绑定身份 |
| `minutes` | 60，0<值≤60 | 会话预算，CLI 可覆盖 |
| `episode_seconds` | 0；非零为 10..1800 | 0 不按单局时长截断，非 180 秒默认限制 |
| `device` | cuda 或 cpu | CUDA 不可用时不静默降级 |
| `input_backend` | dataclass 默认 python；示例 auto | 普通模拟输入实现选择 |
| `inference_backend` | auto / eager / cuda_graph | auto 只针对图捕获限制回退 CUDA eager |
| `goal` | survival / nmnb | NMNB 禁 B；不是通关认证 |
| `planning.enabled` | true | 缓存规划是否参与决策 |
| `planning.beam_width` | 2，1..4 | 后续搜索 beam；当前移动仍 18 种 |
| `planning.collision_backend` | dataclass 默认 python；示例 auto | rust/auto 在可用时启用相应原生路径 |

真实默认值要区分 dataclass、示例配置、已有本机 JSON 和 UI 会话参数，不可只看其中一个就断言所有部署相同。

### 保护字段

| 字段 | 默认 | 配置允许范围 |
| --- | ---: | --- |
| temperature_limit_c | 70°C | 30..70 |
| memory_temperature_limit_c | 80°C | 40..80；硬件提供时使用 |
| vram_fraction_limit | .90 | .1...90 |
| power_limit_ratio | 1.03 | 1..1.03 |
| fan_required_at_c | 60°C | 40..60 |
| poll_seconds | .25 秒 | .05...25 |
| telemetry_timeout_seconds | 60 秒 | .1..60 |
| worker_heartbeat_seconds | 2 秒 | .5..2 |

这些是当前程序的校验边界，不是建议用户寻找的最大性能设置。Web 计算后端接口不提供任意保护阈值修改。

### `.env.example` 的真实含义

项目**没有 dotenv 自动加载器**。复制 `.env.example` 为 `.env` 不会改变配置。它只列出可在启动进程时设置的通用环境：

```dotenv
PYTHONUTF8=1
PYTHONUNBUFFERED=1
UV_CACHE_DIR=.cache/uv
```

PowerShell 显式设置方式：

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
```

不支持 `GAME_EXE=...`、`GPU_LIMIT=...` 等虚构环境接口。保护进程会为所属 worker 设置 CUDA_VISIBLE_DEVICES 绑定实际 GPU；不要把手动覆盖该变量当作可靠的多卡选择方式。

## 首次校准与资源核对

新检出不能复用作者的校准 JSON。这些文件只在本机保存并被 Git 忽略。

### 第一步：GPU 自检

在 UI 中明确点击 GPU 自检，或执行：

```powershell
# 使用 GPU，不操作游戏，不训练用户模型
.\.venv\Scripts\python.exe -m th06ai check --minutes 1
```

自检写 `training_backend.json`。它证明环境可前向／反向，不是已有训练模型或已通关的证明。

### 第二步：移动、弹幕与重开校准

打开游戏，准备 Normal、灵梦 A 的可操作第一关场景，英文输入法，保持前台。明确启动 UI 校准或：

```powershell
# 会发真实按键、移动并重开游戏；只在准备好之后执行
.\.venv\Scripts\python.exe -m th06ai calibrate --minutes 1
```

校准测量高速／低速，确认可见弹幕及 R 重开后的计时复位与推进，最后暂停并写 `native_calibration.json`。校准失败不会生成成功凭据。普通训练允许空弹幕，校准观察到至少一定数量弹幕是一次性接入验收，不是训练每帧必须有弹的限制。

### 第三步：资源读数人工确认

**当前尚无完整的新机器 UI 自助确认流程。** worker 非校准模式要求 `native_resources_calibration.json` 与当前游戏哈希一致，且 `operator_confirmed` 为 true；只跑上述 calibrate 不会创建这份资源确认。

维护者辅助核对时，在游戏显示资源的画面执行下列只读操作。它不发送按键、不连接 CUDA：

```powershell
.\.venv\Scripts\python.exe -c "from th06ai.config import load_config; from th06ai.native import GameReader; from dataclasses import asdict; import json; r=GameReader(load_config().game_path); f=r.read(); print(json.dumps({'game_sha256':r.identity['sha256'],'resources':asdict(f.resources) if f.resources else None},ensure_ascii=False)); r.close()"
```

操作者核对游戏显示的 Power、B 和残机，记录实际证据。确认前的本机文件可使用：

```json
{
  "game_sha256": "复制本次只读核对输出的真实SHA256",
  "operator_confirmed": false
}
```

只有人工核对成功后才将它改为 true，并建议记录核对时间及实际读数。不要用脚本自动把 true 写进去，不要把这份本机凭据提交到 Git。完善正式资源确认交互是发布前已知待办，而不是通过注释掉检查来解决。

## 4. 开发服务与端口

正式本机运行：

```powershell
.\.venv\Scripts\python.exe -m th06ai open
.\.venv\Scripts\python.exe -m th06ai shutdown
```

开发后端终端：

```powershell
.\.venv\Scripts\python.exe -m th06ai web --no-browser
```

另一个终端进入 frontend：

```powershell
npm.cmd ci
npm.cmd run dev
```

Vite `/api` 代理固定默认后端 18765。后端 CLI 改端口时要同步代理；默认端口不是个人目录。代理需要同源控制令牌契约，遇到 403 先看 Origin/Host/token，不删除同源检查。

只读 UI 预览：

```powershell
.\.venv\Scripts\python.exe -m th06ai open --preview
```

默认预览端口 18766。预览不连接真实 GPU／游戏、不启动训练；预览也不是编造训练指标的演示。`legacy-gui` 仅保留兼容入口，不应变成主界面回退方案。

## 5. CLI 操作矩阵

| 命令 | 游戏输入 | GPU／NVML | 更新用户权重 |
| --- | --- | --- | --- |
| open / gui / web | 普通显示不发键；明确 focus/probe 有各自行为 | 服务不持有 CUDA | 否 |
| doctor | 不发键，检查 EXE 与瞬时 GPU 状态 | NVML 单次检查 | 否 |
| monitor | 不发键 | 独立监控 | 否 |
| check | 不发键 | 受保护的 CUDA 自检 | 否 |
| calibrate | 会移动、暂停、重开 | 独立保护 | 否 |
| train | 会控制游戏 | 独立保护 + 学习 | 是 |
| play | 会控制游戏 | 独立保护 + 推理 | 否 |
| evaluate | 会控制游戏、逐局评估 | 独立保护 + 推理 | 否 |
| profile 默认 | 不发键、不读游戏 | 独立保护 + 离线推理 | 否 |
| profile --profile-kind latency | 不发键，但只读游戏 | 独立保护 + 推理 | 否 |

示例，**不要把整段当成一键依次执行**：

```powershell
# 新模型真训练
.\.venv\Scripts\python.exe -m th06ai train --minutes 5 --goal survival
# 从已经存在且校验通过的本项目模型续训
.\.venv\Scripts\python.exe -m th06ai train --minutes 5 --model models/latest.zip
# 固定权重试玩与评估；显式目标避免默认差异
.\.venv\Scripts\python.exe -m th06ai play --minutes 1 --model models/latest.zip --goal nmnb
.\.venv\Scripts\python.exe -m th06ai evaluate --minutes 60 --model models/latest.zip --goal nmnb
# 只读剖析，等待操作者手动解除暂停
.\.venv\Scripts\python.exe -m th06ai profile --profile-kind latency --profile-wait-combat --minutes 1 --model models/latest.zip
```

没有 `--episodes` 公共参数；当前独立评估默认请求 10 局。预算提前到期／中断时读 `evaluation_progress.json`，不能宣称十局全部完成。

## 控制 API

基础路径为本机服务；POST 带 bootstrap 获取的 `X-Console-Token`，浏览器请求遵守同源 Origin。

| 方法与路径 | 请求或行为 |
| --- | --- |
| GET /api/health | 服务名、项目身份、preview、PID |
| GET /api/bootstrap?state=false | token、protocol；省略 state 参数可附初始完整状态 |
| GET /api/state | 原始完整诊断 |
| GET /api/events?protocol=2&hidden=false | 共享增量 SSE |
| GET /api/logs | 有界日志尾，支持 ETag/304 |
| POST /api/models/select | `{model_id}` |
| POST /api/sessions | `{mode,minutes,model_id?,goal?}`；mode=resume/train/play/evaluate/check/monitor/calibrate |
| POST /api/session/pause | `{paused: true/false}`；训练／试玩／评估 |
| POST /api/session/finish | 仅训练的保存结束 |
| POST /api/session/stop | 所属会话 STOP |
| POST /api/session/focus | 向当前会话提交 FOCUS |
| POST /api/game/focus | 明确激活配置游戏；活动会话委托 worker |
| POST /api/game/probe | 无活动会话时做一次只读游戏检查 |
| POST /api/settings/acceleration | `{collision,input}`，无活动会话时保存后端选择 |
| POST /api/server/shutdown | 协调保存退出后关闭服务 |

请求模型禁止未知字段。API 不接受任意命令、任意模型路径或任意保护阈值。无登录系统、远程鉴权或公网部署保证；不要改为 0.0.0.0 后就当作多人服务器。

## 6. 原生构建

```powershell
.\.venv\Scripts\python.exe tools\build_native.py
.\.venv\Scripts\python.exe tools\build_native.py --test
.\.venv\Scripts\python.exe tools\build_native.py --planner-only
```

在已配置 MSVC/Rust 的终端运行，或显式传自己的 `--toolchain` 与 `--vcvars`。默认构建安装普通及独立搜索扩展；planner-only 不覆盖原输入扩展。Windows 不允许覆盖已加载 `.pyd`，需要先结束持有它的会话／后端。详情见[native README](../native/README.md)。

## 7. 构建、Git 发布与部署

### 默认：当前目录直接 push

源码仓库就是当前根目录。`.gitignore` 排除依赖、模型、日志、研究材料、个人配置和编译产物，文件留在本机，不需要复制出一个“干净项目”。

```powershell
# 未初始化时执行一次；准备工作可能已经完成，不必重复
git init -b main
# 先核对范围
git status --short
git status --short --ignored
.\.venv\Scripts\python.exe tools\check_docs.py
# 检查后暂存、提交
git add -A
git diff --cached --stat
git commit -m "docs: prepare TH06NC-Neural-Pilot for open source"
# 用真实托管地址替换变量，不虚构作者账号
$remoteUrl = '<实际仓库的Git地址>'
git remote add origin $remoteUrl
git push -u origin main
```

已有 origin 时用 `git remote -v` 核对，不反复 add 或盲目替换。`.gitignore` 不会移除已经跟踪的文件；已有仓库先审查索引及历史，按需只撤出索引，不删除本机数据。当前准备不会推送到未指定远端。

### 其他开发者部署

clone → install.ps1 → 设置本机 config → build-ui.ps1 → 可选 Rust 构建 → 校准 → 明确启动会话。前端正式产物由 FastAPI 提供；不需要额外数据库或常驻 Vite。

当前没有验证过的单文件 EXE、Docker、完整 wheel 静态资源打包或 Linux 实机方案。项目 ROOT 和 frontend/dist 按源码目录布局解析，单纯 `pip wheel` 不能当作已完成桌面部署。

### 可选归档

如果需要离线审阅的 ZIP，优先从已经核验的 Git 索引／提交生成 `git archive`，不要压缩整个工作目录。Git 归档只包含提交内容，不包含忽略的依赖或模型；先审查 Git 内容再归档。它是可选分发方式，不是正常 push 的前置步骤。

更多发布检查见[开源准备](open-source-readiness.md)。
