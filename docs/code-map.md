# 代码地图与建议阅读顺序

Python 导入包仍叫 `th06ai`，发行项目名是 TH06NC-Neural-Pilot。不要为改展示名称批量改导入、服务身份、互斥量或模型序列化类路径。

## 1. 从哪个入口读

| 目的 | 阅读顺序 |
| --- | --- |
| 跑起来 | README → config.py → __main__.py → console_lifecycle.py |
| 理解训练 | environment.py → observations.py → planning.py → policy.py → pausable_ppo.py → learning.py |
| 排查动作迟到 | native.py → input_timing.py → inference.py → environment.refresh_decision/_step → planning.py |
| 查死亡／资源行为 | diagnostics.py/log_rows.py → rewards.py/objectives.py → planning.py → policy.py |
| 修改 UI | frontend/useConsole.ts → stream.ts/types.ts → App.tsx → console_stream.py → web_console.py |
| 修改显卡保护 | config.py → gpu.py → lease.py → supervisor.py → worker.Heartbeat → win32.OwnedJob |
| 做游戏版本移植 | native.py → combat.py/lasers.py/stage.py/game_phase.py → 只读夹具和实机核对 |

符号级信息由 [code-index.json](code-index.json) 提供：模块、源码哈希、顶层类／函数、方法、签名、行号与直接导入。哈希按 UTF-8/LF 规范化，避免 Windows/Git 换行差异造成假过期。用 `tools/check_docs.py --update-index` 更新。索引不是自动生成的架构判断，职责解释仍在本文。

## 2. 仓库结构

```text
.
├── README.md / CONTRIBUTING.md / DISCLAIMER.md / LICENSE
├── THIRD_PARTY_NOTICES.md / LICENSES/
├── AGENTS.md
├── config.example.json / .env.example / .gitignore
├── pyproject.toml / install.ps1 / build-ui.ps1
├── 启动AI.cmd / 启动后端.cmd / 关闭AI.cmd
├── .github/                   Issue 与 PR 模板
├── docs/                      说明书、交接与生成的符号索引
├── th06ai/                    运行时 Python 包
│   └── native_planner/        独立搜索模块安装位置
├── frontend/
│   ├── src/                   React、状态流、显示派生逻辑和测试
│   ├── public/                自有前端静态资源
│   ├── package*.json          包与 npm 锁文件
│   ├── vite.config.ts / tsconfig.json / index.html
│   └── dist/                  生成产物，本机运行使用，Git 忽略
├── native/
│   ├── src/lib.rs             PyO3 ABI、连续碰撞、原生输入
│   ├── src/planner.rs         整段拥有型搜索实现
│   └── Cargo.toml / Cargo.lock / README.md
├── tests/                     41 个现有测试模块及必要几何 NPZ 夹具
├── tools/                     安装、构建、文档与源码审查工具
├── models/ / runs/            本机产物，Git 忽略
└── .cache/ / .venv/ / research/ / tmp/  本机依赖、缓存和研究，Git 忽略
```

`tests/fixtures/stage5-switch-window.npz` 是特定逻辑缺陷的必要几何夹具，不是训练权重，不应因“所有 NPZ 都是大数据”误删。反之，私人死亡归档和 rollout NPZ 不应整体提交。

## 3. Python 模块逐项说明

### 入口、Web 与本机生命周期

| 文件 | 作用 | 主要关联 |
| --- | --- | --- |
| [__init__.py](../th06ai/__init__.py) | 包入口；避免加入游戏／GPU 初始化副作用 | 所有导入者 |
| [__main__.py](../th06ai/__main__.py) | CLI 参数、open/shutdown/Web/guard 分流 | config、console_lifecycle、supervisor |
| [config.py](../th06ai/config.py) | ROOT、端口、Config/SafetyConfig 与校验 | 保护、服务、worker |
| [console_lifecycle.py](../th06ai/console_lifecycle.py) | 启动复用、身份／端口／启动锁、就绪等待、关闭协调 | CLI、HTTP、所属本机进程 |
| [web_console.py](../th06ai/web_console.py) | 请求模型、回环路由、token/Origin、SSE、静态 UI | ConsoleService、SnapshotBroker |
| [console_service.py](../th06ai/console_service.py) | 模型 ID、会话状态、明确操作、日志与本机配置缓存 | model_store、control、files |
| [console_stream.py](../th06ai/console_stream.py) | 共享采样、DTO、diff、曲线游标、SSE 编码与背压 | 服务快照、前端 stream.ts |
| [dashboard.py](../th06ai/dashboard.py) | 旧 Tk 兼容控制台 | 同一 CLI／保护协议；非主界面 |

### 独立硬件保护与 Windows 边界

| 文件 | 作用 | 修改重点 |
| --- | --- | --- |
| [gpu.py](../th06ai/gpu.py) | GpuSample、SafetyPolicy、NvmlReader、AsyncTelemetry | 锁存故障、可选传感器、读数年龄、不可阻塞 F8 |
| [driver_events.py](../th06ai/driver_events.py) | Windows 驱动事件订阅与错误收集 | 不把事件丢失伪装成没有错误 |
| [lease.py](../th06ai/lease.py) | 共享内存健康授权及命名互斥量 | 唯一 owner、原始采样时刻、LeaseBusy |
| [supervisor.py](../th06ai/supervisor.py) | run_guard、预检、预算、Job、F7/F8、最终状态 | worker 与保护独立，故障不自动续训 |
| [hotkeys.py](../th06ai/hotkeys.py) | 事件监听、边沿、独立分派、轮询回退 | 持 Shift 也能响应，回调不做重工作 |
| [win32.py](../th06ai/win32.py) | ReadOnlyProcess、GameWindow、输入、进程枚举、OwnedJob、GuardLock | 只读权限、指针宽度、结构体 ABI、所属进程 |
| [focus.py](../th06ai/focus.py) | 有界窗口激活和前台确认协调 | 不把普通激活失败变成不可恢复停训 |
| [worker.py](../th06ai/worker.py) | 等保护授权后加载 Torch，分流实机模式；Heartbeat 与校准 | 不绕过授权，不把自检当训练模型 |
| [backend.py](../th06ai/backend.py) | CUDA 前反向与环境自检，写环境准备状态 | 保护下短测，非用户训练权重 |

### 游戏状态和几何语义

| 文件 | 作用 | 输出 |
| --- | --- | --- |
| [native.py](../th06ai/native.py) | 版本白名单、PE/进程特征、批量采集、原生字段与输入读取 | Frame、ControlSample、GameResources |
| [game_phase.py](../th06ai/game_phase.py) | 暂停／对话／菜单／结算等分类 | PhaseState 与 phase_of |
| [stage.py](../th06ai/stage.py) | 原生 Stage 解码、标签、one-hot、上下文边界 | 1–7/None、8 维特征 |
| [combat.py](../th06ai/combat.py) | 攻击目标与本体独立解码、可信伤害、对齐势函数 | enemies、enemy_bodies、damage feedback |
| [lasers.py](../th06ai/lasers.py) | 64 槽直线激光布局、可见状态、碰撞半宽 | 8 列激光数组 |
| [bullet_state.py](../th06ai/bullet_state.py) | pool slot 匹配、运动已知性、合并半径、最近接近估计 | BulletState |
| [body_state.py](../th06ai/body_state.py) | 本体／激光中心、速度、角度及不确定性 | BodyState |
| [motion.py](../th06ai/motion.py) | 输入确认后的稳定移速测量和校准结果处理 | 有效速度样本，不含切键／撞墙截断 |
| [risk.py](../th06ai/risk.py) | 普通弹和当前伤害激光的可见风险反馈 | threat、min_clearance、coverage |

### 决策、奖励与原生加速

| 文件 | 作用 | 相关契约 |
| --- | --- | --- |
| [observations.py](../th06ai/observations.py) | 观测空间、历史控制、编码及 action_keys | 观测 v8／36 动作 |
| [objectives.py](../th06ai/objectives.py) | 库存相关资源权重、选取、Power 势函数 | 不越过碰撞约束 |
| [rewards.py](../th06ai/rewards.py) | RewardTracker、资源预算、驻留／顶墙、奖励分项 | 奖励 v8、gamma |
| [planner_spec.py](../th06ai/planner_spec.py) | 轻量规划常量及 PlanningConfig | 保护进程可安全导入 |
| [navigation.py](../th06ai/navigation.py) | 导航／决策列及出口、条件 B 先验常量 | navigation/decision |
| [planning.py](../th06ai/planning.py) | 连续几何、beam 搜索、时序探测、回退、原生选择 | PlanResult／规划 v7 |
| [acceleration.py](../th06ai/acceleration.py) | 可选扩展发现、拥有型序列化、RustKeyController | API v2／搜索 v1 |
| [native_planner/__init__.py](../th06ai/native_planner/__init__.py) | 独立搜索扩展包入口 | 无二进制时仍可导入包 |
| [policy_spec.py](../th06ai/policy_spec.py) | 轻量策略版本及 BOMB_PRIOR | 策略 v2 |
| [policy.py](../th06ai/policy.py) | BulletEncoder、ResourcePolicy、条件探索 | 同一联合概率分布 |
| [inference.py](../th06ai/inference.py) | 推理缓冲、CUDA Graph、一次小回传、可选事件计时 | Decision，不拥有游戏句柄 |

### 实际采样、学习和持久化

| 文件 | 作用 | 相关组件 |
| --- | --- | --- |
| [environment.py](../th06ai/environment.py) | 游戏状态机、输入、reset/step、时序、奖励、跨关 | 所有 worker 游戏模式共享 |
| [control.py](../th06ai/control.py) | TrainingControl、PAUSE/FOCUS/FINISH 协作、采样中断 | 与紧急保护分离 |
| [input_timing.py](../th06ai/input_timing.py) | 主机时间线、匹配方向位、最近分位数 | 不发键、不做停机决策 |
| [pausable_ppo.py](../th06ai/pausable_ppo.py) | 单环境 rollout gate、真实转移、边界自举 | SB3 派生流程 |
| [ppo_update.py](../th06ai/ppo_update.py) | 资源策略 PPO 更新与探索损失 | 与 ResourcePolicy 保持一致 |
| [learning.py](../th06ai/learning.py) | 模型创建／加载／迁移／保存，回调与独立评估 | 不删除原模型，安全边界更新 |
| [model_store.py](../th06ai/model_store.py) | 检查点兼容检查、目录索引、内容寻址保留 | ZIP + 同名 JSON |
| [diagnostics.py](../th06ai/diagnostics.py) | 原始快照／轻量样本环、死亡归档、有界异步写盘 | 非关键诊断失败标记缺失 |
| [log_rows.py](../th06ai/log_rows.py) | action/episode 字段映射、落盘时序列化 | 统一 CSV 解释 |
| [files.py](../th06ai/files.py) | 原子 JSON、Windows 共享删除读、时间戳和运行目录 | 不把读者锁竞争变成授权故障 |
| [profile_inference.py](../th06ai/profile_inference.py) | 受保护的离线推理基准 | 不接入游戏、不更新用户模型 |
| [profile_latency.py](../th06ai/profile_latency.py) | 只读游戏、规划、GPU 推理链剖析与固定几何对照 | 不发键；不是完整训练链 |

## 4. 前端逐项说明

| 文件 | 作用 |
| --- | --- |
| `main.tsx` | React 根、StrictMode、独立 benchmark 构建分流 |
| `App.tsx` | 训练／模型／诊断页面、操作表单、布局与对话框 |
| `useConsole.ts` | bootstrap、SSE 生命周期、控制请求、连接／日志状态 |
| `stream.ts` | snapshot/delta/heartbeat 校验、引用协调、路径归并、曲线游标 |
| `types.ts` | UI DTO 的 TypeScript 契约；不是全部原生 Frame |
| `logic.ts` | 格式化、状态派生、模型筛选等纯显示逻辑 |
| `style.css` | 主题、布局、响应式、可访问焦点及减少动画偏好 |
| `renderProfile.tsx` | 独立 benchmark 模式的 React Profiler，不在普通生产路径运行 |
| `logic.test.ts / stream.test.ts` | 纯逻辑、增量协议与引用稳定性回归 |
| `vite.config.ts` | React 插件、默认 API 代理、benchmark 别名、正式关闭 sourcemap |

前端不存储训练模型或决定游戏路线。UI 页面状态可以持久化偏好，但不能让刷新／断线重连产生启动、暂停或 F7 操作。

## 5. Rust 和工具

`native/src/lib.rs` 定义 Python 模块、sweep ABI、输入 ABI、互斥 held 状态和平台 SendInput。`planner.rs` 维护拥有型 Scene/Node 与完整搜索，调用同一连续几何内核；不存在一个跨游戏帧共享的可变全局规划缓存。

| 工具 | 用途 |
| --- | --- |
| `tools/fetch_torch_cuda.py` | 官方 wheel 索引、分片、长度与 SHA-256 校验 |
| `tools/build_native.py` | 当前进程设置构建环境、release/test、普通与 planner-only 安装 |
| `tools/check_docs.py` | 文档链接、代码索引、配置示例、AI 上下文与模块覆盖核验 |
| `tools/check_source.py` | 从 Git 获取实际候选／暂存清单，检查生成物、模型、个人路径；不删除或暂存文件 |

根目录三个 CMD 只负责从自身路径启动 Python CLI，不能擅自增加训练命令。`install.ps1` 不覆盖已有 config；`build-ui.ps1` 用锁文件构建 UI。发布主要使用当前 Git 根目录，不需要额外复制工作树。

## 6. 扩展功能时通常改哪里

- **新显示指标**：先保留原始 worker／日志证据，再决定 DTO、types、显示与 SSE 是否需要；不把完整高频数据塞进 UI。
- **新奖励**：rewards/objectives → 相应历史观测 → 评估指标；检查死亡与饱和边界。
- **新攻击几何**：版本映射 → 读取／已知性 → 风险／规划 → Python/Rust 对照 → 实机核对。
- **新模型输入**：数据契约 → 编码器 → 迁移／Adam → 推理缓冲／图 → 轨迹记录 → 测试。
- **新控制操作**：明确授权语义 → 服务 → 标记／worker gate → 生命周期／UI；不要借某个 GET 或刷新事件隐式执行。

具体测试见[测试与性能](testing-and-performance.md)。
