# 五条核心执行链

[English](en/core-logic.md) · 简体中文

阅读顺序建议：状态 → 决策 → 实际转移与学习 → 保护 → 控制台。精确符号见[代码索引](code-index.json)，字段定义见[数据契约](data-contracts.md)。下文的伪代码只解释控制关系，不可直接替换实现。

## 1. 状态读取与观测编码

### `GameReader`：把版本固定的进程内存转换为当前场景

位置：[native.py](../th06ai/native.py)。主要接口：

```python
GameReader(game_exe: Path)
reader.read() -> Frame
reader.read_control() -> ControlSample
reader.close()
```

构造过程先检查磁盘 EXE 的 SHA-256、PE 架构、映像大小和代码特征，再确认恰好一个同名游戏进程，且运行路径与配置一致。之后检查进程内代码特征。RVA 按实际模块基址解释，不把旧版地址加一个猜测的偏移。

`read()` 批量采集大区块并解码普通弹、道具、敌机、激光、资源、原生 tick、阶段及输入位，输出拥有必要独立数组的 `Frame`。内部复用读取缓冲降低系统调用和分配开销，外部不得持有稍后会被复写的临时视图。

`read_control()` 只读自机、时钟、输入及场景等小量控制字段，供保持动作期间和下发前使用，不反复搬运全弹池。轻量读到 Miss 后，即使下一份完整帧已经复活，也保留真实死亡证据。

输出中的 `capture_started_ns/capture_finished_ns` 是单调高分辨率时间戳；`input_observed_ns` 对应游戏输入位被读取的时刻。帧前后 tick 变化只记录跨帧诊断，不因为“读不到严格同帧”退出。

场景识别由 [game_phase.py](../th06ai/game_phase.py) 的 `classify()` 完成。暂停菜单、ECL 脚本停帧、Boss 对话、续关、结算、菜单过渡分别判断。空弹幕不是暂停标志；tick 不动也不是足够的暂停菜单证据。

### `ObservationBuilder.build(frame)`：编码信息，而非决定按键

位置：[observations.py](../th06ai/observations.py)。

输入为当前 `Frame`，状态包括上一份不同 tick 的场景、历史请求动作、墙边堵塞与资源反馈历史。输出为 12 个 float32 数组组成的字典。

1. 用稳定 pool slot 和前一份有效观测估计速度；重复构建同一 tick 时保留此前不同 tick 的历史。
2. 编码近场 64 弹和全池 640 弹，保留“运动已知／几何已知／有效”标记。
3. 编码有用的资源、Boss 优先的攻击目标、库存、自机、上一控制、关卡。
4. 调用 `RollingPlanner.plan()`，缓存 `last_plan`，把规划／导航／决策特征放入观测。
5. 按训练目标设定炸弹可用标志；NMNB 在这里表达目标，策略分布再执行一致的动作屏蔽。

它不通过“AI 应该安全擦弹”之类文字即时教授策略。可以学习的原则必须进入观测、奖励、分布约束或评估指标。

**改动关联**：新增输入字段必须检查网络融合层、模型迁移、Adam 状态扩展、轨迹保存、CPU/GPU 缓冲及测试，不能只改数组长度。

## 2. 规划与神经网络构成同一个决策

### `RollingPlanner.plan(frame, previous=None, control=None, state=None) -> PlanResult`

位置：[planning.py](../th06ai/planning.py)。输入还可复用已经算好的普通弹状态 `state`，减少重复处理。

返回：

| 字段 | 含义 |
| --- | --- |
| `features (18,8)` | 每个移动选择的允许、净空、碰撞时间、目标与偏置等 |
| `paths (18,P,2)` | 为每个初始动作保留的预测折线路径；P 随窗口分段变化 |
| `clearances/collision_times (18,)` | 原始几何指标，未碰撞可为正无穷 |
| `navigation (18,9)` | 稳健安全／回退／旧控制风险／出口等 |
| `decision (8,)` | 回退、迟到、救命探索机会、观测年龄等上下文 |
| `diagnostics` | 使用后端、时间窗口、候选数、已知性和风险解释 |
| `started_ns/completed_ns` | 真实规划时段，供下次延迟估计使用 |

关键步骤：

1. 根据采集年龄、预估本次规划、历史“规划结束到游戏确认输入”的成对记录估算切键窗口。
2. 模拟旧按键继续生效的前缀，再展开全部 18 个当前移动，搜索后续转向。
3. 用连续相对运动检查普通圆形弹，并在撞墙时拆分真实速度；敌机与激光用带朝向／扩张的矩形检测。
4. 仅排除同一有限预测模型下不可能到达的远处对象；不按最近 64 弹裁剪威胁。
5. 在无碰撞路线内比较资源、攻击对齐、净空、墙角／顶墙代价和后续机动空间。
6. 正常切键区间逐 tick 检查；异常宽区间最多取 12 个时刻并标记不完整，不授予稳健安全标签，也不停止训练。
7. 完整路线不可行时进入明确的风险回退，比较新动作可改变的碰撞部分。存在通过所有已测切键时刻的短时脱离选择时优先使用它们；不能把单一典型时刻的可行方向并回集合。
8. 对最终候选做有界出口探测，未知出口保留未知标志。

“未搜到安全路线”不同于证明无解；“通过匀速预测”也不同于实际绝对安全。新生、转向、传送和未知几何仍可能使预测失效。

### Python 与 Rust 的关系

Python 保留参考算法。`acceleration.rust_sweep()` 可只加速碰撞；ABI v3 的 `native_runtime._native_backend` 同时提供整段 `plan_bytes()` 搜索；保留旧独立搜索包作为兼容路径。

新搜索入口接收拥有型序列化快照，Rust 在释放 GIL 后执行候选展开、评分排序、时序探测、回退和出口检查。每次调用独立，不跨线程共享 NumPy 可变缓冲。`collision_backend` 与 `search_backend` 分开记录：看到碰撞为 Rust 不代表整段搜索已经原生化。

规划 v8 在两条搜索路径之后共用激光预警回退规则。完整切键区间内，含预警的本体探测若耗尽候选，先用每个切键时刻的短时伤害探测筛选，再核对当前可碰撞的敌机和伤害激光。候选继续按完整预测排风险，`planner_active_threat_priority` 记录本次重排。

若选择 Python 或没有可选搜索扩展，使用参考路径。默认 ABI v3 统一安装到 native_runtime；仍可用 planner-only 更新旧独立搜索位置，避免覆盖对应会话已映射的 DLL。正在运行的会话在正常结束后才加载新代码。

### `ResourcePolicy` 与 `PolicyRunner`

位置：[policy.py](../th06ai/policy.py)、[inference.py](../th06ai/inference.py)。

`BulletEncoder` 对子弹／道具／敌机逐点编码，采用 masked mean + max 聚合，再融合自机、库存、控制、全场弹、规划、关卡和导航特征。集合编码不依赖输入池槽顺序。

`ResourcePolicy` 将学习到的 logits 与缓存规划评价组合。没有资源或目标为 NMNB 时屏蔽 B；其他情形仍由学习分支决策，救命窗口的条件先验仅影响探索，不是“必死自动按 X”。

```python
runner = PolicyRunner(policy, backend="auto", profile=False)
runner.warmup(observation=None, lease=...)
decision = runner.predict(observation, deterministic=False, lease=...)
# Decision: actions, values, log_probs, bomb_probability, timing
```

`PolicyRunner` 将十二份输入映射到连续 CPU 页锁定缓冲与设备缓冲，CUDA 时使用一次完整 H2D 并可捕获固定形状图；小结果取回也复用固定缓冲。图捕获限制可回退 CUDA eager，设备丢失、非法访问、OOM 不会伪装成正常 CPU 回退。模型更新与图重放使用同一参数存储，避免图继续读旧权重。

**最重要不变量**：PPO 采样、实际执行和 `evaluate_actions()` 计算的分布一致。禁止在拿到 action/log_prob 后再按启发式替换方向或 B。

## 3. 环境、奖励、PPO 与检查点

### `TouhouEnv.step(action)`：把请求变成实际转移

位置：[environment.py](../th06ai/environment.py)，返回 Gym 五元组：

```text
(observation, reward: float, terminated: bool, truncated: bool, info: dict)
```

`action` 是 0–35 的同一已采样动作。`_prepare_action()` 先检查授权、会话预算、暂停请求和当前场景。`_capture_action()` 核对持续射击 Z 的状态，发送输入批次并验证采样。`_record_transition()` 接受已验证的结果，更新奖励、回合统计和下一份观测。

动作保持从下发后的新控制样本起算，正常保持至少两个原生 tick。轻量采样发现 Miss 时可以提前读完整结果，即使完整帧已进入复活状态，也要保留 Miss 证据。`info` 保留实际前一按键、计划、当前与结果激光／本体的只读数组、原始时间戳、奖励分项和生存指标。重型列表转换与动作日志编码由独立 CPU 诊断进程完成。

普通失焦和暂停转为可恢复控制状态；`SamplingInterrupted` 表达需重新观测，不能把中断动作当作正常经验。下发前观测过旧且近危险时最多刷新一次，第二次仍超预算只告警，不无限重试或凭空停训。

### `RewardTracker.step(frame, action, died) -> RewardStep`

位置：[rewards.py](../th06ai/rewards.py)。它基于真实前后观测计算奖励，不发送任何键。

奖励 v9 包括按实际 tick 比例的存活、首次中弹成本、顶墙及驻留反馈、真实库存／点道具净增、Power 势函数进度、资源目标势函数、射击对齐和可信 HP 减少。死亡时终止相关势函数，避免靠“拿 P → 死亡 → 重拿”循环刷反馈。

道具可达性使用经验证的当前自机高速移动速度；未知时保留原 4px/tick 基线。资源边际奖励使用预算递减；库存上限和实际需要还会改变目标权重。原始分数和擦弹主要作为诊断与上下文，不直接为擦弹加分。新手“顺路、安全再擦”的原则体现为不奖励主动贴弹，而不是给网络一段文字。

### `PausablePPO.collect_rollouts()` 与 `TrainingCallback`

位置：[pausable_ppo.py](../th06ai/pausable_ppo.py)、[learning.py](../th06ai/learning.py)。

```text
控制 gate → 必要时刷新观测 → 同分布采样
→ 实际 env.step → 保存实际转移 → 更新计数/回调
→ rollout 满 → 确认暂停 → 复制并排队轨迹 → PPO 更新
→ 保存已更新模型 → 恢复游戏并重新观测
```

实机仅支持一个环境和离散动作。默认 `n_steps=512`、`batch_size=64`、`n_epochs=4`、`learning_rate=3e-4`、`gamma=.999`、`gae_lambda=.95`。算法对资源策略使用 [ppo_update.py](../th06ai/ppo_update.py) 的更新实现，以保持炸弹探索目标和联合概率的一致性。

跨关或丢弃过旧决策时，先用上一段实际观测做价值自举并切断 GAE 连续性。暂停恢复期间发现的关卡交接走同一顺序，最后才刷新观测。已发生动作应先进入缓冲，再处理恰好到期的回调，避免日志／计数比保存转移多一步。

`TrainingControl` 从 worker 心跳线程缓存的看门狗快照读取有效训练时长。达到计划时长时，若仍在已识别的关卡中，锁定本次关卡并通过 `overtime_stage` 上报。剧情确认进入下一关、原生结算或本次尝试结束时，走正常保存流程。Miss 已触发回合结束时，采样器先提交真实终止转移，自动重置跳过重新入场。单关加时最多 600 秒有效训练时间。最后一批未完成的 rollout 仍按现有规则记录实际样本，检查点保存最近完成的参数更新。

`save_model()` 只保存至少完成过一次参数更新的模型。旧 latest 经验证后保留为内容寻址历史副本，保存临时 ZIP 后替换，并写同名 JSON。模型文件和元数据不是跨两个文件的数据库事务；校验不一致时拒绝使用或覆盖，而不是放宽校验。

训练正常结束时先释放按键，再核对保护授权并保存模型。若游戏仍在前台，退出清理会短暂尝试暂停。此时读取或按键失败会记录为 `worker.json` 的 `cleanup_error`，与检查点保存结果分开。

`load_model()` 仅接受项目 `models` 内自行生成且元数据／游戏身份／哈希通过的检查点。旧观测迁移保留原文件、学习量及兼容 Adam 状态；新输入列从零影响开始，策略 v2 不重复重置已学 B 分支。

## 4. 独立保护与共享授权

### `run_guard(config, mode, minutes=None, model=None, ...) -> GuardResult`

位置：[supervisor.py](../th06ai/supervisor.py)。生产调用使用 CLI 可见参数；reader／child command 等工厂注入只供故障测试，不暴露成用户执行任意命令接口。

返回包含 `code/reason/run_dir/child_pid`。保护状态锁存：故障后不自动重启或续训，必须新建保护会话。

`SafetyPolicy.check(sample, now=None)` 检查采样年龄、GPU 身份、有限数值、温度、显存、功耗、降频原因、风扇与可用错误计数。支持可选传感器缺失，但原本可用的能力中途失联不是正常“不支持”。

`SharedLease.publish()` 只允许 owner 写入 GPU UUID、原始采样时刻和保护进程 PID。Windows 命名互斥量保护整条记录，读取非阻塞。遇到 `LeaseBusy` 时只能使用最后一份完整授权的**原始时间戳**，不能把当前读时刻当成新健康采样。

worker 心跳与 GPU 数据超时是不同阈值；暂停不停止保护。F8 的处理不依赖 UI、PPO 回调或主采样循环。硬件紧急终止优先于写模型，保留之前的成功检查点。

`GuardedSessionClock` 是实机会话计时的唯一来源。初始化单独受启动截止约束。看门狗收到 `clock_started` 后开始计会话时间，根据 `clock_paused` 扣除人工等待，并根据 `scene_wait_step` 扣除对话、脚本和转场等待。同一采样步的自动等待合计最多扣除 60 秒。看门狗通过 `status.json` 发布快照，面板显示与 worker 的到期判断取自同一来源。worker 在已有心跳线程中读取，控制链使用缓存。

训练的计划时长、单关加时和执行硬截止基于同一份有效时长。达到计划时间后，训练 worker 报告有效 `overtime_stage` 可获得一次固定的加时额度。执行硬截止额外预留 120 秒给正常收尾；GPU 保护和 F8 仍可随时终止。

## 5. 控制台、会话服务与增量状态

### `create_app()` / `ConsoleService`

位置：[web_console.py](../th06ai/web_console.py)、[console_service.py](../th06ai/console_service.py)。

`ConsoleService.start()` 在会话锁外解析模型 ID 并校验检查点。它在提交启动前重新核对会话、关闭状态和检查点文件，持锁创建并登记保护进程。`control()` 串行提交 PAUSE／FOCUS／FINISH／STOP。关闭请求先于 worker 创建到达时，看门狗读取 `FINISH` 并正常取消启动。窗口激活和只读探测使用独立的游戏操作锁，后者先释放读句柄，启动才可继续。

所有写 HTTP 请求要求同源 `Origin` 和 `X-Console-Token`。令牌从 bootstrap 获取，每次服务创建重新生成。项目服务目前是回环控制工具，不是带用户登录、权限管理和远程隔离的服务器产品。

### `SnapshotBroker` / `applyFrame()`

位置：[console_stream.py](../th06ai/console_stream.py)、[stream.ts](../frontend/src/stream.ts)。

模型库由 CheckpointCatalog 缓存元数据，在后台增量扫描，完成后短暂加锁发布。`snapshot()` 取得会话和配置引用后释放会话锁，再读取状态文件和遥测。日志尾读与快照解析各自独立于会话控制，缓存仅在查询和发布时持锁。多个页面共享一个采样器。首帧 snapshot，之后 delta 包含 `base/revision/set/remove` 和曲线 append/reset，无变化时发 heartbeat。慢客户端合并到最新状态。前端发现版本断层、缺少初帧或游标不一致时重新完整同步。

前端保留未变化对象引用供 memo 跳过重绘。不要在每次心跳都把完整状态深拷贝；也不要为了减少 JSON 成本直接搬运完整弹幕池到浏览器。

## 结构化依赖摘要

```json
{
  "sense": {"entry": "GameReader.read", "feeds": "ObservationBuilder.build", "contract": "Frame"},
  "decide": {"entry": "RollingPlanner.plan + ResourcePolicy", "runner": "PolicyRunner.predict", "output": "Decision"},
  "learn": {"entry": "PausablePPO.collect_rollouts", "environment": "TouhouEnv", "reward": "RewardTracker", "storage": "save_model"},
  "guard": {"entry": "run_guard", "policy": "SafetyPolicy.check", "authorization": "SharedLease", "owner": "OwnedJob"},
  "control": {"entry": "create_app", "service": "ConsoleService", "display": "SnapshotBroker -> applyFrame"}
}
```

用于自动定位的扩展版本见 [ai-context.json](ai-context.json)。
