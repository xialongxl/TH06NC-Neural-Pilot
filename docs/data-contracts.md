# 数据、格式与版本契约

此文的表格用于解释源码，不替代版本常量、NumPy dtype 或读写实现。修改契约时同步更新[代码索引](code-index.json)、迁移和相关测试。

## 1. 版本是相互独立的

| 契约 | 当前值 | 真正来源 |
| --- | ---: | --- |
| 游戏身份 | build 25306795 + 固定 SHA-256 | `native.py:BUILD_ID/SHA256/SIGNATURES` |
| 观测布局 | 8 | `observations.py:OBSERVATION_VERSION` |
| 奖励语义 | 8 | `rewards.py:REWARD_VERSION` |
| 策略／炸弹结构 | 2 | `policy_spec.py:POLICY_VERSION` |
| 规划语义 | 7 | `planner_spec.py:PLANNER_VERSION` |
| 动作计时语义 | 3 | `observations.py:ACTION_TIMING_VERSION` |
| 输入延迟诊断记录 | 2 | `input_timing.py:InputTiming.begin` |
| 碰撞／按键原生 ABI | 2 | `native/src/lib.rs:API_VERSION` |
| 独立搜索 ABI | 1 | `native/src/lib.rs:SEARCH_API_VERSION` |
| UI 增量协议 | 2；保留旧协议 1 | `console_stream.py` / `frontend/src/stream.ts` |

不要因为字段都叫 `version` 就统一递增，也不要改常量来伪装兼容。整段 Rust 搜索是对现有规划语义的优化，因此没有强制重建模型格式。

## 2. `Frame` 与 `ControlSample`

定义在 [native.py](../th06ai/native.py)。`Frame` 是当前已观察状态，不是游戏内存的可写代理。

| 字段组 | 字段与约定 |
| --- | --- |
| 自机 | `tick,x,y,state,focused`；`state==2` 作为 Miss 证据 |
| 重要命名区别 | `Frame.focused` 是游戏自机低速／focus 标记；`reader.window.focused` 才是 Windows 前台窗口状态 |
| 场景 | `paused,stage_active,phase,phase_evidence,menu_selection,menu_ready` |
| 坐标范围 | `bounds=(low_x,low_y,width,height)`；玩家上界为 low+width/height；显示场地总宽高另按 width+2*low 计算 |
| 时序 | `observed_at` 为 monotonic 秒；三个 `*_ns` 字段为 perf-counter 纳秒，不混用 epoch 时间 |
| 关卡 | `stage_id` 为 1–7 或 None；7 代表 EX；`stage_raw` 保留原始读数 |
| 普通弹 | `bullets (N,3)`：pool slot,x,y；`bullet_sizes (N,2)`：width,height；可选 `bullet_states (N,)` |
| 道具 | `items (N,6)`：slot,x,y,type,vx,vy；来自 1024 槽池 |
| 攻击目标 | `enemies (N,8)`：slot,x,y,HP,maxHP,phaseHP参考,战斗flags,Boss |
| 接触本体 | `enemy_bodies (N,6)`：slot,x,y,halfWidth,halfHeight,geometryKnown |
| 直线激光 | `lasers (N,8)`：slot,originX,originY,startDistance,endDistance,angle,collisionHalfWidth,state |
| 资源 | `GameResources` 或 None |
| 输入 | `input_bits` 可为 None；`input_observed_ns=0` 表示没有有效观测时间 |
| 射击诊断 | `player_shots` 可为 None；不能把未知写成 0 弹 |

数组空时保留二维列数，例如 `(0,8)`，不要用任意一维空列表代替。当前字段没有保证跨全部区块严格同帧；跨帧标记保留在 `state_reader` 诊断。

激光 `state=0` 为预告，`1` 为伤害阶段，`2` 为消退。当前解码输出可见预告和伤害阶段，消退排除。预告作为规划障碍，但不能仅凭预告判断已处于当前伤害或提高救命 B 探索。

`ControlSample` 仅包含 tick、自机位置／状态、暂停／场景／关卡、输入位与采集时间。它不含完整子弹、资源、敌机、激光池，不能替代新鲜的完整策略观测。

### 资源范围

```text
GameResources(score, power, bombs, lives, point_items, graze=None)
score: 0 <= value < 2^64
power: 0..128
bombs: 0..8
lives: -1..8
point_items: 0..65535
graze: None 或 0..999999
```

`graze=None` 可以表示原生计数未启用。点道具计数有已处理的 16 位回绕情况。道具类型为 0–6：P、点、大 P、B、满 P、残机、星等分值道具；以实际本版解码和 `objectives.item_weights` 为准，不能套用其他游戏的枚举。

## 3. 观测 v8 的十二个键

`ObservationBuilder.build()` 返回不带 batch 维的 float32 数组。单环境 PPO／PolicyRunner 再补 batch=1。

| 键 | 单观测形状 | 范围 | 内容 |
| --- | --- | --- | --- |
| `bullets` | `(64,5)` | -2..2 | 近场 dx,dy,vx,vy,valid |
| `player` | `(4,)` | 0..1 | x比例,y比例,低速标志,弹数比例 |
| `items` | `(32,12)` | -2..2 | 相对位置2、速度2、类型one-hot7、valid |
| `resources` | `(5,)` | 0..1 | Power/128、B/8、残机/8、对数分数、点道具计数/1000，随后裁剪 |
| `enemies` | `(32,8)` | -2..2 | 相对位置2、速度2、HP比例、Boss、可受伤、valid |
| `control` | `(28,)` | -2..2 | 上次动作与测得运动／反馈历史 |
| `full_bullets` | `(640,10)` | -2..2 | 全可见普通弹的几何／运动／最近接近估计 |
| `planning` | `(18,8)` | -1..1 | 每个当前移动的规划评价 |
| `stage` | `(8,)` | 0..1 | Stage1–6/EX one-hot7 + known |
| `navigation` | `(18,9)` | -1..1 | 安全类别、回退、未来出口等 |
| `decision` | `(8,)` | 0..1 | 当前决策上下文及目标可用性 |

padding 的 valid 为 0；集合池化必须用它过滤。未知运动的速度数值可以是 0，但必须由 known 位区分“实测静止”。重复同 tick 构建不能抹掉上一不同 tick 的运动估计。

### `full_bullets` 列

`dx,dy,vx,vy,motion_known,collision_radius,geometry_known,closest_time,predicted_clearance,valid`。

位置按场地总宽高归一化，速度除 16，半径除 128，最近接近 tick 除 24，净空除 512。几何有效时普通弹合并半径为 `min(width,height)*0.5 + player_radius`；缺失时保留估计及 unknown 标记。投影依据历史速度并假定自机不动，供特征与风险反馈使用，不是实际未来状态。

近场 64 和全场 640 是两个并行分支。扩大感知不能改变近场分支原来的集合和统计。激光不塞进普通弹 640 行，而由本体／规划／导航途径进入当前决策。

### `control` 列

| 切片 | 含义 |
| --- | --- |
| 0:18 | 上次请求 movement one-hot |
| 18 | 上次是否请求 B |
| 19 | 上次请求是否已知 |
| 20:22 | 实测自身速度／16 |
| 22 | 观测间隔 tick／60，裁剪 |
| 23 | 自身运动估计有效 |
| 24:26 | 两轴堵塞记忆／180 |
| 26 | 驻留程度 |
| 27 | 资源奖励递减程度 |

`planning` 列为 `enabled,allowed,clearance,collision_time,blocked,goal,utility,logit_bias`。原始距离／时间经过归一化；未碰撞的原始正无穷在给网络前裁剪为有限值。

`navigation` 列为 `robust_safe,fallback,prefix_risk,action_clearance,action_collision,wall_space,exit_fraction,exit_measured,nominal_collision`。

`decision` 列为 `known,fallback,late_control_risk,rescue_opportunity,bomb_allowed,time_to_contact,observation_age,safe_exit_fraction`。其中 `bomb_allowed` 受训练目标控制；它不是“现在应该按 B”。

## 4. 动作、输入位和时间

网络动作数保持 36，移动序号为 `action % 18`，方向为 `action % 9`。

| 编号范围 | 速度 | B 请求 |
| --- | --- | --- |
| 0–8 | 高速 | 否 |
| 9–17 | 低速，按 Shift | 否 |
| 18–26 | 高速 | 是 |
| 27–35 | 低速，按 Shift | 是 |

方向顺序：停、上、下、左、右、左上、右上、左下、右下。所有正常动作都含 Z。游戏确认 Z、暂停 Esc、重开 R 属于外围场景控制，不增加网络动作编号。F7/F8 属于保护快捷键，不进入网络动作空间。

原生方向／focus 位来自 `input_timing.py`：上 0x10、下 0x20、左 0x40、右 0x80、低速 0x04，`MOVEMENT_MASK=0xF4`。这套诊断匹配不是 Z/X 输入位的完整定义。

`InputTiming` 窗口最近最多 64 个有效样本，区分：

- 观测开始→发送完成；规划开始／结束；策略开始／结束。
- SendInput 调用耗时与整段 key switch 耗时。
- 首次读到匹配方向位的时间与 tick。
- 首次观察位移、保持旧键、没有新事件、未知输入位。

首次读到匹配是采样上界，不是精确硬件生效时刻。原生 tick 基准为 60Hz；`time.perf_counter_ns` 用于主机时段，ISO 时间用于展示，不能混减。

## 5. 模型与迁移

```text
models/
  latest.zip                 SB3 模型、优化器及学习计数
  latest.json                哈希、版本、来源、steps 等元数据
  checkpoints/
    step-000074318-<digest>.zip
    step-000074318-<digest>.json
```

上述步数仅示意命名。源码默认不带用户模型。加载还受 `models` 根目录与元数据校验约束；不能把“位于目录内”理解为任意外部 ZIP 已经过安全审查。

元数据核心键包括 `game_sha256,model_sha256,steps,gradient_updates,format,reward_version,planner_version,action_timing_version,policy_version,source_checkpoint,migrated_from,training_goal`。`gradient_updates` 镜像 SB3 `_n_updates`，不能直接等同采样批次数或每个 optimizer.step 次数。

观测 v1–v7 走显式迁移：兼容参数原样复制，追加融合列从零影响开始，兼容 Adam 状态保留。策略 v2 的 B 分支不反复重置。奖励或目标变化通常用于后续 rollout，不要求删除旧权重；是否值得独立新训练应是明确实验选择。

## 6. 运行目录与训练证据

| 文件 | 内容与解释 |
| --- | --- |
| `runs/current.json` | 最近公开会话目录指针；剖析模式不会覆盖它 |
| `status.json / result.json` | 保护状态／最终结果，带 GPU、worker、reason、code |
| `worker.json / worker.log` | worker 活性、阶段和异常栈 |
| `gpu.jsonl` | 每次实际 GPU 采样记录；不能把旧行当新遥测 |
| `actions.csv` | 每次已发生动作、阶段、奖励／资源／几何摘要和原始时序 |
| `episodes.csv` | 生存、死亡、关卡、B 使用等逐局指标 |
| `sampling_events.jsonl` | 刷新、中断、边界等采样事件 |
| `observations-*.npz` | 实际 on-policy 输入与 actions/rewards/values/log_probs 等 |
| `death-*.npz` | 死亡前短时原始快照及控制样本 |
| `evaluation_progress.json` | 每局落盘的固定权重评估进度，可为未完成 |
| `evaluation.json` | 完整评估报告；只在请求局数全部完成后生成 |
| `training_summary.json` | 正常结束时的训练摘要 |

训练 action/episode CSV 在暂停边界批量写入；急停可能使最近未刷新的行丢失。不能把 CSV 最后一行当作进程真正执行的最后一步，也不能把未完成 rollout 叫作已更新模型。

### PPO NPZ

`obs_<key>` 保留 `steps × env × 原观测形状`，例如 `(512,1,640,10)`。保存实际动作与采样时的 value/log probability。`metadata.gae_ready` 为 false 时不伪造 returns/advantages；GAE 未计算的零值不能当训练目标。

### 死亡 NPZ

`frames` 每行 12 列：tick、stage、自机 x/y、状态、暂停、低速、半径、输入位、Power、B、残机。另有采集纳秒、phase、轻量控制记录。ragged 池使用 `<name>_offsets` 指向拼接数组。

普通弹列为 slot,x,y,width,height,state；本体 6 列、激光 8 列、道具 6 列。归档未保存完整敌人 HP 目标列表，回放不能凭空补造。`metadata.native_damage_source_id` 保持未知，不能因附近有某颗弹就称已查明原生唯一死因。

## 7. Rust 二进制桥接

这些 bytes 用于 Python↔Rust，**不是浏览器传输格式**。

| 接口 | 输入 | 输出 |
| --- | --- | --- |
| `sweep_bytes` | 小端 float32；starts 2N、velocities 2N、durations N、ticks N、bounds 4、circles 6M、bodies 8K | `(N,5)` float32：endX,endY,clearance,collisionTick,blockedFraction |
| `plan_bytes` | 小端 float64 的拥有型快照，固定头＋7M circles＋9K bodies＋5Q items；延迟、分段等另传 | features、paths、净空、碰撞时间、navigation、decision 的 float32 bytes，加诊断 meta |
| `NativeInput.set(mask)` | 支持键集合的位掩码，实例互斥锁串行 | 已持有 mask、事件数；实际调用还有 Python 主机计时 |

搜索快照长度为 `88+7M+9K+5Q` 个 float64；Q≤32。返回路径点数为 `len(durations)+2`。精确头顺序与 meta 顺序见 `acceleration.rust_plan`、`planner.rs`、`plan_bytes`，不要在另一处复制后独立演化。

输入先拥有再 detach；不能把 NumPy view 指针留下供 GIL 外稍后使用。原生缓冲的有限数值、形状和计算预算检查不能省略。

## 8. UI 状态协议

REST 路由和写请求约束见[开发者指南](development.md#控制-api)。bootstrap 返回 `token,protocol`，完整诊断在 `/api/state`，SSE 默认兼容协议 1，当前前端明确请求协议 2。

```json
{"protocol":2,"kind":"snapshot","revision":1,"state":{"...":"完整显示DTO"}}
```

```json
{"protocol":2,"kind":"delta","base":1,"revision":2,"set":[[["active"],true]],"remove":[]}
```

```json
{"protocol":2,"kind":"heartbeat","revision":2}
```

delta 的 set/remove 使用路径数组；前端拒绝危险原型路径。曲线 `history` 使用 append+after 游标或 reset，最多 180 点。版本断层必须重新同步。DTO 刻意不传内部高频时钟和完整弹幕池；不能让 UI 显示协议成为训练记录的唯一来源。
