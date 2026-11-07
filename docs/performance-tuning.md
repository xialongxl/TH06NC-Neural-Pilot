# 控制延迟、诊断与保存配置

[English](en/performance-tuning.md) · 简体中文

本页记录控制链优化、兼容边界与复核方式。离线基准测量指定组件；整条游戏输入链需经操作者授权后独立实测。

## 已实现的路径

| 路径 | 实现与保持的不变量 |
| --- | --- |
| 批量只读采集 | 默认保留复用的大块缓冲，将小块控制/相位读取合并到原生调用；显式 rust 可将大块读取也合并。输入范围和结果均 owned，释放 GIL 时不借用 NumPy；保留全池、本体、激光和跨帧记录 |
| 弹幕与道具解码 | 先确定有效槽位，再读取这些槽位的坐标和其他字段；仍校验全池状态，保留有效槽位的顺序、可见范围和未知几何 |
| 可见弹特征 | 奖励与观测共享 BulletStateCache；按前后快照与几何匹配，不按 tick 单独缓存。原生数组只读，外部可变夹具还核对内容 |
| 控制链诊断 | info 保留只读数组，动作行编码、列表转换、死亡归档与轨迹压缩由独立 CPU 子进程执行；有界队列拥塞/错误明确计数 |
| 网络输入 | 十二份输入映射到连续 pinned host / device 缓冲，CUDA 只复制一次 H2D；小结果也复用缓冲。action/value/log probability 的含义不变 |
| 模型库 | 缓存未变化检查点的元数据，后台扫描后短锁发布；保存修订通知与定期扫描处理新增文件。选择/启动仍重新检查路径，加载仍校验哈希 |
| 模型列表 | 每页最多 25 个；搜索和兼容性过滤仍作用于整个目录 |
| 菜单与奖励 | Q 加入 Python/Rust 输入表；道具可达性采用已验证自机高速速度。奖励版本为 v9，旧权重/Adam 不因此重置 |

只读采集失败仍是读取失败；空弹幕、跨 tick 或正常短暂失焦没有因此新增退出条件。F7/F8、独立 GPU 保护、共享授权和所属 Job 的职责不变。

## 诊断进程的生命周期

`SessionDiagnostics` 在主 worker 保留最近原始帧环，`DiagnosticWriter` 启动自己的 CPU helper，通过私有 stdin/stdout 管道传递内部任务。pickle 仅用于这一内部管道，不接收 HTTP、磁盘或第三方提供的 pickle。子进程不打开游戏、不初始化 CUDA、不更新模型。

主线程只做有界入队。独立发送线程处理传输，子进程生成 CSV、NPZ 和 JSON；结果线程收集计数与错误。最多保留两个待写完整轨迹任务，队列满不会阻塞输入，而会增加 dropped_records。关键死亡记录若无法入队，另报 critical diagnostic 错误；不能把丢失证据的会话说成记录完整。

正常结束会在有限时间内排空队列。写入失败、退出超时或进程异常都记录到诊断摘要/错误计数；不会成为等待 GPU 紧急停止的前提。硬件停止所属 Job 时，helper 同属进程树。不要为了保日志延迟 F8 或强制先做 CUDA 保存。

`diagnostic_worker.log` 是 helper 启动/未捕获错误的排障入口，`diagnostics_summary.json` 是退出时的写入数量、丢失和错误证据。尚未完成异步写入的文件不会冒充成功轨迹。

## 配置

在本机 `config.json` 增加可选 diagnostics 对象；不填写沿用默认：

```json
{
  "diagnostics": {
    "trajectory": "full",
    "trajectory_every_rollouts": 1,
    "compression_level": 1,
    "checkpoint_every_rollouts": 1,
    "thread_profile_steps": 0,
    "readback_ab": false
  }
}
```

这只是配置片段，不能覆盖完整游戏配置。各项在下一次会话生效，已有文件与历史检查点不清理。

顶层 `capture_backend` 默认 auto：大区块保留原 Python/read_into 复用缓冲，小块控制/相位和 HUD 在可用时用 Rust 合并；python 完全使用 Python 采集；rust 显式启用大区块拥有型批量采集。后者不是默认更快的保证。碰撞/按键后端仍由原配置独立选择。

| 字段 | 范围／默认 | 含义 |
| --- | --- | --- |
| trajectory | full / summary / off；默认 full | full 记录实际 PPO 输入及动作/目标；summary 仅标量/动作/目标数组，无法重建完整观测；off 不生成 rollout NPZ。三个级别都不关闭死亡原始快照 |
| trajectory_every_rollouts | 整数 1–100；默认 1 | 每 N 个 rollout 记录一份；跳过的窗口不能用于完整采样回放 |
| compression_level | 整数 0–9；默认 1 | 0 不压缩；1 优先降低压缩时间；更高值可能减少磁盘占用、增加 CPU 成本 |
| checkpoint_every_rollouts | 整数 1–100；默认 1 | 常规已更新模型的保存间隔；正常保存结束仍保存最新已更新模型。放大间隔会增大硬件急停时可能损失的更新量 |
| thread_profile_steps | 整数 0–4096，默认 0 | 设为正数时，下一次会话记录最多这么多条完成动作的控制线程计数 |
| readback_ab | 布尔值，默认 false | 下一次 CUDA Graph 训练在首个 rollout 内采集最多 512 条回读对照样本，并自动启用该窗口的线程计数 |

默认仍保留每轮完整轨迹和每轮模型保存，不悄悄牺牲恢复能力。NPZ 的 metadata 明确 trajectory_level；summary 不应被当成完整观测文件。外部 CPU 测试夹具在没有 SessionDiagnostics 时保留同步写盘兼容路径。

增大模型保存间隔后，未保存的新权重不能冒充旧检查点：相应轨迹的 policy_checkpoint 为 null，last_saved_checkpoint 只记录最近磁盘副本，policy_gradient_epochs 保留真实更新次数。

## 新的时段诊断

输入延迟诊断 v3 保留原始同动作时间戳，新增：

- `input_read_breakdown`：批量采集、解码、相位、输入字段读取；Rust 路径另有 native_compute_ms / native_handoff_ms，后者可定位 GIL 交接及调度等待。
- `input_pre_policy_work`：env_diagnostics_ms、buffer_add_ms、callback_ms，绑定于产生该次决策的同一观测。观测被刷新后不套用旧帧数字。
- worker 状态中的 rollout_sampling_wall_ms、rollout_pause_ms、ppo_update_and_callbacks_ms、checkpoint_save_ms、rollout_resume_and_observe_ms、trajectory_prepare_ms，分别区分采样、暂停、更新、保存、恢复与轨迹准备。
- 诊断 helper 的 last_trace_ms / last_action_info_ms 等记录写入侧耗时；这些不是主控制线程等待耗时。

各段可能有包含关系，独立 P95 不可相加。SendInput 返回、读到游戏输入匹配、首次位移与画面显示仍是不同指标。保持相同键组合而事件数为 0，不应被当作一次零延迟新发键。

### 同步记录控制线程

在本机配置中设 `diagnostics.thread_profile_steps` 为 `512`，再按原方式开始会话。采样依次覆盖采集、规划前准备、规划、规划后准备、推理、发键前检查和下发。只有同一份观测、同一份规划与实际完成动作都对应时，才在 `actions.csv` 的 `input_thread_timing` 列留下记录。训练日志会在 rollout 边界批量刷盘。

```powershell
.\.venv\Scripts\python.exe -m th06ai.thread_profile .\runs\<会话目录>\actions.csv
```

报告给出各段墙钟耗时、线程周期以及最慢 5% 的同组动作。累计 CPU 时间只在至少一秒的同线程窗口中比较，窗口包含动作间隙。Windows 的 CPU 时间可能粗量化，线程周期保留原单位。线程未执行的时段可能来自 I/O、GPU 等待、休眠或调度，区分这些原因需要额外的调度轨迹。 [Windows 线程 CPU 时间](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getthreadtimes) · [Windows 线程周期](https://learn.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-querythreadcycletime)

### CUDA 结果回读对照

训练空闲时可运行以下命令。独立保护进程就绪后，工作进程用现有模型和完整观测归档回放推理，结果写入新会话目录的 `readback_profile.json`。

```powershell
.\.venv\Scripts\python.exe -m th06ai profile --profile-kind readback --minutes 1 --model models/latest.zip
```

两种回读方式使用同一张 CUDA Graph。随机动作的每对回放恢复同一份 CUDA 随机数状态，交替执行顺序，并核对动作、价值、动作对数概率和炸弹概率。GPU 事件覆盖输入传输和前向执行，主机的回读耗时与线程周期另行记录。报告也会用确定性回放验证普通会话的回读配置与策略输出。

普通会话默认使用 `inference_readback: "auto"`。成功启用 CUDA Graph 时，推理将结果异步复制到页锁定主机缓冲，等待同一流上的阻塞事件，再读取结果。CUDA eager 和 CPU 使用同步回读。将本机 `config.json` 的 `inference_readback` 设为 `"synchronous"` 可在下次会话恢复同步回读。

2026-11-07 的两轮对照各使用 64 份归档观测和 512 对随机动作。异步回读加阻塞事件使回读线程周期平均下降 54%–58%，整次推理线程周期下降 14%–17%。两轮推理 P95 各多约 0.03 ms。以上数字来自离线回放。

### 实机回读对照

在本机 `config.json` 中设 `diagnostics.readback_ab` 为 `true`，再正常启动一次续训。首个 rollout 的前 512 条有效动作按相邻两步配对，每对随机决定同步和阻塞事件的先后。两种方式共用当前策略，单次动作只推理和执行一次。采样窗口结束后恢复 `inference_readback` 配置选择的方式。若 CUDA Graph 不可用，工作进程状态记录 `readback_ab: unavailable`。

```powershell
.\.venv\Scripts\python.exe -m th06ai.thread_profile '.\runs\<会话目录>\actions.csv' --readback-ab
```

分析只配对同一关卡和战斗阶段的相邻有效动作，保留排除数量与每个指标的可用样本数。首个 rollout 内的两种方式使用同一组参数，动作仍面对自然推进的游戏帧。完成后将 `diagnostics.readback_ab` 设回 `false`，后续会话恢复默认配置。

2026-11-07 在 Lunatic、灵梦 A、Practice Stage 1、NMNB 的同一 rollout 中收齐 256 对。阻塞事件方案的平均推理线程周期减少 84.2%，采集到发键这段的线程周期减少 45.1%。采集到发键 P95 为同步 17.94 ms、阻塞事件 17.93 ms。两种执行顺序分别统计时，推理周期的降幅均约 84%。

## 本机成对组件对照

以下不是实机接管；每对交替顺序、同输入、先核对输出。不同数据规模和机器不能直接套用。

| 场景 | 样本对数 | 原路径 P95 | 优化／备选 P95 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 295 个模型的完整目录扫描／增量扫描 | 12 | 691.84ms | 25.81ms | 元数据与排序一致，增量扫描另在后台执行 |
| 固定合成 640 弹，奖励＋完整观测／共享特征 | 96 | 6.17ms | 4.99ms | 奖励及全部观测数组一致 |
| 测试进程自身 2.51MiB 内存，复用缓冲／全 Rust owned 结果 | 96 | 1.32ms | 5.08ms | 全 Rust 复制路径更慢，因此默认保留大块复用缓冲 |
| 测试进程自身 4 个小块控制字段，共 449 bytes | 256 | 0.0116ms | 0.0051ms | 内容相同，auto 合并小块读取；不代表完整采集耗时 |

第三项没有读取游戏，不是游戏 ReadProcessMemory 的实测，也未注入人为 GIL 负载。CUDA 连续缓冲只完成 CPU 数值等价验证，新的 CUDA Graph 与整条实机发键路径仍需独立保护下的授权复测。

2026-11-07 对最近一次训练保存的 171 个快照做了 3,420 对交替回放。Rust 规划候选的路径改用定长存储，矩形旋转系数在同一批碰撞检测中复用。两版规划输出逐字节一致。调用 `plan_bytes` 的平均线程周期从约 145 万降到 102 万，耗时 P95 从 1.77 ms 降到 1.24 ms。这里计量的是离线规划调用，实机核占用和采样到按键耗时需单独实测。 [Windows 线程周期计数](https://learn.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-querythreadcycletime)

采集解码先筛选敌机槽位，在激光池没有预警或生效槽位时直接返回。近弹和全场观测复用归一化结果，全场数组在复制近弹行后原地裁剪。普通关卡模拟内存快照做了 768 对交替读取，`GameReader.read()` 的中位数从 0.92 ms 降到 0.80 ms。64 份模拟快照的解码和观测、140 帧历史实机的观测均逐字节一致。模拟采集的耗时覆盖内存缓冲复制和解码，实机采样至发键延迟另行测量。

## 版本与应用方式

规划 v9 适用现有模型。当前使用观测 v9、策略 v2、动作计时 v3、奖励 v9、输入诊断 v3、原生 ABI v3 和 UI SSE v2。

重新构建 Rust 使用 [原生构建说明](../native/README.zh-CN.md)。ABI v3 安装在 native_runtime，首次构建不会覆盖旧 v2 会话已映射的 DLL；后续覆盖同一已加载 v3 DLL 前仍需结束相应进程。正在暂停/运行的 worker 不会热加载新源码；应由操作者保存并结束会话，再正常关闭重开控制台。

离线回归与指标口径见[测试与性能](testing-and-performance.md)。重点覆盖 `test_optimization_contracts.py`、`test_state_capture.py`、`test_observation_traces.py`、`test_latency_profiler.py`、`test_native_search.py`、`test_rust_backend.py`、`test_model_store.py` 和控制台并发测试。CPU 输出一致不能代替新 CUDA 路径的实机验证。
