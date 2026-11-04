# 控制延迟、诊断与保存配置

[English](en/performance-tuning.md) · 简体中文

本页说明 2026-11-03 实施的优化、兼容边界与复核方式。优化代码与离线组件基准不等于已验证整条实机链低于 2ms；真实测试必须另行获得操作者授权。

## 已实现的路径

| 路径 | 实现与保持的不变量 |
| --- | --- |
| 批量只读采集 | 默认保留复用的大块缓冲，将小块控制/相位读取合并到原生调用；显式 rust 可将大块读取也合并。输入范围和结果均 owned，释放 GIL 时不借用 NumPy；保留全池、本体、激光和跨帧记录 |
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
    "checkpoint_every_rollouts": 1
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

默认仍保留每轮完整轨迹和每轮模型保存，不悄悄牺牲恢复能力。NPZ 的 metadata 明确 trajectory_level；summary 不应被当成完整观测文件。外部 CPU 测试夹具在没有 SessionDiagnostics 时保留同步写盘兼容路径。

增大模型保存间隔后，未保存的新权重不能冒充旧检查点：相应轨迹的 policy_checkpoint 为 null，last_saved_checkpoint 只记录最近磁盘副本，policy_gradient_epochs 保留真实更新次数。

## 新的时段诊断

输入延迟诊断 v3 保留原始同动作时间戳，新增：

- `input_read_breakdown`：批量采集、解码、相位、输入字段读取；Rust 路径另有 native_compute_ms / native_handoff_ms，后者可定位 GIL 交接及调度等待。
- `input_pre_policy_work`：env_diagnostics_ms、buffer_add_ms、callback_ms，绑定于产生该次决策的同一观测。观测被刷新后不套用旧帧数字。
- worker 状态中的 rollout_sampling_wall_ms、rollout_pause_ms、ppo_update_and_callbacks_ms、checkpoint_save_ms、rollout_resume_and_observe_ms、trajectory_prepare_ms，分别区分采样、暂停、更新、保存、恢复与轨迹准备。
- 诊断 helper 的 last_trace_ms / last_action_info_ms 等记录写入侧耗时；这些不是主控制线程等待耗时。

各段可能有包含关系，独立 P95 不可相加。SendInput 返回、读到游戏输入匹配、首次位移与画面显示仍是不同指标。保持相同键组合而事件数为 0，不应被当作一次零延迟新发键。

## 本机成对组件对照

以下不是实机接管；每对交替顺序、同输入、先核对输出。不同数据规模和机器不能直接套用。

| 场景 | 样本对数 | 原路径 P95 | 优化／备选 P95 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 295 个模型的完整目录扫描／增量扫描 | 12 | 691.84ms | 25.81ms | 元数据与排序一致，增量扫描另在后台执行 |
| 固定合成 640 弹，奖励＋完整观测／共享特征 | 96 | 6.17ms | 4.99ms | 奖励及全部观测数组一致 |
| 测试进程自身 2.51MiB 内存，复用缓冲／全 Rust owned 结果 | 96 | 1.32ms | 5.08ms | 全 Rust 复制路径更慢，因此默认保留大块复用缓冲 |
| 测试进程自身 4 个小块控制字段，共 449 bytes | 256 | 0.0116ms | 0.0051ms | 内容相同，auto 合并小块读取；不代表完整采集耗时 |

第三项没有读取游戏，不是游戏 ReadProcessMemory 的实测，也未注入人为 GIL 负载。CUDA 连续缓冲只完成 CPU 数值等价验证，新的 CUDA Graph 与整条实机发键路径仍需独立保护下的授权复测。

## 版本与应用方式

观测仍为 v9、策略 v2、规划 v7、动作计时 v3；奖励为 v9、输入诊断为 v3、原生 ABI 为 v3、UI SSE 为 v2。没有为了本次优化删除模型或重置优化器。

重新构建 Rust 使用 [原生构建说明](../native/README.zh-CN.md)。ABI v3 安装在 native_runtime，首次构建不会覆盖旧 v2 会话已映射的 DLL；后续覆盖同一已加载 v3 DLL 前仍需结束相应进程。正在暂停/运行的 worker 不会热加载新源码；应由操作者保存并结束会话，再正常关闭重开控制台。

离线回归与指标口径见[测试与性能](testing-and-performance.md)。重点覆盖 `test_optimization_contracts.py`、`test_state_capture.py`、`test_observation_traces.py`、`test_latency_profiler.py`、`test_native_search.py`、`test_rust_backend.py`、`test_model_store.py` 和控制台并发测试。CPU 输出一致不能代替新 CUDA 路径的实机验证。
