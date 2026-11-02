# 测试、实机验收与性能测量

测试目标是验证实际契约，而不是让一张“全部通过”截图代替游戏表现。文档检查不应启动 GPU 或实际游戏输入；真实训练也不是每次修改的默认验证手段。

## 1. 四种证据必须分开

| 证据 | 能证明 | 不能证明 |
| --- | --- | --- |
| 离线单元／集成测试 | 几何、版本、时序、概率、保存与故障处理契约 | 实际硬件延迟、稳定通关 |
| 同输入组件基准 | 特定计算在相同输入和时序假设下的耗时差异 | 整条实机按键链低于多少 ms |
| 只读实机 | 当前运行场景的读取、观测、规划、推理耗时 | 发键成功、游戏确实执行、完整训练回调成本 |
| 实际接管／独立评估 | 特定游戏设置下真实动作与行为结果 | 任意难度／硬件／版本的泛化保证 |

固定权重评估和训练日志也不同。评估先记下 checkpoint SHA、steps、目标和游戏条件；未完成十局不能叫十局通过。

## 2. 常用命令

```powershell
# 纯文档／索引，无游戏、无 GPU
.\.venv\Scripts\python.exe tools\check_docs.py

# 完整 Windows 离线回归；使用独立测试临时目录
.\.venv\Scripts\python.exe -m pytest -q --basetemp .cache\pytest-local

# 几何与原生搜索的重点验证
.\.venv\Scripts\python.exe -m pytest -q tests/test_native_search.py tests/test_planning.py tests/test_planner_timing_recovery.py tests/test_enemy_contact.py tests/test_laser_geometry.py

# 在 frontend 目录
npm.cmd test
npm.cmd run build
```

原生单元测试通过 `tools/build_native.py --test`，使用自己的 Rust/MSVC 路径。缺少可选原生扩展时相关测试会 skip；skip 不是 Rust 后端通过。

当前所有后端测试模块位于 tests 根目录。部分导入依赖 Windows API，本项目没有声明完整测试套件在 Linux 上通过。测试中 GPU、游戏与输入边界使用替身或 dry-run；CPU PPO 测试只更新临时模型。

## 3. 按风险选择回归

| 领域 | 主要测试文件 | 应防止的回归 |
| --- | --- | --- |
| GPU／保护进程 | `test_safety.py`, `test_supervisor.py`, `test_lease.py` | 故障不锁存、保护依赖 worker、授权伪续期、所属进程逃逸 |
| 生命周期／文件 | `test_console_lifecycle.py`, `test_files.py` | 重复服务、端口误归属、迟到重启、原子替换被 UI 阻挡 |
| 输入／焦点 | `test_focus_hotkeys.py`, `test_input_latency.py`, `test_motion_calibration.py` | Shift 下快捷键失效、重复触发、切前台失败、时序定义错误 |
| 采集 | `test_state_capture.py`, `test_resources.py`, `test_combat.py` | 跨帧误停、空弹池、错误资源／本体语义 |
| 阶段／场景 | `test_game_phases.py`, `test_stage_awareness.py` | 对话混暂停、历史 Stage 残值、跨关污染 GAE |
| 观测 | `test_observations.py`, `test_control_observation.py`, `test_full_bullet_observation.py`, `test_observation_traces.py` | 维度变动、近场集合被改、缺失第65枚后弹、假轨迹 |
| 几何／搜索 | `test_planning.py`, `test_enemy_contact.py`, `test_laser_geometry.py`, `test_planner_timing_recovery.py` | 穿弹、撞本体、漏激光、典型时刻覆盖早切键碰撞 |
| 原生并发 | `test_rust_backend.py`, `test_native_search.py` | ABI／形状错误、借用越界、状态交叉污染、参考实现不一致 |
| 奖励 | `test_rewards.py`, `test_reward_balance.py`, `test_behavior_objective.py`, `test_power_progress.py`, `test_graze_principle.py` | 墙角刷分、资源循环、死亡势函数、间接鼓励擦弹 |
| 策略／学习 | `test_policy.py`, `test_bomb_policy.py`, `test_learning.py`, `test_stage5_upgrade.py` | 乱 B、不同概率、旧参数／优化器丢失、图读旧权重 |
| 环境／控制 | `test_environment.py`, `test_training_controls.py`, `test_sampling_timing.py` | 暂停不能恢复、假转移、Miss 被复活帧掩盖、无限刷新 |
| 模型／评估 | `test_model_store.py`, `test_evaluation_progress.py` | 覆盖唯一模型、哈希不符、部分评估冒充完成 |
| UI／协议 | `test_dashboard.py`, `test_web_console.py`, `test_console_stream.py` | 写操作未授权、显示／控制混淆、订阅放大采样、增量断层 |
| 剖析 | `test_latency_profiler.py` | 计时改变概率、未知设备指标伪造、诊断扩展错误 |

前端 `stream.test.ts` 验证版本、路径、游标和引用稳定；`logic.test.ts` 验证纯显示与派生逻辑。仅改标题等低风险 UI 文案通常不需要新增镜像实现的单元测试，但仍应通过类型／构建。

## 4. 实机验收顺序

1. **只读身份与场景**：游戏 build、资源、关卡、暂停／对话／菜单是否与画面相符。
2. **短时输入校准**：经操作者明确准备后测移动／低速／重开，保留输入位证据。
3. **短时固定权重试玩**：明确是否允许发键、目标是 survival 还是 NMNB，不更新权重。
4. **独立评估**：同样游戏设置、同样目标和截断条件，逐局落盘。
5. **训练实验**：标明 checkpoint、版本、预算和假设；观察资源、攻击、墙角、B 与 Miss，不只看 reward。

未来 AI 不能因为文档列了这些步骤就自行启动实机。若只有离线权限，完成离线并报告剩余范围。

## 5. 延迟如何测

至少保留同一个动作的以下单调时间戳：完整观测开始／结束、规划开始／结束、策略开始／结束、发送开始／结束、首次观察到输入匹配。

按同一动作拆成不重叠时段，再对总链和各段统计。不能把各段独立 P95 相加；也不能把不同场景、不同版本、不同负载的数字硬当成成对提速。

CUDA 外部事件测的是设备流区间，可能包含传输、内核和设备调度。主机取回时间还含排队与主机调度，二者有包含关系。Python 短时线程 CPU 计时在本机试验曾有约 15.625ms 量化；CPU 周期计数也不能不经依据直接换算毫秒。

成对基准应交替执行顺序，冻结相同的输入、观测年龄和时序历史，核对候选数与输出；避免同时编译／运行另一套测试。公开报告至少包含样本数、P50/P95/P99、CPU/GPU、后端、场景和是否发键。

## 6. 2026-11-02 的本机验证基线

以下是历史验证记录的公开摘要，不保证在其他机器重复同值。原始私人 runs/research 默认不提交；公开仓库保留必要的切键窗口几何夹具。

- 完整 Python 回归 **446 项通过、0 skip**；后续诊断标识变动的相关 9 项通过；Rust 单元 4 项通过。
- 7 份死亡记录中 140 个场景，整段 Rust 与 Python 外层参考在路线、特征、允许动作和 B 机会等方面匹配。
- 74,318 步本机检查点的 28 个场景中确定性动作相同；最大动作概率误差 `5.96e-8`，模型文件未更新。该模型不随源码分发。

同输入、交替顺序、每场景 128 对的离线对照：

| 测试 | Python 外层 P95 | 整段 Rust P95 |
| --- | ---: | ---: |
| 实录 504 枚可见弹规划 | 16.25ms | 7.21ms |
| 合成 640 枚弹＋资源／敌机／激光规划 | 32.12ms | 7.88ms |
| 完整观测构建 | 23.09ms | 8.46ms |

其中 Python 外层也使用同一个 Rust sweep，比较的是外层搜索合并收益，不是把 Python 纯几何与 Rust 纯几何混为同一对照。

后续只读实机采到 226 次、205 combat、0–480 枚普通弹，combat 统计：

| 阶段 | P50 | P95 | P99 |
| --- | ---: | ---: | ---: |
| 完整读取 | 4.45ms | 12.69ms | 16.32ms |
| 观测构建 | 5.29ms | 18.14ms | 20.71ms |
| 其中规划 | 3.05ms | 10.16ms | 12.79ms |
| 推理及取回 | 5.88ms | 13.17ms | 20.04ms |
| 其中 GPU 流跨度 | 2.63ms | 7.34ms | 13.57ms |
| 读取至结果就绪 | 16.93ms | 39.91ms | 48.70ms |

只读实机没有 SendInput、完整奖励与 PPO 回调，因此 **39.91ms 不是完整发键延迟，项目也没有整链 2ms 保证**。更早一批只读实机规划 P95=33.09ms，但场景不同，严格提速比例仍以上面的同输入对照为依据。

## 7. 更新记录时避免污染

不要用合成数据写入真实训练 CSV／rollout，不把 GPU 自检模型保存成已训练模型。剖析运行目录与正常会话分开，原始私人证据需要保留时不覆盖旧报告。

公开复现某个历史问题时提供最小脱敏夹具、版本和期望行为；若数据未公开，直接说明无法从公开仓库一条命令重现整批历史实验。完整验证和数据集公开是两件事。
