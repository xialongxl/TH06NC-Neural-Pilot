# 可选 Rust 后端

本项目的碰撞计算与按键后端都可在网页“保护与诊断 → 计算后端”切换为 Python、Rust 或自动。自动模式在扩展可用时选择 Rust，否则使用 Python；修改在下一次会话生效。模型文件、现有观测尺寸和神经网络参数不变。

运行已经编译好的 `th06ai/_native_backend.cp310-win_amd64.pyd` 不需要安装 Rust。该文件用于 Windows x64 / Python 3.10。换 Python 小版本或操作系统时要重新编译。

重新编译需要 Rust 和 MSVC Build Tools；依赖版本由 `Cargo.lock` 固定。使用系统 PATH 中的工具时：

```powershell
.\.venv\Scripts\python.exe tools\build_native.py
.\.venv\Scripts\python.exe tools\build_native.py --test
```

也可传 `--toolchain <含 cargo.exe 和 rustc.exe 的目录> --vcvars <vcvars64.bat 的路径>`，或在已配置好的开发终端运行。脚本只为当前编译子进程设置环境，缓存位于本项目 `.cache/cargo`、`.cache/rust-target`；不会安装系统级 Rust 或修改全局 PATH。重新编译前关闭本项目后端，Windows 不允许覆盖正在加载的 `.pyd`。

既可以使用系统工具链，也可以传入自己的便携 Rust/MSVC 目录。构建脚本没有作者电脑的固定工具链路径；`--toolchain` 指向含 `cargo.exe/rustc.exe` 的目录，`--vcvars` 指向自己的 `vcvars64.bat`。历史验证使用过 Rust 1.99.1，当前依赖由 Cargo.lock 固定；这不是对所有其他编译器版本的兼容保证。

碰撞输入先复制成独立的 float32 快照，Rust 在释放 GIL 后计算，不持有可变 NumPy 内存。碰撞内核没有可变全局状态；按键控制器以互斥锁串行管理 held 状态和 SendInput 批次。Python 包装器另有可重入锁。原有独立 F8 保护仍可直接释放按键，不依赖 Rust 工作线程的锁。

几何包含普通圆形弹幕、敌机本体矩形和有朝向的直线激光矩形，均连续检查路径，并在撞墙处拆分速度。激光宽度、状态和旋转坐标变换依据当前游戏可执行文件映射，不能套用其他版本地址。

整段搜索也可使用 Rust：安装 `th06ai/native_planner/_native_backend.cp310-win_amd64.pyd` 后，碰撞选择 Rust/自动时会一起启用。候选展开、切键时序探测、排序评分、回退和出口检测在一次释放 GIL 的调用中完成，输入是每次调用独立拥有的快照，没有共享可变搜索缓存。Python 参考实现保留；选择 Python 或没有搜索扩展时使用参考路径。诊断字段 `search_backend` 区分整段搜索与仅碰撞内核。

默认构建会安装搜索与原碰撞/按键扩展。只更新独立搜索扩展可使用 `tools/build_native.py --planner-only`（同样支持 `--toolchain` 和 `--vcvars`），无需覆盖正在使用的原按键扩展；正在运行的会话仍需结束后新建会话才会加载新代码。扩展不可用时的回退不重置模型，也不改变保护阈值。

历史研究目录默认不随源码公开。公开的性能范围和复核方式见 [测试与性能](../docs/testing-and-performance.md)。离线按键测试使用 dry run，未发送真实按键；组件耗时不能代表整条控制链。游戏 60Hz 采样、系统调度、读取和模型推理也计入实际体验，SendInput 成功只表示事件进入队列。

参考：[PyO3 并行与 GIL](https://pyo3.rs/v0.30.1/parallelism)、[Microsoft SendInput](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput)。
