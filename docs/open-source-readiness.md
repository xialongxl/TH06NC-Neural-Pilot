# 开源准备、直接推送与发布内容

[English](en/open-source-readiness.md) · 简体中文

项目名 **TH06NC-Neural-Pilot**，作者 **xialongxl**，标准 **MIT**。主要分发方式是当前根目录直接作为 Git 仓库推送，不需要手动复制“干净版”，也不需要先清理本机依赖和模型。

## 1. 这次准备包括什么

- README 与架构、模块、数据、开发、边界、测试及 AI 交接说明。
- 标准 MIT LICENSE、用途／原作权利声明、SB3 派生来源及上游许可保留。
- CONTRIBUTING、Issue 模板、PR 模板。
- 无作者目录的 config.example.json，以及明确不会自动加载的 .env.example。
- Git 忽略规则、代码符号索引、离线文档检查。
- 安装脚本仅在没有 config 时创建示例，不覆盖现有本机配置。

维护者已明确指定并创建公开仓库 [xialongxl/TH06NC-Neural-Pilot](https://github.com/xialongxl/TH06NC-Neural-Pilot)，默认分支为 main。后续同步以本地／远端 Git 提交核对为准；这不代表注册 PyPI 包、分发训练模型或完成通关验收。

## 2. 提交哪些文件

| 应进入 Git | 理由 |
| --- | --- |
| th06ai 的 Python 源码、native 的 Rust 源码 | 主要实现与参考路径 |
| frontend 源码、静态自有图标、Vite/TS 配置 | 可重建界面 |
| package-lock.json、Cargo.lock、安装脚本、pyproject | 已有的依赖与构建信息 |
| tests 与必要 tests/fixtures NPZ | 离线反例与契约验证 |
| README、docs、AGENTS、贡献及许可文件 | 人类和 AI 接手上下文 |
| config.example.json、.env.example、启动脚本 | 通用运行入口 |
| .github 模板、.gitignore | 协作与文件过滤 |

| 留在本机并忽略 | 原因 |
| --- | --- |
| `.venv`、`node_modules` | 可重建依赖，不是项目源码 |
| `target`、`.cache`、`__pycache__`、测试临时目录 | 编译／下载／运行缓存 |
| 整个 `models`，包括任意层级 ZIP | 用户训练权重、元数据及历史，不自动公开 |
| `runs` | 原始训练、GPU、错误、死亡、评估与可能的个人路径 |
| `research`、`tmp` | 本机研究快照、基线、调试产物；公开摘要另整理在 docs |
| `config.json`、各种 native 校准、training_backend、ui_state | 本机路径、人工确认及机器状态 |
| `.env`，不含 .env.example | 本机环境或潜在私密设置 |
| `.pyd/.dll/.exe/.whl` | 二进制／依赖，不在此源码提交 |
| `frontend/dist` | 默认重新构建；不要求每次源码 PR 携带生成包 |

所有模型 ZIP 都被整个 models 目录规则覆盖，不只顶层 `models/*.zip`。必要几何 NPZ 夹具不是模型权重，保留它。

## 3. 当前目录直接 push

首次准备：

```powershell
git init -b main
.\.venv\Scripts\python.exe tools\check_docs.py
git status --short
git status --short --ignored
git add -A
git diff --cached --stat
git diff --cached --name-only
```

自动核对实际清单：`python tools/check_source.py` 检查所有已跟踪及未忽略候选；暂存后用 `python tools/check_source.py --staged` 检查本次新增／修改。工具也扫描必要 NPZ 解压后的路径文本，但不是完整凭据、版权或 Git 历史审查。

Git 已初始化时不必重复 init。查看真正将提交的名称列表，而不是只看文件管理器大小。忽略的缓存继续留在本机，训练仍使用自己的 `.venv` 与 `models`。

确认后由维护者提交与推送：

```powershell
git commit -m "docs: prepare TH06NC-Neural-Pilot for open source"
git remote -v
$remoteUrl = '<维护者实际创建或选定的仓库地址>'
git remote add origin $remoteUrl
git push -u origin main
```

已有 origin 则先核对，不重复添加或盲目覆盖。不自动采用机器上与项目无关的 GitHub 账号。若缺少 Git identity，用维护者自己的姓名／邮件或平台提供的正确 noreply 地址配置；不能随便编造一个邮箱。

准备文档时可以初始化本地仓库和审查暂存区；这不包含向未知远端发布或替用户提交不可见内容的授权。

## 4. 必须用 Git 本身核验过滤

```powershell
git check-ignore -v .venv/Scripts/python.exe frontend/node_modules/react/package.json native/target/release/example.dll runs/current.json .cache/example.bin models/latest.zip models/checkpoints/example.zip config.json
git ls-files
git diff --cached --name-only
```

确认示例／锁文件反而不应被忽略：

```powershell
git check-ignore config.example.json .env.example frontend/package-lock.json native/Cargo.lock tests/fixtures/stage5-switch-window.npz
```

第二条无输出、返回 1 可以表示这些文件未匹配 ignore，不是运行错误。命令参数里的示意文件不需要实际存在即可检查规则。

`.gitignore` 不会移除已经跟踪的文件。旧仓库若已有模型或日志，先核对列表后只撤出索引，例如 `git rm --cached`；`--cached` 不删除本地内容。不要不经审查递归删除目录。已经进入公开历史的敏感内容不能靠加 ignore 解决。

## 5. 路径可移植性

源码通过自身文件位置推导 ROOT，CMD 通过 `%~dp0`，PowerShell 通过 `$PSScriptRoot` 定位。游戏路径是每个人配置中的 `game_exe`，Git 忽略真实 config，仅提交通用示例。

Rust 编译器／MSVC 路径通过命令参数或环境传入，公开说明不再引用作者其他项目目录。默认端口 18765、预览 18766、Vite API 代理是可配置的服务约定，不是个人目录；改端口时同步代理及启动／关闭入口。

新检出验收应至少确认：没有作者盘符依赖、缺少 config 时安装能创建示例、已有配置不被覆盖、UI 可构建、Python 可选路径可用、原生构建使用本机工具链、校准不凭空成功。

## 6. 内容与权利审查

按 [DISCLAIMER](../DISCLAIMER.zh-CN.md) 区分项目 MIT、非商业研究倡议、原作及第三方权利。没有“MIT 严禁商用”附加条款，也不声称工具已获原作方专项授权。

公开前扫描绝对路径、用户名、GPU UUID、控制令牌、账号、下载鉴权、个人截图与输出；尤其注意 JSON/NPZ metadata、日志和生成前端 sourcemap。不能只按文件扩展名判断是否安全。

SB3 派生代码的原版权文件在 LICENSES 保留。未来若分发带依赖的可执行包，需要对实际产物重新收集许可证；本次源码准备没有完成该二进制分发审查。

## 7. 当前已知待办

| 事项 | 当前状态 |
| --- | --- |
| 名称／作者／MIT | 已由维护者确定并落文件 |
| 普通源码 Git 过滤与文档 | 本轮提供，提交前继续核对实际索引 |
| 远程 URL／首次提交／push | 维护者已指定公开仓库与 main；后续使用正常 Git 提交和推送流程 |
| README 徽章／原作具体署名／首次使用 FAQ | 已提供；原作与《新典》开发、发行来源同步到权利说明 |
| 控制台截图／首次操作录像 | 尚未收录，应使用脱敏画面并标明预览或实际运行状态 |
| 新机器资源人工确认 UI | 尚未完善；开发指南写明现有辅助流程 |
| 完整 Python 传递锁／CI | 仍待补充，不假装已有 |
| 跨机器重装验收 | 本轮未新下载完整环境，需在独立机器验证 |
| 稳定通关／完整 NMNB／特殊攻击全覆盖 | 未证明 |
| 单文件 EXE／完整 wheel／远程部署 | 未验证，不作为现成发布方式 |

可选源码归档从审核后的提交用 `git archive` 生成即可。它不是直接 push 的必需步骤；不压缩整个工作目录，不删除本机资料来实现“干净发布”。
