# Third-party notices

English · [简体中文](THIRD_PARTY_NOTICES.zh-CN.md)

TH06NC-Neural-Pilot is maintained by **xialongxl**. The project's MIT license covers its own work. Third-party code, dependencies, and game content retain their respective terms.

## Derived code

The rollout control in `th06ai/pausable_ppo.py` adapts Stable-Baselines3's `OnPolicyAlgorithm.collect_rollouts`. The documented reference environment uses Stable-Baselines3 2.9.2. Retain the source attribution in the file and the upstream copyright and license text when distributing the adaptation. [Stable-Baselines3 MIT notice](LICENSES/stable-baselines3-MIT.txt)

When adding a new adaptation, record its upstream source and version, affected files, and applicable notice.

## Principal dependencies

The following summaries describe the reviewed source setup. A bundled binary requires a separate inventory of the actual contents and upstream notices.

| Component | Role | License summary |
| --- | --- | --- |
| Stable-Baselines3 and Gymnasium | PPO and environment interface | MIT |
| React and React DOM | Web console | MIT |
| FastAPI and Pydantic | Loopback API and validation | MIT |
| Uvicorn and httpx2 | Local service and testing | BSD-3-Clause |
| NumPy | Numerical computing | BSD-family terms plus bundled notices |
| PyTorch CUDA wheel | Training and inference | Multiple licenses and bundled notices |
| Lucide React | Icons | ISC |
| TypeScript | Type checking and tooling | Apache-2.0 |
| Vite and Vitest | Frontend build and tests | MIT |
| Rust, PyO3, and NVIDIA NVML components | Native extension and telemetry | Consult each installed version and distribution's terms |

Check Python `*.dist-info/licenses`, npm package license files, Cargo sources corresponding to `Cargo.lock`, and upstream notices for the version you actually distribute. The source release obtains dependencies during installation and excludes the installed environments and downloaded wheels.

## Game credits and research references

Touhou Project was created by **ZUN** and **Team Shanghai Alice**. New Classic credits Team Shanghai Alice and Shanghai Alice Reprise for development and Alliance Arts for publication. The project distributes source independently of the game. [Touhou Project](https://touhou-project.news/touhou_project/) · [Steam credits](https://store.steampowered.com/app/4659620/) · [Project disclaimer](DISCLAIMER.md)

Game-structure research consulted the linked TH06NC notes. The runtime uses its own executable identity and signature checks, and the repository excludes the referenced project's game patch. [TH06NC development notes](https://github.com/vittorioromeo/th12_hfr/blob/main/docs/games/TH06NC_DEVNOTES.md)
