# 环境与复现

当前没有 GPU 验证过的版本组合，因此不提供声称可用的 CUDA/PyTorch/Triton 固定版本。

首版部署目标为受对应 PyTorch/Triton 版本支持的 NVIDIA GPU 环境，通常使用 Linux。先确认显卡、驱动和所选 wheel 的兼容性，再安装；本地 macOS 可以编辑和运行部分 CPU 参考逻辑，不能据此验证本项目的 NVIDIA GPU kernel。

## 环境记录

首次成功运行后，保存 GPU 型号与显存、操作系统、驱动、Python、PyTorch、PyTorch CUDA runtime、Triton、可选基线库版本；若有系统 CUDA toolkit 也独立记录。驱动显示的 CUDA 能力不等同于 PyTorch 使用的 runtime。

在隔离环境中依照官方安装文档安装计算依赖，再从仓库根目录执行：

```sh
python -m pip install -e .
python -m pip freeze > results/environment-freeze.txt
```

上述第一条只安装本项目包骨架；不自动解决 GPU 依赖。当前仓库没有运行 kernel 的入口。开发测试时另外安装 pytest；绘图工具待确定输出方案后添加。

将首个通过 correctness + benchmark 的环境冻结为可复现配置，记录模型 revision 和外部基线 commit。更换 GPU 或依赖版本后重跑必要测试，不沿用旧性能结论。

官方入口：[PyTorch 安装](https://pytorch.org/get-started/locally/)、[Triton 文档](https://triton-lang.org/main/index.html)。
