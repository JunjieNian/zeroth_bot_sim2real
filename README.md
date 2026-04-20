# Zeroth Bot Sim2Real

这个仓库用于训练 `Zeroth Bot` 行走策略，并将策略部署到真机做 `sim2real` 验证。

当前仓库包含两条主线：

- `sim/genesis/`：本地仿真训练、评估、导出 `ONNX`
- `BetterKOS/`：在 BetterKOS / PyKOS 环境下做真机部署与联调

项目文档可参考：

- Zeroth 机器人文档：<https://docs.kscale.dev/category/zeroth-01/>
- K-Scale 仿真文档：<https://docs.kscale.dev/simulation/isaac>

## 当前状态

- 已支持在本地仿真中训练和回放 Zeroth Bot 行走策略
- 已补齐 `ONNX` 导出中的观测归一化
- 已修复部署侧与训练侧不一致的动作缩放 / 延迟模拟问题
- 当前重点仍是继续收敛真机部署稳定性与 `sim2real` 差异

## 仓库结构

```text
.
├── sim/
│   ├── genesis/                 # Zeroth Bot 训练/评估/导出脚本
│   ├── envs/                    # 通用机器人环境
│   └── resources/               # 机器人资源
├── BetterKOS/                   # 真机部署工作区（已并入当前仓库）
├── examples/                    # 参考模型与示例文件
├── logs/                        # 本地训练输出（已加入忽略）
├── sim/genesis/rsl_rl/          # 唯一保留的子模块
├── analyze_onnx.py              # ONNX 结构检查辅助脚本
└── README.md
```

## 推荐工作流

### 1. 仿真训练

训练入口位于 `sim/genesis/zeroth_train.py`。

典型流程：

1. 配置好仿真环境依赖
2. 运行训练脚本生成 `logs/zeroth-walking/`
3. 用 `sim/genesis/zeroth_eval.py` 回放检查策略

### 2. 导出 ONNX

当前推荐直接从最新 checkpoint 导出：

```bash
python sim/genesis/utils/convert_to_onnx.py \
  --cfg logs/zeroth-walking/cfgs.pkl \
  --model logs/zeroth-walking/model_100.pt \
  --output BetterKOS/model_100.onnx
```

### 3. 真机部署

真机入口位于 `BetterKOS/`。

部署前建议：

1. 先确认电机 ID / 方向 / 零位
2. 让机器人摆到训练默认初始姿态
3. 重新记录关节零位并初始化 IMU
4. 再启动 walking policy

## GitHub 整理说明

为了便于 push，本仓库已经整理了以下规则：

- `logs/`、缓存、`ipynb checkpoints` 不再进入版本控制
- `BetterKOS/config.json` 这类本地运行配置不再提交
- 根 README 明确区分仿真训练与真机部署两条链路

需要特别注意：

- `BetterKOS/` 已经按普通目录并入当前仓库
- 当前仓库仅保留 `sim/genesis/rsl_rl` 这一个子模块
- 如果在新机器克隆仓库，记得额外初始化 `rsl_rl`

```bash
git submodule update --init --recursive sim/genesis/rsl_rl
```

## 后续建议

如果你接下来继续整理仓库，推荐优先做这几件事：

1. 清理 / 归档不再使用的旧模型文件
2. 增加单关节安全诊断脚本，方便真机排查映射和方向
3. 补齐 `rsl_rl` 初始化说明和环境安装脚本

## License

本仓库保留原项目许可证，详见 `LICENSE`。
