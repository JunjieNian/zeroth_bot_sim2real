# BetterKOS 部署工作区

这个目录用于在 `BetterKOS / PyKOS` 环境下部署 Zeroth Bot 的 walking policy，并做真机联调。

## 目录说明

- `BetterKOS/core.py`：真机控制主逻辑、观测拼接、ONNX 推理、动作下发
- `BetterKOS/app.py`：程序入口
- `examples/test.py`：最小运行示例
- `robot_fixed.urdf`：用于读取关节限制

## 运行前准备

### 1. 放置模型

将导出的 `ONNX` 模型放在当前目录，例如：

```text
BetterKOS/model_100.onnx
```

### 2. 配置 `config.json`

在当前目录创建 `config.json`：

```json
{
  "robot_ip": "192.168.42.1",
  "robot_port": 50051,
  "model_file": "model_100.onnx"
}
```

### 3. 编写自己的应用

建议把自定义逻辑放在 `BetterApp/main.py` 中，示例可参考 `examples/test.py`。

主程序结构：

1. 继承 `BetterKOS.core.BetterKOS`
2. 在 `update()` 中写每帧控制逻辑
3. 在 `update_after()` 中写额外后处理
4. 调用 `BetterKOS.app.run(App)`

## 推荐启动流程

当前版本建议按以下顺序启动：

1. 让机器人摆到训练默认初始姿态
2. 启动程序后，按提示重新记录关节零位
3. 初始化 IMU
4. 再进入 walking 循环

这样可以减少零位偏差导致的抽动问题。

## 注意事项

- `config.json` 属于本地配置，默认不提交
- `model_*.onnx` 属于本地部署产物，默认不提交
- 如果真机出现抽动，优先检查：
  - 电机 ID 映射是否正确
  - 关节方向是否相反
  - 启动时机器人姿态是否与训练默认姿态一致

## 参考

- Zeroth 文档：<https://docs.kscale.dev/category/zeroth-01/>
- BetterKOS 原始仓库：<https://github.com/CarrotFish/BetterKOS>
