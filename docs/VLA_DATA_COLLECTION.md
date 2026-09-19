# MuJoCo 遥操与 VLA 数据采集

## 设计依据

本实现参考三个官方开源工作流：

- [robosuite I/O Devices](https://github.com/ARISE-Initiative/robosuite/blob/master/docs/modules/devices.md)：把键盘、SpaceMouse、DualSense 等输入统一为末端 6DoF 增量与夹爪命令；
- [LeRobot record](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_record.py)：以固定频率同步记录 observation、实际下发 action、相机与任务文本，按 episode 保存或重录；
- [openpi 自定义 DROID 微调流程](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md)：小规模自定义数据转换成 LeRobot，再计算 normalization statistics 并微调 π0.5。

LeRobot v3 将数据分为低维 Parquet 信号、多相机 MP4 和元数据。本项目先写一个不依赖 LeRobot 的原始 episode，再通过独立脚本转换，避免采集进程因为训练栈依赖而变得脆弱。

## 安装与启动

```bash
python scripts/setup_assets.py
python -m pip install -e '.[teleop]'
python scripts/teleop_collect.py --device gamepad --episodes 10
```

键盘回退：

```bash
python scripts/teleop_collect.py --device keyboard --episodes 10
```

云容器看不到个人电脑的 USB HID。实际使用手柄时建议在带手柄和显示器的本地机器运行仿真；也可以把容器以支持 USB/桌面转发的方式启动。不要把“手柄已连接本机”误认为远程 SSH 进程能读取该设备。

## 控制映射

所有移动均需按住 deadman，松手立即保持当前位置。

| 功能 | Xbox/SDL 手柄 | 键盘 |
|---|---|---|
| deadman | 按住 LB | 按住 Shift |
| XY 平移 | 左摇杆 | W/S、A/D |
| Z 平移 | LT/RT | Q/E |
| roll/pitch | 右摇杆 | U/O、I/K |
| yaw | D-pad 左/右 | J/L |
| 夹爪开合 | A | Space |
| 保存 episode | Start | Enter |
| 丢弃重录 | Back | Backspace |
| 环境复位 | X | R |
| 退出 | Y | Esc |

命令先积分为目标末端位姿，再经 DLS IK 转成 Panda 关节目标。关节目标通过当前 MuJoCo 碰撞模型验证；抓住杯子后自动切换为完整杯身/杯把载荷检查。数据中保存的是验证后真正下发的关节目标，而不是未执行的原始摇杆输入。

## 原始 episode 格式

```text
datasets/raw/
└── episode_000000/
    ├── metadata.json
    ├── data.npz
    ├── overview.mp4
    └── wrist_rgbd.mp4
```

每个 20 Hz 帧包含：

- `observation_state`：7 个关节位置 + 归一化夹爪位置；
- `observation_velocity`：7 个关节速度；
- `observation_ee_pose`：末端位置与四元数；
- `action`：7 个实际关节位置目标 + 二值夹爪命令；
- 固定视角与手眼视角 RGB；
- 时间戳、阶段、语言任务、成功标签和碰撞指标。

## 标准采集流程

1. 固定相机内外参、动作定义、频率和语言模板，先采 5 条试运行数据。
2. 每条 episode 随机化杯子位置；只保留完整成功轨迹，失败应单独标记，不能混为专家动作。
3. 操作者确认成功后保存；操作失误、画面卡顿或碰撞时丢弃重录。
4. 每批采集后运行一致性检查：

   ```bash
   python scripts/inspect_raw_dataset.py datasets/raw
   ```

5. 检查双相机/动作帧对齐、NaN、时间戳和 idle ratio。长时间不动的数据会降低动作学习效率，openpi 的 DROID 流程也专门过滤 idle action chunk。
6. 转成 LeRobot：

   ```bash
   python -m pip install -e '.[dataset]'
   python scripts/convert_to_lerobot.py \
     --raw-dir datasets/raw \
     --repo-id <hf-user>/panda-cup-mujoco
   ```

7. 在 LeRobot 可视化器中逐集检查，再划分训练/验证场景。验证集应按初始物体位置或场景种子划分，不能随机拆帧，否则同一 episode 会泄漏到两边。
8. 在 openpi 中添加与本项目字段对应的 input/output transform，计算 normalization statistics，然后运行 π0.5 微调。

## 数据量建议

- 管线冒烟测试：5～10 条；
- 单一 pick-and-place 初版：50～100 条成功示范；
- 增加位置、杯把朝向、光照和遮挡覆盖后：200 条以上；
- 保留独立的 20～30 个初始场景，只用于闭环评估。

自动规划轨迹可以快速验证格式并做预训练补充，但不能替代人类遥操数据：其动作分布过于平滑、恢复动作太少。更实用的组合是“脚本专家打底 + 人类遥操补多样性 + 策略上线后 DAgger/HIL 采集失败恢复”。

