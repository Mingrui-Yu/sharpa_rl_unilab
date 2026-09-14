# HORA teacher 与 student

PPO、APPO、FlashSAC 默认均训练同一 HORA Actor，每个旋转算法只保留
`mujoco.yaml`。teacher 输入为147维可测基础历史与独立的当前9维特权信息；
独立 Critic 读取174维 clean 历史。FlashSAC 保留原生分布式双Q和温度损失，
使用公共 HORA Actor 适配。

```bash
uv run sharpa-train --algo appo
uv run sharpa-eval --checkpoint /absolute/path/to/teacher_final.pt
uv run sharpa-distill --checkpoint /absolute/path/to/teacher_final.pt
uv run sharpa-eval --checkpoint /absolute/path/to/student_final.pt
```

三种 teacher 共用蒸馏流程。冻结继承的 Actor 与基础归一化统计，仅通过
latent MSE 训练30帧历史编码器；每个向量步先更新一次，再执行更新后的
student 确定性动作。部署只需要147维基础观测与 [N,30,49] 历史。

checkpoint 保存完整运行配置和归一化统计。旧共享 HORA、flat PPO 和原生
FlashSAC checkpoint 必须重训。定量评估使用固定场景与完整20秒窗口。

完整说明见[协议与迁移记录](../migrations/issue-2.md)。
