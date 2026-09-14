# 参考训练结果

这些完成运行使用 MuJoCo、seed 1 与 `training.no_play=true`。

| 指标 | APPO | FlashSAC |
| --- | ---: | ---: |
| 完成 iterations | 301 | 5,371 |
| Environment steps | 39,452,672 | 11,000,832 |
| 训练 wall time | 1 h 13 m 13 s | 41 m 23 s |
| 最终平均 episode return | 48.8223 | 34.8279 |
| 最佳平均 episode return | 49.2989 | 38.9166 |
| 最终平均 episode 长度 | 390.35 | 351.57 |
| 最终 timeout rate | 0.9636 | 0.8333 |
| 最终 collector 吞吐 | 9,755.5 steps/s | 4,483.6 steps/s |

APPO 在这个固定基准上保持物体更久，并获得更高 return。FlashSAC 以更高的
learner 样本吞吐完成 off-policy 日程。这些是参考数字，不表示在所有任务上
普遍更优。
