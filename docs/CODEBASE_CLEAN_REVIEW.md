# Codebase cleanup review 与 ablation

日期：2026-09-16。审查分支为 `refactor/codebase-clean`，基线为 `bbd18bc`，
原清理实现为 `3dafca7`、`c26bfd5`，共 58 个改动文件。本轮修改接续于 `c26bfd5`。
对照文档：兄弟工作区 `sharpa_rl_unilab/dist/journals/11-plan-codebase_clean.md`。

## 结论与发现

清理方向符合计划，职责拆分和原有冗余删除均有实际用途。本轮发现并修复了两个实现遗漏和一个测试隔离问题：

1. **P2：生成缓存路径仍绑定 manifest 哈希。** 原实现使用
   `<manifest-hash>/generated/`，更新内置资产后，新启动会寻找另一目录并退回内置抓取。
   数据文件仍在磁盘上，但不再参与训练。现改为稳定的
   `$XDG_CACHE_HOME/sharpa-rl-unilab/generated/`；显式资产目录覆盖和绝对输出前缀保持原语义。
   回归测试实际修改临时 manifest、修复资产并确认仍读取生成的数据。
2. **P2：构建期协议校验遗漏 manager 选项。** `priv_info` 的 group/term 历史长度可改为 2，
   `concatenate_terms=false` 可把数组改成字典，原校验均放行直到构建仿真。
   现检查有效 privilege 历史和四个观测组的拼接要求；沿用上游的缺省值和 group 覆盖 term 语义。
   新增用例先复现失败，再验证错误包含字段、当前值和协议要求，且没有进入 `create_env`。
3. **既有测试污染：快慢测试合并运行失败。** 渲染测试直接写入 `MUJOCO_GL=osmesa`，
   当变量原本不存在时，`monkeypatch.delenv(..., raising=False)` 不会登记恢复动作。
   后续 APPO/FlashSAC 子进程继承伪造的后端设置，产生 6 个 MuJoCo 导入失败。
   将该测试的环境映射隔离到副本；没有新增生产环境渲染 fallback。

## 计划逐项核对

| 计划事项 | 审查结果 |
| --- | --- |
| 用户抓取数据与内置资产隔离 | 实现并补上 manifest 更新后的读取持久性；内置资产哈希修复继续有效 |
| 抓取按实际行数计数、截断和停止 | `_num_rows` 统一计数，追加时只保留剩余额度；真实 MuJoCo reset 测试验证落盘和退出 |
| 删除 `disable_tactile_ids` | 旋转/抓取 YAML、参数白名单和无效处理均已删除；旧 v2 快照迁移删除其无效值 |
| 固定 v2 维度、拒绝不兼容配置 | 已集中常量，保留观测张量边界检查；触觉、特权、历史和拼接选项在构建前检查 |
| 删除无消费者的模型输出 | `HoraCoreOutput` 及 student 无用 target/零张量计算已删除，保留 mean 和 std 所需 trunk 特征 |
| 减少 APPO 传输 | 不发送 `behavior_mean/std` 和末帧 proprio 历史；PPO 仍保留计算 KL 必需的行为参数 |
| 公共模型、checkpoint、输入转换 | 归属 `models.py`、`policy.py`、`checkpoints.py`；评估、蒸馏和 FlashSAC 不再依赖 PPO/APPO 调度模块 |
| PPO/APPO 调度与 collector 分离 | 独立算法运行模块和 APPO 进程模块；没有引入通用训练框架 |
| 模型与算法 adapter 分离 | `on_policy.py` / `flashsac.py` 集中上游适配，公共 actor/distribution 保持统一 |
| FlashSAC checkpoint 去重 | learner 与顶层 actor/critic 共用张量存储；旧 schema、推理加载和 learner 状态恢复仍有效 |
| 合并任务配置 | `sharpa_task.yaml` 仅包含共有物理任务；抓取继续使用独立 flat 观测和原生 PPO |
| 无效配置与局部冗余 | 删除无消费者的 seed 字段、重复 `use_amp` 和无用 `variant_index`；缓存内容改为加载时验证 |
| 文档、类型和测试 | 中英文能力/安装说明已对齐；静态检查及快速、仿真、训练集成测试见验证记录 |

## 抽象、helper、wrapper 与防御逻辑的消融决定

| 项目 | 决定与原因 |
| --- | --- |
| `update_ppo` / `update_appo` | 删除。只有各自训练循环一个调用者，却传入 8/9 个参数并修改多份共享状态；合回对应算法循环，保留全部更新顺序 |
| `policy_td` | 删除。采样不用 critic，末帧价值计算不用 actor；采样复用已有推理转换，PPO 只传 critic |
| `evaluation_distribution` | 与采样转换合并，改名 `inference_distribution`；沿用原 helper，没有增加另一层接口 |
| `_scale_tag` | 合入唯一使用它的输出路径函数；尺度合法性校验仍保留 |
| 四处观测配置校验 | 删除 `train_teacher`、`make_models` 中的重复调用；保留 CLI 配置合成和环境构建两个实际边界，覆盖直接构造环境及旧 checkpoint 路径 |
| `algorithm_options` 二次过滤 | 删除。已拒绝全部未知键，直接返回原参数字典；上游签名校验仍保留 |
| APPO staging 字段过滤 | 直接选取必需字段；仍保留 transport 布局转换、短 rollout 更换 pool 和只统计新槽位的逻辑 |
| 缓存路径往返、文件存在性检查 | 去掉 generated 绝对路径转相对路径的往返；resolver 统一报缺失文件，reset 不再重复检查 |
| reset 采样中的缓存处理 | 删除重复非空验证、`np.asarray(cache)` 和随后被完全覆盖的零初始化；完整内容验证留在加载处，索引范围/类型验证保留 |
| recorder 保存转换 | 删除保存时的第二次截断和 float32 转换；追加阶段已保证目标上限和 dtype |
| 旧 tactile 配置迁移 | 保留实际兼容需求，改为操作固定的 v2 actor term；删除遍历任意观测布局的多层 `.get(..., {})` fallback |
| `generated_root`、读/写路径函数 | 保留。资产修复、读取优先级和 recorder 写入路径各有真实消费者；读取不能被复用作写入，以免覆盖内置 fallback |
| `validate_grasp_caches` | 保留。集中加载时的完整扫描，避免每次 reset 扫描全部数据；不再在采样时重复验证 |
| `make_actor` / `make_models` | 保留。加载/采样只构造 actor，新训练才构造 critic/learner 并验证新训练限制，避免旧 checkpoint 被新默认值改写 |
| `tensor_obs`、推理与归一化 helper | 保留实际多调用点；teacher/student 输入有语义差异，归一化只消费新样本的约定不能删 |
| `create_run`、`training_counters`、`append_metrics` | 保留多个运行器共用的目录/元数据、计数字段和 JSONL 写入；没有扩大为通用日志框架 |
| actor、V/Q、Flash 环境/runner wrappers | 保留。它们提供上游不同调用协议、V/Q 结构、终止观测传输和归一化计数，不能互相替换 |
| KL optimizer hook、APPO loss cache/tuple 适配 | 保留。上游缺少公开扩展点；删除会改变 transformed entropy、KL 或学习率时序，已有数学和真实训练回归覆盖 |
| APPO 队列重试、进程错误和关闭处理 | 保留。处理实际 multiprocessing feeder 的 Full/Empty 时序、背压及异常退出；不是未来场景的备用实现 |
| checkpoint format、`legacy.py` | 保留已支持快照的 std、配置、权重迁移；不增加新 schema 或兼容别名 |
| FlashSAC 先初始化再替换 actor | 保留并集中封装。已核对安装的上游构造函数没有 actor/factory 注入；绕过它会改变初始化随机数和原生训练状态 |
| `resolve_scene` | 保留现有公开函数。计划将其列为候选，没有证据证明外部调用可删除；本轮未新增此 API |

本轮没有新增源码函数、类、配置开关或兼容层；净删除四个 helper，源码净减少 21 行，
其中已经计入两处行为修复所需的代码。新增测试只覆盖实际复现的遗漏。

## 验证范围与边界

对 `bbd18bc`、原清理实现及本轮实现分别构造 PPO/APPO/FlashSAC：固定 seed 19，
相同输入/噪声下的 actor/critic 初始化权重、teacher 均值/std/动作、student 动作及随机数状态
逐位一致。旋转与抓取合成配置一致，仅排除基线里明确删除的三个无效字段。
终止处理、归一化计数、KL 调度、collector 同步、旧模型加载均由既有回归测试保留覆盖。

生成路径不再包含 manifest 哈希。未增加对未发布清理提交的旧 `generated/` 路径的自动扫描迁移；
本机默认资产缓存没有此类生成文件，绝对路径和显式资产目录下的生成数据位置未变。
没有运行长训练或性能基准，逐位模型对照和短训练测试不代表学习曲线/吞吐的对比结论。
完整命令与最终结果见 [验证记录](VALIDATION.md)。
