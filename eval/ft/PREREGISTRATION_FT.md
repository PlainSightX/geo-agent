# 预注册附录：Qwen3-8B 工具调用 LoRA 微调评测

本附录、`eval/ft/heldout/questions.json` 与 `eval/ft/heldout/gold.json`（哈希见 `gold.json` 的 `questions_sha256` / `items_sha256`）
在**任何训练步骤之前**提交到 git（以提交时间为准）；评测脚本 `eval/ft/run_eval_ft.py`、判分脚本 `eval/ft/score_ft.py` 同一提交或更早。
之后对 heldout 题目、金标、判分与判定规则的任何改动，都在 README“LoRA 微调”一节逐条披露。
第二轮 72 题（`eval/r2/questions.json`、`gold.json`，冻结于 `08309f2`）不改动，只作次要评测。

## 数据

- train：程序化金标轨迹（`eval/ft/run_build.py`，seed 0），每层配额 tag 250 / join 260 / crs 230 / route 320 / compliance 240 / unanswerable 200，
  实际条数见 `eval/ft/data/build_summary.json`；约 25% 的轨迹含一次可恢复错误（错误轮次不计损失）。
- heldout：每层 25 题，共 150 题；与 train 在实体、设施类型、条文、标签键、分钟数/半径与句式上互不相交，
  与 r2 测试集在实体字面量、设施类型、条文、标签键与数值上互不相交（`eval/ft/disjointness.json`，任一冲突即生成失败）。
- 不读取 r2 测试题面与测试金标作训练数据；r2 题面只在内存中做字符 5-gram Jaccard 检查（阈值 < 0.5）。

## 训练（固定，不在 heldout 或 r2 上调参）

Qwen3-8B 基座，LoRA r=16、alpha=32、dropout 0，目标 q/k/v/o/gate/up/down，lr 1e-4（5% 预热 + 余弦），每次更新 8 条轨迹，2 epoch，seed 0。
按 id 哈希留 5% 训练轨迹做内部验证，只记录损失，不做早停。若 20 步试训外推全量训练超过 6 GPU 小时，先报告再决定，不改配方。
训练与推理格式一致：同一 Qwen3 chat template、`tools=R2_SPECS`（经 vLLM 同样的工具字典转换）、`R2Agent.system_prompt()` 原文、
`enable_thinking=False`；每条训练 completion 经 vLLM hermes 解析器往返解析，失败即丢弃。

## 评测

- 推理：vLLM 0.11.2，同一服务同时提供基座（`qwen3-8b`）与 LoRA（`--enable-lora --lora-modules geo=<adapter> --max-lora-rank 64`）；
  `--tool-call-parser hermes --enable-auto-tool-choice --max-model-len 32768 --gpu-memory-utilization 0.9 --seed 0`；温度 0，非思考模式。
- 臂：基座 Agent、LoRA Agent，各在 heldout 与 r2 上跑 3 次（run1..3）。可选消融臂（Agent 去掉路网/检索/合规工具）单独报告，不进判定。
- 判分：与 r2 相同的确定性规则（`matcher_r2.score_item`；不可回答题须拒答，可回答题被拒答为错）。

## 主指标与判定规则

- **主指标**：heldout 上 LoRA Agent vs 基座 Agent，run1 逐题配对精确 McNemar（双侧），α = 0.05。
- **次要**：r2 72 题上 LoRA vs 基座（run1 配对 McNemar）与 vs DeepSeek Agent（r2run1..3）；3 次合并正确率与 Wilson 95% 区间；按层正确率；
  纯文本工具调用率（无结构化 tool_calls 的模型调用 / 全部模型调用）、工具参数校验错误/题、no_answer 数、模型调用/题、延迟中位数与 p90。
- **判定**（事先写定，结果出来后不改）：
  1. heldout 显著优于基座（LoRA 更好，p < 0.05）**且** r2 不下降（r2 run1 上 LoRA 正确数 ≥ 基座正确数）→ “微调有效”；
  2. heldout 显著提升但 r2 下降 → “对模板分布有效、对未见题面不泛化”；
  3. heldout 不显著 → “未证明有效”。
  与 DeepSeek 的差距无论大小都如实报告。
- 稳健性（不改变判定）：3 次运行逐题多数票的配对 McNemar；heldout 按层结果。

## 已知局限（预先声明）

- heldout 与 train 由同一作者、同一套生成器产生：句式不同、实体与设施不同，但题型、工具链与答案格式相同。
  heldout 衡量的是“同一任务分布内对新实体/新设施/新条文/新句式的泛化”；r2 72 题（另一作者时期写的句式与题型组合）才接近“未见题面”。
- 训练轨迹由确定性求解脚本生成，覆盖的解法比 DeepSeek 的真实轨迹窄；r2 中训练数据没有覆盖的题型（如 Web 墨卡托放大倍数、同名多校区、营业时间）预计不会改善。
- heldout 的“可回答/不可回答”比例（125/25）与 r2（62/10）不同；不可回答题话题与 r2 不重叠。
