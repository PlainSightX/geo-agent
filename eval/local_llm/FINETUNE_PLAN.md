# Qwen3-8B 工具调用 LoRA 微调计划（已准备，未训练）

## 为什么值得做

依据见 README“本地 Qwen3-8B（私有化部署评估）”与 `eval/runs/local/failure_taxonomy.json`：

- 差距很大：同一 72 题冻结集上，Qwen3-8B Agent 3 次合并 43/216 = 19.9% [15.1, 25.7]，DeepSeek Agent 203/216 = 94.0% [90.0, 96.4]；
  run1 配对 54 vs 0，精确 McNemar p ≈ 1e-16。
- 失败集中在**工具调用协议**，不在 GIS 知识。Agent 臂 173 次错题里：
  工具调用写成纯文本、10 次调用用尽 48；调用次数用尽 36（其中 33 次的轨迹里有 find_entity 返回 0 行，多数是把“XX小区”原样当名称、换图层重试而不去掉后缀）；
  实体找不到就拒答 23；`result_as='count'` 用在聚合结果或单行结果上（得到 n=1）17；不可回答题给出答案 12；其余口径/选错工具 29。
  另有每题 0.5 次工具参数校验错误（图层枚举、度量类型与图层不符、pydantic 字段），DeepSeek 为 0.01。
- 这些行为可以从数据学：每种都有确定的正确做法，并且可以由确定性工具**程序化生成金标轨迹**（不需要人工或 DeepSeek 标注）。
- 只修格式不够：客户端把纯文本调用转成结构化调用（`diagnose_text_toolcalls.py --mode salvage`，事后诊断）只把 Agent 从 14/72 提到 19/72，
  说明需要的是行为（名称解析、何时 count、何时调 check_compliance、何时拒答），不只是模板格式。

不做的事：不改 72 道测试题、金标或判分；训练数据不读取 `eval/r2/questions.json` 的题面与 `gold.json`；本任务不训练。

## 不需要训练的替代方案（先记录，不在本计划内实施）

`find_entity` 去掉“小区/地铁站/站”等后缀后再匹配，可能直接消除一大类失败。但这是对 Agent 的改动，需要对 DeepSeek 臂重跑并作为预注册后的改动披露；
它与微调不冲突，建议单独评估，不能和微调效果混在一起报告。

## 数据：三个互不相交的集合

| 集合 | 用途 | 规模 |
|---|---|---|
| train | LoRA 训练（含 5% 内部验证，只用来看损失） | 约 1,500 条轨迹 |
| heldout | **新的冻结集**，训练前冻结哈希，主评测 | 约 150 题（6 层各约 25） |
| r2 test | 现有 72 题，只作次要评测（与 DeepSeek 可比） | 72 |

**不相交规则**（生成器 `eval/ft/build_sets.py` 执行并写入 `eval/ft/disjointness.json`，冲突即失败）：

1. **实体**：从 72 题 `gold_sql` 抽出全部字面量实体（小区 13 个，另有地铁站、POI、公园、区名、标签键值与条文 id，共约 90 个字面量），train 与 heldout 一律排除。
   其余实体按 `sha256(layer:name)` 二分：小区（1,995 个有名称）、地铁站（272）、地标 POI、公园、水面，各约 80% 进 train、20% 进 heldout。
2. **设施类型**：测试集用到的 subtype（bank、cafe、fuel、school 中学/小学、kindergarten、hospital、police、toilets、museum、restaurant、fast_food、
   charging_station、dentist、convenience、university、place_of_worship）不进 train；train 用 supermarket、post_office、library、pharmacy、marketplace、
   mall、community_centre、clinic 等；heldout 用另一组（如 department_store、greengrocer、arts_centre 与 train 未用的部分）。
3. **条文**：测试集合规题用到 7 条（C.0.1 公交车站/初中/卫生服务中心/小学/派出所/轨道交通站点，C.0.2 幼儿园）。train 只用其余条文中的 6 条
   （商场、菜市场、邮政营业场所、门诊部、文化活动中心、小超市），heldout 用剩余 3 条（体育场馆、社区卫生服务站、室外综合健身场地），
   以检验“检索条文 → check_compliance”的流程能迁移到没见过的条文。
4. **题面模板**：train 与 heldout 各自重新写模板（每层 ≥ 8 种句式），不复用 72 题的句式；检查任一 train/heldout 题与任一测试题的
   字符 5-gram Jaccard < 0.5，并人工抽查 50 条。OSM 标签题换键值（如 `cuisine`、`wheelchair`、`internet_access`、`operator`），不用测试题用到的键值对。
5. **步数阈值与半径**：路网分钟数、直线半径在 train/heldout 中从不同取值集合抽样，避免“数字记忆”。

heldout 的金标与 r2 相同方式生成：作者 SQL（`gold_sql.py` 同一口径）→ `freeze_gold.py` 同样的边界裕度检查 → 写入哈希并提交，**早于任何训练**。

## 程序化金标轨迹

每个模板配一个确定性“求解脚本”，按 DeepSeek Agent 成功轨迹的模式（`eval/runs/r2run*/agent.jsonl` 只用来**归纳步骤模式**，不复制题面与答案）
依次调用 `R2Agent._exec_tool` 的真实工具，工具返回值来自数据库，因此轨迹中的结果与校验器提示都是真实的：

- 名称解析：`find_entity`（题面名称去掉“小区/地铁站”等后缀，指定正确图层）→ 用返回的精确名称；
- 路网：`walk_route` / `reachable`（`count_only` 用于“有多少”）/ `isochrone`；
- 合规：`retrieve_standard` → `check_compliance`（clause_id 取检索 top-1）→ `final_answer` 引用含 `compliant` 与 `clause_id` 的结果；
- 标签：`tag_values(key)` → `run_sql` 计数 → `final_answer(result_ref=…)`（结果已是聚合值时**不**设 `result_as='count'`）；
- 不可回答：识别数据缺失的字段后 `final_answer(refuse=true)`，比例与测试集相近（约 14%）。

约 25% 的轨迹加入**可恢复错误**：先用带后缀的名称得到 0 行或校验器报错，下一步改正；先选错图层收到 `layer must be one of …`，下一步改正。
这直接对应观察到的循环失败。轨迹的 `final_answer` 用 matcher_r2 对 heldout/train 自己的金标判分，只保留判对的轨迹。

**格式一致性**（风电 SFT 的教训：训练与推理格式不一致会掉 30 个百分点）：训练样本用与 vLLM 相同的 Qwen3 chat template 渲染，
`tools=` 传入与评测完全相同的 `R2_SPECS`，系统提示用 `R2Agent.system_prompt()` 原文，`enable_thinking=False`；
assistant 工具调用渲染为 `<tool_call>\n{"name": …, "arguments": …}\n</tool_call>`。训练前用 vLLM 的 hermes 解析器对每条渲染后的 completion 做往返解析，失败即丢弃。

## 训练配方

改写作者在 wind-workbench 查询规划微调中使用的 LoRA 训练脚本（保留 causal_flash SDPA、梯度检查点、余弦学习率、梯度累积）：

- `encode` 增加 `tools` 参数与多轮支持：整条轨迹一次前向，对**所有** assistant 轮次的 token 计损失（tool 与 user 消息掩码为 -100）；
  `logits_to_keep` 改为从第一个 assistant token 起，而不是只算末尾；
- Qwen3-8B base，LoRA r=16、alpha=32、dropout 0，目标 q/k/v/o/gate/up/down，lr 1e-4，batch 8（梯度累积），2 epoch，seed 0；
- 每 50 步在 5% 内部验证集上算损失；不在 heldout 或 r2 test 上做任何调参或早停。

## 评测（训练前写入预注册附录）

- 推理：vLLM 0.11.2 `--enable-lora --lora-modules geo=<adapter>`，与本次评测相同的参数、温度 0、非思考模式，Agent 与消融臂各 3 次；
- **主指标**：heldout 上 LoRA Agent vs 基座 Qwen3-8B Agent，配对精确 McNemar；
- 次要：r2 72 题上 LoRA Agent vs DeepSeek Agent（r2run1..3）与 vs 基座；工具调用纯文本率、参数校验错误/题、no_answer、调用次数、延迟；
- 判定：heldout 显著优于基座且 r2 test 不下降，才称“微调有效”；与 DeepSeek 的差距无论多大都如实报告。
  若 heldout 提升但 r2 test 不提升，结论是“对模板分布有效、对未见题面不泛化”。

## GPU 时间估计

- 序列长度：系统提示 + 工具定义约 6.2k token（本次 Agent 每次调用最少 6,175 prompt token），一条 3–4 轮轨迹约 8–10k token；
  1,500 条 × 约 9k token ≈ 13.5M token/epoch，2 epoch ≈ 27M token。
- 吞吐：风电 SFT r1 在同型号 GPU 上用同一脚本，1,600 次样本前向/反向（每条约 23k token）用了 14,370 s，约 2.5k token/s。
  据此训练约 **3–3.5 GPU 小时**；先跑 20 步试训实测吞吐与显存，偏差超过 30% 再调整。
- 评测：本次 9 次 72 题运行并发约 6 分钟；heldout 150 题 × 2 臂 × 3 次 + r2 72 题 × 2 臂 × 3 次，约 0.5 GPU 小时。
- 合计约 **4–5 GPU 小时**（RTX PRO 6000 96 GB），另留 1 小时余量给一次失败重试。数据生成只用本地 CPU 与数据库，不占 GPU。
- 磁盘：adapter 约 0.2 GB；训练日志与数据写到数据盘，不占系统盘。
