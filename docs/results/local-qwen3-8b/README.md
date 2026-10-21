# 本地 Qwen3-8B 完整结果（私有化部署评估）

主 README 的「本地 Qwen3-8B」小节是这份记录的摘要。本文档保留按层结果、失败分类的完整明细、格式修复诊断与预注册后披露。

问题：在内网或私有化部署时，用本地 8B 模型替换 DeepSeek，这套 Agent 还能用吗？
做法：同一 72 题冻结集（金标冻结于 `08309f2`，未改动）、同一提示词、工具、校验器和判分代码，只把模型换成 vLLM 0.11.2 部署的
**Qwen3-8B 基座**（不加 adapter，bf16，单卡 RTX PRO 6000 96 GB，`--tool-call-parser hermes`，`--max-model-len 32768`）。
温度 0，非思考模式（`enable_thinking=false`，与 DeepSeek 臂一致），三臂各跑 3 次。另跑了一次思考模式，单独报告。

**结论：不能直接替换。** Qwen3-8B 的 Agent 比它自己的单次 SQL 还差，比 DeepSeek Agent 低约 74 个百分点。
差距主要来自工具调用协议和多步控制，GIS 知识不是主因。

## 结果（72 题，3 次合并；方括号为 Wilson 95% 区间）

| 指标 | DeepSeek (b) Agent | Qwen3-8B (a) 基线 | Qwen3-8B (b) Agent | Qwen3-8B (c) 消融 |
|---|---|---|---|---|
| **run1 正确率** | **68/72 = 94.4%** [86.6, 97.8] | 25/72 = 34.7% [24.8, 46.2] | **14/72 = 19.4%** [12.0, 30.0] | 13/72 = 18.1% [10.9, 28.5] |
| 3 次逐次 | 94.4 / 93.1 / 94.4% | 34.7 / 36.1 / 37.5% | 19.4 / 19.4 / 20.8% | 18.1 / 16.7 / 16.7% |
| 3 次合并 | 203/216 = 94.0% [90.0, 96.4] | 78/216 = 36.1% [30.0, 42.7] | 43/216 = 19.9% [15.1, 25.7] | 37/216 = 17.1% [12.7, 22.7] |
| 题级一致率（3 次） | — | 97.2% | 98.6% | 95.8% |
| no_answer（216 题次） | 0 | 0 | **89** | 70 |
| 拒答（含不可回答题 30） | 32 | 33 | 44 | 44 |
| 模型调用 / 题 | 3.01 | 1.00 | 6.78 | 6.65 |
| 修复次数 / 题 | 0.16 | — | 1.04 | 1.29 |
| 工具参数校验错误 / 题 | 0.01 | — | 0.50 | 0.78 |
| prompt / completion token / 题 | 20.2k / 389 | 2.0k / 143 | 45.2k / 306 | 34.7k / 321 |
| 延迟中位数 / p90 | 3.1 s / 5.6 s | 1.1 s / 4.4 s | 9.6 s / 15.0 s | 7.5 s / 12.7 s |

DeepSeek 基线与消融见 [第二轮结果](../round2/README.md)（72.7%、87.0%）。Qwen 的延迟是 3 次运行同时进行、每次 4 线程（最多 12 个并发请求）共用一张卡时测得的，单请求延迟会更低。
三臂 9 次运行共约 3,100 次模型调用、约 1,700 万 prompt token，墙钟时间约 6 分钟；本地端点不计费，台账在 `eval/runs/local/ledger_local.jsonl`。

**按层（3 次合并正确率）**

| 层（题数 × 3） | DeepSeek Agent | Qwen 基线 | Qwen Agent | Qwen 消融 |
|---|---|---|---|---|
| tag 原始标签（36） | 75.0% | 36.1% | 25.0% | 16.7% |
| join 多表/别名（36） | 100% | 38.9% | 8.3% | 27.8% |
| crs 投影与度量（30） | 93.3% | **70.0%** | 10.0% | 3.3% |
| route 路网（48） | 95.8% | 0% | 20.8% | 6.2% |
| compliance 合规（36） | 100% | 0% | **0%** | 0% |
| unanswerable 不可回答（30） | 100% | 100% | 60.0% | 56.7% |

**配对（run1，精确 McNemar）**

- DeepSeek Agent vs Qwen Agent：仅 DeepSeek 对 54 题，仅 Qwen 对 0 题，**p ≈ 1×10⁻¹⁶**（3 次多数票配对：53 vs 0）。
- Qwen 基线 vs Qwen Agent：仅基线对 18 题，仅 Agent 对 7 题，**p = 0.043**，方向与 DeepSeek 相反（DeepSeek 上 Agent 显著更好）。
  仅基线对的 18 题：投影/度量 7、别名/多表 4、不可回答 4（Agent 硬答）、标签 3；仅 Agent 对的 7 题：最短路 3、标签 2、度量 1、别名 1。
- Qwen 消融 vs Qwen Agent：6 vs 7，p = 1.0。工具化在 8B 上没有带来可测的收益。
- DeepSeek 基线 vs Qwen 基线：28 vs 2，p = 9×10⁻⁷。

**思考模式**（`--thinking`，各 1 次，不与上表合并）：基线 21/72 = 29.2% [19.9, 40.5]，Agent 13/72 = 18.1% [10.9, 28.5]，
与非思考模式相比都没有显著差异（基线 4 vs 0，p = 0.125；Agent 8 vs 7，p = 1.0）。Agent 延迟中位数升到 26.8 s，completion token 增加约 7 倍。
思考模式基线拒答 41 题（非思考 11 题），常见理由是"数据库中没有营业时间数据"，而这些信息在 `poi_tags` 里。

## Qwen Agent 在哪里失败

事后分类（非预注册，规则见 `eval/local_llm/failure_taxonomy.py`，每个错题次只计一个主因；结果 `eval/runs/local/failure_taxonomy.json`）。
Agent 臂 3 次共错 173 题次：

| 主因 | 题次 | 说明 |
|---|---|---|
| 工具调用写成纯文本 | 48 | 拿到工具结果后输出 `final_answer\n{"result_ref": "R1", …}` 或裸 JSON，不带 `<tool_call>` 标签，vLLM 的 hermes 解析器返回 `tool_calls=None`；循环提醒"请调用 final_answer"后模型重复同一段文本，直到 10 次调用用尽 |
| 调用次数用尽 | 36 | 其中 33 次有 `find_entity` 返回 0 行：题面写"南湾营融康苑小区"，库里名称是"南湾营融康苑"。模型保留"小区"后缀，依次换 `pois/districts/water_areas/parks/roads` 图层重试，不去掉后缀 |
| 实体找不到就拒答 | 23 | 同上的名称问题；或 `check_compliance` 报"not found"后直接 `refuse=true` |
| 计数方式用错 | 17 | 对已经是计数或单行的结果设 `result_as='count'`，程序数出 1 行，答案为 n=1（T07、D06、D12 等） |
| 不可回答题给出答案 | 12 | U01/U05/U07/U08：拿相关但不对应的数据硬答（如用可达设施数回答"客流量"） |
| 选错工具、口径或列名 | 29 | 如用 `distance_between`（直线）回答步行距离；合规题用 `reachable` 计数代替 `check_compliance`，结果里没有 clause_id |
| 同一调用重复 ≥3 次 / 其他拒答 | 5 / 3 | |

对照：DeepSeek Agent 3 次共错 13 题次，11 次是口径比金标宽（T06/T07/T10），2 次是拒答。工具参数校验错误 3 次（Qwen 为 107 次）。

工具参数 JSON 本身很少坏：服务器日志里 hermes 解析器只报了 3 次 JSON 语法错误（`Expecting ',' delimiter`）。
参数问题主要是**语义**错误：图层不在枚举里（`find_entity` 用 `roads`）、度量类型和图层不匹配（对 `pois` 求 `area_km2`）、
`count_by_district` 传 `bus_stops`、半径超出范围。非思考模式 6 次 Agent/消融运行中，合规层都是 0%：没有一次正确走完"检索条文 → `check_compliance` → 引用 clause_id"（思考模式 Agent 对了 1 题）。

**基线（单次 SQL）的失败**：exec_error 平均每次 26 题，3 次合计最常见的是 pgRouting 边查询写错列名（`column "node" does not exist` 17 次）、
`WHERE` 后接了数值表达式（9 次）、没 JOIN 就引用 `poi_tags`（6 次）、`ST_Transform(geography, …)`（4 次）、超时（4 次）。
路网层和合规层都是 0%。投影与度量题 70% 正确，接近 DeepSeek 基线（73%）：短 SQL 能写对，60–100 行的 pgRouting CTE 写不对。

**只修格式能挽回多少**（事后诊断，非预注册，不进主表）：`eval/local_llm/diagnose_text_toolcalls.py --mode salvage` 在客户端把
"工具名 + JSON"形式的纯文本转成结构化调用（Agent 与消融两臂共 140 个文本轮次，救回 30 个），其余不变，Agent 单次 19/72 = 26.4% [17.6, 37.6]、消融 19/72。
`--mode log` 原样复跑一次，Agent 15/72，记录了 145 个文本轮次：72 个是"工具名 + JSON"，37 个是不带工具名的裸 `final_answer` 参数，35 个是直接用中文作答。
所以格式修复只占差距的一小部分，名称解析、计数方式、合规流程和拒答判断都要另外解决。

## 是否微调

值得做，计划见 `eval/local_llm/FINETUNE_PLAN.md`；已按计划训练并评测，结果见 [LoRA 微调](../lora-finetune/README.md)。理由：差距很大（p ≈ 1e-16）；主要失败（纯文本调用、名称后缀循环、
`result_as` 误用、合规链路缺失、不可回答题硬答）都是工具调用行为，正确做法确定，可以用确定性工具程序化生成训练轨迹，
不需要 DeepSeek 或人工标注。计划要求训练集与新的冻结 heldout 集在小区、地点、设施类型、条文和题面模板上与这 72 题**互不相交**，
heldout 在训练前冻结；LoRA r16 改写风电 SFT 的 `train_lora.py`（多轮、对全部 assistant 轮次计损失）。估计训练约 3–3.5 GPU 小时，含评测约 4–5 小时。

一个不需要训练的改动也可能有效：`find_entity` 先去掉"小区/地铁站"等后缀再匹配。这会改 Agent，需要对 DeepSeek 重跑并作为预注册后改动披露，所以没有在本次评测中做。

## 局限

- 只测了一个基座模型、一种 vLLM 工具调用解析器（hermes）和一种提示词。提示词与工具描述是为 DeepSeek 写的，在 12 道开发题上调过；
  没有为 Qwen 调提示词，这是有意的（同一系统换模型），但也意味着本结果是"不改系统直接换模型"的下限，不是 Qwen3-8B 能力的上限。
- 没有测其他工具调用封装（如 Qwen 官方的 Qwen-Agent）或其他 vLLM 解析器；更大的本地模型（Qwen3-32B 等）也没有测。
- 温度 0 下题级一致率 96–99%，3 次运行几乎是同一结果的重复，合并区间低估了提示词或解码设置变化带来的方差。
- 失败分类、格式修复诊断都是看过结果之后设计的（非预注册），只用来解释原因，不改变主结果。
- 本地台账由并发进程追加写入，有 14 行被交错写坏，已按记录起始位置无损重组；台账调用数比逐题记录少约 2%，逐题记录中的 token 与延迟是完整的。
- 运行与文件：`eval/runs/local/local_qwen3-8b_run{1,2,3}`（主结果）、`local_qwen3-8b_think_run1`（思考模式）、`diag_qwen3-8b_{textlog,salvage}`（诊断）；
  汇总 `eval/runs/local/r2_summary_test.json`、与 DeepSeek 的对比 `compare_deepseek.json`（`eval/local_llm/compare_deepseek.py`）。
