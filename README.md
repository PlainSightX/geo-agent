# geo-agent

[![CI](https://github.com/PlainSightX/geo-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/PlainSightX/geo-agent/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Data: OSM ODbL](https://img.shields.io/badge/data-OpenStreetMap%20ODbL-lightgrey)

用中文提问城市空间问题，例如"离新街口地铁站最近的 5 家银行是哪些？"，系统会查询南京的地图数据库，给出带地图的中文答案。背后是一个 LangGraph Agent：它调用类型化 GIS 工具（缓冲区、最近邻、按区统计、面积/长度、拓扑关系）、**步行路网工具**（pgRouting 最短路、等时圈、N 分钟可达设施）、**规划标准条文检索**（GB 50180-2018 摘录 + 本地向量模型 + pgvector）或受限的只读 SQL，由确定性校验器检查单位/CRS、空结果、截断与条文引用，最多修复 3 次。**答案里的数字只能来自一次查询结果，模型手写的裸数字会被剔除并记录**——这是这套系统区别于普通问答的核心约束。

项目的重点是评测：两轮都是先写题和金标并冻结哈希，再让同一模型（`deepseek-chat`）以"单次 text-to-SQL"和"Agent"两种方式作答，用确定性规则判分。

## 项目摘要

**主要工作**：构建 LangGraph 空间 Agent（直线 GIS、pgRouting 路网、规划标准 RAG 检索、受限只读 SQL 四类工具）与确定性校验器（单位/CRS/空结果/条文引用，≤3 次修复）；设计数字绑定机制禁止模型手写裸数字；做两轮预注册评测对照单次 text-to-SQL 与 Agent；在本地 Qwen3-8B 上评测并做 LoRA 微调；把工具集封装为 MCP 服务器对接 Claude Desktop / Cursor。

**成果**：第二轮 72 题上 Agent 94.4% vs 单次 SQL 70.8%，McNemar p = 0.0005，差距主要来自路网与合规题；本地 Qwen3-8B 基座直接替换时 Agent 只有 19.9%。LoRA 微调后，150 道同分布留出题由 39.3% 升到 86.7%；原 72 题由 20.8% 升到 70.8%，仍低于 DeepSeek Agent 的 94.4%。详见下方结果一览。

**架构**：问题 → LangGraph model⇄tools 循环 → 校验器（单位/CRS/条文核验）+ 数字绑定 → 中文答案 + GeoJSON；SQL 走 sqlglot 语法树白名单 + 只读事务 + 数据库角色权限三层防线。

**关键技术**：LangGraph、PostGIS + pgRouting + pgvector、sqlglot SQL 守卫、bge-small-zh ONNX 本地向量检索、PEFT/LoRA + vLLM 微调与推理、MCP（官方 Python SDK）。

![第二轮演示：南湾营融康苑步行 12.5 分钟等时圈与圈内中学，按 GB 50180-2018 初中服务半径判定](docs/screenshot_r2.png)

## 结果一览

- **第一轮**（61 题、详细中文 schema 注释）：两者没有可分辨差异，Agent 57/61 vs 基线 56/61，McNemar p = 1.0。schema 注释太详细，题目太容易，这组题区分不出 Agent 的优势——详见 [第一轮](#第一轮tag-round1)。
- **第二轮**（72 题、精简 schema、原始 OSM 标签/别名/投影陷阱/步行路网/标准合规，各臂 3 次）：Agent **68/72 = 94.4%** [86.6, 97.8] vs 单次 text-to-SQL **51/72 = 70.8%** [59.5, 80.1]（run1），配对 McNemar **p = 0.0005**；3 次均值 94.0% vs 72.7%。差距几乎全部来自路网与合规题；在原始标签题上 Agent 反而更差（75% vs 97%）。代价约 8.4 倍费用、2.8 倍延迟。
- **本地 Qwen3-8B**（vLLM，同一题集与系统，只换模型，各臂 3 次）：Agent **19.9%** [15.1, 25.7]，低于它自己的单次 SQL（36.1%）和 DeepSeek Agent（94.0%），run1 配对 54 vs 0（p ≈ 1e-16）。主要失败是工具调用写成纯文本、名称解析循环和计数方式误用。
- **Qwen3-8B LoRA 微调**（1,342 条程序化金标轨迹，1.5 GPU 小时，预注册判定）：新冻结的 150 题 heldout 上 **130/150 = 86.7%** vs 基座 59/150（p = 1.5e-19）；r2 72 题 **51/72 = 70.8%** vs 基座 15/72（p = 5.6e-9），仍低于 DeepSeek Agent 68/72（19 vs 2，p = 0.0002）。按预注册规则判定为"微调有效"。
- **MCP 服务器**（`geo-agent-mcp`，官方 Python SDK，stdio 为主）：把第二轮 15 个已测试的空间工具、精简 schema 与条文清单暴露给 Claude Desktop、Cursor 等 MCP 客户端，只读、同一 SQL 守卫与限额。

## 快速开始

需要 Docker、[uv](https://docs.astral.sh/uv/)、Git Bash（Windows）或任意 bash。

```bash
uv sync --locked
bash scripts/load_data.sh          # 下载固定日期的 OSM 快照（约 80 MB，校验 sha256）→ GDAL 容器导入 → 建表，约 1 分钟
uv run pytest -q                   # 141 个测试；没有数据库时自动跳过集成测试，不需要 API key

# 需要 DEEPSEEK_API_KEY：
uv run uvicorn geo_agent.api:app --host 127.0.0.1 --port 8765
# 浏览器打开 http://127.0.0.1:8765/?q=离新街口地铁站最近的 5 家银行是哪些？

# 复现第二轮评测（判分不需要 API；运行一臂约 $0.06–0.48）
uv run python eval/r2/freeze_gold.py --check
uv run python eval/r2/run_eval.py --arm agent --set test --run myrun
uv run python eval/r2/score.py --runs myrun
```

## 架构

```
问题 ──► LangGraph：model ⇄ tools，校验 + ≤3 次修复 + 数字绑定（最多 10 次模型调用）
           │
           ├─ 自省：find_entity（名称 + alt_name/old_name/short_name/name:en 别名）· distinct_values · tag_values（原始 OSM 标签）
           ├─ 直线 GIS：缓冲区 · 最近邻 · 按区统计 · 面积/长度 · 拓扑关系 · distance_between
           ├─ 路网（pgRouting 3.8）：walk_route（pgr_dijkstra）· isochrone（pgr_drivingDistance + 边缓冲）· reachable（N 分钟可达设施）
           ├─ 标准检索（RAG）：retrieve_standard（bge-small-zh ONNX 本地向量 → pgvector 余弦 top-k）· check_compliance（半径由程序从条文读取）
           └─ run_sql：sqlglot 守卫（pgRouting 函数的字符串参数递归过同一白名单）→ 单位静态检查 → 只读事务
           │
           ▼  校验器新增：合规结果引用的 clause_id 必须存在；若含 radius_m，必须等于条文数值
       绑定答案 + GeoJSON（线 / 面 / 点）
```

**三层只读防线**：sqlglot 语法树白名单（只允许单条 SELECT/UNION/CTE，禁止 INTO/FOR UPDATE/DML/DDL/COPY/SET，表名限业务表，未识别函数只放行 `st_*` 与少量白名单）→ 只读事务 + 语句超时 → 数据库角色权限。测试覆盖了"绕过守卫直接执行 CREATE TABLE 也会失败"。

**单位/CRS 检查**：所有几何都是 EPSG:4326（度）。对 `ST_Distance/ST_DWithin/ST_Area/ST_Length/ST_Buffer` 的几何参数要求转 `geography` 或 `ST_Transform` 到米制 CRS（32650 / 4547 / 4549）；3857 Web 墨卡托在南京纬度长度放大约 18%，视为错误，在执行前拦截，比"跑出一个看似合理的错数字"更早暴露问题。

**数字绑定**：模型只能写 `{R3.n}`、`{R3.count}`、`{R3.rows[0].name}`、`{R3.rows}` 这类占位符；问题原文以外的裸数字一律剔除并记录。一个真实例子："玄武湖面积多少公顷"，`measure` 只给平方公里，模型在模板里手写"349.7 公顷"被绑定器剔除，判错并计为单位错误——绑定器起了作用，但也暴露了工具缺一个单位选项。

**等时圈计算**：步速 4.8 km/h = 80 m/min；要素代表点吸附到最大连通分量的 KNN 最近顶点，吸附段不计入；等时圈 = 两端可达的边在 EPSG:32650 上 2 m 简化、50 m 缓冲后求并（比先并后缓冲快约 16 倍，面积差 < 1%），是"边缓冲"而不是凹包。

**步行路网**：从南京 OSM 快照的线要素构建，纳入 primary…residential、service、pedestrian、footway、path、steps、track、cycleway；trunk 只在允许步行时纳入，motorway 一律排除。只在 OSM 共享节点与端点处打断（桥隧不与下方道路相交）。结果：**118,304 条边、87,970 个顶点、21,204 km；1,088 个连通分量，最大分量占顶点 94.4%、长度 97.3%**。全部路由在最大分量上进行。

**消融说明收益来源**：第二轮去掉路网与标准工具的消融臂，同样能写 pgRouting SQL、同样有修复循环，路网题仍有 91.7%（修复循环救回了大部分 SQL 错误），但合规题从 100% 掉到 63.9%——13 次失败中 11 次是结论对了但结果里没有条文 id、2 次步数用尽未作答。`check_compliance` 由程序读取条文半径、强制输出 clause_id，正是它补上的部分；这组对照能说明移除这两类工具会让合规题的结果完整性下降，但不能单独分离路网工具、检索工具与 `check_compliance` 各自的贡献。

**本地 Qwen3-8B 的失败分类**（Agent 臂 3 次共错 173 题次，事后分类）：

| 主因 | 题次 | 说明 |
|---|---|---|
| 工具调用写成纯文本 | 48 | vLLM 的 hermes 解析器返回 `tool_calls=None`，模型不带 `<tool_call>` 标签，循环提醒后重复同一段文本直到调用用尽 |
| 调用次数用尽 | 36 | 其中 33 次因 `find_entity` 返回 0 行：题面写"南湾营融康苑小区"，库里名称是"南湾营融康苑"，模型保留"小区"后缀依次换图层重试 |
| 实体找不到就拒答 | 23 | 同上的名称问题；或 `check_compliance` 报"not found"后直接拒答 |
| 计数方式用错 | 17 | 对已经是计数或单行的结果设 `result_as='count'`，程序数出 1 行 |
| 不可回答题给出答案 | 12 | 拿相关但不对应的数据硬答，如用可达设施数回答"客流量" |
| 选错工具、口径或列名 | 29 | 如用直线距离回答步行距离；合规题用可达计数代替合规检查 |
| 同一调用重复 ≥3 次 / 其他拒答 | 5 / 3 | |

对照：DeepSeek Agent 3 次共错 13 题次，11 次是口径比金标宽，2 次是拒答。差距主要来自工具调用协议和多步控制，不是 GIS 知识本身。

**LoRA 微调做法**：用确定性求解脚本驱动真实的 `R2Agent` 工具生成 1,342 条训练轨迹（tag 92 / join 260 / crs 230 / route 320 / compliance 240 / unanswerable 200），与测试集在实体、设施类型、条文、标签键、数值与句式上互不相交。r=16、alpha=32，q/k/v/o/gate/up/down 全部线性层，RTX PRO 6000 上训练+验证共 1.54 小时。微调后主要新失败是误拒答：训练集 200 条不可回答轨迹都用同一句理由，模型在 heldout 遇到没见过的标签键时套用了这句模板拒答——这是微调引入的新失败模式，基座反而没有这个问题。

## 技术栈

Python、LangGraph、PostGIS + pgRouting + pgvector、FastAPI、Leaflet、MCP（官方 Python SDK）、本地向量模型（bge-small-zh ONNX，CPU 推理，不依赖 torch）、PEFT/LoRA + vLLM（本地模型微调）、sqlglot（SQL 语法树校验）、Docker。

## 局限

- 题目、金标与口径均由作者一人决定：区内判定、道路裁剪方式、实体名称识别规则都是一种可辩论的口径；B03、A05、E06 这类失败都与口径有关。
- "合规"只是按标准推荐性服务半径（"不宜大于"）做的空间核查，不是规划审批意义上的合规结论；条文为人工转录，以官方 PDF 为准。
- OSM 在中国的步行网络不完整：缺人行道、过街设施和小区内部道路；部分小区代表点到路网吸附超过 100 米，部分干道因禁止步行标签导致网络距离偏长。
- 单一模型（主评测用 `deepseek-chat`）、单一城市、单一快照；本地模型评测的提示词是为 DeepSeek 写的，没有为 Qwen3-8B 单独调过。
- LoRA 微调的 heldout 集与训练数据出自同一生成器，衡量的是同任务分布内的泛化；r2 72 题更接近独立来源的测试，微调后仍低于 DeepSeek Agent 约 24 个百分点。
- 72 题仍是小样本，单臂区间约 ±10 个百分点；第二轮 Agent 的提示词和工具在 12 道开发题上只跑过一次，没有据此调参，但作者设计工具时已经知道题型分布。
- 演示 API 没有鉴权，只应监听 127.0.0.1；MCP 服务器默认 stdio，HTTP 模式固定绑定 127.0.0.1，没有改绑其他地址的选项。
- MCP 客户端的调度由客户端里的模型自己决定，没有校验/修复/数字绑定，评测数字（94.4% 等）不适用于 MCP 客户端。

## 第一轮（tag `round1`）

**61 题、详细中文 schema 注释**：两者没有可分辨差异，Agent 57/61 vs 基线 56/61，McNemar p = 1.0。给定带中文注释的 schema 时，同一个模型单次写 SQL 已经能答对九成以上；这组题区分不出 Agent 的优势。Agent 在本项目里的实际价值是工程性质的：答案数字可追溯到具体查询、结果能直接上图、写错 SQL 能自我修复、模型手写的数字会被拦截——这些在第二轮的失败案例里都有体现，但没有转化成第一轮这 61 题上的准确率提升。完整结果、逐题记录与预注册后的修订披露见 [docs/results/round1/README.md](docs/results/round1/README.md)。

## 第二轮：路网、标准检索与更难的预注册题集

第一轮的结论是"题太容易、schema 注释太详细"。第二轮按 GIS + AI 岗位常见要求（路径规划、空间 RAG、空间 SQL、GeoJSON）扩展系统，并换一组更接近真实使用的题。基线拿到的规则说明和 Agent 完全一样，包括吸附规则、步速、合规口径和输出列格式，基线只少了工具、检索和修复——这是有意设计的对照，不是让 Agent 赢的裁判偏向。

在路网题上，单次 SQL 的失败不是"不会"，而是一次写对一条 60–100 行的 pgRouting 查询很难：最常见的错误是 `MIN(agg_cost)` 取到路径第一行的 0、`round(double precision, int)` 类型错误、把投影几何转 geography、超时。Agent 的 `walk_route/isochrone/reachable` 把这些口径固化在参数化查询里，模型只填起终点。

原始标签题上 Agent 反而更差，原因是它更主动地探索：用 `tag_values` 看到 `brand=KFC`、`takeaway=only`、"蔚来换电站"后把它们也算进去，口径比金标宽。按预注册规则判错；事后敏感性分析（非预注册）把这三个更宽口径也算对时，Agent 为 71/72，基线 51/72，结论不变——这暴露的是题面歧义，不是 Agent 的推理错误。

逐题失败记录、按层级拆分的完整数据、预注册后的全部改动披露见 [docs/results/round2/README.md](docs/results/round2/README.md)。

## 本地模型评测与微调

本地 Qwen3-8B 不能直接替换 DeepSeek：同一 72 题冻结集、同一提示词、工具、校验器和判分代码，只把模型换成 vLLM 部署的 Qwen3-8B 基座，Agent 比它自己的单次 SQL 还差，比 DeepSeek Agent 低约 74 个百分点。差距主要来自工具调用协议和多步控制（见上文"架构"小节的失败分类表），不是 GIS 领域知识不够。按层结果、完整失败分类与格式修复诊断见 [docs/results/local-qwen3-8b/README.md](docs/results/local-qwen3-8b/README.md)。

用程序化生成的金标轨迹做 LoRA 微调后，heldout 从 39.3% 升到 86.7%，r2 从 20.8% 升到 70.8%，提升在两个集合上都显著，但仍比 DeepSeek Agent 低约 24 个百分点，不能当作等价替代——按预注册规则判定为"微调有效"，定位是内网私有化部署场景。完整训练细节、失败案例分析、数据事故披露见 [docs/results/lora-finetune/README.md](docs/results/lora-finetune/README.md)。

## MCP 服务器

`geo-agent-mcp` 把第二轮已测试的类型化工具按 [Model Context Protocol](https://modelcontextprotocol.io) 暴露出来，Claude Desktop、Cursor、MCP Inspector 等客户端可以直接调用。工具实现不重写：每个 MCP 工具只是把参数交给 `tools` / `routing` / `r2_tools` 中的同一函数。

- **工具（15 个）**：自省 3 个、直线 GIS 6 个、路网 3 个、标准 2 个、受限 SQL 1 个，全部与第二轮 Agent 工具集一致。
- **安全**：只用只读角色 `geo_reader`；`run_sql` 与 Agent 共用同一路径——单位/CRS 静态检查 → sqlglot 语法树白名单 → READ ONLY 事务、5 秒超时、200 行上限。服务器没有鉴权，推荐 stdio；HTTP 模式固定绑定 127.0.0.1。
- **与仓库内 LangGraph Agent 的区别**：Agent 是本项目自己的编排层，评测结果（94.4% 等）测的是整套循环；MCP 只暴露工具，调度交给客户端里的模型，没有校验修复与数字绑定，上面的评测数字不适用于 MCP 客户端。

```bash
uv run geo-agent-mcp                                   # stdio（默认）
uv run python scripts/mcp_demo.py 南湾营融康苑 12.5     # SDK stdio 客户端：列出工具并调用 isochrone
```

一次真实客户端会话的调用与结果大小见 [docs/mcp-demo.md](docs/mcp-demo.md)。

## 许可

代码：MIT（见 `LICENSE`）；数据许可说明见 `DATA_LICENSE.md`。地图数据：© OpenStreetMap contributors，ODbL 1.0。GB 50180-2018 条文摘录来自住房和城乡建设部官网公开文本，仅作引用与研究用途。
