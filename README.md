# geo-agent：南京 OSM + PostGIS 空间问答 Agent（含预注册评测）

用中文提问城市空间问题（“新街口地铁站 500 米内有多少家餐厅？”“哪些区与玄武区相邻？”），
系统把问题交给 LangGraph Agent：它调用类型化 GIS 工具（缓冲区、最近邻、按区统计、面积/长度、拓扑关系）或受限的只读 PostGIS SQL，
由确定性校验器检查单位/CRS、空结果和截断，最多修复 3 次，最后输出一段**数字全部由程序从查询结果中填入**的中文答案和可在 Leaflet 地图上渲染的 GeoJSON。
项目的重点是评测：在任何模型调用之前，我写好了 61 道题和金标 SQL，并冻结了执行结果和哈希；然后让同一模型分别以“单次 text-to-SQL”和“Agent”两种方式作答，用确定性规则判分。
**结果是两者准确率没有可分辨差异（93.4% vs 91.8%，McNemar p = 1.0）**，Agent 的代价是约 7.6 倍费用。下面如实报告这个结果和全部失败案例。

![Leaflet 结果页：离新街口地铁站最近的 5 家银行](docs/screenshot.png)

## 关键结果（test1，2026-10-18，`deepseek-chat`，temperature 0，各臂各跑一次）

| 指标 | (a) 单次 text-to-SQL | (b) Agent（工具 + 校验 + 修复） |
|---|---|---|
| **总体正确率（61 题）** | **56/61 = 91.8%** [82.2, 96.5] | **57/61 = 93.4%** [84.3, 97.4] |
| 可回答题执行准确率（52） | 47/52 = 90.4% [79.4, 95.8] | 48/52 = 92.3% [81.8, 97.0] |
| 不可回答题拒答正确（9） | 9/9 [70.1, 100] | 9/9 [70.1, 100] |
| 有效 SQL 率（模型写的原始 SQL） | 49/49 = 100% | 29/32 = 90.6%（3 条均在下一步修复） |
| 单位/CRS 错误（33 道度量题） | 0/33 [0, 10.4] | 1/33 [0.5, 15.3] |
| 模型调用 / 题 | 1.00 | 2.30 |
| 费用 / 题（估算） | $0.0004 | $0.0030 |
| 延迟中位数 / p90 | 0.98 s / 1.18 s | 2.20 s / 3.48 s |

方括号为 Wilson 95% 区间。配对：两臂都对 53 题、都错 1 题，仅基线对 3 题、仅 Agent 对 4 题，精确 McNemar p = 1.0。
全部 API 花费（含开发调试、测试集、演示）约 **$0.31**（台账：`eval/runs/ledger.jsonl`）。

**怎么理解**：给定带中文注释的 schema 时，同一个模型单次写 SQL 已经能答对九成以上；这组题区分不出 Agent 的优势。
Agent 在本项目里的实际价值是工程性质的：答案数字可追溯到具体查询、结果能直接上图、写错 SQL 能自我修复、模型手写的数字会被拦截。
这些在下面的失败案例里都有体现，但它们没有转化成这 61 题上的准确率提升。

## 快速开始

需要 Docker、[uv](https://docs.astral.sh/uv/)、Git Bash（Windows）或任意 bash。

```bash
uv sync --locked
bash scripts/load_data.sh          # 下载固定日期的 OSM 快照（约 80 MB，校验 sha256）→ GDAL 容器导入 → 建表，约 1 分钟
uv run pytest -q                   # 87 个测试；没有数据库时自动跳过 13 个集成测试，不需要 API key

# 需要 DEEPSEEK_API_KEY：
uv run uvicorn geo_agent.api:app --host 127.0.0.1 --port 8765
# 浏览器打开 http://127.0.0.1:8765/?q=离新街口地铁站最近的 5 家银行是哪些？

# 复现评测（判分不需要 API；运行两臂约 $0.21）
uv run python eval/freeze_gold.py --check          # 重新执行金标 SQL 并做边界裕度检查
uv run python eval/run_eval.py --arm baseline --set test --run myrun
uv run python eval/run_eval.py --arm agent    --set test --run myrun
uv run python eval/score.py --run myrun
```

## 架构

```
问题 ──► LangGraph:  model ──tool_calls──► tools ──► model … ──final_answer──► 绑定答案 + GeoJSON
                       │                     │
                       │       ┌─────────────┴──────────────────────────────┐
                       │       │ 类型化工具 (tools.py，参数化 SQL，geography 米制) │
                       │       │  find_entity / distinct_values（自省）        │
                       │       │  within_distance / nearest_k / count_by_district │
                       │       │  measure / spatial_relation / distance_between │
                       │       │ run_sql：sqlglot AST 白名单 → 单位静态检查 → 执行 │
                       │       └─────────────┬──────────────────────────────┘
                       │                     ▼
                       │       校验器：守卫拒绝 / 执行错误 / 单位·CRS / 空结果 / 截断
                       │       有问题 → 回传给模型并消耗 1 次修复额度（≤3）
                       ▼
               deepseek-chat（每次调用写入费用台账，累计 $3.6 停止）

PostGIS 16-3.5（docker compose，镜像固定 digest）
  geo_reader 角色：只读、无 CREATE、statement_timeout 5s；每次查询再包一层 READ ONLY 事务；最多返回 200 行
```

三层只读防线：sqlglot 语法树白名单（只允许单条 SELECT/UNION/CTE，禁止 INTO/FOR UPDATE/DML/DDL/COPY/SET，表名限业务表，未识别函数只放行 `st_*` 与少量白名单）
→ 只读事务 + 语句超时 → 数据库角色权限。测试覆盖了“绕过守卫直接执行 CREATE TABLE 也会失败”。

**单位/CRS 检查**（`units.py`）：所有几何都是 EPSG:4326（度）。对 `ST_Distance/ST_DWithin/ST_Area/ST_Length/ST_Buffer` 的几何参数要求转 `geography` 或 `ST_Transform` 到米制 CRS（32650 / 4547 / 4549）；
3857 Web 墨卡托在南京纬度长度放大约 18%，视为错误。在执行前拦截，比“跑出一个看似合理的错数字”更早暴露问题。

**数字绑定**（`answer.py`）：模型只能写 `{R3.n}`、`{R3.count}`、`{R3.rows[0].name}`、`{R3.rows}` 这类占位符；
问题原文以外的裸数字一律剔除并记录。最终被判分、被上图的结果，就是答案引用的那次工具结果（可用 `result_rows` 选行、`result_as=count` 让程序计数）。

## 数据

- 来源：Geofabrik 江苏省 OSM 快照 `jiangsu-261017.osm.pbf`（2026-10-17），sha256 `dcba9384…2652e`，md5 与 Geofabrik 发布值一致。见 `data/SOURCE.json`。
- 处理：GDAL 3.10.3 容器按外包框读取 → `sql/build_tables.sql` 按南京市行政边界（OSM relation 2131524）精确裁剪、修复无效几何。原始 PBF 不入库。
- 表：`districts` 11、`pois` 9,088、`metro_stations` 272（换乘站按站名合并）、`roads` 32,153、`waterways` 1,976、`water_areas` 5,727、`parks` 495。
- 11 个区边界拓扑干净：22 对相邻区 `ST_Touches` 为真、交叠面积为 0；面积合计约 6,591 km²（官方 6,587 km²）。
- 许可：© OpenStreetMap contributors，[ODbL 1.0](https://www.openstreetmap.org/copyright)。代码为 MIT；由 OSM 派生的数据库若公开分发，须按 ODbL 提供。

## 评测设计（预注册）

完整预注册见 `eval/PREREGISTRATION.md`。要点：

- **先冻结、后建模**：题目、金标 SQL、执行结果与哈希在提交 `ac3c09c` 中冻结，早于 Agent 代码（`ba0d532`）和任何模型调用。
- **61 题 7 层**：属性过滤 8、距离/缓冲 9、最近邻 8、按区空间连接/聚合 10、多步 9、CRS/单位陷阱 8、数据无法回答 9。另有 10 题开发集（与测试集不重叠）用于调试。
- **边界裕度检查**：冻结前自动检查——半径/阈值题中所有候选对象与阈值相差 ≥ 0.5%，top-k 题第 k 与 k+1 名相差 ≥ 0.5%，避免“合理的不同算法”翻转金标。
  这个检查在冻结前抓到 4 道有歧义的题（如 B08：4 家咖啡馆距新街口 997.8–1004.0 米，正好跨 1 km），改了半径或 k 后才冻结。
- **确定性判分**（`matcher.py`）：预测可多出列；金标每列须单射映射到预测列；行按多重集合比较（D06 要求顺序）；数值相对误差 ≤ 0.5%。不用 LLM 打分。
- **统计**：Wilson 95% 区间；配对精确 McNemar。

## 失败案例（全部保留）

逐题记录在 `eval/runs/test1/per_question.json`，原始轨迹在 `eval/runs/test1/*.jsonl`。

**基线（5 题错）**

- B03 “江苏省人民医院 2 公里内的其他医院”：用 `name <> '江苏省人民医院'` 排除自身，把同名的另一条记录也排除了（9 vs 金标 10）。
- C05、E07、F06：模型输出 JSON 后又追加了“Wait, that returns multiple rows…”之类的自我纠正文字，解析失败。这是单次调用协议的真实失败模式，按预注册判错。
- E06 “鼓楼区道路总长度”：用 `ST_Within` 只算完全在区内的路段（361.7 vs 407.5 km）。

**Agent（4 题错）**

- A05 “菜系标注为 noodle”：写成 `ILIKE '%noodle%'`，多算了 `noodle;chinese` 等多值标签（33 vs 28）。口径之争，也可以认为 33 更合理。
- A08 “名称以南京开头的博物馆”：查询完全正确，但模型反复改写同一查询 7 次，始终没有调用 `final_answer`，步数用尽。这是循环控制失败。
- E06：用 `ST_Intersects` 但没有按区界裁剪（467.3 vs 407.5 km）。两臂都错的唯一一题。
- F02 “玄武湖面积多少公顷”：`measure` 只给平方公里，模型在模板里手写“349.7 公顷”，被绑定器剔除；引用的结果没有公顷列，判错并计为单位错误。绑定器起了作用，但工具缺一个单位选项。

**其他瑕疵**：Agent 有 5 个答案文本含 `?` 或 `□`（无效占位符如 `{R1.row_count}`，或被剔除的手写数字），其中 3 题最终结果仍判对。答案文本质量不在预注册指标里。

## 预注册之后的改动（披露）

1. **单位错误的数量级判定**：首版对“同时返回 m 和 km 两列、且已判对”的 F04 误报。修订为只在判错题上、金标数值在预测中无相等值但恰好差 10^k 时才计。Agent 单位错误 2→1，正确率不变。
2. **基线原始输出日志被截断为 300 字符**，C05、E07 无法事后复核。已修复，test1 不重跑。
3. **事后敏感性分析**（`eval/sensitivity.py`，`runs/test1/sensitivity.json`）：基线解析器放宽为只取第一个 JSON 对象后，F06 仍错；C05、E07 若全算对（上界），基线为 58/61 = 95.1%，McNemar p 仍为 1.0。结论不变。
4. 演示页在最终结果无几何列时改为渲染最近一个带几何的工具结果（只影响地图，不影响评测）。

## 局限

- **样本量小**：单臂区间宽约 ±7 个百分点；两臂只有 7 题不一致，只能检测很大的差异。
- **题目和金标由作者一人编写**，代表一种口径（区内 = ST_Within、道路按区界裁剪、实体按名称识别）；B03、A05、E06 的失败都与口径有关。
- **schema 注释很详细**（中文类别映射、单位提示），这本身帮助了基线；在注释更少、表更多的真实库上结果可能不同，本项目没有测。
- 单一模型、单一城市、单一快照、temperature 0 单次运行，没有测重复运行方差。开发集偏简单（两臂都 10/10），Agent 提示与工具在开发集上调过 3 轮。
- 没有路网，不支持路径/通行时间；“route”类问题归入不可回答。没有空间 RAG（文档检索）部分。
- OSM 在中国数据不完整（很多酒店无名称），数量结论只针对该快照，不代表真实世界。
- 演示 API **没有鉴权**，只应监听 127.0.0.1；每次请求都会消耗 API 额度，公网部署前必须加鉴权与限流。
- 费用按 $0.30 / $1.20 每百万 token 统一估算，未计缓存折扣，实际账单更低。

## 目录

```
docker-compose.yml        PostGIS（固定 digest）
db/init/01_roles.sql      只读角色
scripts/                  fetch_data.sh（固定快照 + sha256）、load_data.sh
sql/build_tables.sql      裁剪、建表、注释、授权
src/geo_agent/            sql_guard · units · tools · agent · baseline · answer · matcher · stats · llm · api · static/
eval/                     PREREGISTRATION.md · questions.json · gold.json · freeze_gold.py · run_eval.py · score.py · sensitivity.py · runs/
tests/                    87 个 pytest（守卫、单位检查、比对器、统计、数字绑定、离线 Agent 循环、API、DB 集成）
```

## 许可

代码：MIT（见 `LICENSE`）。地图数据：© OpenStreetMap contributors，ODbL 1.0。底图瓦片：OpenStreetMap 标准瓦片（需署名，遵守其使用政策）。
