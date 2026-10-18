# geo-agent：南京 OSM + PostGIS / pgRouting / pgvector 空间问答 Agent（两轮预注册评测）

用中文提问城市空间问题，系统把问题交给 LangGraph Agent：它调用类型化 GIS 工具（缓冲区、最近邻、按区统计、面积/长度、拓扑关系）、
**步行路网工具**（pgRouting 最短路、等时圈、N 分钟可达设施）、**规划标准条文检索**（GB 50180-2018 摘录 + 本地向量模型 + pgvector）
或受限的只读 SQL，由确定性校验器检查单位/CRS、空结果、截断与条文引用，最多修复 3 次，最后输出一段**数字全部由程序从查询结果中填入**的中文答案和 GeoJSON。
项目的重点是评测：两轮都是先写题和金标并冻结哈希，再让同一模型（`deepseek-chat`）以“单次 text-to-SQL”和“Agent”两种方式作答，用确定性规则判分。

- **第一轮**（61 题、详细中文 schema 注释）：两者没有可分辨差异，Agent 57/61 vs 基线 56/61，McNemar p = 1.0。
- **第二轮**（72 题、精简 schema、原始 OSM 标签/别名/投影陷阱/步行路网/标准合规，各臂 3 次）：
  Agent **68/72 = 94.4%** [86.6, 97.8] vs 单次 text-to-SQL **51/72 = 70.8%** [59.5, 80.1]（run1），配对 McNemar **p = 0.0005**；
  3 次均值 94.0% vs 72.7%。差距几乎全部来自路网与合规题；在原始标签题上 Agent 反而更差（75% vs 97%）。代价约 8.4 倍费用、2.8 倍延迟。

![第二轮演示：南湾营融康苑步行 12.5 分钟等时圈与圈内中学，按 GB 50180-2018 初中服务半径判定](docs/screenshot_r2.png)

# 第一轮（2026-10-18，tag `round1`）

![第一轮 Leaflet 结果页：离新街口地铁站最近的 5 家银行](docs/screenshot.png)

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


# 第二轮（2026-10-18）：路网、标准检索与更难的预注册题集

第一轮的结论是“题太容易、schema 注释太详细”。第二轮按 GIS + AI 岗位常见要求（路径规划、空间 RAG、空间 SQL、GeoJSON）扩展系统，
并换一组更接近真实使用的题。第一轮的全部文件、结果与 `round1` 标签保持不变，第一轮 52 道可回答题的金标哈希在新镜像上逐题复现。

## 第二轮结果（`deepseek-chat`，temperature 0，每臂 3 次）

| 指标 | (a) 单次 text-to-SQL | (b) Agent（全部工具） | (c) 消融：去掉路网/检索/合规工具 |
|---|---|---|---|
| **run1 正确率（72 题）** | **51/72 = 70.8%** [59.5, 80.1] | **68/72 = 94.4%** [86.6, 97.8] | 62/72 = 86.1% [76.3, 92.3] |
| 3 次逐次 | 70.8% / 75.0% / 72.2% | 94.4% / 93.1% / 94.4% | 86.1% / 86.1% / 88.9% |
| 3 次合并 | 157/216 = 72.7% [66.4, 78.2] | 203/216 = 94.0% [90.0, 96.5] | 188/216 = 87.0% [81.9, 90.9] |
| 题级 3 次一致率 | 81.9% | 97.2% | 79.2% |
| 模型调用 / 题 | 1.00 | 3.01 | 3.33 |
| 费用 / 题（估算） | $0.0008 | $0.0065 | $0.0061 |
| 延迟中位数 / p90（run1） | 1.3 s / 2.4 s | 3.6 s / 5.6 s | 3.9 s / 7.3 s |
| 模型原始 SQL 有效率（run1） | 52/61 = 85.2% | 64/75 = 85.3%（失败均在后续修复） | 135/142 = 95.1% |

**按层（3 次合并，括号为 Wilson 95% 区间）**

| 层（题数） | (a) 基线 | (b) Agent | (c) 消融 |
|---|---|---|---|
| tag 原始标签语义（12） | **97.2%** [85.8, 99.5] | 75.0% [58.9, 86.2] | 77.8% [61.9, 88.3] |
| join 多表/别名（12） | 77.8% [61.9, 88.3] | 100% [90.4, 100] | 100% [90.4, 100] |
| crs 投影与度量（10） | 73.3% [55.6, 85.8] | 93.3% [78.7, 98.2] | 90.0% [74.4, 96.5] |
| route 路网（16） | **35.4%** [23.4, 49.6] | **95.8%** [86.0, 98.8] | 91.7% [80.4, 96.7] |
| compliance 标准合规（12） | 69.4% [53.1, 82.0] | **100%** [90.4, 100] | 63.9% [47.6, 77.5] |
| unanswerable 不可回答（10） | 100% | 100% | 100% |

**配对（run1，精确 McNemar）**：基线 vs Agent：仅基线对 3 题（T06、T07、T10），仅 Agent 对 20 题，p = 0.0005。
消融 vs Agent：仅消融对 1 题，仅 Agent 对 7 题（6 道是合规/可达计数），p = 0.070。基线 vs 消融：8 vs 19，p = 0.052。

第二轮全部 API 花费（开发集 + 9 次测试运行）约 **$3.07**；项目累计 $3.38（台账 `eval/runs/ledger.jsonl`，模型字段全部为 `deepseek-flash`，即 `deepseek-chat`）。

**怎么理解**

- 在路网题上，单次 SQL 的失败不是“不会”，而是一次写对一条 60–100 行的 pgRouting 查询很难：最常见的错误是
  `MIN(agg_cost)` 取到路径第一行的 0（12 次）、`round(double precision, int)` 类型错误（6 次）、把 32650 投影几何转 geography（6 次）、超时（3 次）。
  Agent 的 `walk_route/isochrone/reachable` 把这些口径固化在参数化查询里，模型只填起终点。
- 消融臂说明工具化本身就有价值：同样能写 pgRouting SQL、同样有修复循环，只去掉 5 个工具，路网题仍有 91.7%（修复循环救回了大部分 SQL 错误），
  但**合规题从 100% 掉到 63.9%**——13 次失败中 11 次是结论对了但结果里没有条文 id、2 次步数用尽未作答；`check_compliance` 由程序读取条文半径、强制输出 clause_id，正是它补上的部分。
- 原始标签题上 Agent 更差，原因是它**更主动地探索**：用 `tag_values` 看到 `brand=KFC`、`takeaway=only`、`蔚来换电站` 后把它们也算进去，
  口径比金标宽（83 vs 81、165 vs 164、13 vs 12）。按预注册判错；事后敏感性分析（非预注册，`eval/r2/sensitivity.py`）把这三个更宽口径也算对时，
  Agent run1 为 71/72，基线 51/72，结论不变。这暴露的是题面歧义（“品牌标签为肯德基”是否包含英文品牌值），不是 Agent 的推理错误。
- 不可回答题两臂都是 10/10，第二轮没有区分度；开发集与测试集上都没有出现编造数字的回答。

## 第二轮新增的架构

```
问题 ──► LangGraph（同第一轮：model ⇄ tools，校验 + ≤3 次修复 + 数字绑定；第二轮最多 10 次模型调用）
           │
           ├─ 自省：find_entity（名称 + alt_name/old_name/short_name/name:en 别名）· distinct_values · tag_values（原始 OSM 标签）
           ├─ 直线 GIS：第一轮 6 个工具
           ├─ 路网（pgRouting 3.8）：walk_route（pgr_dijkstra）· isochrone（pgr_drivingDistance + 边缓冲）· reachable（N 分钟可达设施）
           ├─ 标准（RAG）：retrieve_standard（bge-small-zh ONNX 本地向量 → pgvector 余弦 top-k）· check_compliance（半径由程序从条文读取）
           └─ run_sql：sqlglot 守卫（pgRouting 函数的“边查询”字符串参数递归过同一白名单）→ 单位静态检查 → 只读事务
           │
           ▼  校验器新增：合规结果引用的 clause_id 必须存在；若含 radius_m，必须等于条文数值
       绑定答案 + GeoJSON（线 / 面 / 点）
```

- **数据库镜像**：`pgrouting/pgrouting:16-3.5-3.8.0`（digest 固定）+ pgvector 0.8.6（deb sha256 固定，`docker/db.Dockerfile`）。
  选它是因为 PostgreSQL 16.9 / PostGIS 3.5.2 与第一轮完全相同，可直接复用数据卷。
- **步行路网**（`sql/build_walk_network.sql`）：从同一 OSM 快照的线要素构建，纳入 primary…residential、service、pedestrian、footway、path、steps、track、cycleway；
  trunk 只在 `foot=yes|designated|permissive` 时纳入，motorway 一律排除；排除 `foot=no` 与未允许步行的 `access=private|no`。只在 OSM 共享节点与端点处打断（桥隧不与下方道路相交）。
  结果：**118,304 条边、87,970 个顶点、21,204 km；1,088 个连通分量，最大分量占顶点 94.4%、长度 97.3%**。全部路由在最大分量上进行。
- **口径**：步速 4.8 km/h = 80 m/min；要素代表点（面取 `ST_PointOnSurface`，同名取 id 最小）吸附到最大分量的 KNN 最近顶点，吸附段不计入；
  等时圈 = 两端可达的边在 EPSG:32650 上 2 m 简化、50 m 缓冲后求并（比先并后缓冲快约 16 倍，面积差 < 1%），是“边缓冲”而不是凹包。
- **条文语料**（`data/standards/`）：住建部官网公开的 GB 50180-2018 全文 PDF（与教育部转载附件逐字节一致，sha256 `78c0b54c…`）是扫描图像、无文字层，
  因此对照页面图像**人工转录**了 31 条摘录（术语 2.0.2–2.0.5、表 3.0.4、5.0.1、附录 B/C 中与 OSM 可映射设施有关的服务半径与配建要求），每条带页码与来源 URL。
  只摘录所需条文、注明出处，不是全文复制；版权与“宜/不宜”属于推荐性要求的说明见 `data/standards/SOURCE.md`。
- **向量模型**：`BAAI/bge-small-zh-v1.5` 的 ONNX 导出（Hugging Face `Xenova/bge-small-zh-v1.5` 修订 `75c43b06`，文件 sha256 固定），
  onnxruntime + tokenizers 在 CPU 上运行，不依赖 torch、不走 API；CLS 池化 + L2 归一化，查询侧加官方检索指令。31 条语料上，测试里的典型查询 top-1 命中目标条文。
- **精简 schema**（`src/geo_agent/r2_schema.py`）：两臂看到同一份——表/列名、类型、行数、几何类型与 SRID、简短英文字段含义；
  **去掉**第一轮那样的“中文概念 = OSM 标签值”映射和写法提示。另给两臂同一份函数说明（pgRouting 签名、边查询写法、吸附方式、pgvector 运算符）和同一份项目口径，
  也就是一个称职的提示工程师会写给模型的内容；基线缺的只是工具、检索与修复，不是信息。
- **演示**：`/?demo=小区名` 不调用模型，直接渲染等时圈、圈内中学与合规判定（上图）；`/?q=...&agent=round2` 用第二轮 Agent 作答。

## 第二轮评测设计（预注册）

完整内容见 `eval/r2/PREREGISTRATION.md`。题目、金标 SQL、金标结果与判分代码在提交 `08309f2` 中冻结（questions sha256 `857a5193…`），
早于第二轮提示词与 Agent 代码（`688e614`）和任何第二轮模型调用。

- **72 题 6 层**：原始标签 12、多表/别名 12、投影与度量 10、路网 16（最短路 8、等时圈 2、可达设施 6）、标准合规 12（6 符合 / 6 不符合）、不可回答 10。
  题面不提示表名、列名、标签键或函数名；路网题写明步速、合规题不给半径数值。
- **金标**由作者的 SQL 计算（`eval/r2/gold_sql.py` 与 `routing.py` 同一口径，集成测试检查两者对同一输入数值一致）。
- **冻结前检查**：直线阈值题候选与阈值相差 ≥ 0.5%；路网/合规题候选设施网络距离与阈值相差 ≥ 1%；top-k 相邻名次差 ≥ 0.5%；结果非空、计数非 0。
  检查改动了 5 道题（J05 半径 100→120 m；D12/D13/D16 原选点结果为 0；J08 去掉品牌为麦当劳的“自助点餐机”）。
- **判分**（`src/geo_agent/matcher_r2.py`）：沿用第一轮结果集比对；路网层数值容差 2%（其余 0.5%）；合规题要求布尔结论与条文 id 都对，若给出 radius_m 必须等于条文值。
- 开发集 12 题（`eval/r2/dev_questions.json`，实体与测试集不重叠）只跑过一次：基线 8/12、Agent 12/12、消融 10/12。

## 第二轮失败案例

逐题记录在 `eval/runs/r2run{1,2,3}/r2_per_question.json`，原始轨迹在同目录 `*.jsonl`。

**Agent（run1 错 4 题；3 次共错 13 题次）**

- T06 / T07 / T10（3 次全错）：口径比金标宽，见上文“怎么理解”。
- D15 “明都雅苑步行 15 分钟内最近的地铁站”（run1 错，run2、run3 对）：`reachable` 给出“市中医院 265 m”，模型怀疑结果，改写 SQL 用 60/150 m 的错误吸附半径自行计算，
  最后引用了“应天东街 999 m”。工具给了对的答案，模型没有信它。
- C03 “Web 墨卡托面积与放大倍数”（run1 对；run2、run3 拒答）：单位校验器把 `ST_Transform(geom, 3857)` 上的面积量算判为错误，模型据此拒答。
  这是**校验器与题意冲突**：题目本身就是要演示 3857 的放大误差。校验器把 3857 一律视为错误的设计在第一轮是合理的，这里过于严格。

**基线（run1 错 21 题）**：路网 11、合规 4、多表/别名 3、投影 3。除上面的 SQL 错误外，典型例子：
J01 “南京炮院”看到 schema 里有 poi_tags 却认为“没有别名标签”而拒答；J08 把品牌为麦当劳的“自助点餐机”算进最近 3 家；
C07 对多边形用 `ST_Length` 求周长得到 0；E08 合规判定把吸附写成“任意 150 m 内的顶点”导致结论错。

**消融（run1 错 10 题）**：合规 4（E03、E06 结论对但缺条文 id，E02、E11 步数用尽未作答）、可达计数 2（D12 得 9 vs 金标 8、D13 得 1 vs 金标 4，自写 SQL 的设施吸附方式与口径不一致）、
C03 用纬度余弦近似放大倍数（1.393 vs 精确 1.397）、T06/T07/T10。

## 第二轮预注册之后的改动（披露）

1. **单位静态检查补漏**（开发集阶段，测试集运行前）：`ST_Buffer(ST_Simplify(ST_Transform(g, 32650), 2), 50)` 被误判为“度坐标上缓冲”，
   因为 `st_simplify` 不在“保持单位”的函数列表里；补上 `st_simplify / st_simplifypreservetopology / st_snaptogrid / st_linemerge`。
   第一轮两臂 100 条最终 SQL 重新判分，正确与单位错误判定均不变。
2. **事后敏感性分析**（`eval/r2/sensitivity.py` → `eval/runs/r2_sensitivity.json`）：只针对 T06/T07/T10 的更宽口径，主结果不变。
3. 演示 API 新增 `agent=round2` 参数与 `/demo/life-circle`（只影响演示，不影响评测）；第一轮默认行为不变。

## 第二轮局限

- **题目、金标、口径都由作者一人决定**。“中学”按名称含“中学”识别（OSM 不区分学段，名称不含“中学”的初中会漏掉）；“小区”= OSM 有名称的 landuse=residential 面；
  设施只按 OSM 标签。这些都是可辩论的口径，失败案例逐题列出。
- “合规”只是按标准**推荐性**服务半径（“不宜大于”）做的空间核查，不是规划审批意义上的合规结论；条文为人工转录，以官方 PDF 为准。
- OSM 步行网络在中国不完整：缺人行道、过街设施和小区内部道路；部分小区代表点到路网吸附超过 100 m，部分干道因 `foot=no` 不可步行，导致网络距离偏长（如新街口→大行宫 2,031 m，直线 1,014 m）。
  结论只针对该快照与该路网，不代表真实步行时间。
- 72 题仍是小样本（单臂区间约 ±10 个百分点）；单一模型、单一城市；temperature 0 仍有逐次差异（基线题级一致率 82%），所以报告了 3 次。
- 第二轮 Agent 的提示词与工具在 12 道开发题上只运行过一次、没有据此调参，但作者在设计工具时知道题型分布（工具口径与金标口径一致是有意为之，基线也拿到了同样的口径文字）。
  这意味着结果衡量的是“把领域口径固化成工具”的价值，而不是 Agent 在陌生任务上的泛化能力。
- 不可回答题两臂都满分，第二轮没有测出拒答能力的差异。

## 下一步：本地 8B 模型（已准备，未运行）

`eval/local_llm/run_vllm_arms.py` 用同一冻结题集、同一提示词与工具评测 OpenAI 兼容端点（vLLM + Qwen3-8B），带 `--dry-run` 假端点自检（已通过）。
GPU 会话只需提供 `--base-url`，用法见 `eval/local_llm/README.md`。

## 第二轮快速开始

```bash
docker compose build db && docker compose up -d --wait db      # 第二轮镜像（复用第一轮数据卷）
bash scripts/load_data.sh                                       # 第一轮表 + 第二轮表、步行路网、条文向量（下载约 95 MB 向量模型）
uv run python eval/r2/freeze_gold.py --check                    # 复现 72 题金标（逐题比对冻结哈希）
uv run python eval/r2/run_eval.py --arm agent --set test --run myrun   # 需要 DEEPSEEK_API_KEY；每臂每次约 $0.06–0.48
uv run python eval/r2/score.py --runs myrun
uv run uvicorn geo_agent.api:app --host 127.0.0.1 --port 8765   # 打开 /?demo=南湾营融康苑
```

## 目录

```
docker-compose.yml        数据库服务（第二轮：docker/db.Dockerfile = pgRouting 镜像 + pgvector，均固定版本）
db/init/01_roles.sql      只读角色
scripts/                  fetch_data.sh（固定快照 + sha256）· load_data.sh · fetch_model.sh（向量模型 + sha256）· load_standards.py
sql/                      build_tables.sql（第一轮）· build_r2_tables.sql · build_walk_network.sql（第二轮）
data/standards/           GB 50180-2018 条文摘录 clauses.json 与来源/版权说明 SOURCE.md
src/geo_agent/            第一轮：sql_guard · units · tools · agent · baseline · answer · matcher · stats · llm · api · static/
                          第二轮：routing · r2_tools · embed · r2_schema · r2_prompts · r2_baseline · r2_agent · matcher_r2
eval/                     第一轮：PREREGISTRATION.md · questions.json · gold.json · freeze_gold.py · run_eval.py · score.py · sensitivity.py
eval/r2/                  第二轮：PREREGISTRATION.md · questions.json · gold.json · gold_sql.py · build_questions.py · freeze_gold.py · run_eval.py · score.py · sensitivity.py
eval/local_llm/           本地 vLLM（Qwen3-8B）评测脚本与说明（已准备，未运行）
eval/runs/                test1（第一轮）· r2dev1 · r2run1..3 · r2_summary_test.json · ledger.jsonl
tests/                    123 个 pytest；无数据库时跳过 25 个集成测试，不需要 API key、GPU 或向量模型
```

## 许可

代码：MIT（见 `LICENSE`）。地图数据：© OpenStreetMap contributors，ODbL 1.0。底图瓦片：OpenStreetMap 标准瓦片（需署名，遵守其使用政策）。
GB 50180-2018 条文摘录来自住房和城乡建设部官网公开文本，仅作引用与研究用途，见 `data/standards/SOURCE.md`。
