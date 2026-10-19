# MCP 客户端会话演示

目的：确认 `geo-agent-mcp` 能被标准 MCP 客户端以 stdio 子进程方式启动，并通过协议调用已有的 `isochrone` 工具。
环境：本地第二轮数据库（`docker compose`，只读角色 `geo_reader`），`mcp==2.3.0`（官方 Python SDK）。

## 1. SDK stdio 客户端（`scripts/mcp_demo.py`）

```bash
uv run python scripts/mcp_demo.py 南湾营融康苑 12.5
```

脚本用 `mcp.Client(StdioServerParameters(command="uv", args=["run", "--quiet", "geo-agent-mcp"]))` 启动服务器，
列出工具后调用：

```json
{"name": "isochrone",
 "arguments": {"origin": {"layer": "residential_areas", "name": "南湾营融康苑"}, "minutes": 12.5}}
```

结果（`structuredContent`）：

| minutes | network_m | reachable_nodes | area_km2 | snap_m |
|---|---|---|---|---|
| 12.5 | 1000.0 | 114 | 0.7118 | 8.85 |

- 服务器名 `geo-agent`，工具 15 个；`truncated = false`，1 行；
- GeoJSON：1 个 `Polygon`，FeatureCollection 5,643 字节；整个 `structuredContent` 6,871 字节；
- 调用往返 0.49 s（不含子进程启动）；
- 与进程内直接调用 `routing.isochrone` 的结果逐值相同（面积 0.7118420371226929 km²），即 MCP 层没有改变工具口径。

## 2. MCP Inspector CLI（`@modelcontextprotocol/inspector` 2.11.0）

```bash
npx -y @modelcontextprotocol/inspector --cli --strict uv run --directory <repo> geo-agent-mcp --method tools/list
npx -y @modelcontextprotocol/inspector --cli uv run --directory <repo> geo-agent-mcp \
  --method tools/call --tool-name isochrone \
  --tool-arg 'origin={"layer":"residential_areas","name":"南湾营融康苑"}' --tool-arg minutes=12.5
```

- `tools/list --strict`：退出码 0，schema 可移植性检查 0 错误、0 警告，15 个工具（下表），全部带 `readOnlyHint: true`；
- `tools/call isochrone`：`isError: false`，`structuredContent` 与上面 SDK 客户端结果一致；完整响应 28,627 字节
  （文本内容块与结构化内容各一份同样的数据）；
- `resources/list`：`geo-agent://schema/lean`、`geo-agent://standards/clauses`。

| 工具 | 必填参数 |
|---|---|
| find_entity | name |
| distinct_values | layer |
| tag_values | （无） |
| within_distance | anchor, target, radius_m |
| nearest_k | anchor, target, k |
| count_by_district | target |
| measure | layer, name, metric |
| spatial_relation | left_layer, relation, right |
| distance_between | a, b |
| walk_route | a, b |
| isochrone | origin, minutes |
| reachable | origin, target, minutes |
| retrieve_standard | query |
| check_compliance | residential, clause_id, target |
| run_sql | sql |
