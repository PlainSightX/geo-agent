"""MCP 客户端演示：用官方 SDK 的 stdio 客户端以子进程方式启动 geo-agent-mcp，列出工具并调用 isochrone。

用法（仓库根目录，需要本地数据库）：
    uv run python scripts/mcp_demo.py [小区名] [分钟]

输出一段 JSON 摘要（工具数、调用参数、结果列与行、GeoJSON 几何类型与字节数），docs/mcp-demo.md 记录了一次运行结果。
"""

from __future__ import annotations

import json
import sys
import time

import anyio
from mcp import Client, StdioServerParameters


async def main(residential: str, minutes: float) -> None:
    # 与 Claude Desktop / Cursor 的配置相同：由客户端以子进程启动服务器，通过 stdin/stdout 通信
    params = StdioServerParameters(command="uv", args=["run", "--quiet", "geo-agent-mcp"])
    async with Client(params) as client:
        server = client.server_info.name if client.server_info else None
        tools = await client.list_tools()
        args = {"origin": {"layer": "residential_areas", "name": residential}, "minutes": minutes}
        t0 = time.perf_counter()
        res = await client.call_tool("isochrone", args)
        elapsed = time.perf_counter() - t0
    if res.is_error:
        raise SystemExit(res.content[0].text)
    s = res.structured_content
    geom = s["geojson"]["features"][0]["geometry"]
    summary = {
        "server": server,
        "tools": len(tools.tools),
        "call": {"name": "isochrone", "arguments": args},
        "columns": s["columns"],
        "rows": s["rows"],
        "truncated": s["truncated"],
        "geometry_type": geom["type"],
        "polygon_parts": len(geom["coordinates"]) if geom["type"] == "MultiPolygon" else 1,
        "geojson_bytes": len(json.dumps(s["geojson"], ensure_ascii=False).encode("utf-8")),
        "structured_content_bytes": len(json.dumps(s, ensure_ascii=False).encode("utf-8")),
        "round_trip_s": round(elapsed, 2),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "南湾营融康苑"
    mins = float(sys.argv[2]) if len(sys.argv) > 2 else 12.5
    anyio.run(main, name, mins)
