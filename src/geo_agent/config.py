"""运行配置：全部可由环境变量覆盖，默认值对应 docker-compose.yml 中的本地服务。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# 业务表白名单：SQL 守卫与 schema 自省共用这一份，避免两处定义漂移
BUSINESS_TABLES: tuple[str, ...] = (
    "districts",
    "pois",
    "roads",
    "waterways",
    "water_areas",
    "parks",
    "metro_stations",
)


@dataclass(frozen=True)
class Settings:
    # 只读角色连接串：Agent、基线与评测都只用 geo_reader，不使用管理员账号
    dsn: str = field(
        default_factory=lambda: os.environ.get(
            "GEO_AGENT_DSN",
            "postgresql://geo_reader:geo_reader_local_only@127.0.0.1:55442/geo",
        )
    )
    # 只允许 deepseek-chat（Flash）；模型名写死在评测预注册里，改动需同步更新预注册
    model: str = field(default_factory=lambda: os.environ.get("GEO_AGENT_MODEL", "deepseek-chat"))
    base_url: str = field(
        default_factory=lambda: os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    statement_timeout_ms: int = 5000
    row_limit: int = 200
    # 价格按任务约定的保守口径估算（美元/百万 token），不区分缓存命中
    price_in_per_m: float = 0.30
    price_out_per_m: float = 1.20
    # Agent 循环上限：最多 3 次修复、8 次工具调用
    max_repairs: int = 3
    max_tool_steps: int = 8


def get_settings() -> Settings:
    return Settings()
