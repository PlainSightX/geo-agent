#!/usr/bin/env bash
# 把江苏 OSM 快照按南京市范围裁剪后导入 PostGIS：
#   1) GDAL 容器用 -spat 按南京市外包框读取 PBF，写入 staging schema osm_raw；
#   2) sql/build_tables.sql 用南京市行政边界多边形精确裁剪，生成 public 下的业务表；
#   3) 只读角色 geo_reader 只获得 public 业务表的 SELECT 权限。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SNAPSHOT="jiangsu-261017.osm.pbf"
GDAL_IMAGE="ghcr.io/osgeo/gdal:ubuntu-small-3.10.3@sha256:dab45abca3ca83695d442018692f4f8a0f41955871c57e6101d7f89a92375caa"
# 南京市（OSM relation 2131524）外包框，来自 Nominatim，略向外扩 0.01 度
BBOX="118.32 31.21 119.25 32.63"

bash "$ROOT/scripts/fetch_data.sh"

# Git Bash 下：宿主路径用 pwd -W 转成 D:/... 形式交给 docker，
# 同时禁止 MSYS 把 /data 之类的容器内路径改写成 Windows 路径
if pwd -W >/dev/null 2>&1; then
  ROOT="$(cd "$ROOT" && pwd -W)"
  export MSYS_NO_PATHCONV=1
fi
docker compose -f "$ROOT/docker-compose.yml" up -d --wait db

NET="$(docker inspect geo-agent-db --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}')"
PG="PG:host=db port=5432 dbname=geo user=geo_admin password=geo_admin_local_only"

docker exec geo-agent-db psql -U geo_admin -d geo -v ON_ERROR_STOP=1 \
  -c "DROP SCHEMA IF EXISTS osm_raw CASCADE; CREATE SCHEMA osm_raw;"

# 一次 ogr2ogr 调用读取多个图层：OSM 驱动要求交错读取，ogr2ogr 会自动处理
docker run --rm --network "$NET" \
  -v "$ROOT/data/raw:/data:ro" -v "$ROOT/docker:/conf:ro" \
  -e OSM_CONFIG_FILE=/conf/osmconf.ini -e OSM_MAX_TMPFILE_SIZE=1024 \
  "$GDAL_IMAGE" \
  ogr2ogr -f PostgreSQL "$PG" "/data/$SNAPSHOT" points lines multipolygons \
    -spat $BBOX -lco SCHEMA=osm_raw -lco GEOMETRY_NAME=geom -lco SPATIAL_INDEX=GIST \
    -nlt PROMOTE_TO_MULTI -overwrite --config PG_USE_COPY YES -progress

docker exec -i geo-agent-db psql -U geo_admin -d geo -v ON_ERROR_STOP=1 < "$ROOT/sql/build_tables.sql"
docker exec geo-agent-db psql -U geo_admin -d geo -c "SELECT * FROM dataset_summary ORDER BY table_name;"

# ---- 第二轮：原始标签/小区/公交站/条文表、步行路网、条文向量 ----
docker exec -i geo-agent-db psql -U geo_admin -d geo -v ON_ERROR_STOP=1 < "$ROOT/sql/build_r2_tables.sql"
docker exec -i geo-agent-db psql -U geo_admin -d geo -v ON_ERROR_STOP=1 < "$ROOT/sql/build_walk_network.sql"
bash "$ROOT/scripts/fetch_model.sh"
(cd "$ROOT" && uv run python scripts/load_standards.py)
