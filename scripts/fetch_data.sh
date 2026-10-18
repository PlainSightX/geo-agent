#!/usr/bin/env bash
# 下载固定日期的 Geofabrik 江苏省 OSM 快照，并用 data/SOURCE.json 中冻结的 sha256 校验。
# 使用带日期的文件名（而不是 -latest），保证任何人重跑得到的是同一份原始数据。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
SNAPSHOT="jiangsu-261017.osm.pbf"
URL="https://download.geofabrik.de/asia/china/$SNAPSHOT"
EXPECTED_SHA256="dcba9384eb5919ae3bf4aeb209fac913b0754133a5f5d4e2acd2b4d5f1b2652e"

mkdir -p "$RAW"
if [ ! -f "$RAW/$SNAPSHOT" ]; then
  echo "downloading $URL"
  curl -fL --retry 3 -o "$RAW/$SNAPSHOT.part" "$URL"
  mv "$RAW/$SNAPSHOT.part" "$RAW/$SNAPSHOT"
fi

ACTUAL="$(sha256sum "$RAW/$SNAPSHOT" | awk '{print $1}')"
if [ "$ACTUAL" != "$EXPECTED_SHA256" ]; then
  echo "sha256 mismatch: expected $EXPECTED_SHA256 got $ACTUAL" >&2
  echo "Geofabrik 只保留有限天数的日期快照；若已下线，需要换新快照并重新冻结评测金标。" >&2
  exit 1
fi
echo "ok $SNAPSHOT sha256=$ACTUAL"
