-- 第二轮新增业务表（不修改第一轮的 7 张表及其注释，第一轮金标因此可逐题复现）：
--   poi_tags          POI 的原始 OSM 标签（键值对，一行一个标签），用于“原始标签语义”题与别名查找
--   residential_areas 有名称的居住用地（OSM landuse=residential），即题目中的“小区”
--   bus_stops         公交站点（OSM highway=bus_stop，未合并，同名站常有上下行两处）
--   standard_clauses  规划标准条文摘录（pgvector 向量列由 scripts/load_standards.py 写入）
-- 依赖：sql/build_tables.sql 已运行（osm_raw.city、pois 存在）。
\set ON_ERROR_STOP on
CREATE EXTENSION IF NOT EXISTS hstore;
CREATE EXTENSION IF NOT EXISTS pgrouting;
CREATE EXTENSION IF NOT EXISTS vector;
BEGIN;

DROP TABLE IF EXISTS poi_tags, residential_areas, bus_stops CASCADE;

-- 1) poi_tags：pois.id 与原始要素一一对应（osm_type + osm_id），展开 other_tags 并补上已拆成列的标签
CREATE TABLE poi_tags AS
WITH raw AS (
  SELECT 'node' AS osm_type, osm_id::bigint AS osm_id, other_tags::hstore AS h, name_en
  FROM osm_raw.points WHERE amenity IS NOT NULL OR shop IS NOT NULL OR tourism IS NOT NULL
  UNION ALL
  SELECT CASE WHEN osm_id IS NOT NULL THEN 'relation' ELSE 'way' END,
         COALESCE(osm_id, osm_way_id)::bigint, other_tags::hstore, name_en
  FROM osm_raw.multipolygons WHERE amenity IS NOT NULL OR shop IS NOT NULL OR tourism IS NOT NULL
), kv AS (
  SELECT p.id AS poi_id, (each(r.h)).key AS key, (each(r.h)).value AS value
  FROM pois p JOIN raw r ON r.osm_type = p.osm_type AND r.osm_id = p.osm_id
  WHERE r.h IS NOT NULL
  UNION ALL
  SELECT p.id, 'name:en', r.name_en
  FROM pois p JOIN raw r ON r.osm_type = p.osm_type AND r.osm_id = p.osm_id
  WHERE r.name_en IS NOT NULL
)
SELECT DISTINCT poi_id, key, value FROM kv;
ALTER TABLE poi_tags ADD PRIMARY KEY (poi_id, key);
CREATE INDEX ON poi_tags (key, value);

-- 2) residential_areas：有名称的居住用地面，修复几何后按市界裁剪
CREATE TABLE residential_areas AS
WITH v AS (
  SELECT COALESCE(osm_id, osm_way_id)::bigint AS osm_id, name,
         ST_CollectionExtract(ST_MakeValid(geom), 3) AS geom
  FROM osm_raw.multipolygons WHERE landuse = 'residential' AND name IS NOT NULL
)
SELECT row_number() OVER (ORDER BY v.osm_id)::int AS id, v.osm_id, v.name,
       ST_Multi(ST_CollectionExtract(
         CASE WHEN ST_Within(v.geom, c.geom) THEN v.geom ELSE ST_Intersection(v.geom, c.geom) END, 3
       ))::geometry(MultiPolygon, 4326) AS geom
FROM v, osm_raw.city c
WHERE NOT ST_IsEmpty(v.geom) AND ST_Intersects(v.geom, c.geom);
DELETE FROM residential_areas WHERE ST_IsEmpty(geom);
ALTER TABLE residential_areas ADD PRIMARY KEY (id);
CREATE INDEX ON residential_areas USING gist (geom);
CREATE INDEX ON residential_areas (name);

-- 3) bus_stops：公交站点（点要素），不合并同名站
CREATE TABLE bus_stops AS
SELECT row_number() OVER (ORDER BY p.osm_id::bigint)::int AS id, p.osm_id::bigint AS osm_id, p.name,
       ST_GeometryN(p.geom, 1)::geometry(Point, 4326) AS geom
FROM osm_raw.points p, osm_raw.city c
WHERE p.highway = 'bus_stop' AND ST_Intersects(p.geom, c.geom);
ALTER TABLE bus_stops ADD PRIMARY KEY (id);
CREATE INDEX ON bus_stops USING gist (geom);
CREATE INDEX ON bus_stops (name);

-- 4) standard_clauses：结构化条文摘录；embedding 维度 512 对应 BAAI/bge-small-zh-v1.5
CREATE TABLE IF NOT EXISTS standard_clauses (
  clause_id text PRIMARY KEY,
  standard text NOT NULL,
  section text NOT NULL,
  title text NOT NULL,
  body text NOT NULL,
  facility text,
  circle_min int,
  max_radius_m int,
  walk_m_min int,
  walk_m_max int,
  strength text,
  page int,
  source_url text NOT NULL,
  embedding vector(512)
);

GRANT SELECT ON poi_tags, residential_areas, bus_stops, standard_clauses TO geo_reader;
COMMIT;
ANALYZE poi_tags;
ANALYZE residential_areas;
ANALYZE bus_stops;
SELECT 'poi_tags' AS t, count(*) FROM poi_tags UNION ALL
SELECT 'residential_areas', count(*) FROM residential_areas UNION ALL
SELECT 'bus_stops', count(*) FROM bus_stops;
