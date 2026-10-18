-- 从 staging schema osm_raw 构建 public 业务表。
-- 约定：所有 geom 均为 EPSG:4326（经纬度，单位为度）。
-- 业务表刻意不预存“所属行政区”字段，按区统计必须做空间连接，这正是评测要考察的能力。
\set ON_ERROR_STOP on
BEGIN;

DROP VIEW IF EXISTS dataset_summary;
DROP TABLE IF EXISTS districts, pois, roads, waterways, water_areas, parks, metro_stations CASCADE;

-- 南京市边界（OSM relation 2131524），只用于裁剪，不对外暴露
DROP TABLE IF EXISTS osm_raw.city;
CREATE TABLE osm_raw.city AS
SELECT ST_MakeValid(geom) AS geom FROM osm_raw.multipolygons WHERE osm_id = '2131524';

-- 1) 行政区：11 个市辖区（admin_level=6）
CREATE TABLE districts AS
SELECT row_number() OVER (ORDER BY osm_id::bigint)::int AS id,
       osm_id::bigint AS osm_id, name, name_en,
       ST_Multi(ST_MakeValid(geom))::geometry(MultiPolygon, 4326) AS geom
FROM osm_raw.multipolygons
WHERE boundary = 'administrative' AND admin_level = '6'
  AND ST_Within(ST_PointOnSurface(geom), (SELECT geom FROM osm_raw.city));

-- 2) POI：点要素 + 面要素（取面内点），类别取 amenity > shop > tourism 的第一个非空值
CREATE TABLE pois AS
WITH src AS (
  SELECT 'node' AS osm_type, osm_id::bigint AS osm_id, name, amenity, shop, tourism, cuisine,
         ST_GeometryN(geom, 1) AS geom
  FROM osm_raw.points
  WHERE amenity IS NOT NULL OR shop IS NOT NULL OR tourism IS NOT NULL
  UNION ALL
  SELECT CASE WHEN osm_id IS NOT NULL THEN 'relation' ELSE 'way' END,
         COALESCE(osm_id, osm_way_id)::bigint, name, amenity, shop, tourism, NULL,
         ST_PointOnSurface(ST_MakeValid(geom))
  FROM osm_raw.multipolygons
  WHERE amenity IS NOT NULL OR shop IS NOT NULL OR tourism IS NOT NULL
)
SELECT row_number() OVER (ORDER BY osm_type, osm_id)::int AS id, osm_type, osm_id, name,
       CASE WHEN amenity IS NOT NULL THEN 'amenity' WHEN shop IS NOT NULL THEN 'shop' ELSE 'tourism' END AS category,
       COALESCE(amenity, shop, tourism) AS subtype,
       cuisine,
       geom::geometry(Point, 4326) AS geom
FROM src
WHERE GeometryType(geom) = 'POINT'
  AND ST_Intersects(geom, (SELECT geom FROM osm_raw.city));

-- 3) 道路：机动车可通行等级，按市界裁剪
CREATE TABLE roads AS
SELECT row_number() OVER (ORDER BY osm_id::bigint)::int AS id, osm_id::bigint AS osm_id,
       name, highway, ref, oneway,
       ST_Multi(ST_CollectionExtract(
         CASE WHEN ST_Within(l.geom, c.geom) THEN l.geom ELSE ST_Intersection(l.geom, c.geom) END, 2
       ))::geometry(MultiLineString, 4326) AS geom
FROM osm_raw.lines l, osm_raw.city c
WHERE highway IN ('motorway','motorway_link','trunk','trunk_link','primary','primary_link',
                  'secondary','secondary_link','tertiary','tertiary_link',
                  'unclassified','residential','living_street')
  AND ST_Intersects(l.geom, c.geom);
DELETE FROM roads WHERE ST_IsEmpty(geom);

-- 4) 线状水系：河流/运河/溪流，按市界裁剪
CREATE TABLE waterways AS
SELECT row_number() OVER (ORDER BY osm_id::bigint)::int AS id, osm_id::bigint AS osm_id,
       name, waterway,
       ST_Multi(ST_CollectionExtract(
         CASE WHEN ST_Within(l.geom, c.geom) THEN l.geom ELSE ST_Intersection(l.geom, c.geom) END, 2
       ))::geometry(MultiLineString, 4326) AS geom
FROM osm_raw.lines l, osm_raw.city c
WHERE waterway IN ('river','canal','stream')
  AND ST_Intersects(l.geom, c.geom);
DELETE FROM waterways WHERE ST_IsEmpty(geom);

-- 5) 面状水体：natural=water（湖泊、水库、河面等），修复无效几何后按市界裁剪
CREATE TABLE water_areas AS
WITH v AS (
  SELECT COALESCE(osm_id, osm_way_id)::bigint AS osm_id, name, water,
         ST_CollectionExtract(ST_MakeValid(geom), 3) AS geom
  FROM osm_raw.multipolygons WHERE "natural" = 'water'
)
SELECT row_number() OVER (ORDER BY v.osm_id)::int AS id, v.osm_id, v.name, v.water AS water_type,
       ST_Multi(ST_CollectionExtract(
         CASE WHEN ST_Within(v.geom, c.geom) THEN v.geom ELSE ST_Intersection(v.geom, c.geom) END, 3
       ))::geometry(MultiPolygon, 4326) AS geom
FROM v, osm_raw.city c
WHERE NOT ST_IsEmpty(v.geom) AND ST_Intersects(v.geom, c.geom);
DELETE FROM water_areas WHERE ST_IsEmpty(geom);

-- 6) 公园：leisure=park
CREATE TABLE parks AS
WITH v AS (
  SELECT COALESCE(osm_id, osm_way_id)::bigint AS osm_id, name,
         ST_CollectionExtract(ST_MakeValid(geom), 3) AS geom
  FROM osm_raw.multipolygons WHERE leisure = 'park'
)
SELECT row_number() OVER (ORDER BY v.osm_id)::int AS id, v.osm_id, v.name,
       ST_Multi(ST_CollectionExtract(
         CASE WHEN ST_Within(v.geom, c.geom) THEN v.geom ELSE ST_Intersection(v.geom, c.geom) END, 3
       ))::geometry(MultiPolygon, 4326) AS geom
FROM v, osm_raw.city c
WHERE NOT ST_IsEmpty(v.geom) AND ST_Intersects(v.geom, c.geom);
DELETE FROM parks WHERE ST_IsEmpty(geom);

-- 7) 地铁站：OSM 换乘站常按线路各有一个节点，这里按站名合并为一行，几何取各节点的质心
CREATE TABLE metro_stations AS
SELECT row_number() OVER (ORDER BY name)::int AS id, name,
       CASE WHEN bool_or(station = 'subway') THEN 'subway' ELSE 'light_rail' END AS station_type,
       count(*)::int AS osm_node_count,
       ST_Centroid(ST_Collect(ST_GeometryN(geom, 1)))::geometry(Point, 4326) AS geom
FROM osm_raw.points
WHERE railway = 'station' AND station IN ('subway', 'light_rail') AND name IS NOT NULL
  AND ST_Intersects(geom, (SELECT geom FROM osm_raw.city))
GROUP BY name;

-- 主键与空间索引
ALTER TABLE districts ADD PRIMARY KEY (id);
ALTER TABLE pois ADD PRIMARY KEY (id);
ALTER TABLE roads ADD PRIMARY KEY (id);
ALTER TABLE waterways ADD PRIMARY KEY (id);
ALTER TABLE water_areas ADD PRIMARY KEY (id);
ALTER TABLE parks ADD PRIMARY KEY (id);
ALTER TABLE metro_stations ADD PRIMARY KEY (id);
CREATE INDEX ON districts USING gist (geom);
CREATE INDEX ON pois USING gist (geom);
CREATE INDEX ON pois (subtype);
CREATE INDEX ON roads USING gist (geom);
CREATE INDEX ON waterways USING gist (geom);
CREATE INDEX ON water_areas USING gist (geom);
CREATE INDEX ON parks USING gist (geom);
CREATE INDEX ON metro_stations USING gist (geom);

-- 表/列注释：schema 自省工具会把它们原样提供给模型（两个评测臂看到的 schema 文本相同）
COMMENT ON TABLE districts IS '南京市 11 个市辖区边界（OSM admin_level=6）';
COMMENT ON TABLE pois IS '兴趣点：OSM amenity/shop/tourism 要素，面要素已转为面内点；无所属区字段，需空间连接 districts';
COMMENT ON COLUMN pois.category IS '一级类别：amenity | shop | tourism';
COMMENT ON COLUMN pois.subtype IS 'OSM 标签值（英文小写）。常用：hospital=医院, clinic=诊所, pharmacy=药店, school=中小学, kindergarten=幼儿园, university=大学, college=学院, library=图书馆, bank=银行, atm=取款机, restaurant=餐厅, fast_food=快餐, cafe=咖啡馆, bar=酒吧, fuel=加油站, charging_station=充电站, parking=停车场, police=派出所/警察, post_office=邮局, cinema=电影院, theatre=剧院, toilets=厕所, supermarket=超市, convenience=便利店, mall=商场, marketplace=市场, hotel=酒店, hostel=青年旅舍, museum=博物馆, attraction=景点, viewpoint=观景点';
COMMENT ON TABLE roads IS '机动车道路（motorway..residential），已按南京市界裁剪';
COMMENT ON COLUMN roads.highway IS 'OSM highway 等级：motorway, trunk, primary, secondary, tertiary, unclassified, residential, living_street 及 *_link';
COMMENT ON TABLE waterways IS '线状水系：river | canal | stream，已按市界裁剪';
COMMENT ON TABLE water_areas IS '面状水体（OSM natural=water），water_type 如 lake, reservoir, river, pond';
COMMENT ON TABLE parks IS '公园面（OSM leisure=park）';
COMMENT ON TABLE metro_stations IS '地铁/轻轨站，按站名合并（换乘站一行）；station_type: subway | light_rail；name 通常不带“站”字（如 新街口），个别站本名含“站”（如 南京站、南京南站）';
COMMENT ON COLUMN pois.cuisine IS 'OSM cuisine 标签，如 chinese, coffee_shop, chicken, burger, noodle；大多为空';
COMMENT ON COLUMN districts.geom IS 'EPSG:4326 经纬度（度）。距离/长度/面积须用 ::geography（米）或 ST_Transform(geom, 32650)（UTM 50N，米）';

CREATE VIEW dataset_summary AS
SELECT 'districts' AS table_name, count(*) AS n FROM districts UNION ALL
SELECT 'pois', count(*) FROM pois UNION ALL
SELECT 'roads', count(*) FROM roads UNION ALL
SELECT 'waterways', count(*) FROM waterways UNION ALL
SELECT 'water_areas', count(*) FROM water_areas UNION ALL
SELECT 'parks', count(*) FROM parks UNION ALL
SELECT 'metro_stations', count(*) FROM metro_stations;

-- 只读角色：只能读 public 业务表，看不到 osm_raw
REVOKE ALL ON SCHEMA osm_raw FROM PUBLIC;
GRANT SELECT ON districts, pois, roads, waterways, water_areas, parks, metro_stations, dataset_summary
  TO geo_reader;
GRANT SELECT ON spatial_ref_sys, geometry_columns TO geo_reader;

COMMIT;
ANALYZE;
