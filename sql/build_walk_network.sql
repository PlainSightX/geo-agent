-- 第二轮：从 OSM 原始线要素（osm_raw.lines）构建步行路网 walk_edges / walk_vertices（pgRouting 3.8）。
--
-- 构建参数（README 与 eval/r2/PREREGISTRATION.md 引用此处，修改需重新冻结金标）：
--   1) 纳入的 highway 等级：primary(_link)、secondary(_link)、tertiary(_link)、unclassified、residential、
--      living_street、service、pedestrian、footway、path、steps、track、cycleway、road；
--      trunk/trunk_link 只在显式 foot=yes|designated|permissive 时纳入；motorway* 一律排除。
--   2) 排除 foot=no；access=private|no 且未显式允许步行（foot=yes|designated|permissive）的路段排除。
--   3) 按南京市行政边界（osm_raw.city）裁剪；坐标先 ST_SnapToGrid 到 1e-7 度（OSM 原生精度）。
--   4) 拓扑：只在 OSM 共享节点（坐标完全相同的顶点）与路段端点处打断，不在几何交叉处打断
--      ——桥梁/隧道与下方道路不相交是正确的；这与 osm2pgrouting 的做法一致。
--   5) 步行无单向限制：cost = reverse_cost = 路段 geography 长度（米）。
--   6) 连通分量：pgr_connectedComponents；walk_vertices.in_main 标记最大连通分量。
--      吸附约定：任何点（POI 点、面要素的 ST_PointOnSurface）吸附到 in_main 顶点中 geom <-> 最近的一个。
\set ON_ERROR_STOP on
BEGIN;

DROP TABLE IF EXISTS walk_edges, walk_vertices CASCADE;
DROP TABLE IF EXISTS osm_raw.walk_src, osm_raw.walk_pts;

-- 1) 选取可步行路段并按市界裁剪，拆成单条 LineString
CREATE TABLE osm_raw.walk_src AS
WITH sel AS (
  SELECT l.osm_id::bigint AS osm_id, l.highway, l.name, l.geom,
         substring(l.other_tags from '"foot"=>"([^"]*)"') AS foot,
         substring(l.other_tags from '"access"=>"([^"]*)"') AS access
  FROM osm_raw.lines l
  WHERE l.highway IN ('primary','primary_link','secondary','secondary_link','tertiary','tertiary_link',
                      'unclassified','residential','living_street','service','pedestrian','footway',
                      'path','steps','track','cycleway','road','trunk','trunk_link')
), ok AS (
  SELECT * FROM sel
  WHERE COALESCE(foot, '') <> 'no'
    AND (highway NOT IN ('trunk','trunk_link') OR foot IN ('yes','designated','permissive'))
    AND (COALESCE(access, '') NOT IN ('private','no') OR foot IN ('yes','designated','permissive'))
), clipped AS (
  SELECT ok.osm_id, ok.highway, ok.name,
         CASE WHEN ST_Within(ok.geom, c.geom) THEN ok.geom ELSE ST_Intersection(ok.geom, c.geom) END AS geom
  FROM ok, osm_raw.city c
  WHERE ST_Intersects(ok.geom, c.geom)
)
SELECT row_number() OVER (ORDER BY osm_id, d.path)::int AS sid, osm_id, highway, name,
       ST_SnapToGrid(d.geom, 1e-7) AS geom
FROM clipped, LATERAL ST_Dump(ST_CollectionExtract(clipped.geom, 2)) d
WHERE GeometryType(d.geom) = 'LINESTRING';
DELETE FROM osm_raw.walk_src WHERE ST_NPoints(geom) < 2 OR ST_Length(geom) = 0;

-- 2) 拆点：标记打断点（端点，或被 ≥2 条路段/同一路段多次使用的顶点 = OSM 共享节点）
CREATE TABLE osm_raw.walk_pts AS
SELECT s.sid, (dp).path[1] AS idx, (dp).geom AS geom
FROM (SELECT sid, ST_DumpPoints(geom) AS dp FROM osm_raw.walk_src) s;
CREATE INDEX ON osm_raw.walk_pts (sid, idx);
ALTER TABLE osm_raw.walk_pts ADD COLUMN k text;
UPDATE osm_raw.walk_pts SET k = encode(ST_AsBinary(geom), 'hex');
CREATE INDEX ON osm_raw.walk_pts (k);
ALTER TABLE osm_raw.walk_pts ADD COLUMN is_break boolean;
WITH shared AS (SELECT k FROM osm_raw.walk_pts GROUP BY k HAVING count(*) > 1),
     ends AS (SELECT sid, min(idx) AS a, max(idx) AS b FROM osm_raw.walk_pts GROUP BY sid)
UPDATE osm_raw.walk_pts p
SET is_break = (p.idx = e.a OR p.idx = e.b OR p.k IN (SELECT k FROM shared))
FROM ends e WHERE e.sid = p.sid;

-- 3) 相邻打断点之间组成一条边
CREATE TABLE walk_edges AS
WITH b AS (
  SELECT sid, idx, lead(idx) OVER (PARTITION BY sid ORDER BY idx) AS nidx
  FROM osm_raw.walk_pts WHERE is_break
), seg AS (
  SELECT b.sid, b.idx, ST_MakeLine(p.geom ORDER BY p.idx) AS geom
  FROM b JOIN osm_raw.walk_pts p ON p.sid = b.sid AND p.idx BETWEEN b.idx AND b.nidx
  WHERE b.nidx IS NOT NULL
  GROUP BY b.sid, b.idx
)
SELECT row_number() OVER (ORDER BY seg.sid, seg.idx)::bigint AS id,
       NULL::bigint AS source, NULL::bigint AS target,
       s.osm_id, s.highway, s.name,
       ST_Length(seg.geom::geography) AS cost_m,
       seg.geom::geometry(LineString, 4326) AS geom
FROM seg JOIN osm_raw.walk_src s ON s.sid = seg.sid
WHERE ST_NPoints(seg.geom) >= 2;
DELETE FROM walk_edges WHERE cost_m <= 0;
ALTER TABLE walk_edges ADD PRIMARY KEY (id);

-- 4) 顶点：pgr_extractVertices 从边的端点提取，再回填 source/target
CREATE TABLE walk_vertices AS
SELECT id, geom::geometry(Point, 4326) AS geom
FROM pgr_extractVertices('SELECT id, geom FROM walk_edges ORDER BY id');
ALTER TABLE walk_vertices ADD PRIMARY KEY (id);
CREATE INDEX ON walk_vertices USING gist (geom);

UPDATE walk_edges e SET source = v.id FROM walk_vertices v WHERE ST_StartPoint(e.geom) = v.geom;
UPDATE walk_edges e SET target = v.id FROM walk_vertices v WHERE ST_EndPoint(e.geom) = v.geom;
DELETE FROM walk_edges WHERE source IS NULL OR target IS NULL OR source = target;
ALTER TABLE walk_edges ALTER COLUMN source SET NOT NULL, ALTER COLUMN target SET NOT NULL;
CREATE INDEX ON walk_edges (source);
CREATE INDEX ON walk_edges (target);
CREATE INDEX ON walk_edges USING gist (geom);

-- 5) 连通分量
ALTER TABLE walk_vertices ADD COLUMN component bigint, ADD COLUMN in_main boolean DEFAULT false;
UPDATE walk_vertices v SET component = c.component
FROM pgr_connectedComponents('SELECT id, source, target, cost_m AS cost, cost_m AS reverse_cost FROM walk_edges') c
WHERE c.node = v.id;
WITH big AS (SELECT component FROM walk_vertices GROUP BY component ORDER BY count(*) DESC LIMIT 1)
UPDATE walk_vertices SET in_main = true WHERE component = (SELECT component FROM big);
CREATE INDEX walk_vertices_main_gix ON walk_vertices USING gist (geom) WHERE in_main;

COMMENT ON TABLE walk_edges IS '步行路网边（pgRouting）：source/target 为 walk_vertices.id；cost_m 为长度（米），双向通行';
COMMENT ON TABLE walk_vertices IS '步行路网顶点；component 为连通分量编号，in_main=true 为最大连通分量';

-- 构建统计（写入 walk_network_stats 供 README 与测试引用）
DROP TABLE IF EXISTS walk_network_stats;
CREATE TABLE walk_network_stats AS
SELECT (SELECT count(*) FROM walk_edges) AS edges,
       (SELECT count(*) FROM walk_vertices) AS vertices,
       (SELECT round((sum(cost_m) / 1000)::numeric, 1) FROM walk_edges) AS total_km,
       (SELECT count(DISTINCT component) FROM walk_vertices) AS components,
       (SELECT count(*) FROM walk_vertices WHERE in_main) AS main_vertices,
       (SELECT round((count(*) FILTER (WHERE in_main))::numeric / count(*), 4) FROM walk_vertices) AS main_vertex_share,
       (SELECT round((sum(e.cost_m) FILTER (WHERE v.in_main) / sum(e.cost_m))::numeric, 4)
          FROM walk_edges e JOIN walk_vertices v ON v.id = e.source) AS main_length_share;

GRANT SELECT ON walk_edges, walk_vertices, walk_network_stats TO geo_reader;
COMMIT;
ANALYZE walk_edges;
ANALYZE walk_vertices;
SELECT * FROM walk_network_stats;
