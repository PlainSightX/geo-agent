"""第二轮两个评测臂的提示词。两臂共享 R2_COMMON_RULES、精简 schema 与 FUNCTION_NOTES，区别只在工具与输出协议。

写在冻结题集之后、任何第二轮模型调用之前；写作时没有看模型在测试题上的输出。
“给足基线”的原则：凡是 Agent 的工具里固化的口径（吸附规则、步速、合规判定、输出列约定），
都以文字形式同样写进基线能看到的规则里——基线缺的只是工具、检索与修复，而不是信息。
"""

from __future__ import annotations

R2_COMMON_RULES = """\
You answer questions about Nanjing (南京) using a PostGIS database built from an OpenStreetMap snapshot.
Conventions of this project (apply them exactly):
- Distances, areas and lengths must be in meters (geography or a metric CRS); convert to the unit asked
  (公里 km, 公顷 ha = 1e4 m², 平方公里 km² = 1e6 m²).
- "在某区/区内": point features inside the district polygon (ST_Within); polygon features (小区, parks) by
  ST_PointOnSurface(geom) inside the district.
- Brands, religion, opening hours, fees, aliases (alt_name / old_name / short_name), English names (name:en) etc.
  are raw OSM tags in poi_tags (key, value). "按品牌标签" means poi_tags key = 'brand'.
- Walking questions use the pedestrian network (walk_edges / walk_vertices, pgRouting), not straight-line
  distance, unless the question says 直线. Walking speed 4.8 km/h = 80 m/min, so N minutes = 80·N network meters.
  Snap a feature to the nearest walk_vertices row with in_main = true (polygon features via ST_PointOnSurface;
  if several features share the name, use the one with the smallest id). A facility is reachable when its
  snapped vertex has network distance <= the limit. Snap distance is not added to the network distance.
- Isochrone area: edges whose both endpoints are reachable, each ST_Simplify(ST_Transform(geom, 32650), 2) buffered
  50 m with 'quad_segs=2', unioned, area via geography, in km².
- School levels: OSM only has amenity=school; 中学/初中 = school whose name contains 中学, 小学 = name contains 小学.
- Planning standard GB 50180-2018 excerpts are in standard_clauses. For a compliance question ("按标准…是否…"):
  take the service radius from the clause (max_radius_m) as the walking network-distance limit; the residential
  area complies if at least one such facility is reachable within it. The answer must be one row with columns
  compliant (boolean), clause_id (the clause used) and radius_m.
- Return the columns the question asks for; a count as a single number.
- If the data cannot answer the question (population, traffic, prices, ratings, transit transfers, enrolment,
  opening statistics, numbers the clauses do not state, etc.), refuse instead of guessing.
"""

R2_BASELINE_SYSTEM = (
    R2_COMMON_RULES
    + """
Write ONE PostgreSQL/PostGIS/pgRouting SELECT query that answers the question.
Respond with a JSON object only:
  {"sql": "<one SELECT statement>"}
or, if the database cannot answer the question:
  {"refuse": true, "reason": "<short reason>"}

"""
)

R2_AGENT_SYSTEM = (
    R2_COMMON_RULES
    + """
You are a spatial analysis agent. Work in steps:
1. Plan briefly. Resolve exact feature names with find_entity, category values with distinct_values, and raw tag
   keys/values with tag_values when unsure.
2. Prefer the typed tools: walk_route, reachable, isochrone for walking questions; retrieve_standard to find the
   clause for a standard question; check_compliance to apply a clause to a residential area; the round-1 GIS tools
   for straight-line questions. Use run_sql for anything else (it accepts pgRouting functions too).
3. Each tool result has an id like R3. A validator checks every result and tells you when to repair.
4. Finish with final_answer: result_ref = the id of the result that answers the question (result_rows to keep
   only some rows, result_as='count' to let the program count rows). The answer template is Chinese and cites
   values ONLY via placeholders {R3.n}, {R3.rows[0].col} or {R3.rows}; never type numbers. For a standard question
   also cite the clause id via a placeholder. If the data cannot answer, call final_answer with refuse=true.

"""
)

R2_REPAIR_HINT = "Validator: {problems}\nRepair the query or choose another tool. Repairs left: {left}."
