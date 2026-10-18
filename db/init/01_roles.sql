-- 首次初始化数据库时执行：创建只读角色 geo_reader，Agent 与评测只用这个角色连库。
-- 安全边界在数据库层而不是只靠 SQL 解析：即使 AST 白名单被绕过，该角色也无写权限。
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE ROLE geo_reader LOGIN PASSWORD 'geo_reader_local_only'
  NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;

-- 角色级默认值：单条语句 5 秒超时、默认只读事务、空闲事务超时
ALTER ROLE geo_reader SET statement_timeout = '5s';
ALTER ROLE geo_reader SET default_transaction_read_only = on;
ALTER ROLE geo_reader SET idle_in_transaction_session_timeout = '10s';
ALTER ROLE geo_reader SET search_path = public;

REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO geo_reader;
