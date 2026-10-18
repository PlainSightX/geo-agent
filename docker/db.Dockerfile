# 第二轮数据库镜像：PostGIS 16-3.5 + pgRouting 3.8.0（官方镜像，固定 digest）+ pgvector 0.8.6（固定 deb 版本与 sha256）。
# 选择 pgrouting/pgrouting:16-3.5-3.8.0：其 PostgreSQL 16.9 / PostGIS 3.5.2 与第一轮镜像完全一致，
# 可以直接复用第一轮的数据卷，且第一轮金标在新镜像上可逐题复现（见 README“第二轮”）。
FROM pgrouting/pgrouting:16-3.5-3.8.0@sha256:4c480c7f9e25a3490573cd6ef86b286fd21162e288cc355e8da2b52518ef6ab8

# 镜像内的 apt.postgresql.org bullseye 源已下线，改用官方归档源；只下载一个固定版本的 deb 并校验 sha256，
# 不做 apt-get upgrade，保证 PostgreSQL/PostGIS 版本不被连带升级。
ARG PGVECTOR_DEB=postgresql-16-pgvector_0.8.6-1.pgdg11+1_amd64.deb
ARG PGVECTOR_SHA256=1752b51bf59f94f1f0b26cb8ba2bbd80ac7e98d017407f03130fb86b5a7e5c8f
RUN set -eux; \
    sed -i 's#http://apt.postgresql.org/pub/repos/apt#https://apt-archive.postgresql.org/pub/repos/apt#' \
        /etc/apt/sources.list.d/pgdg.list; \
    apt-get update; \
    cd /tmp; \
    apt-get download postgresql-16-pgvector=0.8.6-1.pgdg11+1; \
    echo "${PGVECTOR_SHA256}  ${PGVECTOR_DEB}" | sha256sum -c -; \
    dpkg -i "${PGVECTOR_DEB}"; \
    rm -f "${PGVECTOR_DEB}"; \
    rm -rf /var/lib/apt/lists/*; \
    test "$(dpkg-query -W -f='${Version}' postgresql-16)" = "16.9-1.pgdg110+1"; \
    test "$(dpkg-query -W -f='${Version}' postgresql-16-postgis-3)" = "3.5.2+dfsg-1.pgdg110+1"
