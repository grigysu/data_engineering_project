-- Postgres init: create the Hive Metastore backing database and user.
-- Runs automatically on the first time the postgres volume is initialized
-- (via the bind mount on /docker-entrypoint-initdb.d). On existing volumes
-- this file is ignored; create the user+db manually with psql.

CREATE USER hive WITH PASSWORD 'hive';
CREATE DATABASE metastore_db OWNER hive;
GRANT ALL PRIVILEGES ON DATABASE metastore_db TO hive;
