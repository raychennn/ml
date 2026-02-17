#!/bin/sh
# Seed references.json if not present on the persistent volume
if [ ! -f /data/references/references.json ]; then
    mkdir -p /data/references
    cp /app/data/references/references.json /data/references/references.json
    echo "Seeded references.json from Docker image"
fi

# Ensure all data directories exist
mkdir -p /data/parquet /data/models /data/images /data/cache /data/logs /data/references

exec "$@"
