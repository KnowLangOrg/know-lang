FROM postgres:15

# Set environment variables
ENV POSTGRES_PASSWORD=postgres
ENV POSTGRES_USER=postgres
ENV POSTGRES_DB=postgres

# Install build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       postgresql-server-dev-15 \
       git \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone and install pgvector
RUN git clone --branch v0.5.0 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install \
    && cd .. \
    && rm -rf pgvector

# Add custom configurations to allow remote connections
RUN echo "listen_addresses = '*'" >> /usr/share/postgresql/postgresql.conf.sample
RUN echo "host all all 0.0.0.0/0 md5" >> /usr/share/postgresql/pg_hba.conf.sample

# Create a script to initialize the vector extension
COPY <<EOF /docker-entrypoint-initdb.d/01-init-vector.sql
CREATE EXTENSION IF NOT EXISTS vector;
EOF

# Expose PostgreSQL port
EXPOSE 5432

# Set healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD pg_isready -U postgres -h localhost