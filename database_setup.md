# Database Setup

KnowLang supports multiple database providers for vector embeddings and source indexing.

## Sqlite (w/ sqlite-vec)

KnowLang use [sqlite-vec](https://github.com/asg017/sqlite-vec) to support the vector emebdding in sqlite.

### Settings

Following is the example `settings/.env.app` to use sqlite

```
DB__COLLECTION_NAME=code
DB__CODEBASE_DIRECTORY='.'
DB__DB_PROVIDER=sqlite
DB__CONNECTION_URL=sqlite:///knowlang_sqlite.db
# State Store Configuration
DB__STATE_STORE__PROVIDER=sqlite
DB__STATE_STORE__CONNECTION_URL=sqlite:///knowlang_sqlite.db
```

### Troubleshotting sqlite-vec

You may encouter following error when using sqlite.

```sh
AttributeError: 'sqlite3.Connection' object has no attribute 'enable_load_extension'
```

The error occurs because the Python `sqlite3` module is not always built with loadable extension support enabled by default. On many platforms (notably macOS and some Linux distributions), the system SQLite library and the Python module are compiled without the ENABLE_LOAD_EXTENSION flag. As a result, the enable_load_extension method is missing from the sqlite3.Connection object.
For mac users, the eaiest fix will be using the python installed throgh homebrew, which normally has built with loadable extension support.

```sh
brew install python3.12
uv venv --python /opt/homebrew/bin/python3.12
source .venv/bin/activate
uv sync --all-groups
```

## Postgres

KnowLang uses PostgreSQL with pgvector extension for efficient vector storage and retrieval. You can easily set up the database using Docker:

### Prerequisites

1. Make sure you have Docker and Docker Compose installed:

   ```bash
   docker --version
   docker compose --version
   ```

2. Start the PostgreSQL database:

   ```bash
   # From the root of the know-lang repository
   docker compose -f docker/application/docker-compose.app.yml up -d
   ```

3. Verify the database is running:
   ```bash
   docker ps | grep pgvector
   ```

You should see the pgvector container running on port 5432.

> ⚠️ **Important**: The database must be running before you use any KnowLang commands like `parse` or `chat` that require database access.

### Settings

Following is the example `settings/.env.app` to use postgres

```
# Database Configuration
DB__COLLECTION_NAME=code
DB__CODEBASE_DIRECTORY='.'
DB__DB_PROVIDER=postgres
DB__CONNECTION_URL=postgresql://postgres:postgres@localhost:5432/postgres

# State Store Configuration
DB__STATE_STORE__PROVIDER=postgres
DB__STATE_STORE__CONNECTION_URL=postgresql://postgres:postgres@localhost:5432/postgres
```
