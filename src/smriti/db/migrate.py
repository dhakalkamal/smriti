from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

import asyncpg
import click

from smriti.config import Settings, get_settings

MIGRATION_FILENAME_PATTERN = re.compile(r"^(?P<version>\d+)_(?P<name>.+)\.sql$")


@dataclass(frozen=True)
class MigrationFile:
    version: int
    filename: str
    path: Path


def _load_migration_files(migrations_dir: Path) -> list[MigrationFile]:
    migration_files: list[MigrationFile] = []

    for path in migrations_dir.glob("*.sql"):
        match = MIGRATION_FILENAME_PATTERN.match(path.name)
        if match is None:
            continue

        migration_files.append(
            MigrationFile(
                version=int(match.group("version")),
                filename=path.name,
                path=path,
            )
        )

    migration_files.sort(key=lambda migration: (migration.version, migration.filename))
    return migration_files


async def _ensure_schema_migrations_table(connection: asyncpg.Connection) -> None:
    await connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )


async def _fetch_applied_versions(connection: asyncpg.Connection) -> set[int]:
    rows = await connection.fetch("SELECT version FROM schema_migrations;")
    return {row["version"] for row in rows}


async def apply_migrations(settings: Settings, migrations_dir: Path) -> list[MigrationFile]:
    """Apply pending migrations in order and return applied files."""

    connection = await asyncpg.connect(
        dsn=settings.database_url,
        timeout=settings.database_connect_timeout,
        command_timeout=settings.database_command_timeout,
    )

    applied_now: list[MigrationFile] = []
    try:
        await _ensure_schema_migrations_table(connection)
        applied_versions = await _fetch_applied_versions(connection)

        for migration in _load_migration_files(migrations_dir):
            if migration.version in applied_versions:
                continue

            sql = migration.path.read_text(encoding="utf-8")
            async with connection.transaction():
                await connection.execute(sql)
                await connection.execute(
                    "INSERT INTO schema_migrations (version, filename) VALUES ($1, $2);",
                    migration.version,
                    migration.filename,
                )

            applied_now.append(migration)

    finally:
        await connection.close()

    return applied_now


async def collect_migration_status(
    settings: Settings,
    migrations_dir: Path,
) -> tuple[list[MigrationFile], set[int]]:
    """Return filesystem migrations and DB-applied migration versions."""

    connection = await asyncpg.connect(
        dsn=settings.database_url,
        timeout=settings.database_connect_timeout,
        command_timeout=settings.database_command_timeout,
    )

    try:
        await _ensure_schema_migrations_table(connection)
        applied_versions = await _fetch_applied_versions(connection)
    finally:
        await connection.close()

    return _load_migration_files(migrations_dir), applied_versions


@click.group(name="migrate")
def migrate() -> None:
    """Database migration commands."""


@migrate.command("up")
@click.option("--database-url", type=str, default=None, help="Override SMRITI_DATABASE_URL")
@click.option("--migrations-dir", type=click.Path(path_type=Path), default=None)
def migrate_up(database_url: str | None, migrations_dir: Path | None) -> None:
    """Apply pending numbered SQL migrations."""

    settings = get_settings()
    if database_url is not None:
        settings = settings.model_copy(update={"database_url": database_url})

    resolved_migrations_dir = migrations_dir or settings.migrations_dir
    applied_now = asyncio.run(apply_migrations(settings=settings, migrations_dir=resolved_migrations_dir))

    if not applied_now:
        click.echo("No pending migrations.")
        return

    for migration in applied_now:
        click.echo(f"Applied {migration.filename}")


@migrate.command("status")
@click.option("--database-url", type=str, default=None, help="Override SMRITI_DATABASE_URL")
@click.option("--migrations-dir", type=click.Path(path_type=Path), default=None)
def migrate_status(database_url: str | None, migrations_dir: Path | None) -> None:
    """Show migration status for files on disk versus DB state."""

    settings = get_settings()
    if database_url is not None:
        settings = settings.model_copy(update={"database_url": database_url})

    resolved_migrations_dir = migrations_dir or settings.migrations_dir
    files, applied_versions = asyncio.run(
        collect_migration_status(settings=settings, migrations_dir=resolved_migrations_dir)
    )

    if not files:
        click.echo("No migration files found.")
        return

    for migration in files:
        status_label = "applied" if migration.version in applied_versions else "pending"
        click.echo(f"[{status_label}] {migration.filename}")


if __name__ == "__main__":
    migrate()
