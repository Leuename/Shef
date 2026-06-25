from __future__ import annotations

import os
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env", override=True)


def supabase_integration_enabled() -> bool:
    return os.getenv("RUN_SUPABASE_USAGE_INTEGRATION") == "1"


class TestSupabaseUsageIntegration(unittest.TestCase):
    def setUp(self) -> None:
        if not supabase_integration_enabled():
            self.skipTest("Set RUN_SUPABASE_USAGE_INTEGRATION=1 to run Supabase integration tests.")

        import usage_logging

        self.usage_logging = usage_logging
        if not usage_logging.usage_database_url():
            self.skipTest("Set SHEF_USAGE_DATABASE_URL or SUPABASE_DATABASE_URL in .env.")
        if usage_logging.psycopg is None:
            self.skipTest("Install psycopg to run Supabase integration tests.")

    def _connect(self):
        try:
            return self.usage_logging.psycopg.connect(
                self.usage_logging.postgres_dsn(),
                connect_timeout=10,
            )
        except Exception:
            raise AssertionError(
                "Could not connect to Supabase Postgres. Check the database connection string, "
                "username, and password in .env."
            ) from None

    def test_env_selects_postgres_backend_and_admin_token_exists(self) -> None:
        self.assertEqual(self.usage_logging.usage_storage_backend(), "postgres")
        self.assertTrue(os.getenv("ADMIN_DASHBOARD_TOKEN", "").strip())
        self.assertIn("sslmode=", self.usage_logging.postgres_dsn())

        parsed = urlsplit(self.usage_logging.usage_database_url())
        username = unquote(parsed.username or "")
        hostname = parsed.hostname or ""
        if hostname.endswith(".pooler.supabase.com"):
            self.assertRegex(
                username,
                r"^postgres\.[a-z0-9]+$",
                "Supabase pooler URLs usually need user postgres.<project-ref>, not plain postgres.",
            )

    def test_usage_events_table_has_safe_schema_and_rls(self) -> None:
        required_columns = {
            "id",
            "created_at",
            "session_id",
            "event_type",
            "response_mode",
            "model_provider",
            "success",
            "attachment_type",
            "status_code",
            "error_category",
            "user_agent_family",
        }
        forbidden_columns = {
            "message",
            "prompt",
            "model_response",
            "raw_response",
            "ip_address",
            "raw_ip",
            "raw_user_agent",
            "file_name",
            "file_path",
            "blob",
            "content",
        }

        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute("select to_regclass('public.usage_events')")
                table_name = cursor.fetchone()[0]
                self.assertIsNotNone(table_name)

                cursor.execute(
                    """
                    select column_name
                    from information_schema.columns
                    where table_schema = 'public'
                      and table_name = 'usage_events'
                    """
                )
                columns = {row[0] for row in cursor.fetchall()}
                self.assertTrue(required_columns.issubset(columns))
                self.assertTrue(forbidden_columns.isdisjoint(columns))

                cursor.execute(
                    """
                    select c.relrowsecurity
                    from pg_class c
                    join pg_namespace n on n.oid = c.relnamespace
                    where n.nspname = 'public'
                      and c.relname = 'usage_events'
                    """
                )
                rls_enabled = cursor.fetchone()[0]
                self.assertTrue(rls_enabled)
        finally:
            connection.close()

    @unittest.skipUnless(
        os.getenv("RUN_SUPABASE_USAGE_WRITE_TEST") == "1",
        "Set RUN_SUPABASE_USAGE_WRITE_TEST=1 to insert one anonymous test event.",
    )
    def test_can_write_and_read_anonymous_usage_event(self) -> None:
        session_id = f"integration-test-{self.usage_logging.new_session_id()}"

        logged = self.usage_logging.log_usage_event(
            event_type="session_started",
            session_id=session_id,
            response_mode="integration_test",
            model_provider="integration_test",
            success=True,
            status_code=200,
            user_agent_family="IntegrationTest",
        )
        self.assertTrue(logged)

        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    select count(*)
                    from public.usage_events
                    where session_id = %s
                      and event_type = 'session_started'
                    """,
                    (session_id,),
                )
                self.assertEqual(cursor.fetchone()[0], 1)
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
