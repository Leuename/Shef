from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


class UsageLoggingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "usage.sqlite3"
        self.env_patch = patch.dict(
            os.environ,
            {
                "SHEF_USAGE_DB_PATH": str(self.db_path),
                "SHEF_USAGE_DATABASE_URL": "",
                "SUPABASE_DATABASE_URL": "",
            },
            clear=False,
        )
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.tmp.cleanup()

    def fetch_events(self) -> list[sqlite3.Row]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            return connection.execute(
                "SELECT * FROM usage_events ORDER BY id"
            ).fetchall()
        finally:
            connection.close()


class TestUsageEventStorage(UsageLoggingTestCase):
    def test_records_only_safe_event_metadata(self) -> None:
        import usage_logging

        usage_logging.log_usage_event(
            event_type="chat_submitted",
            session_id="session-test",
            response_mode="full_recipe",
            model_provider="NVIDIA NIM",
            success=True,
            attachment_type="image",
            status_code=200,
            error_category=None,
            user_agent_family="Chrome",
        )

        connection = sqlite3.connect(self.db_path)
        try:
            columns = [
                row[1]
                for row in connection.execute("PRAGMA table_info(usage_events)").fetchall()
            ]
        finally:
            connection.close()

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
        self.assertTrue(forbidden_columns.isdisjoint(columns))

        [event] = self.fetch_events()
        self.assertEqual(event["event_type"], "chat_submitted")
        self.assertEqual(event["session_id"], "session-test")
        self.assertEqual(event["attachment_type"], "image")

    def test_usage_summary_counts_core_events(self) -> None:
        import usage_logging

        for event_type, session_id in [
            ("session_started", "session-a"),
            ("session_started", "session-b"),
            ("chat_submitted", "session-a"),
            ("recipe_selected", "session-a"),
            ("image_uploaded", "session-a"),
            ("audio_uploaded", "session-b"),
            ("chat_error", "session-b"),
        ]:
            usage_logging.log_usage_event(event_type=event_type, session_id=session_id)

        summary = usage_logging.usage_summary(limit=5)

        self.assertEqual(summary["total_sessions"], 2)
        self.assertEqual(summary["total_chats"], 1)
        self.assertEqual(summary["recipe_selections"], 1)
        self.assertEqual(summary["uploads"], 2)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(len(summary["recent_activity"]), 5)

    def test_supabase_database_url_selects_postgres_backend(self) -> None:
        import usage_logging

        with patch.dict(
            os.environ,
            {
                "SUPABASE_DATABASE_URL": "postgresql://postgres.example:secret@db.example.supabase.co:5432/postgres",
            },
            clear=False,
        ):
            self.assertEqual(usage_logging.usage_storage_backend(), "postgres")

    def test_postgres_logging_uses_safe_columns_only(self) -> None:
        import usage_logging

        executed_sql: list[str] = []

        class FakeCursor:
            def execute(self, sql, params=None):
                del params
                executed_sql.append(sql)

            def fetchone(self):
                return None

            def fetchall(self):
                return []

        class FakeConnection:
            def __init__(self):
                self.cursor_obj = FakeCursor()

            def cursor(self, row_factory=None):
                del row_factory
                return self.cursor_obj

            def commit(self):
                pass

            def rollback(self):
                pass

            def close(self):
                pass

        fake_psycopg = MagicMock()
        fake_psycopg.connect.return_value = FakeConnection()

        with (
            patch.dict(
                os.environ,
                {"SUPABASE_DATABASE_URL": "postgresql://postgres.example:secret@db.example.supabase.co:5432/postgres"},
                clear=False,
            ),
            patch.object(usage_logging, "psycopg", fake_psycopg),
        ):
            usage_logging.log_usage_event(
                event_type="chat_submitted",
                session_id="session-test",
                response_mode="full_recipe",
                model_provider="NVIDIA NIM",
                success=True,
                attachment_type="image",
                status_code=200,
                user_agent_family="Chrome",
            )

        all_sql = "\n".join(executed_sql).lower()
        self.assertIn("insert into public.usage_events", all_sql)
        for forbidden in [
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
        ]:
            self.assertNotIn(forbidden, all_sql)

    def test_postgres_schema_initialization_is_cached_after_first_success(self) -> None:
        import usage_logging

        executed_sql: list[str] = []

        class FakeCursor:
            def execute(self, sql, params=None):
                del params
                executed_sql.append(sql.strip())

        class FakeConnection:
            def cursor(self, row_factory=None):
                del row_factory
                return FakeCursor()

            def commit(self):
                pass

            def rollback(self):
                pass

            def close(self):
                pass

        fake_psycopg = MagicMock()
        fake_psycopg.connect.return_value = FakeConnection()

        with (
            patch.dict(
                os.environ,
                {
                    "SUPABASE_DATABASE_URL": (
                        "postgresql://postgres.cachetest:secret@db.cachetest.supabase.co:5432/postgres"
                    )
                },
                clear=False,
            ),
            patch.object(usage_logging, "psycopg", fake_psycopg),
        ):
            for session_id in ["session-one", "session-two"]:
                self.assertTrue(
                    usage_logging.log_usage_event(
                        event_type="chat_submitted",
                        session_id=session_id,
                        response_mode="full_recipe",
                        model_provider="NVIDIA NIM",
                    )
                )

        normalised_sql = [sql.lower() for sql in executed_sql]
        self.assertEqual(
            sum(sql.startswith("create table") for sql in normalised_sql),
            1,
        )
        self.assertEqual(
            sum(sql.startswith("alter table") for sql in normalised_sql),
            1,
        )
        self.assertEqual(
            sum(sql.startswith("create index") for sql in normalised_sql),
            3,
        )
        self.assertEqual(
            sum(sql.startswith("insert into public.usage_events") for sql in normalised_sql),
            2,
        )

    def test_supabase_migration_enables_rls_without_public_policies(self) -> None:
        migration_dir = Path(__file__).resolve().parent / "supabase" / "migrations"
        migrations = list(migration_dir.glob("*usage_events*.sql"))
        self.assertEqual(len(migrations), 1)
        sql = migrations[0].read_text(encoding="utf-8").lower()

        self.assertIn("create table if not exists public.usage_events", sql)
        self.assertIn("alter table public.usage_events enable row level security", sql)
        self.assertNotIn("create policy", sql)


class TestAdminUsageDashboard(UsageLoggingTestCase):
    def test_dashboard_requires_admin_token(self) -> None:
        import usage_logging
        import lc

        usage_logging.log_usage_event(event_type="session_started", session_id="session-a")
        client = TestClient(lc.app)

        with patch.dict(os.environ, {"ADMIN_DASHBOARD_TOKEN": "admin-token"}, clear=False):
            unauthenticated = client.get("/admin/usage")
            self.assertEqual(unauthenticated.status_code, 401)
            self.assertNotIn("Total sessions", unauthenticated.text)

            wrong_token = client.post("/admin/usage", data={"token": "wrong"})
            self.assertEqual(wrong_token.status_code, 403)
            self.assertNotIn("Total sessions", wrong_token.text)

            dashboard = client.post("/admin/usage", data={"token": "admin-token"})
            self.assertEqual(dashboard.status_code, 200)
            self.assertIn("Total sessions", dashboard.text)
            self.assertIn("Recent activity", dashboard.text)


class TestChatUsageEvents(UsageLoggingTestCase):
    def test_chat_request_logs_safe_events_without_message_text(self) -> None:
        import lc

        client = TestClient(lc.app)
        client.cookies.set("shef_usage_session", "session-chat")

        with (
            patch.object(lc, "enforce_rate_limit", lambda request: None),
            patch.object(lc, "recipe_search_sync", return_value="Filipino recipe context"),
            patch.object(
                lc,
                "invoke_recipe_agent_sync",
                return_value="Chicken Tinola\nIngredients: chicken, ginger, sayote",
            ),
        ):
            response = client.post(
                "/api/chat",
                data={
                    "message": "Show me the recipe for Secret Family Tinola.",
                    "thread_id": "local-chat",
                    "history": "[]",
                    "response_mode": "full_recipe",
                    "usage_event": "recipe_selected",
                },
            )

        self.assertEqual(response.status_code, 200)
        events = self.fetch_events()
        self.assertEqual(
            [event["event_type"] for event in events],
            ["chat_submitted", "recipe_selected", "chat_success"],
        )

        stored_values = " ".join(
            str(value) for event in events for value in tuple(event) if value is not None
        )
        self.assertNotIn("Secret Family Tinola", stored_values)
        self.assertNotIn("chicken, ginger, sayote", stored_values)

    def test_non_ingredient_upload_rejection_is_logged_without_file_content(self) -> None:
        import lc

        client = TestClient(lc.app)
        client.cookies.set("shef_usage_session", "session-upload")

        with (
            patch.object(lc, "enforce_rate_limit", lambda request: None),
            patch.object(
                lc,
                "extract_image_ingredients_sync",
                return_value="NOT_INGREDIENTS: no edible cooking ingredients are visible.",
            ),
        ):
            response = client.post(
                "/api/chat",
                data={
                    "message": "",
                    "thread_id": "local-chat",
                    "history": "[]",
                    "response_mode": "auto",
                },
                files={"image": ("portrait.jpg", b"not really an image", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 400)
        events = self.fetch_events()
        self.assertEqual(
            [event["event_type"] for event in events],
            [
                "chat_submitted",
                "image_uploaded",
                "non_ingredient_upload_rejected",
                "chat_error",
            ],
        )
        self.assertEqual(events[-1]["error_category"], "non_ingredient_upload")
        stored_values = " ".join(
            str(value) for event in events for value in tuple(event) if value is not None
        )
        self.assertNotIn("portrait.jpg", stored_values)
        self.assertNotIn("not really an image", stored_values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
