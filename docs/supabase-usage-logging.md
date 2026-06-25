# Supabase Usage Logging

Shef can store anonymous usage events in Supabase Postgres. The app still falls back to local SQLite when no Postgres connection string is configured.

## Schema

Run this migration in your Supabase project:

```text
supabase/migrations/20260625124020_create_usage_events.sql
```

The table is `public.usage_events`. It stores only anonymous metadata:

- timestamp
- anonymous session ID
- event type
- response mode
- model provider
- success/failure
- attachment type
- status code or short error category
- rough user agent family

It does not store chat messages, uploaded files, transcripts, generated recipes, prompts, model responses, exact IP addresses, personal identifiers, filenames, or raw user agents.

## Environment

Set these only in your local `.env` or deployment provider secrets:

```env
SHEF_USAGE_DATABASE_URL=postgresql://...
ADMIN_DASHBOARD_TOKEN=...
```

`SUPABASE_DATABASE_URL` also works, but `SHEF_USAGE_DATABASE_URL` is preferred because it makes the variable's purpose clear.

Do not put Supabase credentials in frontend JavaScript, static files, or committed config. The frontend does not need any Supabase key.

## Connection String

Use a Supabase Postgres connection string from the project Connect panel. For many hosted environments, the session pooler connection string is the practical default because it works in IPv4-only environments. If your deployment supports IPv6 or your Supabase project has the IPv4 add-on, the direct connection string is also fine.

The app adds `sslmode=require` automatically when the connection string does not already include an SSL mode.

## Dashboard

After setting `ADMIN_DASHBOARD_TOKEN`, open:

```text
/admin/usage
```

The dashboard queries the same backend selected by the environment:

- Supabase/Postgres when `SHEF_USAGE_DATABASE_URL` or `SUPABASE_DATABASE_URL` is present
- local SQLite otherwise

## Verification

After deploying:

1. Open the app once to create a `session_started` event.
2. Submit one chat to create `chat_submitted` and either `chat_success` or `chat_error`.
3. Open `/admin/usage` with the admin token and confirm the counts changed.
4. In Supabase SQL Editor, run:

```sql
select event_type, count(*)
from public.usage_events
group by event_type
order by event_type;
```

Do not query for message text or uploaded file data; those columns intentionally do not exist.

## Local Integration Test

After setting `.env`, run the read-only Supabase integration test:

```powershell
$env:RUN_SUPABASE_USAGE_INTEGRATION="1"
python -m unittest test_supabase_usage_integration.py
```

This verifies:

- the app selects the Postgres backend
- `ADMIN_DASHBOARD_TOKEN` is present
- `public.usage_events` exists
- the table has only safe metadata columns
- RLS is enabled

It does not insert data by default. To intentionally insert one anonymous test event, also set:

```powershell
$env:RUN_SUPABASE_USAGE_WRITE_TEST="1"
```

If authentication fails, check the database password in Supabase's project settings. If the password contains special URL characters such as `@`, `#`, `?`, `/`, or `:`, use the fully encoded connection string from Supabase's Connect panel or URL-encode the password before putting it in `.env`.
