# Web service: read-only API that serves pre-computed predictions instantly.
web: uvicorn api:app --host 0.0.0.0 --port $PORT

# Cron service: the daily write-side job (Statcast snapshot + inference).
# Create a SEPARATE Railway service from this repo, set its start command to
# this process, and give it a Cron Schedule (see railway.toml) so it runs once
# each morning before lineups lock. It exits when done — it is NOT long-running.
cron: python cron_daily.py
