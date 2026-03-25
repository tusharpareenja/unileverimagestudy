# Celery Deployment Guide

## Overview

Celery handles background tasks:
- **Task Generation** (`celery_job.generate_tasks`) — generates study task designs
- **AI Simulation** (`celery_job.simulate_synthetic_respondents`) — simulates AI respondents

## Requirements

- **RabbitMQ** — message broker (e.g. [CloudAMQP](https://www.cloudamqp.com/), Azure Service Bus with AMQP, or self-hosted RabbitMQ)
- **Python 3.10+** with virtualenv
- **systemd** — for running Celery as a service (Linux)
- **Redis** (optional) — only for app cache + WebSocket pub/sub, **not** for Celery broker

## Environment variables

| Variable | Purpose |
|----------|---------|
| `CELERY_BROKER_URL` | RabbitMQ URL (e.g. `amqps://user:pass@host/vhost`). One URL for dev and prod is fine; use a second instance later if you want isolation. |
| `CELERY_RESULT_BACKEND` | Optional: default `rpc://` (results via RabbitMQ). Set `redis://...` if you want Redis for task results |
| `USE_CELERY` | `true` to dispatch jobs from FastAPI to workers |
| `REDIS_URL` | App cache / pub-sub only |

## Quick Start (Azure Ubuntu VM)

### 1. Install dependencies

```bash
cd /home/azureuser/uniliverimagestudy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env`

```bash
USE_CELERY=true
CELERY_BROKER_URL=amqps://user:pass@fly.rmq.cloudamqp.com/vhost

# Redis: caching / WebSocket (not Celery broker)
REDIS_URL=rediss://:key@your-redis.redis.cache.windows.net:6380/0
```

### 3. Run setup script

```bash
sudo bash deploy/setup-celery.sh
```

### 4. Verify

```bash
sudo systemctl status celery-worker
```

## Manual Commands

```bash
# Start worker manually (for testing)
celery -A app.celery_app worker --loglevel=info --pool=prefork --concurrency=6

# Check registered tasks
celery -A app.celery_app inspect registered

# Monitor tasks
celery -A app.celery_app events

# Purge all pending tasks (careful!)
celery -A app.celery_app purge
```

## Scaling Guidelines

| VM Size | Cores | RAM | Recommended Concurrency |
|---------|-------|-----|------------------------|
| 4 vCPU  | 4     | 16GB | 2-3 |
| 8 vCPU  | 8     | 32GB | 6 |
| 16 vCPU | 16    | 64GB | 12 |

Leave 2 cores for FastAPI/Gunicorn and system overhead.

## Time Limits

Configured in `app/celery_app.py`:
- **Hard limit:** 8 hours (`task_time_limit=28800`)
- **Soft limit:** 7h50m (`task_soft_time_limit=28200`)

## Troubleshooting

### Cannot connect to broker
1. Check URL uses `amqps://` (SSL) or `amqp://` as required by your host
2. CloudAMQP: use the full URL from the dashboard (includes vhost path)
3. Firewall: outbound 5671/5672 to RabbitMQ host

### Tasks not executing
1. `systemctl status celery-worker`
2. `tail -f /var/log/celery/worker.log`
3. Ensure `USE_CELERY=true` on the API server

### Memory issues
Reduce `--concurrency` or increase VM RAM.

## Architecture

```
┌─────────────┐    ┌────────────┐    ┌────────────────┐
│   FastAPI   │───▶│  RabbitMQ  │◀───│ Celery Worker  │
│  (Gunicorn) │    │   (AMQP)   │    │  (prefork x6)  │
└─────────────┘    └────────────┘    └────────────────┘
       │                  │                    │
       │    ┌─────────────┴─────────────┐      │
       │    ▼                           │      │
       └──▶ │ Redis (cache / pub-sub)   │ ◀────┘
            └───────────────────────────┘
                       │
                       ▼
               ┌──────────────┐
               │  PostgreSQL  │
               └──────────────┘
```

FastAPI enqueues tasks to RabbitMQ. Workers consume, run jobs, and update PostgreSQL. Task results default to `rpc://` (via the broker). Redis is independent (cache + WebSockets).
