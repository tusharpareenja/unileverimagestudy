# Deploy Celery + RabbitMQ on your Azure Linux server (simple)

Your server has the backend here:

`/var/www/unileverimagestudy-test`

You log in as user: **`mindsurve`**

Everything below can run on **the same VM**: RabbitMQ + FastAPI + Celery worker.

---

## Part 1 — RabbitMQ on this same VM (your setup)

### Install RabbitMQ (Ubuntu)

```bash
sudo apt update
sudo apt install -y rabbitmq-server
sudo systemctl enable rabbitmq-server
sudo systemctl start rabbitmq-server
sudo systemctl status rabbitmq-server
```

You should see **active (running)**.

### Default login (OK for same machine only)

RabbitMQ ships with user **`guest`** / password **`guest`**. It is **only allowed from localhost** — perfect when API and Celery both run on this VM.

In **`.env`**:

```env
CELERY_BROKER_URL=amqp://guest:guest@127.0.0.1:5672//
```

(`//` at the end = default virtual host `/`)

### Optional: stronger setup (own user + vhost)

If you want a dedicated user (recommended before opening RabbitMQ to the network):

```bash
sudo rabbitmqctl add_user mindsurve_celery YOUR_STRONG_PASSWORD
sudo rabbitmqctl add_vhost mindsurve
sudo rabbitmqctl set_permissions -p mindsurve mindsurve_celery ".*" ".*" ".*"
```

Then in **`.env`**:

```env
CELERY_BROKER_URL=amqp://mindsurve_celery:YOUR_STRONG_PASSWORD@127.0.0.1:5672/mindsurve
```

*(If the password has special characters, URL-encode them in the URL.)*

### Optional: web UI (management plugin)

```bash
sudo rabbitmq-plugins enable rabbitmq_management
```

Then open in browser (only if your firewall allows it): `http://YOUR_VM_IP:15672`  
Default login: `guest` / `guest` (localhost only — for remote access create another user).

### Azure network note

- **Same VM:** API and Celery use `127.0.0.1` — no extra firewall rule.
- **Do not** expose port **5672** or **15672** to the whole internet unless you know what you’re doing.

---

## Part 2 — CloudAMQP instead (optional)

If you use **CloudAMQP** in the cloud, you **do not** install RabbitMQ on the VM. Only set:

```env
CELERY_BROKER_URL=amqps://USER:PASSWORD@HOST/VHOST
```

---

## Part 3 — What Celery does on the server

1. **FastAPI** receives “run this in background”.
2. It sends a message to **RabbitMQ** (on this VM at `127.0.0.1`, or CloudAMQP).
3. **Celery worker** on this VM reads the message and runs the job.
4. Worker uses the **same** `.env` as the API (DB, `CELERY_BROKER_URL`, etc.).

---

## Part 4 — Step-by-step (app + Celery)

### Step 1 — Project folder

```bash
cd /var/www/unileverimagestudy-test
```

### Step 2 — Python venv

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

(If `venv` exists, only `activate` + `pip install`.)

### Step 3 — `.env`

```env
USE_CELERY=true
CELERY_BROKER_URL=amqp://guest:guest@127.0.0.1:5672//
DATABASE_URL=...
REDIS_URL=...
# ... rest of your vars
```

### Step 4 — Test Celery once

```bash
cd /var/www/unileverimagestudy-test
source venv/bin/activate
celery -A app.celery_app worker --loglevel=info --pool=prefork --concurrency=4
```

You should see **connected** and **ready**. `Ctrl+C` to stop.

### Step 5 — systemd: start Celery **after** RabbitMQ

Create log dir:

```bash
sudo mkdir -p /var/log/celery
sudo chown mindsurve:mindsurve /var/log/celery
```

Service file:

```bash
sudo nano /etc/systemd/system/celery-worker-test.service
```

Paste (**note `After=` includes RabbitMQ**):

```ini
[Unit]
Description=Celery worker (unileverimagestudy-test)
After=network.target rabbitmq-server.service
Requires=rabbitmq-server.service

[Service]
Type=simple
User=mindsurve
Group=mindsurve
WorkingDirectory=/var/www/unileverimagestudy-test
Environment="PATH=/var/www/unileverimagestudy-test/venv/bin"
EnvironmentFile=/var/www/unileverimagestudy-test/.env
ExecStart=/var/www/unileverimagestudy-test/venv/bin/celery -A app.celery_app worker --loglevel=info --pool=prefork --concurrency=4
Restart=always
RestartSec=10
StandardOutput=append:/var/log/celery/worker-test.log
StandardError=append:/var/log/celery/worker-test.log

[Install]
WantedBy=multi-user.target
```

If you use **CloudAMQP only**, remove the `Requires=` line and use `After=network.target` only.

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable celery-worker-test
sudo systemctl start celery-worker-test
sudo systemctl status celery-worker-test
```

Logs:

```bash
tail -f /var/log/celery/worker-test.log
```

---

## Part 5 — FastAPI

Same `.env`, `USE_CELERY=true`, same `CELERY_BROKER_URL`. Restart the API after changes.

---

## Part 6 — Quick checklist

| Piece        | On your VM                         |
|-------------|-------------------------------------|
| RabbitMQ    | `apt install rabbitmq-server`       |
| Broker URL  | `amqp://guest:guest@127.0.0.1:5672//` (or your user/vhost) |
| Celery      | systemd `celery-worker-test`        |
| FastAPI     | your existing process               |
| `.env`      | one file shared by API + worker     |

---

## If something fails

1. `sudo systemctl status rabbitmq-server` — RabbitMQ running?
2. `sudo systemctl status celery-worker-test` — worker running?
3. `CELERY_BROKER_URL` must use **`127.0.0.1`** (or `localhost`) when RabbitMQ is local.
4. Lower concurrency if RAM is tight: `--concurrency=2`.

---

## One-sentence summary

**Install RabbitMQ on the VM, point `CELERY_BROKER_URL` at `127.0.0.1`, run Celery with systemd and `After=rabbitmq-server.service`, and keep one `.env` for both API and worker.**
