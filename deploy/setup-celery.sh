#!/bin/bash
# Setup script for Celery on Azure Ubuntu VM (8 cores, 32GB RAM)
# Run as root or with sudo
set -e
APP_USER="azureuser"
APP_DIR="/home/$APP_USER/uniliverimagestudy"

echo "=== Setting up Celery for Mindsurve ==="

# 1. Create log directory
echo "Creating log directory..."
mkdir -p /var/log/celery
chown $APP_USER:$APP_USER /var/log/celery

# 2. Optional: local Redis (only if you want cache/pub-sub on this VM; Celery uses RabbitMQ from .env)
read -p "Install local redis-server for app cache? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    apt-get update
    apt-get install -y redis-server
    systemctl enable redis-server
    systemctl start redis-server
fi

# 3. Copy systemd service file
echo "Installing systemd service..."
cp "$APP_DIR/deploy/celery-worker.service" /etc/systemd/system/celery-worker.service
systemctl daemon-reload

# 4. Enable and start Celery worker
echo "Enabling Celery worker service..."
systemctl enable celery-worker
systemctl start celery-worker
echo "=== Setup complete ==="
echo ""
echo "Ensure .env includes:"
echo "  CELERY_BROKER_URL=amqps://..."
echo ""
echo "Commands:"
echo "  sudo systemctl status celery-worker"
echo "  sudo systemctl restart celery-worker"
echo "  sudo journalctl -u celery-worker -f"
echo "  tail -f /var/log/celery/worker.log"