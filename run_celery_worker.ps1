# Run Celery worker (Windows-friendly)
# On Windows use --pool=solo (prefork is unsupported).
# If you see MovedError: try --without-mingle or disable Clustering in Azure Portal.

$pool = if ($env:CELERY_POOL) { $env:CELERY_POOL } elseif ($IsWindows -ne $false) { "solo" } else { "prefork" }
$concurrency = if ($env:CELERY_CONCURRENCY) { $env:CELERY_CONCURRENCY } else { 1 }
$withoutMingle = if ($env:CELERY_WITHOUT_MINGLE -eq "1") { "--without-mingle --without-gossip" } else { "" }

Write-Host "Starting Celery worker (pool=$pool, concurrency=$concurrency)..."
celery -A app.celery_app worker --loglevel=info --pool=$pool --concurrency=$concurrency $withoutMingle.Split()
