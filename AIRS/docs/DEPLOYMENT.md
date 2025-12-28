# Deployment Guide

This document describes how to deploy AIRS in production.

## Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │              Load Balancer              │
                    └─────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
              ┌─────┴─────┐      ┌─────┴─────┐      ┌─────┴─────┐
              │  API Pod  │      │  API Pod  │      │  API Pod  │
              └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
                    │                  │                  │
                    └──────────────────┼──────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
              ┌─────┴─────┐      ┌─────┴─────┐      ┌─────┴─────┐
              │ PostgreSQL│      │   Redis   │      │  MLflow   │
              └───────────┘      └───────────┘      └───────────┘
                                       │
                              ┌────────┴────────┐
                              │     Airflow     │
                              │   (Scheduler)   │
                              └─────────────────┘
```

## Prerequisites

- Docker & Docker Compose
- Kubernetes (optional, for production)
- PostgreSQL 14+
- Redis 7+
- API Keys: FRED, Alpha Vantage

## Docker Compose Deployment

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/airs.git
cd airs

# Configure environment
cp .env.example .env
# Edit .env with your API keys and credentials

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| api | 8000 | FastAPI application |
| postgres | 5432 | PostgreSQL database |
| redis | 6379 | Celery broker |
| mlflow | 5000 | Experiment tracking |
| airflow-webserver | 8080 | Airflow UI |
| airflow-scheduler | - | DAG scheduler |

### Configuration

Key environment variables in `.env`:

```bash
# Database
POSTGRES_USER=airs
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=airs
DATABASE_URL=postgresql://airs:secure_password@postgres:5432/airs

# Redis
REDIS_URL=redis://redis:6379/0

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# API Keys (required)
FRED_API_KEY=your_fred_api_key
ALPHA_VANTAGE_API_KEY=your_av_api_key

# Airflow
AIRFLOW__CORE__EXECUTOR=CeleryExecutor
AIRFLOW__CORE__SQL_ALCHEMY_CONN=${DATABASE_URL}
AIRFLOW__CELERY__BROKER_URL=${REDIS_URL}
```

## Kubernetes Deployment

### Namespace

```bash
kubectl create namespace airs
```

### Secrets

```bash
# Create secrets
kubectl create secret generic airs-secrets \
  --from-literal=postgres-password=secure_password \
  --from-literal=fred-api-key=your_key \
  --from-literal=alpha-vantage-api-key=your_key \
  -n airs
```

### Deployment Manifests

Example API deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: airs-api
  namespace: airs
spec:
  replicas: 3
  selector:
    matchLabels:
      app: airs-api
  template:
    metadata:
      labels:
        app: airs-api
    spec:
      containers:
      - name: api
        image: your-registry/airs:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: airs-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/v1/health/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: airs-api
  namespace: airs
spec:
  selector:
    app: airs-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: airs-ingress
  namespace: airs
  annotations:
    kubernetes.io/ingress.class: nginx
spec:
  rules:
  - host: airs.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: airs-api
            port:
              number: 80
```

## Database Setup

### Initial Schema

```bash
# Run migrations
docker-compose exec api alembic upgrade head

# Or manually
python -m alembic upgrade head
```

### Backup Strategy

```bash
# Daily backup
pg_dump -h postgres -U airs airs > backup_$(date +%Y%m%d).sql

# Restore
psql -h postgres -U airs airs < backup_20240115.sql
```

## Airflow Configuration

### DAG Schedule

| DAG | Schedule | Description |
|-----|----------|-------------|
| daily_pipeline | 0 18 * * 1-5 | Daily data + prediction |
| model_retrain | 0 6 * * 1 | Weekly model check |
| data_quality | 0 17 * * 1-5 | Pre-pipeline checks |

### Worker Scaling

```bash
# Scale Celery workers
docker-compose up -d --scale airflow-worker=3
```

## Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/api/v1/health

# Airflow health
curl http://localhost:8080/health
```

### Logging

Logs are written to stdout in JSON format:

```json
{
  "timestamp": "2024-01-15T18:00:00Z",
  "level": "INFO",
  "message": "Prediction generated",
  "probability": 0.62,
  "alert_level": "moderate"
}
```

### Metrics

Key metrics to monitor:

| Metric | Alert Threshold |
|--------|-----------------|
| API latency p99 | >500ms |
| Prediction latency | >1000ms |
| Data freshness | >2 days |
| Model precision (rolling) | <0.60 |
| Error rate | >1% |

### Alerting

Configure alerts in your monitoring system for:

1. **Service Health**
   - API pods down
   - Database connection failures
   - Redis unavailable

2. **Data Pipeline**
   - DAG failures
   - Data fetch errors
   - Missing data

3. **Model Performance**
   - Precision degradation
   - Feature drift
   - Prediction drift

## Security

### Network Security

- Keep PostgreSQL and Redis internal
- Use TLS for all external connections
- Implement rate limiting

### Access Control

- Use service accounts for K8s
- Rotate API keys regularly
- Audit log access

### Secrets Management

- Use K8s secrets or Vault
- Never commit credentials
- Encrypt at rest

## Scaling Considerations

### Horizontal Scaling

| Component | Scale Factor |
|-----------|--------------|
| API | CPU usage > 70% |
| Airflow Workers | DAG queue depth |
| Database | Connection count |

### Vertical Scaling

| Component | Recommended |
|-----------|-------------|
| API | 1 CPU, 2GB RAM |
| Database | 4 CPU, 16GB RAM |
| Airflow Scheduler | 2 CPU, 4GB RAM |

## Troubleshooting

### Common Issues

**API not responding:**
```bash
# Check logs
docker-compose logs api

# Restart
docker-compose restart api
```

**Database connection error:**
```bash
# Check PostgreSQL
docker-compose exec postgres pg_isready

# Check connection
docker-compose exec api python -c "from airs.db.session import engine; print(engine.url)"
```

**Airflow DAG not running:**
```bash
# Check scheduler logs
docker-compose logs airflow-scheduler

# Trigger manually
docker-compose exec airflow-scheduler airflow dags trigger airs_daily_pipeline
```

**Model not loading:**
```bash
# Check MLflow
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Verify model exists
docker-compose exec api python -c "import mlflow; print(mlflow.list_registered_models())"
```

## Disaster Recovery

### Backup Checklist

- [ ] Daily PostgreSQL backups
- [ ] MLflow artifact backup
- [ ] Configuration backup

### Recovery Steps

1. Provision new infrastructure
2. Restore PostgreSQL from backup
3. Restore MLflow artifacts
4. Deploy application
5. Verify health checks
6. Resume Airflow DAGs

## Maintenance

### Regular Tasks

| Task | Frequency |
|------|-----------|
| Database vacuum | Weekly |
| Log rotation | Daily |
| Security patches | Monthly |
| Dependency updates | Quarterly |

### Upgrade Process

1. Create database backup
2. Deploy to staging
3. Run integration tests
4. Deploy to production (rolling)
5. Monitor for issues
6. Rollback if needed
