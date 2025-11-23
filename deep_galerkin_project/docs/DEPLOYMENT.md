# Cloud Deployment Guide

This guide covers deploying the Deep Galerkin Pricing system to major cloud platforms.

## 📋 Prerequisites

- Trained model checkpoint (`checkpoints/bs_european/best_model.pt`)
- Docker installed locally (for testing)
- Cloud platform account (AWS, GCP, or Azure)
- Domain name (optional, for production)

---

## 🐳 Docker Deployment (All Platforms)

### Build and Test Locally

```bash
# Build Docker image
docker build -t dgm-pricing:latest .

# Test locally
docker run -p 8000:8000 dgm-pricing:latest

# Visit http://localhost:8000/docs to verify
```

---

## ☁️ AWS Deployment

### Option 1: AWS Elastic Beanstalk (Easiest)

**1. Install AWS CLI and EB CLI:**
```bash
pip install awscli awsebcli
aws configure
```

**2. Initialize Elastic Beanstalk:**
```bash
eb init -p docker dgm-pricing --region us-east-1
```

**3. Create environment and deploy:**
```bash
eb create dgm-pricing-prod
eb open
```

**4. Update deployment:**
```bash
eb deploy
```

**Cost**: ~$25-50/month (t2.micro instance)

### Option 2: AWS ECS (Production)

**1. Create ECR repository:**
```bash
aws ecr create-repository --repository-name dgm-pricing
```

**2. Build and push image:**
```bash
# Get login credentials
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag dgm-pricing:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/dgm-pricing:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/dgm-pricing:latest
```

**3. Create ECS task definition:**
```json
{
  "family": "dgm-pricing",
  "containerDefinitions": [
    {
      "name": "dgm-api",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/dgm-pricing:latest",
      "memory": 2048,
      "cpu": 1024,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ]
    }
  ]
}
```

**4. Create ECS service with load balancer**

**Cost**: ~$70-120/month (Fargate)

### Option 3: AWS Lambda (Serverless)

**1. Create Lambda deployment package:**
```bash
# Install dependencies
pip install -t package -r requirements.txt

# Create deployment package
cd package
zip -r ../lambda_function.zip .
cd ..
zip -g lambda_function.zip api/main.py dgmlib/
```

**2. Create Lambda function:**
```bash
aws lambda create-function \
  --function-name dgm-pricing \
  --runtime python3.10 \
  --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-execution-role \
  --handler api.main.handler \
  --zip-file fileb://lambda_function.zip \
  --memory-size 3008 \
  --timeout 60
```

**3. Set up API Gateway:**
Use AWS Console to create REST API → Lambda proxy integration

**Cost**: Pay per request (~$0-10/month for low traffic)

---

## 🌐 Google Cloud Platform (GCP)

### Option 1: Cloud Run (Recommended)

**1. Install gcloud CLI:**
```bash
# Follow: https://cloud.google.com/sdk/docs/install
gcloud init
```

**2. Build and deploy:**
```bash
# Enable required APIs
gcloud services enable run.googleapis.com containerregistry.googleapis.com

# Build image with Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/dgm-pricing

# Deploy to Cloud Run
gcloud run deploy dgm-pricing \
  --image gcr.io/YOUR_PROJECT_ID/dgm-pricing \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated
```

**Cost**: ~$10-30/month (pay per use)

### Option 2: Google Kubernetes Engine (GKE)

**1. Create GKE cluster:**
```bash
gcloud container clusters create dgm-cluster \
  --num-nodes 2 \
  --machine-type e2-medium \
  --region us-central1
```

**2. Deploy with Kubernetes:**
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dgm-pricing
spec:
  replicas: 2
  selector:
    matchLabels:
      app: dgm-pricing
  template:
    metadata:
      labels:
        app: dgm-pricing
    spec:
      containers:
      - name: dgm-api
        image: gcr.io/YOUR_PROJECT_ID/dgm-pricing:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: dgm-pricing-service
spec:
  type: LoadBalancer
  selector:
    app: dgm-pricing
  ports:
  - port: 80
    targetPort: 8000
```

```bash
kubectl apply -f deployment.yaml
```

**Cost**: ~$150-250/month

---

## 🔷 Microsoft Azure

### Option 1: Azure Container Instances (Simplest)

**1. Install Azure CLI:**
```bash
# Follow: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
az login
```

**2. Create resource group:**
```bash
az group create --name dgm-pricing-rg --location eastus
```

**3. Create container registry:**
```bash
az acr create --resource-group dgm-pricing-rg \
  --name dgmpricing --sku Basic
```

**4. Build and push:**
```bash
az acr build --registry dgmpricing --image dgm-pricing:latest .
```

**5. Deploy container:**
```bash
az container create \
  --resource-group dgm-pricing-rg \
  --name dgm-pricing \
  --image dgmpricing.azurecr.io/dgm-pricing:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --dns-name-label dgm-pricing-api
```

**Cost**: ~$40-80/month

### Option 2: Azure App Service

**1. Create App Service plan:**
```bash
az appservice plan create \
  --name dgm-pricing-plan \
  --resource-group dgm-pricing-rg \
  --is-linux \
  --sku B1
```

**2. Deploy web app:**
```bash
az webapp create \
  --resource-group dgm-pricing-rg \
  --plan dgm-pricing-plan \
  --name dgm-pricing-api \
  --deployment-container-image-name dgmpricing.azurecr.io/dgm-pricing:latest
```

**Cost**: ~$15-60/month

---

## 🔒 Security Best Practices

### 1. Environment Variables
Never hardcode secrets. Use environment variables:

```python
# In your code
import os

SECRET_KEY = os.getenv("SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
```

**AWS**: Systems Manager Parameter Store or Secrets Manager
**GCP**: Secret Manager
**Azure**: Key Vault

### 2. API Authentication

Add API key authentication:

```python
# api/main.py
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Use in endpoints
@app.post("/price/european")
async def price_european(request: PriceRequest, api_key: str = Depends(verify_api_key)):
    # ...
```

### 3. Rate Limiting

```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/price/european")
@limiter.limit("100/minute")
async def price_european(request: Request, ...):
    # ...
```

### 4. HTTPS/SSL

**AWS**: Use ACM (Amazon Certificate Manager)
**GCP**: Cloud Run provides automatic HTTPS
**Azure**: App Service automatic HTTPS

---

## 📊 Monitoring & Logging

### Application Monitoring

```python
# Add health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }
```

### Cloud-Specific Monitoring

**AWS CloudWatch**:
```bash
# Enable logging
aws logs create-log-group --log-group-name /aws/dgm-pricing
```

**GCP Cloud Monitoring**:
```bash
# Automatic with Cloud Run
gcloud logging read "resource.type=cloud_run_revision" --limit 50
```

**Azure Application Insights**:
```bash
az monitor app-insights component create \
  --app dgm-pricing \
  --resource-group dgm-pricing-rg
```

---

## 🚀 Production Checklist

- [ ] Environment variables configured
- [ ] API authentication enabled
- [ ] Rate limiting configured
- [ ] HTTPS/SSL enabled
- [ ] Monitoring and alerting set up
- [ ] Backup strategy for model checkpoints
- [ ] Auto-scaling configured
- [ ] Health checks implemented
- [ ] Error tracking (e.g., Sentry)
- [ ] Load testing completed
- [ ] Documentation updated with API endpoint
- [ ] Cost monitoring alerts enabled

---

## 💰 Cost Comparison

| Platform | Option | Monthly Cost | Best For |
|----------|--------|--------------|----------|
| **AWS** | Lambda | $0-10 | Sporadic usage |
| **AWS** | Elastic Beanstalk | $25-50 | Getting started |
| **AWS** | ECS Fargate | $70-120 | Production |
| **GCP** | Cloud Run | $10-30 | Cost-effective prod |
| **GCP** | GKE | $150-250 | Enterprise scale |
| **Azure** | Container Instances | $40-80 | Simple deployment |
| **Azure** | App Service | $15-60 | Managed service |

---

## 🧪 Testing Production Deployment

```bash
# Test health endpoint
curl https://your-api.com/health

# Test pricing endpoint
curl -X POST "https://your-api.com/price/european" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "S": 100,
    "K": 100,
    "r": 0.05,
    "sigma": 0.2,
    "T": 1.0,
    "option_type": "call",
    "method": "dgm"
  }'

# Load test
pip install locust
locust -f tests/load_test.py --host https://your-api.com
```

---

## 🆘 Troubleshooting

### Issue: Out of Memory
**Solution**: Increase memory allocation or use smaller batch sizes

### Issue: Cold start latency
**Solution**: Use provisioned concurrency (AWS Lambda) or min instances (GCP Cloud Run)

### Issue: Model file too large
**Solution**: Store model in cloud storage (S3/GCS/Blob) and download on startup

### Issue: Slow inference
**Solution**: Enable GPU instances or use model quantization

---

## 📚 Additional Resources

- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [GCP Best Practices](https://cloud.google.com/architecture/framework)
- [Azure Architecture Center](https://docs.microsoft.com/en-us/azure/architecture/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

**Need help?** Open an issue on [GitHub](https://github.com/atharvajoshi01/deep-galerkin-pricing/issues) or contact the maintainers.
