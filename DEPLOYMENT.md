# Deployment Guide

This guide covers deploying Stoic Emperor to production using Railway or Fly.io with PostgreSQL + pgvector.

## Prerequisites

- PostgreSQL database with pgvector extension (Supabase, Neon, or Railway PostgreSQL)
- OpenAI API key (and optionally Anthropic API key for Aegean consensus)
- Supabase Auth configured (for production authentication)

## Option A: Railway Deployment

Railway provides the simplest deployment with automatic PostgreSQL + pgvector.

### 1. Install Railway CLI

```bash
npm install -g @railway/cli
railway login
```

### 2. Create New Project

```bash
railway init
```

### 3. Add PostgreSQL

In Railway dashboard:
- Click "New" → "Database" → "PostgreSQL"
- Railway will automatically create `DATABASE_URL` variable

### 4. Enable pgvector Extension

Connect to your Railway PostgreSQL:

```bash
railway connect PostgreSQL
```

Then run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 5. Set Environment Variables

```bash
railway variables set ENVIRONMENT=production
railway variables set OPENAI_API_KEY=your-key
railway variables set ANTHROPIC_API_KEY=your-key
railway variables set LLM_MAIN_MODEL=gpt-4o
railway variables set LLM_REVIEWER_MODEL=claude-3-5-sonnet
railway variables set SUPABASE_JWT_SECRET=your-secret
railway variables set SUPABASE_URL=https://your-project.supabase.co
railway variables set SUPABASE_ANON_KEY=your-anon-key
```

### 6. Deploy

```bash
railway up
```

Railway will:
- Build your Docker image
- Deploy to production
- Provide a public URL

## Option B: Fly.io Deployment

Fly.io offers more control and edge deployment.

### 1. Install Fly CLI

```bash
curl -L https://fly.io/install.sh | sh
fly auth login
```

### 2. Create App

```bash
fly launch --no-deploy
```

This creates `fly.toml` (already included).

### 3. Create PostgreSQL Database

You have two options:

**Option 3a: Fly Postgres (self-managed)**

```bash
fly postgres create --name stoic-emperor-db
fly postgres attach stoic-emperor-db
```

Then enable pgvector:

```bash
fly postgres connect -a stoic-emperor-db
```

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Option 3b: Supabase (recommended)**

- Create project at https://supabase.com
- Get connection string from Settings → Database
- Copy the "Connection pooling" URL (port 6543)

### 4. Set Secrets

```bash
fly secrets set DATABASE_URL="postgresql://..." \
  ENVIRONMENT=production \
  OPENAI_API_KEY=your-key \
  ANTHROPIC_API_KEY=your-key \
  LLM_MAIN_MODEL=gpt-4o \
  LLM_REVIEWER_MODEL=claude-3-5-sonnet \
  SUPABASE_JWT_SECRET=your-secret \
  SUPABASE_URL=https://your-project.supabase.co \
  SUPABASE_ANON_KEY=your-anon-key
```

### 5. Deploy

```bash
fly deploy
```

Your app will be available at `https://stoic-emperor.fly.dev`

## Local Development with Docker

Test the production setup locally:

```bash
# Copy environment file
cp .env.example .env

# Add your API keys to .env

# Start services
docker-compose up

# Access at http://localhost:8000
```

## Database Migrations

The app automatically runs migrations on startup. To manually migrate:

```bash
# Railway
railway run python -c "from src.infrastructure.database import Database; Database()"

# Fly.io
fly ssh console
python -c "from src.infrastructure.database import Database; Database()"
```

## Monitoring

### Railway
- Dashboard: https://railway.app/dashboard
- Logs: `railway logs`

### Fly.io
- Dashboard: `fly dashboard`
- Logs: `fly logs`
- Metrics: `fly status`

## Scaling

### Railway
- Automatic horizontal scaling
- Upgrade plan for more resources

### Fly.io
Edit `fly.toml`:

```toml
[[vm]]
  cpu_kind = "shared"
  cpus = 2
  memory_mb = 1024
```

Then redeploy:

```bash
fly deploy
```

## Supabase Auth Setup

1. Go to https://supabase.com/dashboard
2. Create new project (or use existing)
3. Navigate to Settings → API
4. Copy:
   - URL (SUPABASE_URL)
   - anon public key (SUPABASE_ANON_KEY)
   - JWT Secret (SUPABASE_JWT_SECRET)
5. Configure authentication providers in Authentication → Providers

## Cost Estimates

### Option A: Railway + Supabase (Recommended for MVP)
- Railway: $5-20/month (usage-based)
- Supabase: Free tier (upgrade to Pro $25/month for production)
- **Total: $5-45/month**

### Option B: Fly.io + Supabase
- Fly.io: $5-30/month (1 VM)
- Supabase: Free tier (Pro $25/month)
- **Total: $5-55/month**

### Option C: Fly.io + Fly Postgres
- Fly.io app: $5-30/month
- Fly Postgres: $15-30/month
- **Total: $20-60/month**

## Troubleshooting

### Database Connection Issues

Check DATABASE_URL format:

```bash
# Railway/Fly
railway run env | grep DATABASE_URL
fly ssh console -C "env | grep DATABASE_URL"
```

### pgvector Extension Missing

```bash
# Connect to database
railway connect PostgreSQL  # or fly postgres connect

# Enable extension
CREATE EXTENSION IF NOT EXISTS vector;
```

### Authentication Errors

Verify JWT secret matches Supabase:

```bash
# Get from Supabase dashboard
Settings → API → JWT Secret

# Set in deployment
railway variables set SUPABASE_JWT_SECRET=...
fly secrets set SUPABASE_JWT_SECRET=...
```

## Zero Data Retention (ZDR)

For OpenAI Enterprise with ZDR:

1. Contact OpenAI sales for Enterprise agreement
2. Enable ZDR in your organization settings
3. Verify in API usage dashboard
4. Document compliance in your Terms of Service

## Next Steps

- Set up monitoring (Sentry, LogRocket)
- Configure custom domain
- Enable CORS for web client
- Set up CI/CD pipeline
- Add rate limiting
- Configure backup strategy
