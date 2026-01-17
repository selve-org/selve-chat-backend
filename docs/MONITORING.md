# Health Monitoring & CI/CD

This directory contains comprehensive health checks and CI/CD workflows to prevent issues like package version mismatches, broken imports, and service degradation.

## GitHub Actions Workflows

### 1. Health & Compatibility Check (`.github/workflows/health-check.yml`)

**Runs on:**
- Every push to `main` or `develop`
- Every pull request
- Daily at 2 AM UTC (catches drift)

**What it checks:**
- ✅ **Dependency Compatibility**: Ensures all packages are compatible
  - Validates Qdrant client version (must be ≥1.12.0 for server 1.16.x)
  - Checks for security vulnerabilities with `pip-audit`
  - Detects conflicting dependencies
  
- ✅ **Syntax & Imports**: Validates Python code
  - Compiles all Python files
  - Tests critical service imports
  - Catches syntax errors before deployment
  
- ✅ **Integration Tests**: Mocked service testing
  - Spins up Qdrant container (v1.16.3)
  - Tests RAG service initialization
  - Validates service connectivity
  - Runs pytest suite
  
- ✅ **Tool & Agent Validation**: Ensures agentic tools work
  - Validates function definitions
  - Checks thinking engine integrity
  - Ensures all agent methods exist
  
- ✅ **Production Health** (main branch only): Live service checks
  - Checks `api.selve.me` (main backend)
  - Checks `api-chat.selve.me` (chat backend)
  - **Verifies Qdrant connection status**

### 2. Schema Sync Check (`.github/workflows/schema-sync.yml`)

Ensures Prisma schema stays in sync between `selve` and `selve-chat-backend` repositories.

## Local Health Monitor Script

### `scripts/health_monitor.py`

Run comprehensive health checks locally or in production:

```bash
# Check all services
python scripts/health_monitor.py

# Check specific service
python scripts/health_monitor.py --check qdrant
python scripts/health_monitor.py --check chat

# Verbose output with details
python scripts/health_monitor.py -v

# Exit with error on failures (for CI)
python scripts/health_monitor.py --alert
```

**Checks:**
- `selve-backend` - Main API health + database connection
- `selve-chat-backend` - Chat API health + **Qdrant connection**
- `qdrant` - Direct Qdrant connection, collection counts
- `openai` - OpenAI API connectivity

## Key Protection: Qdrant Version Mismatch

The incident that prompted this:
- **Problem**: `qdrant-client` 1.7.3 was incompatible with Qdrant server 1.16.3
- **Symptom**: Health check always showed "degraded", Pydantic validation errors
- **Impact**: RAG retrieval failed silently, chatbot responses lacked knowledge grounding

**How we prevent this now:**

1. **requirements.txt** now specifies compatible range:
   ```
   qdrant-client>=1.12.0,<2.0.0
   ```

2. **CI validates Qdrant client version**:
   - Fails if installed version < 1.12.0
   - Checks compatibility on every commit

3. **Integration tests** spin up Qdrant container:
   - Tests actual connectivity
   - Validates RAG service initialization
   - Catches breaking changes early

4. **Production health checks** verify:
   - Qdrant connection status in API response
   - Fails the build if Qdrant is disconnected

## Running Checks Locally

Before pushing code:

```bash
# 1. Check Python syntax
cd selve-chat-backend
find app -name "*.py" -exec python -m py_compile {} \;

# 2. Test critical imports
python -c "from app.services.rag_service import RAGService; print('✅ RAG OK')"
python -c "from app.services.thinking_engine import ThinkingEngine; print('✅ Engine OK')"

# 3. Check dependencies
pip install pip-audit pip-check
pip-audit --desc
pip-check

# 4. Run health monitor
python scripts/health_monitor.py -v

# 5. Run tests
pytest tests/ -v
```

## Setting Up GitHub Secrets

For production health checks to work, set these secrets in your GitHub repo:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add secrets:
   - `SELVE_PAT` - Personal access token for cross-repo checks (optional, uses GITHUB_TOKEN otherwise)

## Daily Monitoring

The workflow runs daily at 2 AM UTC to catch:
- Dependency drift
- Security vulnerabilities
- Production service degradation
- Package updates that break compatibility

Check the **Actions** tab in GitHub to see daily reports.

## Alert on Failures

Failed checks will:
- ❌ Block PR merges (required status check)
- 📧 Send email notifications to committers
- 📊 Show in GitHub Actions dashboard
- 🔴 Mark build as failed

## What Gets Tested

| Component | Test Coverage |
|-----------|--------------|
| Package versions | ✅ Version compatibility, security audit |
| Python syntax | ✅ All `.py` files compiled |
| Critical imports | ✅ Core services, tools, agents |
| Qdrant connectivity | ✅ Direct connection + health endpoint |
| RAG service | ✅ Initialization + collection access |
| Tool definitions | ✅ JSON schema validation |
| Thinking engine | ✅ Method existence, initialization |
| Production APIs | ✅ Live health checks (main branch) |
| Database schema | ✅ Prisma validation + sync check |

## Adding New Checks

To add a new health check:

1. **For services**: Add to `scripts/health_monitor.py`:
   ```python
   def check_new_service(self) -> HealthStatus:
       # Your check logic
       return HealthStatus(...)
   ```

2. **For CI**: Add job to `.github/workflows/health-check.yml`:
   ```yaml
   new-check:
     runs-on: ubuntu-latest
     name: New Check
     steps:
       - name: Run check
         run: |
           # Your check commands
   ```

3. **Update summary** job to include new check in `needs:` array

## Troubleshooting

**Q: CI fails with "Qdrant connection timeout"**  
A: The Qdrant container might not be ready. Increase the wait time in the workflow.

**Q: Health check passes but production is degraded**  
A: Run `python scripts/health_monitor.py --check all -v` on the server to see detailed status.

**Q: How do I skip checks temporarily?**  
A: Add `[skip ci]` to your commit message. **Not recommended for production branches.**

## Monitoring Dashboard

View all health check results:
- **GitHub Actions**: `https://github.com/selve-org/selve-chat-backend/actions`
- **Production health**: Run health monitor script on server

## Cost Optimization

- Workflows use caching for pip dependencies (faster, cheaper)
- Integration tests use lightweight Qdrant container
- Daily checks run during low-traffic hours
- Production checks only on `main` branch pushes
