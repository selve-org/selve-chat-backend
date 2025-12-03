# Clerk Satellite Authentication Testing Guide

**Status**: ✅ Implementation Complete
**Last Updated**: 2025-12-02

## Overview

This guide covers testing the Clerk satellite domain authentication setup for chat.selve.me. The implementation uses Clerk's satellite domain feature with a "toll gate" pattern to ensure only legitimate SELVE users can access the chatbot.

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│              Clerk Development Instance                          │
│         (pretty-boxer-70.clerk.accounts.dev)                     │
└─────────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
       PRIMARY DOMAIN              SATELLITE DOMAIN
       selve.me                    chat.selve.me
  ┌────────────────┐          ┌────────────────────┐
  │   Auth UI      │          │  No Auth UI        │
  │   Port 3000    │          │  Port 4000         │
  └────────────────┘          └────────────────────┘
          │                           │
          ▼                           ▼
  ┌────────────────┐          ┌────────────────────┐
  │ SELVE Backend  │◄─────────│ Chatbot Backend    │
  │ Port 8000      │  Toll    │ Port 9000          │
  │                │  Gate    │                    │
  └────────────────┘          └────────────────────┘
```

## Implementation Status

✅ **Completed**:
- [x] Frontend satellite configuration (selve-chat-frontend/.env.local)
- [x] Frontend ClerkProvider setup (app/layout.tsx)
- [x] Frontend middleware (middleware.ts)
- [x] Backend toll gate authentication (app/core/auth.py)
- [x] Main SELVE toll gate endpoint (selve/backend/app/api/routes/users.py)
- [x] Environment variables configured
- [x] JWKS public key added
- [x] Syntax validation (all files compile successfully)
- [x] Import validation (auth module imports successfully)

⏳ **Pending**:
- [ ] Full integration testing with all 4 services running
- [ ] Authentication flow testing
- [ ] Toll gate security testing

## Prerequisites

### 1. Environment Variables

Verify all environment variables are set:

**selve-chat-backend/.env**:
```bash
# Verify these are set
grep -E "(CLERK_PUBLIC_KEY|CLERK_DOMAIN|INTERNAL_API_SECRET|MAIN_SELVE_API_URL)" .env
```

**selve/backend/.env**:
```bash
# Verify internal secret is set
grep "INTERNAL_API_SECRET" .env
```

**selve-chat-frontend/.env.local**:
```bash
# Verify satellite configuration
grep -E "(CLERK_IS_SATELLITE|CLERK_DOMAIN|CLERK_SIGN_IN_URL)" .env.local
```

### 2. Dependencies

**Chatbot backend dependencies**:
```bash
cd /home/chris/selve-org/selve-chat-backend
source venv/bin/activate
pip list | grep -E "(python-jose|httpx|fastapi)"
```

**Expected output**:
- `python-jose 3.3.0`
- `httpx 0.26.0`
- `fastapi 0.109.0`

### 3. Database

Ensure the main SELVE database is accessible and has user data:
```bash
cd /home/chris/selve-org/selve/backend
# Check database connection
venv/bin/python -c "from app.db import prisma; import asyncio; asyncio.run(prisma.connect()); print('✓ Database connected')"
```

## Testing Steps

### Step 1: Start All Services

You need 4 services running simultaneously on different ports.

#### Terminal 1: Main SELVE Backend (Port 8000)

```bash
cd /home/chris/selve-org/selve/backend
source venv/bin/activate
uvicorn app.main:app --port 8000 --reload
```

**Expected output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Application startup complete.
```

**Verify**:
```bash
curl http://localhost:8000/docs
# Should return HTML with title containing "SELVE"
```

#### Terminal 2: Main SELVE Frontend (Port 3000)

```bash
cd /home/chris/selve-org/selve/frontend
pnpm dev
```

**Expected output**:
```
▲ Next.js running on http://localhost:3000
```

**Verify**:
```bash
curl -I http://localhost:3000
# Should return 200 OK
```

#### Terminal 3: Chatbot Backend (Port 9000)

**IMPORTANT**: The chatbot backend must run on port 9000 (not 8000).

```bash
cd /home/chris/selve-org/selve-chat-backend
source venv/bin/activate
uvicorn app.main:app --port 9000 --reload
```

**Expected output**:
```
INFO:     Uvicorn running on http://127.0.0.1:9000 (Press CTRL+C to quit)
```

**Verify**:
```bash
curl http://localhost:9000/docs
# Should return HTML with title "SELVE Chatbot API"
```

#### Terminal 4: Chatbot Frontend (Port 4000)

**IMPORTANT**: Next.js defaults to port 3000, so you must specify port 4000.

```bash
cd /home/chris/selve-org/selve-chat-frontend
pnpm dev -p 4000
```

**Expected output**:
```
▲ Next.js running on http://localhost:4000
```

**Verify**:
```bash
curl -I http://localhost:4000
# Should return 200 OK
```

#### Verify All Ports

```bash
netstat -tuln | grep -E ":(3000|4000|8000|9000)" | grep LISTEN
```

**Expected output**:
```
tcp  0.0.0.0:3000  LISTEN  # Main SELVE frontend
tcp  0.0.0.0:4000  LISTEN  # Chatbot frontend
tcp  127.0.0.1:8000  LISTEN  # Main SELVE backend
tcp  127.0.0.1:9000  LISTEN  # Chatbot backend
```

### Step 2: Test Toll Gate Endpoint

The toll gate endpoint is the foundation of the security model. Test it first.

#### Test with Internal Secret (Should Succeed)

```bash
# Replace with actual Clerk user ID from your database
CLERK_USER_ID="user_2xxxxxxxxxxxxxxxxxxxxx"
INTERNAL_SECRET="F2c3K84IKGfg74P-TT-QhLC4mpsIvW6kB-MVItd16-Q"

curl -X GET "http://localhost:8000/api/users/verify/${CLERK_USER_ID}" \
  -H "X-Internal-Secret: ${INTERNAL_SECRET}" \
  -v
```

**Expected response** (200 OK):
```json
{
  "id": "cm4abc123...",
  "clerk_user_id": "user_2xxxxxxxxxxxxxxxxxxxxx",
  "email": "user@example.com",
  "has_completed_assessment": true,
  "created_at": "2025-11-28T12:34:56.789Z"
}
```

#### Test without Internal Secret (Should Fail)

```bash
curl -X GET "http://localhost:8000/api/users/verify/${CLERK_USER_ID}" \
  -v
```

**Expected response** (403 Forbidden):
```json
{
  "detail": "Invalid internal secret"
}
```

#### Test with Invalid Secret (Should Fail)

```bash
curl -X GET "http://localhost:8000/api/users/verify/${CLERK_USER_ID}" \
  -H "X-Internal-Secret: wrong-secret" \
  -v
```

**Expected response** (403 Forbidden):
```json
{
  "detail": "Invalid internal secret"
}
```

#### Test with Non-Existent User (Should Fail)

```bash
curl -X GET "http://localhost:8000/api/users/verify/user_nonexistent123" \
  -H "X-Internal-Secret: ${INTERNAL_SECRET}" \
  -v
```

**Expected response** (404 Not Found):
```json
{
  "detail": "User not found"
}
```

### Step 3: Test Authentication Flow

This tests the complete end-to-end authentication flow.

#### 3.1 Visit Chatbot Frontend (Unauthenticated)

1. Open browser: `http://localhost:4000`
2. **Expected behavior**: Automatic redirect to `http://localhost:3000/sign-in`
3. The URL should change to the main SELVE sign-in page

#### 3.2 Sign In on Primary Domain

1. On `http://localhost:3000/sign-in`, enter your Clerk credentials
2. Sign in successfully
3. **Expected behavior**: Redirect back to `http://localhost:4000`
4. You should see the chatbot interface

#### 3.3 Verify Session Persistence

1. While authenticated at `http://localhost:4000`:
   - Open new tab: `http://localhost:3000`
   - **Expected**: Already signed in (no redirect to sign-in)
2. Refresh `http://localhost:4000`
   - **Expected**: Still authenticated (no redirect)
3. Open DevTools → Application → Cookies
   - Look for `__clerk_db_jwt` cookie
   - Should be present on both `localhost:3000` and `localhost:4000`

#### 3.4 Test Protected Endpoint

With a valid session, test a protected chatbot endpoint:

```bash
# Get session token from browser
# 1. Open DevTools → Application → Cookies
# 2. Copy value of __clerk_db_jwt cookie
# 3. Use it in the Authorization header

TOKEN="<your-clerk-jwt-token>"

curl -X POST "http://localhost:9000/api/chat" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}' \
  -v
```

**Expected behavior**:
- Chatbot backend verifies JWT with Clerk public key
- Chatbot backend calls toll gate endpoint on main SELVE backend
- Request succeeds if user exists in database
- Request fails with 403 if user doesn't exist

### Step 4: Test Toll Gate Security

Test that the toll gate properly blocks unauthorized users.

#### 4.1 Test with Valid Clerk User (Not in SELVE DB)

This simulates a random Clerk user trying to access the chatbot.

**Setup**:
1. Create a new Clerk user in the development instance
2. This user should NOT exist in the main SELVE database
3. Get the JWT token for this user

**Test**:
```bash
# Use JWT from the new user
TOKEN="<new-user-jwt-token>"

curl -X POST "http://localhost:9000/api/chat" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}' \
  -v
```

**Expected response** (403 Forbidden):
```json
{
  "detail": "Access denied. Please complete your SELVE assessment first at selve.me"
}
```

#### 4.2 Test with Expired JWT

**Test**:
```bash
# Use an old/expired JWT token
TOKEN="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.expired..."

curl -X POST "http://localhost:9000/api/chat" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}' \
  -v
```

**Expected response** (401 Unauthorized):
```json
{
  "detail": "Invalid token: ..."
}
```

#### 4.3 Test without JWT

**Test**:
```bash
curl -X POST "http://localhost:9000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}' \
  -v
```

**Expected response** (401 Unauthorized):
```json
{
  "detail": "Missing authorization header"
}
```

### Step 5: Test Cross-Domain Sign Out

1. Sign in at `http://localhost:4000` (should redirect to 3000 and back)
2. Click sign out button
3. **Expected**: Signed out from both domains
4. Visit `http://localhost:3000` → should show sign-in page
5. Visit `http://localhost:4000` → should redirect to sign-in page

## Troubleshooting

### Issue: Port Already in Use

**Symptom**: `Error: address already in use :::8000`

**Solution**:
```bash
# Find process using the port
lsof -ti:8000

# Kill the process
kill -9 $(lsof -ti:8000)
```

### Issue: "Failed to verify user" Error

**Symptom**: Chatbot backend returns 500 error with "Failed to verify user"

**Possible causes**:
1. Main SELVE backend is not running
2. Main SELVE backend is not on port 8000
3. INTERNAL_API_SECRET mismatch

**Solution**:
```bash
# Verify main backend is running
curl http://localhost:8000/docs

# Verify secrets match
diff <(grep INTERNAL_API_SECRET /home/chris/selve-org/selve-chat-backend/.env) \
     <(grep INTERNAL_API_SECRET /home/chris/selve-org/selve/backend/.env)
```

### Issue: JWT Verification Fails

**Symptom**: `Invalid token: ...` error

**Possible causes**:
1. CLERK_PUBLIC_KEY is incorrect
2. JWT is expired
3. JWT is from wrong Clerk instance

**Solution**:
```bash
# Verify public key is set
grep -A 10 "CLERK_PUBLIC_KEY" /home/chris/selve-org/selve-chat-backend/.env

# Decode JWT to check issuer (requires jq)
echo "<your-jwt-token>" | cut -d. -f2 | base64 -d 2>/dev/null | jq .

# Should show issuer: "https://pretty-boxer-70.clerk.accounts.dev"
```

### Issue: Redirect Loop

**Symptom**: Browser keeps redirecting between localhost:3000 and localhost:4000

**Possible causes**:
1. `isSatellite` not set correctly
2. Clerk domain mismatch
3. Cookies not being set

**Solution**:
1. Check `.env.local`:
   ```bash
   grep CLERK_IS_SATELLITE /home/chris/selve-org/selve-chat-frontend/.env.local
   # Should output: NEXT_PUBLIC_CLERK_IS_SATELLITE=true
   ```

2. Clear cookies:
   - Open DevTools → Application → Cookies
   - Delete all Clerk cookies
   - Try signing in again

3. Check browser console for errors

### Issue: CORS Errors

**Symptom**: Browser console shows CORS errors

**Solution**:
1. Verify CORS_ORIGINS in backend .env files
2. Main SELVE backend should allow `http://localhost:4000`
3. Chatbot backend should allow `http://localhost:4000`

```bash
# Check CORS configuration
grep CORS_ORIGINS /home/chris/selve-org/selve-chat-backend/.env
# Should include: http://localhost:4000,http://localhost:3000
```

## Security Checklist

Before deploying to production:

- [ ] INTERNAL_API_SECRET is a strong random string (32+ characters)
- [ ] INTERNAL_API_SECRET is never committed to git
- [ ] CLERK_PUBLIC_KEY matches the development instance
- [ ] Toll gate endpoint requires X-Internal-Secret header
- [ ] Toll gate endpoint returns 404 for non-existent users
- [ ] JWT verification checks token expiration
- [ ] JWT verification checks token signature
- [ ] Protected endpoints use `Depends(get_current_user)`
- [ ] Rate limiting is configured on toll gate endpoint
- [ ] HTTPS is enabled in production
- [ ] CORS origins are restricted to known domains

## Next Steps

After testing is complete:

1. **Clerk Dashboard Configuration**:
   - Add `chat.selve.me` as satellite domain
   - Configure CORS allowed origins
   - Update redirect URLs for production

2. **Production Environment**:
   - Update `.env.production` files with production URLs
   - Change `localhost:3000` → `https://selve.me`
   - Change `localhost:4000` → `https://chat.selve.me`
   - Change `localhost:8000` → production API URL
   - Change `localhost:9000` → production chatbot API URL

3. **Deployment**:
   - Deploy main SELVE backend to production
   - Deploy chatbot backend to production
   - Deploy chatbot frontend to production
   - Configure DNS for `chat.selve.me`
   - Set up SSL certificates

4. **Monitoring**:
   - Set up logging for toll gate requests
   - Monitor toll gate rejection rate
   - Set up alerts for authentication failures
   - Track JWT verification errors

## Resources

- **Clerk Satellite Domains**: https://clerk.com/docs/deployments/satellite-domains
- **Clerk JWT Verification**: https://clerk.com/docs/backend-requests/handling/manual-jwt
- **FastAPI Dependencies**: https://fastapi.tiangolo.com/tutorial/dependencies/
- **Next.js Middleware**: https://nextjs.org/docs/app/building-your-application/routing/middleware

## Support

If you encounter issues not covered in this guide:

1. Check Clerk Dashboard logs
2. Check backend server logs
3. Check browser console for errors
4. Verify all environment variables are set
5. Verify all services are running on correct ports
