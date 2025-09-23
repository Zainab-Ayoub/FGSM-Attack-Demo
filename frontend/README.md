### Frontend setup

1. Copy `.env.local.example` to `.env.local` and set:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```
2. Install deps:
```
npm install
```
3. Start dev server:
```
npm run dev
```

When deploying as a Static Site (Render):
- Build Command: `npm ci --include=dev && npm run build`
- Publish Directory: `out`
- Env var: `NEXT_PUBLIC_API_URL=https://fgsmbackend.onrender.com`



