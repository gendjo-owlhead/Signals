---
description: Restart both Backend and Frontend components together
---

This workflow ensures that both the Backend (FastAPI) and Frontend (Vite) are restarted together to maintain synchronization, as required by project rules.

1. **Restart Backend**
   ```bash
   # Kill existing backend process
   lsof -ti:8000 | xargs kill -9 2>/dev/null
   
   # Wait for port to clear
   sleep 2
   
   # Start Backend
   cd ../backend
   source venv/bin/activate
   python main.py &
   ```

2. **Wait for Backend Health**
   ```bash
   sleep 3
   curl -s http://localhost:8000/
   ```

3. **Restart Frontend**
   ```bash
   # Kill existing frontend process and lingering ports
   lsof -ti:3000 | xargs kill -9 2>/dev/null
   
   # Start Frontend on port 3000
   cd ../frontend
   npm run dev -- --port 3000
   ```
