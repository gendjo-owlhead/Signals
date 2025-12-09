#!/bin/bash

echo "🛑 MAX POWER: Stopping all services..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

echo "🚀 Starting Backend (Port 8000)..."
cd backend
source venv/bin/activate
python main.py > ../backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

echo "🎨 Starting Frontend (Port 3000)..."
cd ../frontend
npm run dev -- --port 3000 > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

echo "✅ All systems go! Monitoring logs..."
echo "-----------------------------------"
tail -f ../backend.log ../frontend.log
