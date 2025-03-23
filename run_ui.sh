#!/bin/bash

# Function to stop all processes on exit
cleanup() {
    echo "Stopping all processes..."
    kill $API_PID $UI_PID 2>/dev/null
    exit
}

# Set trap for cleanup on script exit
trap cleanup EXIT INT TERM

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -r api/requirements.txt
else
    source venv/bin/activate
fi

# Check for frontend dependencies
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Create necessary directories if they don't exist
mkdir -p data
mkdir -p logs
mkdir -p order_logs

# Start the API server
echo "Starting API server..."
cd api
python main.py &
API_PID=$!
cd ..

# Wait a bit for the API to start
sleep 2

# Check if the API server started successfully
if ! kill -0 $API_PID 2>/dev/null; then
    echo "Failed to start API server. Exiting."
    exit 1
fi

# Start the Next.js frontend
echo "Starting Next.js frontend..."
cd frontend
npm run dev &
UI_PID=$!
cd ..

echo "------------------------------------------------------------"
echo "🚀 Trading Bot UI is now running!"
echo "API: http://localhost:8000"
echo "UI:  http://localhost:3000"
echo "------------------------------------------------------------"
echo "Press Ctrl+C to stop the servers"

# Wait for both processes to complete
wait $API_PID $UI_PID 