#!/bin/bash

echo "========================================="
echo "Starting Streamlit App..."
echo "========================================="

cd /home/site/wwwroot

source /home/site/wwwroot/.venv/bin/activate

python3 -m streamlit run lmk_final_dashboard_final.py \
    --server.port 8000 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
