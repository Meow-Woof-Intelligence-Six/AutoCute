#!/bin/bash
cd /app/code/src
echo "Starting Stage 5"
python stage51_make_top10_updown_results.py
python stage52_predict_price_change.py
python stage55_make_simple.py

# python stage54_make_results.py
# python stage59.py

echo "Starting Stage 6"
python stage62_super_ensemble.py