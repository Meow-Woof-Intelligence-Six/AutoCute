#!/bin/bash
cd /app/code/src
echo "Running AutoCute training pipeline..."
echo "Current directory: $(pwd)"

echo "Starting Stage 0"
python stage00_csv2qlib.py
python stage01_qlib2alpha158.py
python stage02_make_labels.py
python stage04_make_test1_set_competition.py

echo "Starting Stage 1"
python stage11_add_stock_info.py
python stage12_add_financial_info.py

# 检查无误

echo "Starting Stage 2"
python stage20_distribution.py
python stage21_make_lag_data.py
python stage22_add_timestamp_features.py
python stage23_select_features.py

echo "Starting Stage 3"
python stage30_train_test_split.py
python stage31_get_vetted_data.py
python stage35_simple_price_data.py

echo "Starting Stage 4"
python stage40_train_updown.py
python stage41_train_top10.py
python stage42_train_price_change.py
python stage43_train_price.py
python stage44_train_price_agts_best_quality.py
python stage45_train_price_agts_single_best.py
python stage45_train_price_agts_stats.py
python stage49_infras.py

