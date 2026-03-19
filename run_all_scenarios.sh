#!/bin/bash
echo "=========================================="
echo "批量评估 V3 Final (改进版) - 场景 4-13"
echo "=========================================="

for s in 4 5 6 7 8 9 10 11 12 13; do
    echo ""
    echo ">>> 评估场景 $s ..."
    python main_improved_v3_final.py --test_scenarios $s 2>&1 | grep -E "(AUC|Precision|Recall|F1-Score|智能阈值|Optimal Thresh|估计异常|中位数分离|分布重叠|决策)"
    echo "---"
done
