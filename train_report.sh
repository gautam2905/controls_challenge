#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm

# 1. Run temp.py and WAIT for it to finish completely
# (We removed the '&' and the 'sleep')
echo "Running temp.py..."
python3 temp.py 

# 2. Only run eval.py if temp.py succeeded
if [ $? -eq 0 ]; then
    echo "temp.py finished successfully. Starting evaluation..."
    python eval.py \
        --model_path ./models/tinyphysics.onnx \
        --data_path ./data \
        --num_segs 5000 \
        --test_controller controller \
        --baseline_controller pid
else
    echo "temp.py failed! Skipping evaluation."
fi