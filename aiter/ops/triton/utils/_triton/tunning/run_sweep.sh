#!/bin/bash

echo "K: $K"
echo "N: $N"
echo "G: $G"

# Array of M values to iterate over
# M_VALUES=(8 16 32)
M_VALUES=(4 8 16)

# Loop over each M value
for M in "${M_VALUES[@]}"; do
    echo "Running with M=$M..."
    export M=$M
    python3 screen.py $M $N $K $G ut_afp4wfp4_gemm_preshuffle.py \
        --block-size-k-range 256 512 1024 \
        --block-size-n-range 16 32 64 128 \
        --num-warps-range 2 4 8 \
        --waves-per-eu-range 1 2 4 6 \
        --overwrite --verbose > example_M${M}_K${K}_N${N}.out
    echo "Completed M=$M (output: example3_M${M}_K${K}_N${N}.out)"
done

echo "All runs completed!"
