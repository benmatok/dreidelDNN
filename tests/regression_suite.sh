#!/bin/bash
# set -e

OUTPUT_FILE="test_outputs.txt"

echo "Running Regression Suite..." > "$OUTPUT_FILE"
echo "Date: $(date)" >> "$OUTPUT_FILE"
echo "--------------------------------------------------" >> "$OUTPUT_FILE"

# Automatically find and run all C++ tests
for test_file in tests/*.cpp; do
    test_name=$(basename "$test_file" .cpp)
    echo "Running Test $test_name..." | tee -a "$OUTPUT_FILE"

    # Log Resources
    echo "Disk Usage:" >> "$OUTPUT_FILE"
    df -h | grep overlay >> "$OUTPUT_FILE"
    echo "Memory Usage:" >> "$OUTPUT_FILE"
    free -h >> "$OUTPUT_FILE"

    echo "Compiling $test_file..." >> "$OUTPUT_FILE"
    g++ -std=c++17 -O3 -fopenmp -mavx2 -mfma -Iinclude "$test_file" -o "tests/$test_name" >> "$OUTPUT_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "Running tests/$test_name..." >> "$OUTPUT_FILE"
        ./tests/$test_name >> "$OUTPUT_FILE" 2>&1
        RET_CODE=$?
        if [ $RET_CODE -eq 0 ]; then
            echo "[PASS] Test $test_name" | tee -a "$OUTPUT_FILE"
        else
            echo "[FAIL] Test $test_name" | tee -a "$OUTPUT_FILE"
        fi
    else
        echo "[FAIL] Compilation of $test_name" | tee -a "$OUTPUT_FILE"
    fi
    echo "--------------------------------------------------" >> "$OUTPUT_FILE"
done

# Critical Benchmarks
echo "Running Critical Benchmarks..." | tee -a "$OUTPUT_FILE"
BENCHMARKS=("benchmarks/benchmark_zenith_ablation.cpp" "benchmarks/benchmark_zenith_restoration.cpp")

for bench_file in "${BENCHMARKS[@]}"; do
    bench_name=$(basename "$bench_file" .cpp)
    echo "Running Benchmark $bench_name..." | tee -a "$OUTPUT_FILE"

    echo "Compiling $bench_file..." >> "$OUTPUT_FILE"
    g++ -std=c++17 -O3 -fopenmp -mavx2 -mfma -Iinclude "$bench_file" -o "$bench_name" >> "$OUTPUT_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "Executing $bench_name..." >> "$OUTPUT_FILE"
        ./"$bench_name" >> "$OUTPUT_FILE" 2>&1
        RET_CODE=$?
        if [ $RET_CODE -eq 0 ]; then
             echo "[PASS] Benchmark $bench_name" | tee -a "$OUTPUT_FILE"
             rm "$bench_name" # Cleanup
        else
             echo "[FAIL] Benchmark $bench_name" | tee -a "$OUTPUT_FILE"
        fi
    else
        echo "[FAIL] Compilation of $bench_name" | tee -a "$OUTPUT_FILE"
    fi
    echo "--------------------------------------------------" >> "$OUTPUT_FILE"
done

# Python Tests
echo "Running Python Tests..." | tee -a "$OUTPUT_FILE"
if [ -f "test_expressivity.py" ]; then
    python3 test_expressivity.py >> "$OUTPUT_FILE" 2>&1
    if [ $? -eq 0 ]; then
        echo "[PASS] test_expressivity.py" | tee -a "$OUTPUT_FILE"
    else
        echo "[FAIL] test_expressivity.py" | tee -a "$OUTPUT_FILE"
    fi
fi
echo "--------------------------------------------------" >> "$OUTPUT_FILE"

echo "Regression Suite Completed." | tee -a "$OUTPUT_FILE"
