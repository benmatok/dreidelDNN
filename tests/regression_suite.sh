#!/bin/bash
set -e

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
    g++ -std=c++17 -O3 -fopenmp -Iinclude "$test_file" -o "tests/$test_name" >> "$OUTPUT_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "Running tests/$test_name..." >> "$OUTPUT_FILE"
        ./tests/$test_name >> "$OUTPUT_FILE" 2>&1
        if [ $? -eq 0 ]; then
            echo "[PASS] Test $test_name" | tee -a "$OUTPUT_FILE"
        else
            echo "[FAIL] Test $test_name" | tee -a "$OUTPUT_FILE"
        fi
    else
        echo "[FAIL] Compilation of $test_name" | tee -a "$OUTPUT_FILE"
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
