# Running Zenith Tests

This document describes how to build and run the Zenith reconstruction tests.

## Prerequisites

- C++17 compliant compiler (GCC 7+ recommended)
- CMake 3.10+
- OpenMP (optional but recommended for performance)

## Quick Start (Bash Script)

To quickly build and run the 512x512 reconstruction test:

```bash
./run_zenith_512.sh
```

This script will:
1. Create a `build` directory.
2. Configure the project using CMake.
3. Compile the `test_zenith_reconstruction_512` executable.
4. Run the test.

Output images (`test_512_input.png`, `test_512_target.png`, `test_512_output.png`) will be generated in the `build` directory.

## Manual Build with CMake

You can also build manually:

```bash
mkdir build
cd build
cmake ..
make
```

### Running Specific Tests

After building, you can run any of the compiled executables:

```bash
# Run the 512x512 reconstruction test
./test_zenith_reconstruction_512

# Run the comparative autoencoder training
./train_comparative_ae

# Run regression tests
cd ..
./tests/regression_suite.sh
```
