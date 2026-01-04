#!/bin/bash
set -e

# Create build directory
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Run CMake
echo "Configuring with CMake..."
cmake ..

# Build
echo "Building..."
make test_zenith_reconstruction_512

# Run
echo "Running Zenith 512x512 Reconstruction Test..."
./test_zenith_reconstruction_512

cd ..
echo "Done. Output images should be in the 'build' directory."
