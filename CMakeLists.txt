cmake_minimum_required(VERSION 3.10)
project(dreidelDNN CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
# Added -mf16c for F16 intrinsics
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -mavx2 -mfma -mf16c -fopenmp -Wall -Wextra)
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Include directories
include_directories(include)

# Helper function to add examples
function(add_dreidel_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
endfunction()

# --- Examples ---
add_dreidel_executable(train_zenith_nano examples/train_zenith_nano.cpp)
add_dreidel_executable(validate_zenith_nano examples/validate_zenith_nano.cpp)
add_dreidel_executable(benchmark_zenith_nano_full examples/benchmark_zenith_nano_full.cpp)
add_dreidel_executable(zenith_nano_pipelined examples/ZenithNano_Pipelined.cpp)

# --- Benchmarks ---
# Explicitly add F16 benchmark
add_dreidel_executable(benchmark_zenith_nano_f16 benchmarks/benchmark_zenith_nano_f16.cpp)

# --- Tests ---
file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_dreidel_executable(${test_name} ${test_src})
endforeach()
