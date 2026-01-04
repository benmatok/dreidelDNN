cmake_minimum_required(VERSION 3.10)
project(dreidelDNN CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -mavx2 -mfma -Wall -Wextra)
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Include directories
include_directories(include)

# Helper function to add examples
function(add_dreidel_example name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
endfunction()

# Examples
add_dreidel_example(test_zenith_reconstruction_512 examples/test_zenith_reconstruction_512.cpp)
add_dreidel_example(train_wavelet_autoencoder examples/train_wavelet_autoencoder.cpp)
add_dreidel_example(train_zenith_autoencoder examples/train_zenith_autoencoder.cpp)
add_dreidel_example(train_comparative_ae examples/train_comparative_ae.cpp)

# Tests (Optional, can be added similarly)
file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
endforeach()
