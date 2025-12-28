#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include "dreidel/jules/Agent.hpp"
#include "dreidel/core/Arena.hpp"

// Define static memory
// 2MB Workspace
static uint8_t NETWORK_WORKSPACE[2 * 1024 * 1024];

// Inputs/Outputs
static float SENSOR_INPUT[1024];
static float ACTUATOR_OUTPUT[16];

// 1. Initialize Arena
static dreidel::core::Arena arena(NETWORK_WORKSPACE, sizeof(NETWORK_WORKSPACE));

// 2. Instantiate Agent Globally (Data Segment, not Stack)
// Input: 1024, Hidden: 1024, Output: 16
// This avoids stack overflow from large weight arrays in SparseBlock (~16MB)
static dreidel::jules::AgentJules<1024, 1024, 16> agent(&arena);

int main() {
    std::cout << "Initializing Project Jules (Alien Architecture)..." << std::endl;

    // 3. Initialize Agent
    agent.init();

    // 4. Mock Inputs
    for(int i=0; i<1024; ++i) SENSOR_INPUT[i] = std::sin(i * 0.01f);

    // 5. Run Control Loop Benchmark
    int steps = 1000;
    std::cout << "Running " << steps << " steps..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for(int i=0; i<steps; ++i) {
        agent.step(SENSOR_INPUT, ACTUATOR_OUTPUT);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Completed in " << diff.count() << " s." << std::endl;
    std::cout << "Frequency: " << steps / diff.count() << " Hz" << std::endl;

    // Verify Output
    float sum = 0;
    for(int i=0; i<16; ++i) sum += std::abs(ACTUATOR_OUTPUT[i]);
    std::cout << "Output checksum: " << sum << std::endl;

    return 0;
}
