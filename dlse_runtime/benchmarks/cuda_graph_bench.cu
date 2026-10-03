#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

__global__ void decode_step(
    std::uint64_t* state,
    std::uint64_t* counter) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const std::uint64_t step = *counter;

        *state =
            (*state ^
             (step + 0x9e3779b97f4a7c15ULL)) *
            0xbf58476d1ce4e5b9ULL +
            0x94d049bb133111ebULL;

        *counter = step + 1;
    }
}

static void check(
    cudaError_t error,
    const char* label) {
    if (error != cudaSuccess) {
        std::cerr << label << ": "
                  << cudaGetErrorString(error)
                  << "\n";
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    const int iterations =
        argc > 1
            ? std::max(100, std::atoi(argv[1]))
            : 10000;

    std::uint64_t* state = nullptr;
    std::uint64_t* counter = nullptr;

    check(
        cudaMalloc(&state, sizeof(*state)),
        "cudaMalloc state");

    check(
        cudaMalloc(&counter, sizeof(*counter)),
        "cudaMalloc counter");

    cudaStream_t stream = nullptr;
    check(
        cudaStreamCreate(&stream),
        "cudaStreamCreate");

    const std::uint64_t seed = 7;
    const std::uint64_t zero = 0;

    check(
        cudaMemcpy(
            state,
            &seed,
            sizeof(seed),
            cudaMemcpyHostToDevice),
        "copy seed");

    check(
        cudaMemcpy(
            counter,
            &zero,
            sizeof(zero),
            cudaMemcpyHostToDevice),
        "reset counter");

    const auto baseline_start =
        std::chrono::steady_clock::now();

    for (int i = 0;
         i < iterations;
         ++i) {
        decode_step<<<1, 1, 0, stream>>>(
            state, counter);

        check(
            cudaGetLastError(),
            "baseline launch");
    }

    check(
        cudaStreamSynchronize(stream),
        "baseline synchronize");

    const auto baseline_end =
        std::chrono::steady_clock::now();

    check(
        cudaMemcpy(
            state,
            &seed,
            sizeof(seed),
            cudaMemcpyHostToDevice),
        "reset graph state");

    check(
        cudaMemcpy(
            counter,
            &zero,
            sizeof(zero),
            cudaMemcpyHostToDevice),
        "reset graph counter");

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    check(
        cudaStreamBeginCapture(
            stream,
            cudaStreamCaptureModeGlobal),
        "begin capture");

    decode_step<<<1, 1, 0, stream>>>(
        state, counter);

    check(
        cudaGetLastError(),
        "capture launch");

    check(
        cudaStreamEndCapture(
            stream,
            &graph),
        "end capture");

    check(
        cudaGraphInstantiate(
            &executable,
            graph,
            nullptr,
            nullptr,
            0),
        "graph instantiate");

    const auto graph_start =
        std::chrono::steady_clock::now();

    for (int i = 0;
         i < iterations;
         ++i) {
        check(
            cudaGraphLaunch(
                executable,
                stream),
            "graph launch");
    }

    check(
        cudaStreamSynchronize(stream),
        "graph synchronize");

    const auto graph_end =
        std::chrono::steady_clock::now();

    const double baseline_us =
        std::chrono::duration<double, std::micro>(
            baseline_end -
            baseline_start).count() /
        static_cast<double>(iterations);

    const double graph_us =
        std::chrono::duration<double, std::micro>(
            graph_end -
            graph_start).count() /
        static_cast<double>(iterations);

    const double reduction =
        baseline_us > 0.0
            ? 1.0 - graph_us / baseline_us
            : 0.0;

    std::cout
        << std::fixed
        << std::setprecision(4)
        << "iterations=" << iterations
        << " baseline_host_us="
        << baseline_us
        << " graph_host_us="
        << graph_us
        << " host_enqueue_reduction="
        << reduction
        << "\n";

    check(
        cudaGraphExecDestroy(executable),
        "destroy executable");
    check(
        cudaGraphDestroy(graph),
        "destroy graph");
    check(
        cudaStreamDestroy(stream),
        "destroy stream");
    check(
        cudaFree(state),
        "free state");
    check(
        cudaFree(counter),
        "free counter");

    return 0;
}
