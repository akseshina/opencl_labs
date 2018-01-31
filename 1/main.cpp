#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include <CL/cl.h>
#include "cl.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <math.h>

inline size_t to_1D(size_t y, size_t x, size_t n) {
    return y * n + x;
}

inline float round3(float x) {
    return roundf(x * 1000) / 1000;
}

int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0]);

        // load opencl source
        std::ifstream cl_file("matrix_conv.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        // create program
        cl::Program program(context, source);


        // compile opencl source
        try {
            program.build(devices);
        }
        catch (cl::Error const &e) {
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 0;
        }


        // create a message to send to kernel
        std::ifstream in("input.txt");
        size_t Nt, Mt;
        in >> Nt >> Mt;
        size_t const N = Nt, M = Mt;
        size_t const matrix_size = N * N;
        size_t const kernel_size = M * M;
        std::vector<float> a(matrix_size, 0);
        std::vector<float> b(kernel_size, 0);
        std::vector<float> c(matrix_size, 0);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                in >> a[to_1D(i, j, N)];
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < M; ++j)
                in >> b[to_1D(i, j, M)];


        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(float) * matrix_size);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(float) * matrix_size);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * matrix_size);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * matrix_size, a.data());
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * matrix_size, b.data());

        // load named kernel from opencl source
        cl::Kernel kernel(program, "convolution");
        cl::KernelFunctor convolution(kernel, queue, cl::NullRange, cl::NDRange(N, N), cl::NullRange);
        convolution(dev_a, dev_b, dev_c, (int) N, (int) M);

        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * matrix_size, c.data());

        std::ofstream out("output.txt");
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                std::cout << round3(c[to_1D(i, j, N)]) << " ";
                out << round3(c[to_1D(i, j, N)]) << " ";
            }
            out << std::endl;
            std::cout << std::endl;
        }

    }
    catch (cl::Error e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}