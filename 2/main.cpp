#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <assert.h>
#include <math.h>


size_t const block_size = 256;


inline float round3(float x) {
    return roundf(x * 1000) / 1000;
}


std::vector<float> subsums(std::vector<float> & input) {

    /*for (int i = 0; i < input.size(); ++i)
        std::cout << input[i] << ' ';
    std::cout << std::endl;*/

    size_t const N = input.size();
    size_t const N2 = (N / block_size) * block_size + (int)(N % block_size > 0) * block_size;
    for (int i = N; i < N2; ++i)
        input.push_back(0);

    std::vector <cl::Platform> platforms;
    std::vector <cl::Device> devices;
    std::vector <cl::Kernel> kernels;

    try {

        // initialize kernel
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        cl::Context context(devices);
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));
        cl::Program program(context, source);
        program.build(devices);

        // allocate device buffer to hold message
        size_t const array_size = input.size();
        cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * array_size);
        cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * array_size);
        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * array_size, &input[0]);
        queue.finish();

        // load named kernel from opencl source
        cl::Kernel kernel_hs(program, "scan_hillis_steele");
        cl::KernelFunctor scan_hs(kernel_hs, queue, cl::NullRange, cl::NDRange(array_size),
                                  cl::NDRange(block_size));
        cl::Event event = scan_hs(dev_input, dev_output, cl::__local(sizeof(float) * block_size),
                                  cl::__local(sizeof(float) * block_size));
        event.wait();

        std::vector<float> cur_output(array_size, 0);
        queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * array_size, &cur_output[0]);


        // recursion ends
        if (array_size <= block_size)
            return cur_output;


        // if we need to add a value for each block (in parallel)
        size_t const num_of_blocks = array_size / block_size;
        std::vector<float> blocks_end_elems(num_of_blocks, 0);
        for (int i = 0; i < num_of_blocks; ++i)
            blocks_end_elems[i] = cur_output[(i+1) * block_size - 1];

        std::vector<float> block_ends_sums = subsums(blocks_end_elems);

        // copy from cpu to gpu
        cl::Buffer dev_input2(context, CL_MEM_READ_ONLY, sizeof(float) * num_of_blocks);
        queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * array_size, &cur_output[0]);
        queue.enqueueWriteBuffer(dev_input2, CL_TRUE, 0, sizeof(float) * num_of_blocks, &block_ends_sums[0]);
        queue.finish();

        // load named kernel from opencl source
        cl::Kernel kernel_sum_up(program, "sum_up");
        cl::KernelFunctor sum_up(kernel_sum_up, queue, cl::NullRange, cl::NDRange(array_size),
                                  cl::NDRange(block_size));
        cl::Event event2 = sum_up(dev_input, dev_input2, dev_output);
        event2.wait();

        std::vector<float> final_output(array_size, 0);
        queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * array_size, &final_output[0]);

        return final_output;


    }
    catch (cl::Error e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }
}

int main() {

    std::ifstream in("input.txt");
    size_t Nt;
    in >> Nt;
    size_t const N = Nt;
    std::vector<float> input(N, 0);
    for (size_t i = 0; i < N; ++i)
        in >> input[i];

    std::vector<float> output = subsums(input);
    std::ofstream out("output.txt");
    for (int i = 0; i < N; ++i) {
        std::cout << round3(output[i]) << ' ';
        out << round3(output[i]) << ' ';
    }
    std::cout << std::endl;

    return 0;
}