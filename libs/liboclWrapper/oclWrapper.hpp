#ifndef OCL_H
#define OCL_H

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "CL/opencl.hpp"

namespace ocl{

    typedef enum{
        CPU2GPU = CL_MEM_READ_ONLY,  //GPU Input, the data is copied from CPU to GPU.
        GPU2CPU = CL_MEM_WRITE_ONLY, //GPU Output, the data is copied from GPU to CPU.
        GPU_INT = CL_MEM_READ_WRITE, //GPU internal memory.
    } Direction_t;

    typedef enum{
        SYNC = 0,
        ASYNC = 1,
    } Blocking_t;

    typedef enum{
        NOT_PROFILING = 0,
        PROFILING = 1,
    } Profiling_t;

}

class oclWrapper{
private:
    cl::Device m_device;
    cl::Context m_context;
    cl::Program m_program;
    cl::CommandQueue m_queue;

public:
    oclWrapper() {}
    ~oclWrapper() {}

    bool Init(const ocl::Blocking_t block, const ocl::Profiling_t profiling = ocl::NOT_PROFILING);
    bool LoadSourceCode(const std::string &KernelFile);
    cl::Context GetContext();
    cl::Program GetProgram();
    cl::CommandQueue GetQueue();
    void Flush();
    double ElapsedTime(const cl::Event &event);
};

#endif /* OCL_H */
