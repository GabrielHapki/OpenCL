#include "oclWrapper.hpp"
#include <iostream>
#include <fstream>

bool oclWrapper::Init(const ocl::Blocking_t block, const ocl::Profiling_t profiling)
{
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    if (all_platforms.size()==0) {
            std::cout<<" No platforms found. Check OpenCL installation!\n";
            return false;
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
            std::cout<<" No devices found. Check OpenCL installation!\n";
            return false;
    }

    this->m_device=all_devices[0];
    std::cout<< "Using device: "<<this->m_device.getInfo<CL_DEVICE_NAME>()<<"\n";

    this->m_context = cl::Context({this->m_device});

    cl_command_queue_properties properties = 0;
    if (block == ocl::ASYNC)
        properties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    if (profiling == ocl::PROFILING)
        properties |= CL_QUEUE_PROFILING_ENABLE;
    this->m_queue = cl::CommandQueue(this->m_context, this->m_device, properties);

    return true;
}

bool oclWrapper::LoadSourceCode(const std::string& KernelFile)
{
    std::string KernelFileSrc = KernelFile + ".cl";
    std::ifstream file(KernelFileSrc.data());
    std::string prog(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources sources;
    sources.push_back({prog.c_str(), prog.length() + 1});
    this->m_program = cl::Program(this->m_context, sources);
    file.close();

    if (this->m_program.build({this->m_device}) != CL_SUCCESS){
        std::cout << "Error building: " << this->m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->m_device) << std::endl;
        return false;
    }
    return true;
}

cl::Context oclWrapper::GetContext()
{
    return this->m_context;
}

cl::Program oclWrapper::GetProgram()
{
    return this->m_program;
}

cl::CommandQueue oclWrapper::GetQueue()
{
    return this->m_queue;
}

void oclWrapper::Flush()
{
    auto properties = this->m_queue.getInfo<CL_QUEUE_PROPERTIES>();
    if (!(properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)) // if it is Synchronous
        this->m_queue.flush();
}

double oclWrapper::ElapsedTime(const cl::Event &event)
{
    auto properties = this->m_queue.getInfo<CL_QUEUE_PROPERTIES>();
    if (properties & CL_QUEUE_PROFILING_ENABLE){ // if it is Synchronous
        cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        double elapsedtime = (endTime - startTime)/1000.f;
        return elapsedtime;
    }
    return -1.f;
}