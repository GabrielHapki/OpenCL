#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <iostream>
#include <string.h>
#include "oclWrapper.hpp"

int main(int argc, char** argv) {
    bool loadsrc = false;
    cl_int ret = 0;

    oclWrapper ocl;
    if (ocl.Init(ocl::ASYNC, ocl::PROFILING)){
        loadsrc = ocl.LoadSourceCode("../../examples/oclBuffer/buffer");
    }

    if (loadsrc){
        cl::Context context = ocl.GetContext();
        cl::Program program = ocl.GetProgram();
        cl::CommandQueue queue = ocl.GetQueue();

        uint8_t A[16*16] = {0};
        uint8_t D[16*16] = {0};
        for (uint16_t i = 0; i < 16 * 16; i++)
            A[i] = i;        

        uint8_t mult = 2;
        uint8_t add = 1;

        cl::Buffer buff_A(context, ocl::CPU2GPU, 16 * 16);
        cl::Buffer buff_B(context, ocl::GPU_INT, 16 * 16);

        cl::Kernel kmult = cl::Kernel(program, "mult_buff", &ret);
        if (ret != CL_SUCCESS)
            return 0;
        kmult.setArg(0, buff_A);
        kmult.setArg(1, buff_B);
        kmult.setArg(2, sizeof(mult), &mult);

        cl::Buffer buff_D(context, ocl::GPU2CPU, 16 * 16);

        cl::Kernel kadd = cl::Kernel(program, "add_buff", &ret);
        if (ret != CL_SUCCESS)
            return 0;
        kadd.setArg(0, buff_B);
        kadd.setArg(1, buff_D);
        kadd.setArg(2, sizeof(add), &add);

        cl::Event ev_in, ev_kmult, ev_kadd, ev_out;

        cl::UserEvent ev_user = cl::UserEvent(context);
        ev_user.setStatus(CL_COMPLETE);

        std::vector<cl::Event> evs_in = {ev_user};
        queue.enqueueWriteBuffer(buff_A, CL_TRUE, 0, buff_A.getInfo<CL_MEM_SIZE>(), A, &evs_in, &ev_in);
        std::vector<cl::Event> evs_kmult = {ev_in};
        queue.enqueueNDRangeKernel(kmult, cl::NullRange, cl::NDRange(16 * 16), cl::NullRange, &evs_kmult, &ev_kmult);
        ocl.Flush();
        std::vector<cl::Event> evs_kadd = {ev_kmult};
        queue.enqueueNDRangeKernel(kadd, cl::NullRange, cl::NDRange(16 * 16), cl::NullRange, &evs_kadd, &ev_kadd);
        ocl.Flush();
        std::vector<cl::Event> evs_out = {ev_kadd};
        queue.enqueueReadBuffer(buff_D, CL_TRUE, 0, buff_D.getInfo<CL_MEM_SIZE>(), D, &evs_out, &ev_out);

        ev_out.wait();

        printf("Elapsed time to Write = %fus\n", ocl.ElapsedTime(ev_in));
        printf("Elapsed time to process kmult = %fus\n", ocl.ElapsedTime(ev_kmult));
        printf("Elapsed time to process kadd = %fus\n", ocl.ElapsedTime(ev_kadd));
        printf("Elapsed time to Read = %fus\n", ocl.ElapsedTime(ev_out));

    }

    return 0;
}

