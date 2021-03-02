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
        loadsrc = ocl.LoadSourceCode("../../examples/oclImage/image");
    }

    if (loadsrc){
        cl::Context context = ocl.GetContext();
        cl::Program program = ocl.GetProgram();
        cl::CommandQueue queue = ocl.GetQueue();

        const uint32_t Width = 16;
	    const uint32_t Height = 16;
	    const uint32_t MatSize = Width * Height;
	    float A[MatSize] = {0};
        float D[MatSize] = {0};
        for (uint32_t i = 0; i < MatSize; i++)
            A[i] = (float)i;

        float mult = 2.0f;
        float add = 0.5f;

        cl::Image2D img_A(context, ocl::CPU2GPU, cl::ImageFormat(CL_R, CL_FLOAT), Width, Height);
        cl::Image2D img_B(context, ocl::GPU_INT, cl::ImageFormat(CL_R, CL_FLOAT), Width, Height);

        cl::Kernel kmult = cl::Kernel(program, "mult_img", &ret);
        if (ret != CL_SUCCESS)
            return 0;
        kmult.setArg(0, img_A);
        kmult.setArg(1, img_B);
        kmult.setArg(2, sizeof(mult), &mult);

        cl::Image2D img_D(context, ocl::GPU2CPU, cl::ImageFormat(CL_R, CL_FLOAT), Width, Height);

        cl::Kernel kadd = cl::Kernel(program, "add_img", &ret);
        if (ret != CL_SUCCESS)
            return false;
        kadd.setArg(0, img_B);
        kadd.setArg(1, img_D);
        kadd.setArg(2, sizeof(add), &add);

        cl::Event ev_in, ev_kmult, ev_kadd, ev_out;

        cl::UserEvent ev_user = cl::UserEvent(context);
        ev_user.setStatus(CL_COMPLETE);

        //auto Width = img_A.getImageInfo<CL_IMAGE_WIDTH>();
        //auto Height = img_A.getImageInfo<CL_IMAGE_HEIGHT>();

        std::vector<cl::Event> evs_in = {ev_user};
        queue.enqueueWriteImage(img_A, CL_TRUE, {0, 0, 0}, {Width, Height, 1}, 0, 0, A, &evs_in, &ev_in);
        std::vector<cl::Event> evs_kmult = {ev_in};
        queue.enqueueNDRangeKernel(kmult, cl::NullRange, cl::NDRange(16, 16), cl::NullRange, &evs_kmult, &ev_kmult);
        ocl.Flush();
        std::vector<cl::Event> evs_kadd = {ev_kmult};
        queue.enqueueNDRangeKernel(kadd, cl::NullRange, cl::NDRange(16, 16), cl::NullRange, &evs_kadd, &ev_kadd);
        ocl.Flush();

        //auto Width = out.getImageInfo<CL_IMAGE_WIDTH>();
        //auto Height = out.getImageInfo<CL_IMAGE_HEIGHT>();
        std::vector<cl::Event> evs_out = {ev_kadd};
        queue.enqueueReadImage(img_D, CL_TRUE, {0, 0, 0}, {Width, Height, 1}, 0, 0, D, &evs_out, &ev_out);

        ev_out.wait();

        printf("Elapsed time to Write = %fus\n", ocl.ElapsedTime(ev_in));
        printf("Elapsed time to process kmult = %fus\n", ocl.ElapsedTime(ev_kmult));
        printf("Elapsed time to process kadd = %fus\n", ocl.ElapsedTime(ev_kadd));
        printf("Elapsed time to Read = %fus\n", ocl.ElapsedTime(ev_out));

    }

    return 0;
}

