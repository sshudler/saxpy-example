cmake_minimum_required(VERSION 3.13)

Project(SaxpyExample)

set(CMAKE_VERBOSE_MAKEFILE TRUE)

if(ENABLE_CUDA)
    message("Using CUDA")
    enable_language(CUDA)

    set(SRC
        saxpy.cu
    )
    add_executable(saxpy ${SRC})
endif()

if(ENABLE_HIP)
    # HIP options
endif()

if(ENABLE_ONEAPI)
    # OneAPI options
endif()
