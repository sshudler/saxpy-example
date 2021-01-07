cmake_minimum_required(VERSION 3.13)

Project(SaxpyExample)

set(CMAKE_VERBOSE_MAKEFILE ON)

if(ENABLE_CUDA)
    message("Using CUDA")
    enable_language(CUDA)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES OFF)
    endif()

    set(SRC
        saxpy.cu
    )
    add_executable(saxpy-cuda ${SRC})
    target_compile_options(saxpy-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)

    install(TARGETS saxpy-cuda DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endif()

if(ENABLE_HIP)
    # HIP options
    message("Using HIP")
    set(HIP_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
    find_package(HIP QUIET)
    if(HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_VERSION})
    else()
        message(FATAL_ERROR "Could not find HIP. Ensure that the variable ROCM_PATH is set to point to the right location.")
    endif()

    set(SRC
        saxpy.cpp
    )
    set_source_files_properties(${SRC} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_add_executable(saxpy-hip ${SRC})

    install(TARGETS saxpy-hip DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endif()

if(ENABLE_ONEAPI)
    # OneAPI options
endif()
