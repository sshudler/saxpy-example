cmake_minimum_required(VERSION 3.13)

Project(SaxpyExample)

set(CMAKE_VERBOSE_MAKEFILE ON)

if(ENABLE_CUDA)
    message("Using CUDA")

    if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
        message(FATAL_ERROR "CUDA root directory should be specified in CUDA_TOOLKIT_ROOT_DIR.")
    endif()
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "Found CUDA: " ${CUDA_VERSION})
    else()
        message(FATAL_ERROR "Could not find CUDA. Ensure that CUDA_TOOLKIT_ROOT_DIR points to the right location.")
    endif()

    set(SRC
        saxpy-cuda.cu
    )
    cuda_add_executable(saxpy-cuda ${SRC})

    install(TARGETS saxpy-cuda DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endif()

if(ENABLE_HIP)
    message("Using HIP")

    if(NOT DEFINED HIP_ROOT_DIR)
        message(FATAL_ERROR "HIP directory should be specified in HIP_ROOT_DIR.")
    endif()
    set(CMAKE_MODULE_PATH "${HIP_ROOT_DIR}/cmake" ${CMAKE_MODULE_PATH})
    find_package(HIP QUIET)
    if(HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_VERSION})
    else()
        message(FATAL_ERROR "Could not find HIP. Ensure that the variable ROCM_PATH is set to point to the right location.")
    endif()

    set(SRC
        saxpy-hip.cpp
    )
    set_source_files_properties(${SRC} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_add_executable(saxpy-hip ${SRC})

    install(TARGETS saxpy-hip DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endif()

if(ENABLE_ONEAPI)
    message("Using OneAPI")

    set(SRC
        saxpy-dpcpp.cpp
    )
    set(CMAKE_CXX_COMPILER "dpcpp")
    set(CMAKE_CXX_STANDARD 17)
    add_executable(saxpy-dpcpp ${SRC})

    install(TARGETS saxpy-dpcpp DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endif()
