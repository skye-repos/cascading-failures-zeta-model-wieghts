cmake_minimum_required(VERSION 3.16)
project(CascadingFailures LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(NATIVE_OPT "Optimize for the build host CPU" ON)
option(ENABLE_LTO "Enable interprocedural optimization in Release" ON)

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS version")
endif()
find_package(OpenMP REQUIRED)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()
find_package(Boost REQUIRED)

# On Linux with a manually-downloaded Boost (see README), pass:
#   cmake -B build -DBOOST_ROOT=/path/to/boost

add_executable(cascade
    src/main.cpp
    src/simulate.cpp
    src/betweenness.cpp
    src/graph_generators.cpp
    src/util.cpp
)

target_link_libraries(cascade PRIVATE
    Boost::headers
    OpenMP::OpenMP_CXX
)

target_include_directories(cascade PRIVATE src)

if(NATIVE_OPT AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        target_compile_options(cascade PRIVATE -march=native -mtune=native)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        target_compile_options(cascade PRIVATE -mcpu=native)
    endif()
    target_compile_options(cascade PRIVATE -fno-math-errno)
endif()

if(ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _ipo_supported OUTPUT _ipo_error)
    if(_ipo_supported)
        set_property(TARGET cascade PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "LTO not supported by this toolchain; building without LTO")
    endif()
endif()
