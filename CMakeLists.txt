cmake_minimum_required(VERSION 3.16)
project(CascadingFailures LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    src/graph_generators.cpp
    src/util.cpp
)

target_link_libraries(cascade PRIVATE
    Boost::headers
    OpenMP::OpenMP_CXX
)

target_include_directories(cascade PRIVATE src)
