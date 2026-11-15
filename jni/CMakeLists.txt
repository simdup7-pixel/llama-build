cmake_minimum_required(VERSION 3.22)
project(llamabridge C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP          OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE          OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

add_library(llamabridge SHARED llamabridge.cpp)
target_compile_definitions(llamabridge PRIVATE BRIDGE_CLASS=\"${BRIDGE_CLASS}\")
target_link_libraries(llamabridge PRIVATE llama log android)
target_link_options(llamabridge PRIVATE -Wl,-s -Wl,-z,max-page-size=16384)
