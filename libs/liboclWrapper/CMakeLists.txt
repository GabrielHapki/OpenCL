project (oclWrap)

# The version number
set (oclWrap_VERSION_MAJOR 1)
set (oclWrap_VERSION_MINOR 0)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "-Wall")

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package (OpenCL REQUIRED)
include_directories (${CMAKE_INCLUDE_DIR})

message ("##################################################################################################")
message ("# lib OpenCLWrapper                                                                              #")
message ("CMAKE_HOME_DIRECTORY => ${CMAKE_HOME_DIRECTORY}")
message ("CMAKE_INCLUDE_DIR =>    ${CMAKE_INCLUDE_DIR}")
message ("CMAKE_SOURCE_DIR =>     ${CMAKE_SOURCE_DIR}")
message ("CMAKE_BINARY_DIR =>     ${CMAKE_BINARY_DIR}")
message ("CMAKE_LIB_OUTPUT_DIR => ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

add_library (oclWrap SHARED oclWrapper.cpp)
target_link_libraries (oclWrap OpenCL)

