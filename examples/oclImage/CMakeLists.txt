project (oclImage)

# The version number
set (oclImage_VERSION_MAJOR 1)
set (oclImage_VERSION_MINOR 0)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "-Wall")

set (EXECUTABLE_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/bin)
#set (CMAKE_INCLUDE_DIR ${CMAKE_HOME_DIRECTORY}/include)

message ("############################################################")
message ("# Example OpenCL Image                                     #")
message ("CMAKE_HOME_DIRECTORY => ${CMAKE_HOME_DIRECTORY}")
message ("CMAKE_INCLUDE_DIR =>    ${CMAKE_INCLUDE_DIR}")
message ("CMAKE_SOURCE_DIR =>     ${CMAKE_SOURCE_DIR}")
message ("CMAKE_BINARY_DIR =>     ${CMAKE_BINARY_DIR}")
message ("EXE_OUTPUT_PATH =>      ${EXECUTABLE_OUTPUT_PATH}")

include_directories (${CMAKE_SOURCE_DIR}/libs/liboclWrapper)
link_directories (${CMAKE_BINARY_DIR}/lib)

add_executable (oclImage image.cpp)

target_link_libraries (oclImage oclWrap)