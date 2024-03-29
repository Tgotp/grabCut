cmake_minimum_required(VERSION 3.20)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

project(Grabcut)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(maxflow-v3.01)
add_executable(${PROJECT_NAME} test4-OpenCv-Grabcut.cpp GMM.cpp)

# add_executable(${PROJECT_NAME}  test5-OpenCv-TempGrabcut.cpp TempGMM.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,--stack,436207616" )

target_link_libraries(${PROJECT_NAME} maxflow ${OpenCV_LIBS})

target_include_directories(${PROJECT_NAME} PUBLIC
						${PROJECT_SOURCE_DIR}/maxflow-v3.01
						)