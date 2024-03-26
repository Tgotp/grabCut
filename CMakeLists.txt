cmake_minimum_required(VERSION 3.20)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)

project(Grabcut)

# set(OpenCV_DIR D:/OpenCV/opencv/build/x64/vc16/lib)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(maxflow-v3.01)
add_executable(${PROJECT_NAME} test4-OpenCV-Grabcut.cpp GMM.cpp)

target_link_libraries(${PROJECT_NAME} maxflow ${OpenCV_LIBS})

target_include_directories(${PROJECT_NAME} PUBLIC
						${PROJECT_SOURCE_DIR}/maxflow-v3.01
						)