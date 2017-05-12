cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)
set(CAMEK_CXXSTANDARD-REQUIRED ON)
project(RTInference CXX)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(${PROJECT_SOURCE_DIR}/cmake/FindGlog.cmake)

find_package(OpenCV REQUIRED COMPONENTS core imgproc video highgui features2d ml)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Boost REQUIRED COMPONENTS system 
                              filesystem 
                              log 
                              log_setup 
                              thread 
                              date_time 
                              regex 
                              unit_test_framework
                              program_options)

set(OPENNI2_INCLUDE_DIR "/home/zhi/Downloads/OpenNI-Linux-x64-2.3/Include")
set(OPENNI2_LIB_DIR "/home/zhi/Downloads/OpenNI-Linux-x64-2.3/Redist")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OPENNI2_INCLUDE_DIR})
include_directories(${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${GLOG_INCLUDE_DIRS})
include_directories(${Caffe_INCLUDE_DIRS})


set(RTI_LIBRARIES 
  /home/zhi/Downloads/OpenNI-Linux-x64-2.3/Redist/libOpenNI2.so
  /home/zhi/Downloads/OpenNI-Linux-x64-2.3/Redist/OpenNI2/Drivers/libOniFile.so
  /home/zhi/Downloads/OpenNI-Linux-x64-2.3/Redist/OpenNI2/Drivers/liborbbec.so
  ${OpenCV_LIBS}
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARY}
  ${Boost_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${Caffe_LIBRARIES})

set(TOOLS_LIBRARIES RTILibrary ${RTI_LIBRARIES})

# set(TOOLS_LIBRARIES ${TOOLS_LIBRARIES} PARENT_SCOPE)

add_subdirectory(src)
add_subdirectory(tools)
