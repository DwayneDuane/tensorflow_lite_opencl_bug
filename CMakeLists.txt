cmake_minimum_required(VERSION 3.10)
project(tflite_test)

set(CMAKE_CXX_STANDARD 17)

message(WARNING $ENV{TF_ROOT})

find_library(TENSORFLOW_LIB tensorflowlite HINT $ENV{TF_ROOT}/lib/${CMAKE_SYSTEM_PROCESSOR} NO_CMAKE_FIND_ROOT_PATH)
find_library(TENSORFLOW_GPU_DELEGATE tensorflowlite_gpu_delegate HINT $ENV{TF_ROOT}/lib/${CMAKE_SYSTEM_PROCESSOR} NO_CMAKE_FIND_ROOT_PATH)

add_executable(tflite_test main.cpp)
target_include_directories(tflite_test PRIVATE $ENV{TF_ROOT}/include)

if(ANDROID)
    set(ADDITIONAL_LIBS log EGL GLESv2 GLESv3)
endif()

target_link_libraries(tflite_test PRIVATE ${TENSORFLOW_GPU_DELEGATE} ${TENSORFLOW_LIB} ${ADDITIONAL_LIBS})

#cmake -G "Unix Makefiles" -DCMAKE_SYSTEM_NAME=Android -DANDROID_ABI=arm64-v8a -DANDROID_STL=c++_shared -DANDROID_NATIVE_API_LEVEL=27 -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release ..
