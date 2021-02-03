cmake_minimum_required(VERSION 3.10)
project(tflite_test)

set(CMAKE_CXX_STANDARD 17)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS NO_OUTPUT_DIRS KEEP_RPATHS)

add_executable(tflite_test main.cpp)
target_include_directories(tflite_test PRIVATE $CONAN_INCLUDE_DIRS_TENSORFLOW_LITE)

if(ANDROID)
    set(ADDITIONAL_LIBS log EGL GLESv2 GLESv3)
endif()

target_link_libraries(tflite_test PRIVATE CONAN_PKG::tensorflow_lite ${ADDITIONAL_LIBS})

#cmake -G "Unix Makefiles" -DCMAKE_SYSTEM_NAME=Android -DANDROID_ABI=arm64-v8a -DANDROID_STL=c++_shared -DANDROID_NATIVE_API_LEVEL=27 -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release ..
