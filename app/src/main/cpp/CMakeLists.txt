# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        zbarjni

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        zbar/img_scanner.c
        zbar/decoder.c
        zbar/image.c
        zbar/symbol.c
        zbar/convert.c
        zbar/config.c
        zbar/scanner.c
        zbar/error.c
        zbar/refcnt.c
        zbar/video.c
        zbar/video/null.c
        zbar/decoder/code128.c
        zbar/decoder/code39.c
        zbar/decoder/code93.c
        zbar/decoder/codabar.c
        zbar/decoder/databar.c
        zbar/decoder/ean.c
        zbar/decoder/i25.c
        zbar/decoder/qr_finder.c
        zbar/qrcode/bch15_5.c
        zbar/qrcode/binarize.c
        zbar/qrcode/isaac.c
        zbar/qrcode/qrdec.c
        zbar/qrcode/qrdectxt.c
        zbar/qrcode/rs.c
        zbar/qrcode/util.c

        native-lib.cpp
        zbarjni.c
        )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.
add_library(libiconv SHARED IMPORTED )
set_target_properties(libiconv PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libiconv.so)
find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.
INCLUDE_DIRECTORIES(
        ${CMAKE_SOURCE_DIR}/../jniLibs/include
        ${CMAKE_SOURCE_DIR}/../jniLibs/libiconv/include)

target_link_libraries( # Specifies the target library.
        zbarjni
        libiconv
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})