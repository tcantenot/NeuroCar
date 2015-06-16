# Based on the CMake config file of SFML:
# https://github.com/LaurentGomila/SFML/blob/master/cmake/Config.cmake

# Improvements: cf https://github.com/petroules/solar-cmake/blob/master/TargetArch.cmake


### Platform detection ###

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

    set(NEURO_CAR_OS_WINDOWS TRUE)

elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

    set(NEURO_CAR_OS_UNIX TRUE)
    set(NEURO_CAR_OS_LINUX TRUE)

else()

    message(FATAL_ERROR "Unsupported operating system")
    return()

endif()

### Architecture detection (fails with cross-compilation) ###
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ARCH_64_BITS TRUE)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(ARCH_32_BITS TRUE)
else()
    message(FATAL_ERROR "Unsupported architecture")
    return()
endif()


### Compiler detection ###

# Note: on some platforms (OS X), CMAKE_COMPILER_IS_GNUCXX is true
# even when CLANG is used, therefore the Clang test is done first
if(CMAKE_CXX_COMPILER MATCHES ".*clang[+][+]" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

   # CMAKE_CXX_COMPILER_ID is an internal CMake variable subject to change,
   # but there is no other way to detect Clang at the moment
   set(NEURO_CAR_COMPILER_CLANG TRUE)
   execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "--version" OUTPUT_VARIABLE CLANG_VERSION_OUTPUT)
   string(REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+).*" "\\1" NEURO_CAR_CLANG_VERSION "${CLANG_VERSION_OUTPUT}")

elseif(CMAKE_COMPILER_IS_GNUCXX)

    set(NEURO_CAR_COMPILER_GCC TRUE)

    # GCC version
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "-dumpversion" OUTPUT_VARIABLE GCC_VERSION)

    # Check for C++11 support
    if(NOT(GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7))
        message(FATAL_ERROR "HYBRID requires g++ 4.7 or greater (for c++11 support).")
    endif ()

    string(REGEX REPLACE "([0-9]+\\.[0-9]+).*" "\\1" NEURO_CAR_GCC_VERSION "${GCC_VERSION}")

    # TDM-GCC version
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "--version" OUTPUT_VARIABLE GCC_COMPILER_VERSION)
    string(REGEX MATCHALL ".*(tdm[64]*-[1-9]).*" NEURO_CAR_COMPILER_GCC_TDM "${GCC_COMPILER_VERSION}")

    # GCC architecture
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "-dumpmachine" OUTPUT_VARIABLE GCC_MACHINE)
    string(STRIP "${GCC_MACHINE}" GCC_MACHINE)
    if(${GCC_MACHINE} MATCHES ".*w64.*")
        set(NEURO_CAR_COMPILER_GCC_W64 1)
    endif()

elseif(MSVC)

    set(NEURO_CAR_COMPILER_MSVC TRUE)

    if(MSVC_VERSION EQUAL 1400)
        set(NEURO_CAR_MSVC_VERSION 8)
    elseif(MSVC_VERSION EQUAL 1500)
        set(NEURO_CAR_MSVC_VERSION 9)
    elseif(MSVC_VERSION EQUAL 1600)
        set(NEURO_CAR_MSVC_VERSION 10)
    elseif(MSVC_VERSION EQUAL 1700)
        set(NEURO_CAR_MSVC_VERSION 11)
    elseif(MSVC_VERSION EQUAL 1800)
        set(NEURO_CAR_MSVC_VERSION 12)
    endif()

else()

    message(FATAL_ERROR "Unsupported compiler")
    return()

endif()
