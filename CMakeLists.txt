cmake_minimum_required(VERSION 2.8)

# Some useful macros
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/macros.cmake)

################################################################################
#                            GLOBAL CONFIGURATION                              #
################################################################################

# Ensures that we are doing an out-of-source build
MACRO_ENSURE_OUT_OF_SOURCE_BUILD("Requires an out of source build.")

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(
      CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
    )
endif()


# Project name
project(NeuroCar)

# Version
set(NEURO_CAR_MAJOR_VERSION 0)
set(NEURO_CAR_MINOR_VERSION 1)
set(NEURO_CAR_PATCH_VERSION 0)
set(NEURO_CAR_VERSION
  ${NEURO_CAR_MAJOR_VERSION}.${NEURO_CAR_MINOR_VERSION}.${NEURO_CAR_PATCH_VERSION}
)

# Configuration file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)

# Static library name
set(NEURO_CAR_STATIC_LIBRARY NeuroCarLib)

# Executable name
set(NEURO_CAR_EXECUTABLE_NAME NeuroCar)

# Binary directory
set(NEURO_CAR_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})


### Definitions ###

# Enable/disable graphic mode
if(NOT DEFINED NEURO_CAR_GRAPHIC_MODE)
    set(NEURO_CAR_GRAPHIC_MODE ON CACHE BOOL "Enable/Disable graphic mode")
endif()

if(NEURO_CAR_GRAPHIC_MODE)
    set(CAR_PHYSICS_GRAPHIC_MODE ON)
    add_definitions(-DNEURO_CAR_GRAPHIC_MODE=1)
    add_definitions(-DCAR_PHYSICS_GRAPHIC_MODE_SFML=1)
else()
    set(CAR_PHYSICS_GRAPHIC_MODE OFF)
    add_definitions(-DNEURO_CAR_GRAPHIC_MODE=0)
    add_definitions(-DCAR_PHYSICS_GRAPHIC_MODE_SFML=0)
endif()


################################################################################
#                             COMPILATION FLAGS                                #
################################################################################

if(NEURO_CAR_COMPILER_GCC OR NEURO_CAR_COMPILER_CLANG)

    # C++11 flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

    # Base compiler flags
    # https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
    # https://gcc.gnu.org/onlinedocs/gcc/C_002b_002b-Dialect-Options.html
    # http://stackoverflow.com/a/9862800
    set(CXX_BASE_FLAGS
        "-W -Wall -Wextra -pedantic"
        "-Wold-style-cast -Wparentheses -Wlogical-op -Woverloaded-virtual"
        "-Wfloat-equal -Wpointer-arith -Wcast-align -Wcast-qual"
        "-Winit-self -Wwrite-strings -Wstrict-null-sentinel"
        "-Wredundant-decls -Wswitch-default -Wundef -Wunused-parameter"
        "-Wmissing-include-dirs"
        "-Wdisabled-optimization"
        "-Wnoexcept"

        #"-Weffc++"
        #"-Wshadow"
        #"-Wsign-conversion"
        #"-Wconversion"
        #"-Wstrict-overflow=5"
        #"-Winline"
        #"-Wpadded"
    )
    string(REGEX REPLACE ";" " " CXX_BASE_FLAGS "${CXX_BASE_FLAGS}")

    # Release mode
    set(CXX_FLAGS_RELEASE "-Werror ${CXX_BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CXX_DEFINES} ${CXX_FLAGS_RELEASE}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CXX_FLAGS_RELEASE}")

    # Debug mode
    set(CXX_FLAGS_DEBUG ${CXX_BASE_FLAGS})
    set(CMAKE_CXX_FLAGS_DEBUG "${CXX_DEFINES} ${CXX_FLAGS_DEBUG}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CXX_FLAGS_DEBUG}")

    # Release with debug symbols
    set(CXX_FLAGS_RELWITHDEBINFO "${CXX_BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
        "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CXX_FLAGS_RELWITHDEBINFO}"
    )

    # Release with stripping of extra symbols
    set(CXX_FLAGS_MINSIZEREL "${CXX_BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_MINSIZEREL
        "${CMAKE_CXX_FLAGS_MINSIZEREL} ${CXX_FLAGS_MINSIZEREL}"
    )

elseif(NEURO_CAR_COMPILER_MSVC)

    # Base compiler flags
    set(CXX_BASE_FLAGS
        "/MP4"
		"/W0"
		#"/Wall"
    )
    string(REGEX REPLACE ";" " " CXX_BASE_FLAGS "${CXX_BASE_FLAGS}")

    # Release mode
    set(CXX_FLAGS_RELEASE ${CXX_BASE_FLAGS})
    set(CXX_FLAGS_RELEASE ${CXX_FLAGS_RELEASE} "/O2")
	string(REGEX REPLACE ";" " " CXX_FLAGS_RELEASE "${CXX_FLAGS_RELEASE}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CXX_FLAGS_RELEASE}")

    # Debug mode
    set(CXX_FLAGS_DEBUG ${CXX_BASE_FLAGS})
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CXX_FLAGS_DEBUG}")

    # Release with debug symbols
    set(CXX_FLAGS_RELWITHDEBINFO "${CXX_BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
        "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CXX_FLAGS_RELWITHDEBINFO}"
    )

    # Release with stripping of extra symbols
    set(CXX_FLAGS_MINSIZEREL "${CXX_BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_MINSIZEREL
        "${CMAKE_CXX_FLAGS_MINSIZEREL} ${CXX_FLAGS_MINSIZEREL}"
    )


	# Remove SL security warnings with Visual C++
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS)

else()

    message(FATAL_ERROR "Unsupported compiler")
    return()

endif()


################################################################################
#                                 CUSTOM COMMANDS                              #
################################################################################

if(NEURO_CAR_OS_LINUX)

    # Clean up all generated files
    add_custom_target(
        trash
        COMMAND rm -r ${CMAKE_CURRENT_SOURCE_DIR}/bin/* ${CMAKE_CURRENT_SOURCE_DIR}/build/*
        COMMENT "Deleting generated files"
    )

    # Delete generated files and reset project settings
    add_custom_target(
        reset
        COMMAND make trash\; cmake ..
        COMMENT "Resetting project settings"
    )

endif()


################################################################################
#                                   EXECUTABLE                                 #
################################################################################

set(NEURO_CAR_INCLUDE_DIR ./code/include)
set(NEURO_CAR_SOURCE_DIR ./code/src)

# Include directories
include_directories(${NEURO_CAR_INCLUDE_DIR} ${NEURO_CAR_SOURCE_DIR})
include_directories(SYSTEM ${NEURO_CAR_EXTERN_INCLUDE_DIRS})

# Source files
set(NEURO_CAR_HEADERS
    ${NEURO_CAR_INCLUDE_DIR}/neuro_controller.hpp
)

# Source files
set(NEURO_CAR_SOURCES
    ${NEURO_CAR_SOURCE_DIR}/neuro_controller.cpp
)

# Static library
set(LIBRARY_OUTPUT_PATH ${NEURO_CAR_BINARY_DIR})
add_library(${NEURO_CAR_STATIC_LIBRARY}
    STATIC ${NEURO_CAR_HEADERS} ${NEURO_CAR_SOURCES}
)
target_link_libraries(${NEURO_CAR_STATIC_LIBRARY} ${NEURO_CAR_EXTERN_LIBRARIES})


# Submodules
set(NEURO_CAR_SUBMODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)

set(NEURO_EVOLUTION_SUBMODULE_DIR ${NEURO_CAR_SUBMODULE_DIR}/neuro_evo)
add_subdirectory(${NEURO_EVOLUTION_SUBMODULE_DIR})
include_directories(SYSTEM ${NEURO_EVOLUTION_INCLUDE_DIR} ${NEURO_EVOLUTION_EXTERN_INCLUDE_DIR})

set(CAR_PHYSICS_SUBMODULE_DIR ${NEURO_CAR_SUBMODULE_DIR}/car_physics)
add_subdirectory(${CAR_PHYSICS_SUBMODULE_DIR})
include_directories(SYSTEM ${CAR_PHYSICS_INCLUDE_DIR} ${CAR_PHYSICS_EXTERN_INCLUDE_DIR})

# Executable
set(EXECUTABLE_OUTPUT_PATH ${NEURO_CAR_BINARY_DIR})
add_executable(${NEURO_CAR_EXECUTABLE_NAME} ${NEURO_CAR_SOURCE_DIR}/main.cpp)
target_link_libraries(${NEURO_CAR_EXECUTABLE_NAME}
    ${NEURO_EVOLUTION_STATIC_LIBRARY}
    ${CAR_PHYSICS_STATIC_LIBRARY}
    ${NEURO_CAR_STATIC_LIBRARY}
)
