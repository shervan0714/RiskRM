# File: CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(RSWCovariateShift VERSION 1.0)

# Enforce modern C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add maximum performance optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Find OpenMP for multithreading
find_package(OpenMP REQUIRED)

# --- EIGEN FOLDER SETUP ---
# IMPORTANT: Make sure this is the exact path to your Eigen folder!
include_directories("C:/Users/sherv/Downloads/rsw/eigen")

# Notice how the "src/" prefix is gone because everything is in the same folder now
add_executable(rsw_benchmark 
    main.cpp 
    rsw_model.cpp 
    dataset.cpp 
)

# Link OpenMP to our executable
if(OpenMP_CXX_FOUND)
    target_link_libraries(rsw_benchmark PUBLIC OpenMP::OpenMP_CXX)
endif()
# --- STEP 3: PYBIND11 PYTHON WRAPPER ---
# Automatically download PyBind11 so we don't have to install it manually
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Build the Python module (we don't need dataset.cpp here because Python handles the data!)
pybind11_add_module(rsw_engine bindings.cpp rsw_model.cpp)

# Make sure the Python module knows where Eigen is
target_include_directories(rsw_engine PRIVATE "C:/Users/sherv/Downloads/rsw/eigen")

# Link OpenMP for multithreaded Python training
if(OpenMP_CXX_FOUND)
    target_link_libraries(rsw_engine PRIVATE OpenMP::OpenMP_CXX)
endif()