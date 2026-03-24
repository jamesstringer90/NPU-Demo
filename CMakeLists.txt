cmake_minimum_required(VERSION 3.16)
project(npu_water_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_executable(npu_water_sim src/main.cpp)

target_link_libraries(npu_water_sim PRIVATE
    openvino::runtime
    d3d11
    dxgi
    d3dcompiler
)

# Copy the ONNX model next to the executable
if(EXISTS "${CMAKE_SOURCE_DIR}/water_physics.onnx")
    add_custom_command(TARGET npu_water_sim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/water_physics.onnx"
            "$<TARGET_FILE_DIR:npu_water_sim>/water_physics.onnx"
        COMMENT "Copying water_physics.onnx to output directory"
    )
endif()
