cmake_minimum_required(VERSION 3.16)
project(npu_water_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_OPENVINO "Enable OpenVINO backend (Intel NPU)" ON)
option(USE_ONNXRT   "Enable ONNX Runtime + DirectML backend (AMD/Qualcomm NPU)" OFF)

add_executable(npu_water_sim src/main.cpp)

target_link_libraries(npu_water_sim PRIVATE
    d3d11
    dxgi
    d3dcompiler
)

if(USE_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
    target_link_libraries(npu_water_sim PRIVATE openvino::runtime)
    target_compile_definitions(npu_water_sim PRIVATE HAS_OPENVINO)
endif()

if(USE_ONNXRT)
    find_package(onnxruntime REQUIRED)
    target_link_libraries(npu_water_sim PRIVATE onnxruntime::onnxruntime)
    target_compile_definitions(npu_water_sim PRIVATE HAS_ONNXRT)
endif()

# Copy the ONNX model next to the executable
if(EXISTS "${CMAKE_SOURCE_DIR}/water_physics.onnx")
    add_custom_command(TARGET npu_water_sim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/water_physics.onnx"
            "$<TARGET_FILE_DIR:npu_water_sim>/water_physics.onnx"
        COMMENT "Copying water_physics.onnx to output directory"
    )
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/ball_physics.onnx")
    add_custom_command(TARGET npu_water_sim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/ball_physics.onnx"
            "$<TARGET_FILE_DIR:npu_water_sim>/ball_physics.onnx"
        COMMENT "Copying ball_physics.onnx to output directory"
    )
endif()

