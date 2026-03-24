# NPU Rubber Duck Bath Simulator

A proof-of-concept demonstrating the Intel NPU as a general-purpose co-processor for real-time "simulation" workloads. A single FP16 ONNX inference call per frame computes all physics (water waves, ripples, duck movement, caustics, refraction) entirely on the NPU via OpenVINO. The CPU only copies data and builds vertex buffers, the GPU handles shading.

**There is no neural network here.** The ONNX graph contains zero learned weights — it's a hand-authored physics simulation (Gerstner waves, Verlet integration, Newtonian mechanics, Snell's law) expressed as 208 tensor operations. The NPU doesn't know it's not running a neural network. This treats ONNX as a "compute shader" for the NPU, showing that any computation expressible as a tensor graph can run on hardware that the industry markets exclusively for AI inference.

![Windows](https://img.shields.io/badge/platform-Windows-blue)
![OpenVINO](https://img.shields.io/badge/OpenVINO-2025.0-green)
![D3D11](https://img.shields.io/badge/GPU-D3D11-orange)

https://github.com/user-attachments/assets/6e632d36-5d03-4ab2-be6c-53e98c07bb05

## What runs on the NPU

- **32 Gerstner waves** — Phillips spectrum with deep-water dispersion, packed into batch tensors
- **Interactive ripple layer** — 8-substep Verlet wave equation with damping
- **Duck physics** — hull displacement, directional wake injection, Newtonian drift, slope sampling, wall bounce
- **Ball splash** — directional ring impulse with path interpolation scaling
- **Caustics** — Laplacian convolution with Gaussian pre-smoothing
- **Refraction** — Snell's law with view-angle correction (per-cell sec(theta))
- **Surface normals** — finite-difference derivatives for lighting

All 204 ONNX nodes execute in a single inference call at FP16 precision.

## Controls

| Input | Action |
|-------|--------|
| Left-click drag | Drag ball through water |
| Right-click drag | Rotate camera |
| Scroll wheel | Zoom in/out |
| Space | Splash at center |
| R | Reset simulation |
| T | Toggle tuning slider panel |
| Escape | Quit |
| `--device NPU\|GPU\|CPU` | Select compute device (default: NPU) |

## Prerequisites

- Windows 10/11 (x64)
- Visual Studio 2022 (v143 toolset, C++17)
- Python 3.10+ with `onnx` and `numpy`
- Intel CPU with NPU (Core Ultra series) — or use `--device CPU`/`--device GPU`

## Setup

### 1. OpenVINO C++ Runtime

Download the [OpenVINO 2025.0 Windows archive](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) (C++ archive, not pip) and extract it to:

```
openvino_rt\openvino_toolkit_windows_2025.0.0.17942.1f68be9f594_x86_64\
```

The `.vcxproj` expects this path relative to the project root. The build automatically copies required DLLs to the output directory.

### 2. Generate the ONNX model

```bash
pip install onnx numpy
python generate_model.py
```

This produces `water_physics.onnx` (256x256 grid, 32 waves, ~4.5 MB).

### 3. Build and run

Open `npu_water_sim.sln` in Visual Studio 2022, select **Debug** or **Release** x64, and build. The post-build step copies the ONNX model, `plugins.xml`, and all OpenVINO/TBB DLLs to the output directory.

```bash
bin\Release\npu_water_sim.exe
bin\Release\npu_water_sim.exe --device CPU   # run on CPU instead
```

## Architecture

```
generate_model.py          ONNX model generator (Python)
src/main.cpp               Application: D3D11 rendering + OpenVINO inference
plugins.xml                OpenVINO plugin registry (NPU/GPU/CPU)
npu_water_sim.vcxproj      VS2022 project (copies DLLs + model on build)
```

### Data flow per frame

```
CPU: state + wave_phase + camera + duck + ball + dt
  |
  v
NPU: single ONNX inference (204 nodes, FP16)
  |   - Gerstner wave synthesis (batch Sin + 1x1 Conv)
  |   - Ripple propagation (8x Verlet substeps)
  |   - Hull displacement + wake injection
  |   - Ball splash (ring impulse + facing dot)
  |   - Duck slope sampling + physics + wall bounce
  |   - Caustics + refraction + normals
  |
  v
CPU: read state_out + render_out + duck_out + wave_phase_out
  |
  v
GPU: D3D11 vertex buffer update + shading
```

## Tested hardware

| CPU | NPU | Status |
|-----|-----|--------|
| Intel Core Ultra 9 285HX | Intel AI Boost (Arrow Lake) | 30 FPS |
| Intel Core Ultra 7 256V | Intel AI Boost (Lunar Lake) | 30 FPS |

## License

MIT
