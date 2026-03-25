# NPU Rubber Duck Bath Simulator

A proof-of-concept demonstrating the Intel NPU (Neural Processing Unit) as a general-purpose co-processor for real-time "simulation" workloads. A single FP16 ONNX inference call per frame computes all physics (water waves, ripples, duck movement, foam, caustics, refraction, choppy displacement) entirely on the Intel AI Boost NPU via the OpenVINO C++ inference API. The CPU only copies data, the GPU handles shading.

**There is no neural network here.** The ONNX model contains zero learned weights — it's a hand-authored physics simulation (Gerstner waves, Verlet integration, Newtonian mechanics, Snell's law) expressed as a 465-node tensor graph running at FP16 precision. The NPU doesn't know it's not running a neural network; it just sees tensor operations.

I'm not a game developer and I have no background in simulation — I just wanted to see if the NPU could do something other than AI. Turns out it can.

![Windows](https://img.shields.io/badge/platform-Windows-blue)
![OpenVINO](https://img.shields.io/badge/OpenVINO-2025.0-green)
![D3D11](https://img.shields.io/badge/GPU-D3D11-orange)

https://github.com/user-attachments/assets/3c648a21-815f-4de4-a3e5-f68ef0072412  
https://github.com/user-attachments/assets/f6dcd307-a957-454c-a543-4d9d7e9846a7

## What runs on the NPU


Everything below executes as a single OpenVINO inference request on the Intel NPU — no CPU fallback, no neural network, just tensor math at FP16 half-precision:

- **32 Gerstner waves** — Phillips spectrum with deep-water dispersion, packed into batch tensors
- **Interactive ripple layer** — 8-substep Verlet wave equation with damping
- **Duck physics** — 2 ducks with hull displacement, directional wake, bow wave, Newtonian drift, slope sampling, wall bounce, duck-duck collision, buoyancy bob, tilt smoothing
- **Foam / Bubble Bath** — emission from wave height×velocity + collisions, advection by water slope, duck-foam dispersal, ripple breakup, wave peak grip, age-dependent thresholding, 3D bubble dome enhancement via unsharp mask
- **Splash** — Gaussian impulse heightfield deformation
- **Choppy displacement** — Gerstner-style horizontal vertex displacement computed on NPU
- **Surface normals** — finite-difference derivatives with runtime normalY scaling
- **Caustics** — Laplacian convolution with Gaussian pre-smoothing
- **Refraction** — Snell's law with view-angle correction (per-cell sec(theta))

All 465 ONNX nodes execute in a single synchronous `infer()` call at FP16 precision on the NPU device.

## Controls

| Input | Action |
|-------|--------|
| Left-click drag | Grab and drag a duck through water |
| Right-click drag | Orbit camera |
| Scroll wheel | Zoom in/out |
| Space | Splash at center |
| R | Reset simulation |
| T | Toggle tuning slider panel |
| F | Toggle Bubble Bath mode |
| Escape | Quit |
| `--device NPU\|CPU` | Select compute device (default: NPU) |

## Prerequisites

- Windows 10/11 (x64)
- Visual Studio 2022 (v143 toolset, C++17)
- Python 3.10+ with `onnx` and `numpy`
- [OpenVINO 2025.0](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) C++ runtime
- Intel Core Ultra processor with NPU / Intel AI Boost (Arrow Lake, Lunar Lake, Meteor Lake) — or use `--device CPU` to run on CPU via OpenVINO

## Setup

### 1. OpenVINO C++ runtime

Download the [OpenVINO 2025.0 Windows archive](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) (C++ archive, not the pip package) and extract it to:

```
openvino_rt\openvino_toolkit_windows_2025.0.0.17942.1f68be9f594_x86_64\
```

The `.vcxproj` expects this path relative to the project root. The build automatically copies required DLLs to the output directory.

### 2. Generate the ONNX model

```bash
pip install onnx numpy
python generate_model.py
```

This produces `water_physics.onnx` — a hand-built ONNX graph (opset 13, 256×256 grid, 32 waves, ~4.7 MB) with no trained weights.

### 3. Build and run

Open `npu_water_sim.sln` in Visual Studio 2022, select **Debug** or **Release** x64, and build. The post-build step copies the ONNX model, `plugins.xml`, and all OpenVINO/TBB DLLs to the output directory.

```bash
bin\Release\npu_water_sim.exe
bin\Release\npu_water_sim.exe --device CPU    # run on CPU instead
```

## Architecture

```
generate_model.py          ONNX model builder (Python, onnx.helper API)
src/main.cpp               D3D11 rendering + OpenVINO C++ inference API
plugins.xml                OpenVINO plugin config (NPU/CPU devices)
npu_water_sim.vcxproj      VS2022 project (copies DLLs + model on build)
```

### Data flow per frame

```
CPU: pack inputs (state, wave_phase, camera, duck1, dt, duck2, splash, foam_params, render_params)
  |
  v
NPU: single OpenVINO infer() call (465 ONNX nodes, FP16)
  |   - Gerstner wave synthesis (batch Sin + 1x1 Conv)
  |   - Ripple propagation (8× Verlet substeps)
  |   - Hull displacement + wake + bow wave (×2 ducks)
  |   - Splash (Gaussian impulse)
  |   - Duck-duck collision (branchless overlap + push)
  |   - Duck buoyancy + tilt smoothing
  |   - Duck slope sampling + Newtonian physics + wall bounce
  |   - Foam: emission, advection, decay, duck dispersal, ripple breakup
  |   - Choppy displacement + surface normals
  |   - Caustics + refraction
  |
  v
CPU: read outputs (state_out, render_out, duck1_out, duck2_out, wave_phase_out)
  |
  v
GPU: D3D11 vertex buffer update + pixel shading
```

## Tested hardware

| CPU | NPU | Status |
|-----|-----|--------|
| Intel Core Ultra 9 285HX | Intel AI Boost (Arrow Lake) | 30 FPS |
| Intel Core Ultra 7 256V | Intel AI Boost (Lunar Lake) | 30 FPS |

## License

MIT
