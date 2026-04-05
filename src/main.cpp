/*
 * NPU Ocean Simulation
 *
 * Gerstner wave ocean synthesis on the Intel NPU via OpenVINO.
 * A single NPU inference call per frame performs:
 *   - 32 Gerstner waves with Phillips spectrum + deep-water dispersion
 *   - 8-step interactive ripple layer (for duck/splash interaction)
 *   - Render output: scaled heights + surface derivatives for normals/choppiness
 *
 * The CPU only copies data in/out and builds the vertex buffer.
 * The GPU handles shading via D3D11.
 *
 * Controls:
 *   Left-drag   = grab and drag a duck through water
 *   Space       = splash at center
 *   R           = reset simulation
 *   T           = toggle tuning slider panel
 *   F           = toggle Bubble Bath mode
 *   Escape      = quit
 */

#include <windows.h>
#include <intrin.h>
#include <commctrl.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <DirectXPackedVector.h>

#ifdef HAS_OPENVINO
#include <openvino/openvino.hpp>
#endif

#ifdef HAS_WINDOWSML
#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Microsoft.Windows.AI.MachineLearning.h>
#include <winml/onnxruntime_cxx_api.h>
namespace WinML = winrt::Microsoft::Windows::AI::MachineLearning;
#endif

#include <vector>
#include <random>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <string>
#include <memory>
#include <unordered_map>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "comctl32.lib")

using namespace DirectX;

// ---------------------------------------------------------------------------
// Inference backend abstraction — allows OpenVINO or ONNX Runtime
// ---------------------------------------------------------------------------
struct InferenceBackend {
    virtual ~InferenceBackend() = default;
    virtual uint16_t* getInput(int index) = 0;
    virtual const uint16_t* getOutput(int index) = 0;
    virtual void infer() = 0;
    virtual std::string name() const = 0;
    virtual bool isBf16() const { return false; }
    virtual bool isFp32Model() const { return false; }
};

// FP16 conversion
inline uint16_t f32_to_fp16(float v) { return PackedVector::XMConvertFloatToHalf(v); }
inline float fp16_to_f32(uint16_t v) { return PackedVector::XMConvertHalfToFloat(v); }

// BF16: upper 16 bits of IEEE 754 float32
inline uint16_t f32_to_bf16(float v) { uint32_t u; memcpy(&u, &v, 4); return uint16_t(u >> 16); }
inline float bf16_to_f32(uint16_t v) { uint32_t u = uint32_t(v) << 16; float f; memcpy(&f, &u, 4); return f; }

// Active conversion functions — set once when backend is initialized
static uint16_t (*f32_to_f16)(float) = f32_to_fp16;
static float (*f16_to_f32)(uint16_t) = fp16_to_f32;

#ifdef HAS_OPENVINO
class OpenVinoBackend : public InferenceBackend {
    ov::InferRequest req_;
    std::string device_;
public:
    OpenVinoBackend(const std::string& modelPath, const std::string& device)
        : device_(device)
    {
        ov::Core core;
        auto devices = core.get_available_devices();
        printf("OpenVINO devices:\n");
        bool deviceFound = false;
        for (auto& d : devices) {
            std::string caps;
            try {
                auto opt = core.get_property(d, ov::device::capabilities);
                for (auto& c : opt) { if (!caps.empty()) caps += ", "; caps += c; }
            } catch (...) {}
            printf("  %-8s  [%s]\n", d.c_str(), caps.c_str());
            if (d.find(device) != std::string::npos) deviceFound = true;
        }
        if (!deviceFound) {
            std::string msg = "Device '" + device + "' not found.\nAvailable: ";
            for (auto& d : devices) msg += d + " ";
            throw std::runtime_error(msg);
        }
        auto model = core.read_model(modelPath);
        printf("Model loaded, compiling for %s ...\n", device.c_str());
        auto compiled = core.compile_model(model, device);
        req_ = compiled.create_infer_request();
    }

    uint16_t* getInput(int i) override {
        return reinterpret_cast<uint16_t*>(req_.get_input_tensor(i).data<ov::float16>());
    }
    const uint16_t* getOutput(int i) override {
        return reinterpret_cast<const uint16_t*>(req_.get_output_tensor(i).data<ov::float16>());
    }
    void infer() override { req_.infer(); }
    std::string name() const override { return "OpenVINO (" + device_ + ")"; }

    static bool hasIntelNPU() {
        try {
            ov::Core probe;
            for (auto& d : probe.get_available_devices())
                if (d.find("NPU") != std::string::npos) return true;
        } catch (...) {}
        return false;
    }
};
#endif

// ---------------------------------------------------------------------------
// Windows ML backend — auto-selects vendor EP (QNN, Vitis AI, DirectML, etc.)
// Uses ONNX Runtime bundled in Microsoft.WindowsAppSDK.ML NuGet package.
// The EP catalog downloads vendor-specific NPU drivers via Windows Update.
// ---------------------------------------------------------------------------
#ifdef HAS_WINDOWSML
class WindowsMLBackend : public InferenceBackend {
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memInfo_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<std::vector<uint16_t>> inputBufs_, outputBufs_;
    std::vector<std::vector<float>> floatInBufs_, floatOutBufs_;  // for FP32 model path
    std::vector<std::vector<int64_t>> inputShapes_, outputShapes_;
    std::vector<std::string> inputNames_, outputNames_;
    std::vector<const char*> inputNamePtrs_, outputNamePtrs_;
    std::string epName_, devType_;
    bool useFp32Model_ = false;  // model is FP32, need float conversion buffers
    bool hostBf16_ = false;      // host data is BF16 (Vitis AI); false = FP16

public:
    WindowsMLBackend(const std::string& modelPath, const std::string& device = "NPU")
        : env_(ORT_LOGGING_LEVEL_WARNING, "npu_bath")
    {
        printf("Windows ML: init\n");
        fflush(stdout);

        // Use WinRT ExecutionProviderCatalog to discover, download, and register
        // all certified vendor EPs (QNN, Vitis AI, OpenVINO, etc.)
        printf("Windows ML: discovering and registering execution providers...\n");
        fflush(stdout);
        bool hasVitisAI = false;
        try {
            winrt::init_apartment(winrt::apartment_type::multi_threaded);
            printf("  WinRT apartment initialized\n");
            fflush(stdout);
            WinML::ExecutionProviderCatalog catalog = WinML::ExecutionProviderCatalog::GetDefault();
            printf("  Got EP catalog, calling EnsureAndRegisterCertifiedAsync...\n");
            fflush(stdout);
            auto op = catalog.EnsureAndRegisterCertifiedAsync();
            op.Progress([](auto const&, double progress) {
                printf("  EP registration progress: %.0f%%\n", progress);
                fflush(stdout);
            });
            winrt::Windows::Foundation::Collections::IVector<WinML::ExecutionProvider> eps = op.get();
            printf("  EP registration complete. Found %u EPs:\n", eps.Size());
            fflush(stdout);
            for (uint32_t i = 0; i < eps.Size(); i++) {
                WinML::ExecutionProvider ep = eps.GetAt(i);
                std::string name = winrt::to_string(ep.Name());
                WinML::ExecutionProviderReadyState rs = ep.ReadyState();
                const char* state = "unknown";
                if (rs == WinML::ExecutionProviderReadyState::Ready)      state = "ready+registered";
                else if (rs == WinML::ExecutionProviderReadyState::NotReady)  state = "not ready";
                else if (rs == WinML::ExecutionProviderReadyState::NotPresent) state = "not present";
                printf("  EP[%u]: %-30s [%s]\n", i, name.c_str(), state);
                fflush(stdout);
                if (name.find("VitisAI") != std::string::npos &&
                    rs == WinML::ExecutionProviderReadyState::Ready) {
                    hasVitisAI = true;
                }
            }
            if (eps.Size() == 0) printf("  (no vendor EPs found)\n");
        } catch (const winrt::hresult_error& e) {
            printf("  EP catalog error: %ls (0x%08X)\n", e.message().c_str(), static_cast<unsigned>(e.code()));
        }
        fflush(stdout);

        // Select FP32 model when needed (Vitis AI or GPU/DirectML)
        std::string actualModel = modelPath;
        std::string fp32Path = modelPath.substr(0, modelPath.rfind('.')) + "_fp32.onnx";
        bool fp32Exists = GetFileAttributesA(fp32Path.c_str()) != INVALID_FILE_ATTRIBUTES;

        if (hasVitisAI) {
            if (fp32Exists) {
                actualModel = fp32Path;
                useFp32Model_ = true;
                hostBf16_ = true;  // Vitis AI uses BF16 encoding
                printf("  Vitis AI detected — using FP32 model: %s\n", fp32Path.c_str());
            } else {
                printf("  WARNING: Vitis AI detected but %s not found — using FP16\n", fp32Path.c_str());
            }
        } else if (device == "GPU" && fp32Exists) {
            // DirectML has FP16 Conv precision issues — use FP32 model
            actualModel = fp32Path;
            useFp32Model_ = true;
            hostBf16_ = false;  // host stays FP16
            printf("  GPU mode — using FP32 model (avoids DirectML FP16 Conv issues): %s\n", fp32Path.c_str());
        }
        printf("  Model file: %s\n", actualModel.c_str());
        fflush(stdout);

        // Configure session
        printf("  Creating session options...\n");
        fflush(stdout);
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (device == "GPU") {
            printf("  Device: GPU (DirectML)\n");
            opts.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);
        } else if (hasVitisAI && device != "CPU") {
            // Write minimal Vitis AI config — no target section (Windows ML provides its own)
            const char* configJson = R"({
  "passes": [
    {"name": "init", "plugin": "vaip-pass_init"},
    {"name": "vaiml_partition", "plugin": "vaip-pass_vaiml_partition",
     "vaiml_config": {"enable_f32_to_bf16_conversion": true}}
  ]
})";
            FILE* f = fopen("vai_ep_config.json", "w");
            if (f) { fputs(configJson, f); fclose(f); }
            printf("  Wrote vai_ep_config.json (enable_f32_to_bf16_conversion=true)\n");
            fflush(stdout);

            // Use generic C API — EP already registered by Windows ML
            const char* keys[] = {"config_file"};
            const char* vals[] = {"vai_ep_config.json"};
            OrtStatus* status = Ort::GetApi().SessionOptionsAppendExecutionProvider(
                opts, "VitisAI", keys, vals, 1);
            if (status) {
                printf("  AppendExecutionProvider(VitisAI) failed: %s\n",
                       Ort::GetApi().GetErrorMessage(status));
                Ort::GetApi().ReleaseStatus(status);
                printf("  Falling back to PREFER_NPU\n");
                opts.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_NPU);
            } else {
                printf("  Appended VitisAI EP with config_file\n");
            }
            fflush(stdout);
        } else {
            opts.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_NPU);
        }

        // Load model — this is where the EP compiler runs (may take minutes on first run)
        printf("  Creating ORT session (EP compilation may take a few minutes on first run)...\n");
        fflush(stdout);
        auto t0 = std::chrono::steady_clock::now();
        std::wstring wpath(actualModel.begin(), actualModel.end());
        session_ = Ort::Session(env_, wpath.c_str(), opts);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        printf("  Session created in %.1f seconds\n", secs);
        fflush(stdout);

        // Query input/output names and shapes from model
        Ort::AllocatorWithDefaultOptions alloc;
        size_t numIn = session_.GetInputCount();
        size_t numOut = session_.GetOutputCount();
        printf("  Model: %zu inputs, %zu outputs\n", numIn, numOut);

        for (size_t i = 0; i < numIn; i++) {
            auto n = session_.GetInputNameAllocated(i, alloc);
            inputNames_.push_back(n.get());
            auto info = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
            auto elemType = info.GetElementType();
            inputShapes_.push_back(info.GetShape());
            int64_t count = 1;
            std::string shapeStr = "[";
            for (auto d : inputShapes_.back()) { count *= d; shapeStr += std::to_string(d) + ","; }
            shapeStr.back() = ']';
            printf("  Input[%zu]: %-20s shape=%-20s type=%d elements=%lld\n",
                   i, inputNames_.back().c_str(), shapeStr.c_str(), (int)elemType, (long long)count);
            inputBufs_.emplace_back(count, uint16_t(0));
        }
        for (size_t i = 0; i < numOut; i++) {
            auto n = session_.GetOutputNameAllocated(i, alloc);
            outputNames_.push_back(n.get());
            auto info = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
            auto elemType = info.GetElementType();
            outputShapes_.push_back(info.GetShape());
            int64_t count = 1;
            std::string shapeStr = "[";
            for (auto d : outputShapes_.back()) { count *= d; shapeStr += std::to_string(d) + ","; }
            shapeStr.back() = ']';
            printf("  Output[%zu]: %-20s shape=%-20s type=%d elements=%lld\n",
                   i, outputNames_.back().c_str(), shapeStr.c_str(), (int)elemType, (long long)count);
            outputBufs_.emplace_back(count, uint16_t(0));
        }

        for (auto& s : inputNames_) inputNamePtrs_.push_back(s.c_str());
        for (auto& s : outputNames_) outputNamePtrs_.push_back(s.c_str());

        // Pre-allocate float buffers for FP32 model path (Vitis AI or GPU)
        if (useFp32Model_) {
            for (auto& buf : inputBufs_)  floatInBufs_.emplace_back(buf.size(), 0.0f);
            for (auto& buf : outputBufs_) floatOutBufs_.emplace_back(buf.size(), 0.0f);
            printf("  Allocated FP32 conversion buffers\n");
        }
        fflush(stdout);

        // Report which EP/device was selected for the model
        printf("  Querying EP device assignment...\n");
        fflush(stdout);
        try {
            auto devices = session_.GetEpDeviceForInputs();
            printf("  %zu device assignment(s):\n", devices.size());
            for (auto& dev : devices) {
                const char* devType = "unknown";
                switch (dev.Device().Type()) {
                    case 0: devType = "CPU"; break;
                    case 1: devType = "GPU"; break;
                    case 2: devType = "NPU"; break;
                }
                printf("  -> EP: %s  vendor: %s  device: %s\n", dev.EpName(), dev.EpVendor(), devType);
                epName_ = dev.EpName();
                devType_ = devType;
            }
        } catch (const Ort::Exception& e) {
            printf("  GetEpDeviceForInputs failed: %s\n", e.what());
        } catch (...) {
            printf("  GetEpDeviceForInputs failed (unknown error)\n");
        }

        printf("Windows ML: ready (%s)\n", name().c_str());
        fflush(stdout);
    }

    uint16_t* getInput(int i) override { return inputBufs_[i].data(); }
    const uint16_t* getOutput(int i) override { return outputBufs_[i].data(); }

    void infer() override {
        if (useFp32Model_) {
            // FP32 model path: convert host uint16_t inputs → float, run, convert back
            std::vector<Ort::Value> inputs;
            for (size_t i = 0; i < inputBufs_.size(); i++) {
                if (hostBf16_) {
                    // BF16 → float32 (shift left 16 bits)
                    for (size_t j = 0; j < inputBufs_[i].size(); j++) {
                        uint32_t u = uint32_t(inputBufs_[i][j]) << 16;
                        memcpy(&floatInBufs_[i][j], &u, 4);
                    }
                } else {
                    // FP16 → float32 (proper half-float conversion)
                    for (size_t j = 0; j < inputBufs_[i].size(); j++)
                        floatInBufs_[i][j] = fp16_to_f32(inputBufs_[i][j]);
                }
                inputs.push_back(Ort::Value::CreateTensor(
                    memInfo_, floatInBufs_[i].data(), floatInBufs_[i].size() * sizeof(float),
                    inputShapes_[i].data(), inputShapes_[i].size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
            }

            auto outputs = session_.Run(Ort::RunOptions{nullptr},
                inputNamePtrs_.data(), inputs.data(), inputs.size(),
                outputNamePtrs_.data(), outputNamePtrs_.size());

            for (size_t i = 0; i < outputBufs_.size(); i++) {
                auto* src = outputs[i].GetTensorData<float>();
                if (hostBf16_) {
                    // float32 → BF16 (shift right 16 bits)
                    for (size_t j = 0; j < outputBufs_[i].size(); j++) {
                        uint32_t u;
                        memcpy(&u, &src[j], 4);
                        outputBufs_[i][j] = uint16_t(u >> 16);
                    }
                } else {
                    // float32 → FP16 (proper conversion)
                    for (size_t j = 0; j < outputBufs_[i].size(); j++)
                        outputBufs_[i][j] = f32_to_fp16(src[j]);
                }
            }
        } else {
            // FP16 model path: direct uint16_t tensors
            std::vector<Ort::Value> inputs;
            for (size_t i = 0; i < inputBufs_.size(); i++) {
                inputs.push_back(Ort::Value::CreateTensor(
                    memInfo_, inputBufs_[i].data(), inputBufs_[i].size() * sizeof(uint16_t),
                    inputShapes_[i].data(), inputShapes_[i].size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
            }

            auto outputs = session_.Run(Ort::RunOptions{nullptr},
                inputNamePtrs_.data(), inputs.data(), inputs.size(),
                outputNamePtrs_.data(), outputNamePtrs_.size());

            for (size_t i = 0; i < outputBufs_.size(); i++) {
                auto* src = outputs[i].GetTensorData<uint16_t>();
                memcpy(outputBufs_[i].data(), src, outputBufs_[i].size() * sizeof(uint16_t));
            }
        }
    }

    std::string name() const override {
        if (!epName_.empty()) return "Windows ML (" + epName_ + ", " + devType_ + ")";
        return "Windows ML";
    }
    bool isBf16() const override { return hostBf16_; }
    bool isFp32Model() const override { return useFp32Model_; }
};
#endif

enum class RuntimeChoice { Auto, OpenVINO, WindowsML };
static RuntimeChoice g_runtimeChoice = RuntimeChoice::Auto;

enum class Scene { Water, Balls };
static Scene g_scene = Scene::Water;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static constexpr int   GRID     = 256;
static constexpr int   N_WAVES  = 32;
static constexpr int   WIN_W    = 1280;
static constexpr int   WIN_H    = 720;

// Runtime-adjustable parameters (controlled by on-screen sliders)
// Defaults tuned for gentle Sea-of-Thieves-style ocean (model HEIGHT_SCALE=3)
static float g_normalY      = 0.5f;
static float g_chopScale    = 3.815f;
static float g_heightScale  = 2.06f;
static float g_fresnelPow   = 1.0f;
static float g_fresnelMin   = 0.107f;
static float g_fresnelMax   = 0.595f;
static float g_specPow      = 10.0f;
static float g_specStr      = 1.826f;
static float g_camDist      = 1.267f;   // fraction of GRID
static float g_camHeight    = 0.575f;   // fraction of GRID
static float g_camSpeed     = 0.0f;
static float g_camAngle     = 0.0f;     // manual orbit angle (radians)
static float g_foamThreshold = 0.028f;  // minimum foam density to display
static float g_foamCoarseness = 0.0f;   // noise scale for organic foam breakup
static float g_foamDecay = 0.971f;      // foam decay per frame
static float g_foamGeneration = 0.083f; // foam particles generated when threshold met
static float g_foamOpacity = 4.395f;    // foam transparency curve (higher = more opaque)
static bool  g_foamEnabled  = false;    // F key toggles Bubble Bath mode
static bool  g_rDragging    = false;    // right-click drag active
static float g_rDragStartX  = 0.0f;    // mouse X at right-click start
static float g_rDragStartAngle = 0.0f; // camAngle at right-click start


// ---------------------------------------------------------------------------
// Vertex layout
// ---------------------------------------------------------------------------
struct Vertex {
    XMFLOAT3 pos;
    XMFLOAT3 nrm;
    float    foam;   // sea foam intensity [0..1] — white foam at wave crests + wakes
};

// ---------------------------------------------------------------------------
// Constant buffer (matches HLSL layout exactly)
// ---------------------------------------------------------------------------
struct alignas(16) CB {
    XMFLOAT4X4 wvp;                // 64 bytes — world*view*proj
    XMFLOAT4X4 world;              // 64 bytes — world matrix (identity for water)
    XMFLOAT3   lightDir; float _0; // 16 bytes
    XMFLOAT3   eye;      float _1; // 16 bytes
    float      time;     float fresnelPow; float fresnelMin; float fresnelMax; // 16 bytes
    float      specPow;  float specStr;    float foamOpacity; float _3; // 16 bytes  (total: 192)
};

// ---------------------------------------------------------------------------
// Ball pit configuration
// ---------------------------------------------------------------------------
static constexpr int   NUM_BALLS        = 1024;
static constexpr float BALL_RADIUS      = 2.0f;
static constexpr float BOX_HALF         = 30.0f;
static constexpr float BOX_HEIGHT       = 10.0f;
static constexpr float BALL_GRAVITY     = -49.05f;
static constexpr float BALL_RESTITUTION = 0.6f;
static constexpr float BALL_FRICTION    = 0.98f;

struct BallVertex {
    XMFLOAT3 pos;
    XMFLOAT3 nrm;
};

struct SphereInstance {
    XMFLOAT3 pos;
    float    radius;
    XMFLOAT3 color;
};

struct Ball {
    XMFLOAT3 pos;
    XMFLOAT3 vel;
    float    radius;
    XMFLOAT3 color;
};

// ---------------------------------------------------------------------------
// HLSL sources (embedded)
// ---------------------------------------------------------------------------
static const char* g_vsSource = R"(
cbuffer CB : register(b0) {
    float4x4 wvp;
    float4x4 world;
    float3   lightDir; float _0;
    float3   eye;      float _1;
    float    time;     float fresnelPow; float fresnelMin; float fresnelMax;
    float    specPow;  float specStr;    float foamOpacity; float _3;
};
struct I { float3 p : POSITION; float3 n : NORMAL; float foam : FOAM; };
struct O { float4 sv : SV_POSITION; float3 n : NORMAL; float3 wp : TEXCOORD0; float foam : TEXCOORD1; };
O main(I i) {
    O o;
    o.wp = mul(float4(i.p, 1), world).xyz;
    o.sv = mul(float4(i.p, 1), wvp);
    o.n  = mul(float4(i.n, 0), world).xyz;
    o.foam = i.foam;
    return o;
}
)";

static const char* g_psSource = R"(
static const float PI = 3.141592653589793;

cbuffer CB : register(b0) {
    float4x4 wvp;
    float4x4 world;
    float3   lightDir; float _0;
    float3   eye;      float _1;
    float    time;     float fresnelPow; float fresnelMin; float fresnelMax;
    float    specPow;  float specStr;    float foamOpacity; float _3;
};

float3 getSkyColor(float3 e) {
    e.y = max(e.y, 0.0);
    return float3(pow(1.0 - e.y, 2.0), 1.0 - e.y, 0.6 + (1.0 - e.y) * 0.4) * 1.1;
}

struct I { float4 sv : SV_POSITION; float3 n : NORMAL; float3 wp : TEXCOORD0; float foam : TEXCOORD1; };

float4 main(I i) : SV_TARGET {
    float3 N = normalize(i.n);
    float3 L = normalize(lightDir);
    float3 V = normalize(eye - i.wp);

    float NdotV = saturate(dot(N, V));
    float fresnel = pow(1.0 - NdotV, fresnelPow) * fresnelMax + fresnelMin;

    float3 reflected = reflect(-V, N);
    float3 skyRefl = getSkyColor(reflected);

    float3 seaBase  = float3(0.0, 0.09, 0.18);
    float3 seaWater = float3(0.8, 0.9, 0.6) * 0.6;

    float depth = saturate(-N.y * 0.5 + 0.5);
    float3 waterColor = seaBase + seaWater * depth * 0.18;

    float3 color = lerp(waterColor, skyRefl, fresnel);

    float NdotL = dot(N, L);
    float diffuse = pow(saturate(NdotL * 0.4 + 0.6), 30.0);
    color += seaWater * diffuse * 0.08;

    float3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float spec = pow(NdotH, specPow);
    float dist = length(eye - i.wp);
    spec *= specStr / (1.0 + dist * 0.004);
    color += float3(1.0, 1.0, 1.0) * spec;

    float h = i.wp.y;
    color += seaWater * saturate(h * 0.02) * 0.15;

    // Sea foam — NPU-simulated bubble layer
    // foamOpacity controls transparency: thin foam fades, thick foam stays opaque
    float foamAmt = saturate(i.foam * foamOpacity);

    float foamDiffuse = saturate(dot(N, L));
    float foamHalf    = foamDiffuse * 0.6 + 0.4;
    float3 foamLit    = float3(0.95, 0.97, 1.0) * foamHalf;
    float3 foamShadow = float3(0.55, 0.62, 0.72);
    float3 foamColor  = lerp(foamShadow, foamLit, foamDiffuse * 0.7 + 0.3);
    float foamSpec = pow(NdotH, 80.0) * 1.5;
    foamColor += float3(1.0, 1.0, 1.0) * foamSpec;

    color = lerp(color, foamColor, foamAmt);

    // Gamma correction
    color = pow(saturate(color), 0.65);

    float alpha = saturate(fresnel + 0.25);
    alpha = lerp(alpha, 1.0, foamAmt);

    return float4(color, alpha);
}
)";


// Tile floor pixel shader — blue tiles + white grout + NPU-computed caustics
static const char* g_tilePsSource = R"(
Texture2D    causticTex : register(t0);
SamplerState samp       : register(s0);
cbuffer CB : register(b0) {
    float4x4 wvp;
    float4x4 world;
    float3   lightDir; float _0;
    float3   eye;      float _1;
    float    time;     float fresnelPow; float fresnelMin; float fresnelMax;
    float    specPow;  float specStr;    float foamOpacity; float _3;
};
struct I { float4 sv : SV_POSITION; float3 n : NORMAL; float3 wp : TEXCOORD0; };
float4 main(I i) : SV_TARGET {
    // Sample NPU data: (caustic, refract_x, refract_z, 0)
    float gridSize = 256.0;
    float2 npuUV = float2(i.wp.x + gridSize * 0.5, i.wp.z + gridSize * 0.5) / gridSize;
    float4 npuData = causticTex.Sample(samp, npuUV);

    // NPU-computed Snell's law refraction — shift tile pattern through water
    float2 refractedWP = float2(i.wp.x + npuData.g, i.wp.z + npuData.b);

    // Tile grid at refracted position
    float tileSize = 16.0;
    float2 uv = refractedWP / tileSize;
    float2 f = frac(uv + 0.5);
    float grout = 0.04;
    float isGrout = (f.x < grout || f.x > 1.0 - grout || f.y < grout || f.y > 1.0 - grout) ? 1.0 : 0.0;

    float3 tileColor = float3(0.42, 0.72, 0.88);
    float3 groutColor = float3(0.92, 0.93, 0.95);
    float3 base = lerp(tileColor, groutColor, isGrout);

    // NPU-computed caustics — Laplacian of wave surface = light focusing
    base += float3(0.9, 0.95, 1.0) * saturate(npuData.r) * 0.5;

    // Simple lighting
    float3 N = float3(0, 1, 0);
    float NdotL = saturate(dot(N, normalize(lightDir)));
    base *= NdotL * 0.4 + 0.6;

    return float4(base, 1.0);
}
)";

// Duck body pixel shader — glossy rubber duck yellow
static const char* g_duckPsSource = R"(
cbuffer CB : register(b0) {
    float4x4 wvp;
    float4x4 world;
    float3   lightDir; float _0;
    float3   eye;      float _1;
    float    time;     float fresnelPow; float fresnelMin; float fresnelMax;
    float    specPow;  float specStr;    float foamOpacity; float _3;
};
struct I { float4 sv : SV_POSITION; float3 n : NORMAL; float3 wp : TEXCOORD0; };
float4 main(I i) : SV_TARGET {
    float3 N = normalize(i.n);
    float3 L = normalize(lightDir);
    float3 V = normalize(eye - i.wp);
    float NdotL = saturate(dot(N, L) * 0.5 + 0.5);
    float3 H = normalize(L + V);
    // Glossy rubber specular — broad soft highlight
    float spec = pow(max(dot(N, H), 0.0), 30.0) * 0.5;
    // Rich warm yellow like a real rubber duck
    float3 yellow = float3(1.0, 0.85, 0.05);
    // Subtle rim light for that plastic toy sheen
    float rim = pow(1.0 - saturate(dot(N, V)), 3.0) * 0.15;
    float3 color = yellow * (NdotL * 0.8 + 0.2) + float3(1,1,0.8) * spec + yellow * rim;
    return float4(color, 1.0);
}
)";

// Duck bill pixel shader — orange
static const char* g_billPsSource = R"(
cbuffer CB : register(b0) {
    float4x4 wvp;
    float4x4 world;
    float3   lightDir; float _0;
    float3   eye;      float _1;
    float    time;     float fresnelPow; float fresnelMin; float fresnelMax;
    float    specPow;  float specStr;    float foamOpacity; float _3;
};
struct I { float4 sv : SV_POSITION; float3 n : NORMAL; float3 wp : TEXCOORD0; };
float4 main(I i) : SV_TARGET {
    float3 N = normalize(i.n);
    float3 L = normalize(lightDir);
    float3 V = normalize(eye - i.wp);
    float NdotL = saturate(dot(N, L) * 0.5 + 0.5);
    float3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 30.0) * 0.4;
    float3 orange = float3(1.0, 0.55, 0.05);
    float3 color = orange * (NdotL * 0.8 + 0.2) + float3(1,1,0.8) * spec;
    return float4(color, 1.0);
}
)";

// Ball pit instanced vertex shader
static const char* g_ballVsSource = R"(
cbuffer CB : register(b0) {
    float4x4 wvp;
    float4x4 world;
    float3   lightDir; float _0;
    float3   eye;      float _1;
    float    time;     float fresnelPow; float fresnelMin; float fresnelMax;
    float    specPow;  float specStr;    float foamOpacity; float _3;
};
struct VSIn {
    float3 pos : POSITION;
    float3 nrm : NORMAL;
    float3 iPos : INST_POS;
    float  iRad : INST_RAD;
    float3 iCol : INST_COL;
};
struct VSOut {
    float4 sv  : SV_POSITION;
    float3 n   : NORMAL;
    float3 wp  : TEXCOORD0;
    float3 col : TEXCOORD1;
};
VSOut main(VSIn i) {
    VSOut o;
    float3 wp = i.pos * i.iRad + i.iPos;
    o.wp = wp;
    o.sv = mul(float4(wp, 1), wvp);
    o.n  = i.nrm;
    o.col = i.iCol;
    return o;
}
)";

// Ball pit sphere pixel shader (Blinn-Phong with rim light)
static const char* g_ballPsSource = R"(
cbuffer CB : register(b0) {
    float4x4 wvp;
    float4x4 world;
    float3   lightDir; float _0;
    float3   eye;      float _1;
    float    time;     float fresnelPow; float fresnelMin; float fresnelMax;
    float    specPow;  float specStr;    float foamOpacity; float _3;
};
struct PSIn {
    float4 sv  : SV_POSITION;
    float3 n   : NORMAL;
    float3 wp  : TEXCOORD0;
    float3 col : TEXCOORD1;
};
float4 main(PSIn i) : SV_TARGET {
    float3 N = normalize(i.n);
    float3 L = normalize(lightDir);
    float3 V = normalize(eye - i.wp);
    float3 R = reflect(-L, N);
    float3 H = normalize(L + V);
    float diff = saturate(dot(N, L)) * 0.6 + 0.35;
    float spec = pow(max(dot(N, H), 0.0), 256.0) * 1.2;
    float spec2 = pow(max(dot(V, R), 0.0), 64.0) * 0.4;
    float rim = pow(1.0 - saturate(dot(N, V)), 4.0) * 0.3;
    float fresnel = lerp(0.04, 0.6, pow(1.0 - saturate(dot(N, V)), 5.0));
    float3 envUp = float3(0.6, 0.7, 0.8);
    float3 envDn = float3(0.2, 0.2, 0.25);
    float3 env = lerp(envDn, envUp, N.y * 0.5 + 0.5);
    float3 color = i.col * diff + float3(1,1,1) * (spec + spec2) + fresnel * env + i.col * rim;
    return float4(color, 1.0);
}
)";

// Box pixel shader (solid gray with simple lighting)
static const char* g_boxPsSource = R"(
cbuffer CB : register(b0) {
    float4x4 wvp;
    float4x4 world;
    float3   lightDir; float _0;
    float3   eye;      float _1;
    float    time;     float fresnelPow; float fresnelMin; float fresnelMax;
    float    specPow;  float specStr;    float foamOpacity; float _3;
};
struct PSIn {
    float4 sv : SV_POSITION;
    float3 n  : NORMAL;
    float3 wp : TEXCOORD0;
    float  foam : TEXCOORD1;
};
float4 main(PSIn i) : SV_TARGET {
    float3 N = normalize(i.n);
    float3 L = normalize(lightDir);
    float diff = saturate(dot(N, L)) * 0.5 + 0.5;
    float3 color = float3(0.85, 0.85, 0.88) * diff;
    return float4(color, 1.0);
}
)";

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static HWND                    g_hwnd   = nullptr;
static ID3D11Device*           g_dev    = nullptr;
static ID3D11DeviceContext*    g_ctx    = nullptr;
static IDXGISwapChain*         g_sc     = nullptr;
static ID3D11RenderTargetView* g_rtv    = nullptr;
static ID3D11DepthStencilView* g_dsv    = nullptr;
static ID3D11Buffer*           g_vb     = nullptr;
static ID3D11Buffer*           g_ib     = nullptr;
static ID3D11Buffer*           g_cb     = nullptr;
static ID3D11InputLayout*      g_il     = nullptr;
static ID3D11VertexShader*     g_vs     = nullptr;
static ID3D11PixelShader*      g_ps     = nullptr;
static ID3D11RasterizerState*  g_rs     = nullptr;
static ID3D11BlendState*       g_blend  = nullptr;


// Tile floor
static ID3D11Buffer*           g_tileVB  = nullptr;
static ID3D11Buffer*           g_tileIB  = nullptr;
static ID3D11PixelShader*      g_tilePS  = nullptr;
static UINT                    g_tileNumIdx = 0;
static constexpr float         FLOOR_Y   = -20.0f;

// Caustic texture (NPU-computed, uploaded each frame)
static ID3D11Texture2D*             g_causticTex = nullptr;
static ID3D11ShaderResourceView*    g_causticSRV = nullptr;
static ID3D11SamplerState*          g_sampler    = nullptr;

// Rubber duck (body + bill drawn separately for different colors)
static ID3D11Buffer*           g_duckVB  = nullptr;
static ID3D11Buffer*           g_duckIB  = nullptr;
static ID3D11PixelShader*      g_duckPS  = nullptr;
static ID3D11PixelShader*      g_billPS  = nullptr;
static UINT                    g_duckBodyNumIdx = 0;
static UINT                    g_duckBillStart  = 0;
static UINT                    g_duckBillNumIdx = 0;
// Duck 1
static float g_duckX = 30.0f, g_duckZ = -20.0f, g_duckY = 0.0f;
static float g_duckVX = 0.0f, g_duckVZ = 0.0f;
static float g_duckPrevX = 30.0f, g_duckPrevZ = -20.0f;
static float g_duckTiltX = 0.0f, g_duckTiltZ = 0.0f;
// Duck 2
static float g_duck2X = -30.0f, g_duck2Z = 20.0f, g_duck2Y = 0.0f;
static float g_duck2VX = 0.0f, g_duck2VZ = 0.0f;
static float g_duck2PrevX = -30.0f, g_duck2PrevZ = 20.0f;
static float g_duck2TiltX = 0.0f, g_duck2TiltZ = 0.0f;
static constexpr float DUCK_SCALE = 45.0f;
// Mouse drag state
static int   g_draggedDuck = -1;  // -1 = none, 0 = duck1, 1 = duck2

static std::unique_ptr<InferenceBackend> g_backend;
static std::vector<uint16_t>        g_state;      // [4 * GRID * GRID] — FP16 ripple state
static std::vector<uint16_t>        g_wavePhase;  // [32] — per-wave wrapped omega*time (maintained on NPU)
static std::vector<uint16_t>        g_renderBuf;  // [10 * GRID * GRID] — NPU render output (pos, nrm, caustic, refract, foam)
static std::vector<Vertex>          g_verts;       // [GRID * GRID]
static UINT                         g_numIdx = 0;

static XMMATRIX g_lastVP = XMMatrixIdentity();
static float    g_time    = 0.0f;
static float    g_dt      = 0.033f;  // frame delta time (passed to NPU for rate-independent physics)
static bool     g_running = true;
static std::string g_device = "NPU";  // OpenVINO device: "NPU" or "CPU"
static std::mt19937 g_rng{42};

// Ball pit state
static std::vector<Ball>           g_balls;
static std::vector<SphereInstance>  g_ballInstances;
static ID3D11Buffer*               g_ballSphereVB = nullptr;
static ID3D11Buffer*               g_ballSphereIB = nullptr;
static ID3D11Buffer*               g_ballInstBuf  = nullptr;
static ID3D11VertexShader*         g_ballVS       = nullptr;
static ID3D11PixelShader*          g_ballPS       = nullptr;
static ID3D11PixelShader*          g_boxPS        = nullptr;
static ID3D11InputLayout*          g_ballIL       = nullptr;
static ID3D11RasterizerState*      g_noCullRS     = nullptr;
static ID3D11Buffer*               g_boxVB        = nullptr;
static ID3D11Buffer*               g_boxIB        = nullptr;
static UINT                        g_ballSphereNumIdx = 0;
static UINT                        g_boxNumIdx    = 0;
static int                         g_grabbedBall  = -1;
static float                       g_grabY        = 0.0f;
static float                       g_ballCamDist  = 80.0f;
static float                       g_ballCamHeight = 50.0f;
static float                       g_ballCamAngle = 0.0f;
static bool                        g_ballRDrag    = false;
static float                       g_ballRDragStartX = 0.0f;
static float                       g_ballRDragStartAngle = 0.0f;
static std::unique_ptr<InferenceBackend> g_ballBackend;

// Splash state (consumed by NPU each frame, then cleared)
static float g_splashX      = 0.0f;
static float g_splashZ      = 0.0f;
static float g_splashRadius = 0.0f;
static float g_splashHeight = 0.0f;

// ---------------------------------------------------------------------------
// Slider panel
// ---------------------------------------------------------------------------
static HWND g_panelHwnd = nullptr;
static bool g_showPanel = true;

struct Slider {
    const char* label;
    float* value;
    float  min, max;
    HWND   track;
    HWND   labelHwnd;
    HWND   valHwnd;
    int    id;
};

static Slider g_sliders[] = {
    {"Normal Y",      &g_normalY,      0.5f,  15.0f, nullptr, nullptr, nullptr, 100},
    {"Chop Scale",    &g_chopScale,    0.0f,   5.0f, nullptr, nullptr, nullptr, 101},
    {"Height Scale",  &g_heightScale,  0.1f,   5.0f, nullptr, nullptr, nullptr, 102},
    {"Fresnel Pow",   &g_fresnelPow,   1.0f,  10.0f, nullptr, nullptr, nullptr, 103},
    {"Fresnel Min",   &g_fresnelMin,   0.0f,   0.3f, nullptr, nullptr, nullptr, 104},
    {"Fresnel Max",   &g_fresnelMax,   0.0f,   1.0f, nullptr, nullptr, nullptr, 105},
    {"Spec Power",    &g_specPow,     10.0f, 500.0f, nullptr, nullptr, nullptr, 106},
    {"Spec Strength", &g_specStr,      0.0f,   2.0f, nullptr, nullptr, nullptr, 107},
    {"Cam Distance",  &g_camDist,      0.3f,   2.0f, nullptr, nullptr, nullptr, 108},
    {"Cam Height",    &g_camHeight,    0.05f,  1.0f, nullptr, nullptr, nullptr, 109},
    {"Cam Speed",     &g_camSpeed,     0.0f,   1.0f, nullptr, nullptr, nullptr, 110},
    {"Foam Threshold",&g_foamThreshold,0.0f,   0.3f, nullptr, nullptr, nullptr, 111},
    {"Foam Size",     &g_foamCoarseness,0.0f, 1.0f, nullptr, nullptr, nullptr, 112},
    {"Foam Decay",    &g_foamDecay,    0.95f, 0.999f, nullptr, nullptr, nullptr, 113},
    {"Foam Amount",   &g_foamGeneration,0.0f, 0.15f, nullptr, nullptr, nullptr, 114},
    {"Foam Opacity",  &g_foamOpacity,  0.5f, 10.0f, nullptr, nullptr, nullptr, 115},
};
static constexpr int NUM_SLIDERS = sizeof(g_sliders) / sizeof(g_sliders[0]);

static float sliderToValue(const Slider& s, int pos) {
    return s.min + (s.max - s.min) * pos / 1000.0f;
}
static int valueToSlider(const Slider& s) {
    return static_cast<int>((*s.value - s.min) / (s.max - s.min) * 1000.0f);
}

static void updateSliderLabel(Slider& s) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.3f", *s.value);
    SetWindowTextA(s.valHwnd, buf);
}

static LRESULT CALLBACK panelProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_HSCROLL) {
        HWND trackHwnd = reinterpret_cast<HWND>(lp);
        for (int i = 0; i < NUM_SLIDERS; i++) {
            if (g_sliders[i].track == trackHwnd) {
                int pos = static_cast<int>(SendMessageW(trackHwnd, TBM_GETPOS, 0, 0));
                *g_sliders[i].value = sliderToValue(g_sliders[i], pos);
                updateSliderLabel(g_sliders[i]);
                break;
            }
        }
        return 0;
    }
    if (msg == WM_CLOSE) {
        ShowWindow(hwnd, SW_HIDE);
        g_showPanel = false;
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static void createSliderPanel(HINSTANCE hInst) {
    INITCOMMONCONTROLSEX icex{sizeof(icex), ICC_BAR_CLASSES};
    InitCommonControlsEx(&icex);

    WNDCLASSW wc{};
    wc.lpfnWndProc  = panelProc;
    wc.hInstance     = hInst;
    wc.lpszClassName = L"NPUPanel";
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);

    int panelW = 400, rowH = 28;
    RECT rc = {0, 0, panelW, NUM_SLIDERS * rowH + 10};
    AdjustWindowRect(&rc, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, FALSE);
    int panelH = rc.bottom - rc.top;

    g_panelHwnd = CreateWindowW(
        L"NPUPanel", L"Tuning",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, panelW, panelH,
        nullptr, nullptr, hInst, nullptr);

    for (int i = 0; i < NUM_SLIDERS; i++) {
        int y = i * rowH + 4;
        Slider& s = g_sliders[i];

        s.labelHwnd = CreateWindowA("STATIC", s.label,
            WS_CHILD | WS_VISIBLE | SS_LEFT,
            4, y + 2, 120, 20, g_panelHwnd, nullptr, hInst, nullptr);

        s.track = CreateWindowW(TRACKBAR_CLASSW, L"",
            WS_CHILD | WS_VISIBLE | TBS_HORZ | TBS_NOTICKS,
            126, y, 200, 22, g_panelHwnd,
            reinterpret_cast<HMENU>(static_cast<intptr_t>(s.id)), hInst, nullptr);
        SendMessageW(s.track, TBM_SETRANGE, TRUE, MAKELPARAM(0, 1000));
        SendMessageW(s.track, TBM_SETPOS, TRUE, valueToSlider(s));

        s.valHwnd = CreateWindowA("STATIC", "",
            WS_CHILD | WS_VISIBLE | SS_LEFT,
            330, y + 2, 55, 20, g_panelHwnd, nullptr, hInst, nullptr);
        updateSliderLabel(s);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
[[noreturn]] static void fail(const char* msg) {
    fprintf(stderr, "FATAL: %s\n", msg);
    MessageBoxA(nullptr, msg, "NPU Water Sim — Error", MB_ICONERROR);
    ExitProcess(1);
}


// ---------------------------------------------------------------------------
// Reset simulation state
// ---------------------------------------------------------------------------
static void resetState() {
    std::fill(g_state.begin(), g_state.end(), uint16_t(0));
}

// ---------------------------------------------------------------------------
// Unproject mouse coords to the Y=0 water plane, returns true if hit
// ---------------------------------------------------------------------------
static bool mouseToWaterPlane(float mx, float my, float& outX, float& outZ) {
    float ndcX =  2.0f * mx / WIN_W - 1.0f;
    float ndcY =  1.0f - 2.0f * my / WIN_H;

    XMMATRIX inv = XMMatrixInverse(nullptr, g_lastVP);
    XMVECTOR nearPt = XMVector3TransformCoord(XMVectorSet(ndcX, ndcY, 0.0f, 1.0f), inv);
    XMVECTOR farPt  = XMVector3TransformCoord(XMVectorSet(ndcX, ndcY, 1.0f, 1.0f), inv);
    XMVECTOR dir    = XMVector3Normalize(farPt - nearPt);

    float oy = XMVectorGetY(nearPt);
    float dy = XMVectorGetY(dir);
    if (fabsf(dy) < 0.001f) return false;
    float t = -oy / dy;
    if (t <= 0.0f) return false;

    XMVECTOR hit = nearPt + XMVectorScale(dir, t);
    outX = XMVectorGetX(hit);
    outZ = XMVectorGetZ(hit);
    return true;
}

// Unproject mouse coords to a horizontal plane at given Y
static bool mouseToHorizPlane(float mx, float my, float planeY, float& outX, float& outZ) {
    float ndcX = 2.0f * mx / WIN_W - 1.0f;
    float ndcY = 1.0f - 2.0f * my / WIN_H;
    XMMATRIX inv = XMMatrixInverse(nullptr, g_lastVP);
    XMVECTOR nearPt = XMVector3TransformCoord(XMVectorSet(ndcX, ndcY, 0.0f, 1.0f), inv);
    XMVECTOR farPt  = XMVector3TransformCoord(XMVectorSet(ndcX, ndcY, 1.0f, 1.0f), inv);
    XMVECTOR dir = XMVector3Normalize(farPt - nearPt);
    float oy = XMVectorGetY(nearPt);
    float dy = XMVectorGetY(dir);
    if (fabsf(dy) < 0.001f) return false;
    float t = (planeY - oy) / dy;
    if (t <= 0.0f) return false;
    XMVECTOR hit = nearPt + XMVectorScale(dir, t);
    outX = XMVectorGetX(hit);
    outZ = XMVectorGetZ(hit);
    return true;
}

// ---------------------------------------------------------------------------
// Initialize D3D11
// ---------------------------------------------------------------------------
static void initD3D() {
    DXGI_SWAP_CHAIN_DESC scd{};
    scd.BufferCount       = 2;
    scd.BufferDesc.Width  = WIN_W;
    scd.BufferDesc.Height = WIN_H;
    scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferUsage       = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.OutputWindow      = g_hwnd;
    scd.SampleDesc.Count  = 1;
    scd.Windowed          = TRUE;
    scd.SwapEffect        = DXGI_SWAP_EFFECT_FLIP_DISCARD;

    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
        &fl, 1, D3D11_SDK_VERSION, &scd,
        &g_sc, &g_dev, nullptr, &g_ctx);
    if (FAILED(hr)) fail("D3D11CreateDeviceAndSwapChain failed");

    ID3D11Texture2D* bb = nullptr;
    g_sc->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&bb));
    g_dev->CreateRenderTargetView(bb, nullptr, &g_rtv);
    bb->Release();

    D3D11_TEXTURE2D_DESC dd{};
    dd.Width            = WIN_W;
    dd.Height           = WIN_H;
    dd.MipLevels        = 1;
    dd.ArraySize        = 1;
    dd.Format           = DXGI_FORMAT_D24_UNORM_S8_UINT;
    dd.SampleDesc.Count = 1;
    dd.Usage            = D3D11_USAGE_DEFAULT;
    dd.BindFlags        = D3D11_BIND_DEPTH_STENCIL;
    ID3D11Texture2D* ds = nullptr;
    g_dev->CreateTexture2D(&dd, nullptr, &ds);
    g_dev->CreateDepthStencilView(ds, nullptr, &g_dsv);
    ds->Release();

    D3D11_VIEWPORT vp{0.0f, 0.0f, float(WIN_W), float(WIN_H), 0.0f, 1.0f};
    g_ctx->RSSetViewports(1, &vp);

    D3D11_RASTERIZER_DESC rd{};
    rd.FillMode        = D3D11_FILL_SOLID;
    rd.CullMode        = D3D11_CULL_BACK;
    rd.DepthClipEnable = TRUE;
    g_dev->CreateRasterizerState(&rd, &g_rs);

    D3D11_BLEND_DESC bd{};
    bd.RenderTarget[0].BlendEnable           = TRUE;
    bd.RenderTarget[0].SrcBlend              = D3D11_BLEND_SRC_ALPHA;
    bd.RenderTarget[0].DestBlend             = D3D11_BLEND_INV_SRC_ALPHA;
    bd.RenderTarget[0].BlendOp               = D3D11_BLEND_OP_ADD;
    bd.RenderTarget[0].SrcBlendAlpha         = D3D11_BLEND_ONE;
    bd.RenderTarget[0].DestBlendAlpha        = D3D11_BLEND_ZERO;
    bd.RenderTarget[0].BlendOpAlpha          = D3D11_BLEND_OP_ADD;
    bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    g_dev->CreateBlendState(&bd, &g_blend);

    // NPU data texture — RGBA16F: (caustic, refract_x, refract_z, 0)
    D3D11_TEXTURE2D_DESC ctd{};
    ctd.Width            = GRID;
    ctd.Height           = GRID;
    ctd.MipLevels        = 1;
    ctd.ArraySize        = 1;
    ctd.Format           = DXGI_FORMAT_R16G16B16A16_FLOAT;
    ctd.SampleDesc.Count = 1;
    ctd.Usage            = D3D11_USAGE_DYNAMIC;
    ctd.BindFlags        = D3D11_BIND_SHADER_RESOURCE;
    ctd.CPUAccessFlags   = D3D11_CPU_ACCESS_WRITE;
    g_dev->CreateTexture2D(&ctd, nullptr, &g_causticTex);
    g_dev->CreateShaderResourceView(g_causticTex, nullptr, &g_causticSRV);

    D3D11_SAMPLER_DESC sd{};
    sd.Filter   = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    g_dev->CreateSamplerState(&sd, &g_sampler);
}

// ---------------------------------------------------------------------------
// Compile shaders and create input layout
// ---------------------------------------------------------------------------
static void initShaders() {
    auto compile = [](const char* src, const char* target, const char* entry) -> ID3DBlob* {
        ID3DBlob* blob = nullptr;
        ID3DBlob* err  = nullptr;
        HRESULT hr = D3DCompile(src, strlen(src), nullptr, nullptr, nullptr,
                                 entry, target, 0, 0, &blob, &err);
        if (FAILED(hr)) {
            std::string msg = "Shader compile error: ";
            if (err) { msg += static_cast<const char*>(err->GetBufferPointer()); err->Release(); }
            fail(msg.c_str());
        }
        if (err) err->Release();
        return blob;
    };

    ID3DBlob* vsBlob = compile(g_vsSource, "vs_5_0", "main");
    g_dev->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &g_vs);

    D3D11_INPUT_ELEMENT_DESC layout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"FOAM",     0, DXGI_FORMAT_R32_FLOAT,       0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    g_dev->CreateInputLayout(layout, 3, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &g_il);
    vsBlob->Release();

    ID3DBlob* psBlob = compile(g_psSource, "ps_5_0", "main");
    g_dev->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &g_ps);
    psBlob->Release();

    ID3DBlob* duckBlob = compile(g_duckPsSource, "ps_5_0", "main");
    g_dev->CreatePixelShader(duckBlob->GetBufferPointer(), duckBlob->GetBufferSize(), nullptr, &g_duckPS);
    duckBlob->Release();

    ID3DBlob* billBlob = compile(g_billPsSource, "ps_5_0", "main");
    g_dev->CreatePixelShader(billBlob->GetBufferPointer(), billBlob->GetBufferSize(), nullptr, &g_billPS);
    billBlob->Release();

    ID3DBlob* tileBlob = compile(g_tilePsSource, "ps_5_0", "main");
    g_dev->CreatePixelShader(tileBlob->GetBufferPointer(), tileBlob->GetBufferSize(), nullptr, &g_tilePS);
    tileBlob->Release();

    // Ball pit shaders
    ID3DBlob* ballVsBlob = compile(g_ballVsSource, "vs_5_0", "main");
    g_dev->CreateVertexShader(ballVsBlob->GetBufferPointer(), ballVsBlob->GetBufferSize(), nullptr, &g_ballVS);

    D3D11_INPUT_ELEMENT_DESC ballLayout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA,   0},
        {"NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA,   0},
        {"INST_POS", 0, DXGI_FORMAT_R32G32B32_FLOAT, 1, 0,  D3D11_INPUT_PER_INSTANCE_DATA, 1},
        {"INST_RAD", 0, DXGI_FORMAT_R32_FLOAT,       1, 12, D3D11_INPUT_PER_INSTANCE_DATA, 1},
        {"INST_COL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 1, 16, D3D11_INPUT_PER_INSTANCE_DATA, 1},
    };
    g_dev->CreateInputLayout(ballLayout, 5, ballVsBlob->GetBufferPointer(), ballVsBlob->GetBufferSize(), &g_ballIL);
    ballVsBlob->Release();

    ID3DBlob* ballPsBlob = compile(g_ballPsSource, "ps_5_0", "main");
    g_dev->CreatePixelShader(ballPsBlob->GetBufferPointer(), ballPsBlob->GetBufferSize(), nullptr, &g_ballPS);
    ballPsBlob->Release();

    ID3DBlob* boxPsBlob = compile(g_boxPsSource, "ps_5_0", "main");
    g_dev->CreatePixelShader(boxPsBlob->GetBufferPointer(), boxPsBlob->GetBufferSize(), nullptr, &g_boxPS);
    boxPsBlob->Release();
}

// ---------------------------------------------------------------------------
// Create grid mesh buffers
// ---------------------------------------------------------------------------
static void initMesh() {
    g_verts.resize(GRID * GRID);
    for (int j = 0; j < GRID; j++) {
        for (int i = 0; i < GRID; i++) {
            float x = float(i) - GRID * 0.5f;
            float z = float(j) - GRID * 0.5f;
            g_verts[j * GRID + i] = {{x, 0.0f, z}, {0.0f, 1.0f, 0.0f}};
        }
    }

    std::vector<uint32_t> idx;
    idx.reserve((GRID - 1) * (GRID - 1) * 6);
    for (int j = 0; j < GRID - 1; j++) {
        for (int i = 0; i < GRID - 1; i++) {
            uint32_t tl = j * GRID + i;
            uint32_t tr = tl + 1;
            uint32_t bl = tl + GRID;
            uint32_t br = bl + 1;
            idx.push_back(tl); idx.push_back(bl); idx.push_back(tr);
            idx.push_back(tr); idx.push_back(bl); idx.push_back(br);
        }
    }
    g_numIdx = static_cast<UINT>(idx.size());

    D3D11_BUFFER_DESC vbd{};
    vbd.ByteWidth      = static_cast<UINT>(g_verts.size() * sizeof(Vertex));
    vbd.Usage          = D3D11_USAGE_DYNAMIC;
    vbd.BindFlags      = D3D11_BIND_VERTEX_BUFFER;
    vbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    D3D11_SUBRESOURCE_DATA vsd{g_verts.data()};
    g_dev->CreateBuffer(&vbd, &vsd, &g_vb);

    D3D11_BUFFER_DESC ibd{};
    ibd.ByteWidth = static_cast<UINT>(idx.size() * sizeof(uint32_t));
    ibd.Usage     = D3D11_USAGE_DEFAULT;
    ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    D3D11_SUBRESOURCE_DATA isd{idx.data()};
    g_dev->CreateBuffer(&ibd, &isd, &g_ib);

    D3D11_BUFFER_DESC cbd{};
    cbd.ByteWidth      = sizeof(CB);
    cbd.Usage          = D3D11_USAGE_DYNAMIC;
    cbd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    g_dev->CreateBuffer(&cbd, nullptr, &g_cb);

    // --- Rubber duck from STL file ---
    {
        FILE* stlf = fopen("rubber_duck.stl", "rb");
        if (!stlf) fail("Cannot open rubber_duck.stl");

        fseek(stlf, 80, SEEK_SET);  // skip header
        uint32_t numTri;
        fread(&numTri, 4, 1, stlf);

        // Read triangles, converting Blender Z-up → D3D Y-up: (X, Z, -Y)
        std::vector<XMFLOAT3> triPos(numTri * 3);
        std::vector<XMFLOAT3> faceNrm(numTri);

        for (uint32_t ti = 0; ti < numTri; ti++) {
            float d[12]; uint16_t attr;
            fread(d, sizeof(float), 12, stlf);
            fread(&attr, 2, 1, stlf);

            triPos[ti*3+0] = {d[3],  d[5],  -d[4]};
            triPos[ti*3+1] = {d[6],  d[8],  -d[7]};
            triPos[ti*3+2] = {d[9],  d[11], -d[10]};

            XMVECTOR e1 = XMVectorSubtract(XMLoadFloat3(&triPos[ti*3+1]),
                                            XMLoadFloat3(&triPos[ti*3+0]));
            XMVECTOR e2 = XMVectorSubtract(XMLoadFloat3(&triPos[ti*3+2]),
                                            XMLoadFloat3(&triPos[ti*3+0]));
            XMStoreFloat3(&faceNrm[ti], XMVector3Normalize(XMVector3Cross(e1, e2)));
        }
        fclose(stlf);

        // Bounding box
        XMFLOAT3 bmin = {1e9f, 1e9f, 1e9f}, bmax = {-1e9f, -1e9f, -1e9f};
        for (auto& p : triPos) {
            bmin.x = std::min(bmin.x, p.x); bmax.x = std::max(bmax.x, p.x);
            bmin.y = std::min(bmin.y, p.y); bmax.y = std::max(bmax.y, p.y);
            bmin.z = std::min(bmin.z, p.z); bmax.z = std::max(bmax.z, p.z);
        }

        // Center and normalize: centroid at origin, max extent = 1.0
        float mcx = (bmin.x + bmax.x) * 0.5f;
        float mcy = (bmin.y + bmax.y) * 0.5f;
        float mcz = (bmin.z + bmax.z) * 0.5f;
        float ext = std::max({bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z});
        float msc = 1.0f / ext;

        for (auto& p : triPos) {
            p.x = (p.x - mcx) * msc;
            p.y = (p.y - mcy) * msc;
            p.z = (p.z - mcz) * msc;
        }

        // Weld vertices by quantized position and compute smooth normals
        std::unordered_map<uint64_t, uint32_t> vertMap;
        std::vector<Vertex> duckV;
        std::vector<uint32_t> duckI;

        for (uint32_t ti = 0; ti < numTri; ti++) {
            for (int vi = 0; vi < 3; vi++) {
                auto& p = triPos[ti * 3 + vi];
                int32_t qx = static_cast<int32_t>(roundf(p.x * 100000.0f));
                int32_t qy = static_cast<int32_t>(roundf(p.y * 100000.0f));
                int32_t qz = static_cast<int32_t>(roundf(p.z * 100000.0f));
                uint64_t key = (uint64_t(qx + 500000) * 1000001ULL +
                                uint64_t(qy + 500000)) * 1000001ULL +
                                uint64_t(qz + 500000);

                auto it = vertMap.find(key);
                uint32_t idx;
                if (it != vertMap.end()) {
                    idx = it->second;
                } else {
                    idx = static_cast<uint32_t>(duckV.size());
                    vertMap[key] = idx;
                    duckV.push_back({{p.x, p.y, p.z}, {0, 0, 0}});
                }
                duckI.push_back(idx);
                duckV[idx].nrm.x += faceNrm[ti].x;
                duckV[idx].nrm.y += faceNrm[ti].y;
                duckV[idx].nrm.z += faceNrm[ti].z;
            }
        }

        // Normalize accumulated vertex normals
        for (auto& v : duckV) {
            XMVECTOR n = XMVector3Normalize(XMLoadFloat3(&v.nrm));
            XMStoreFloat3(&v.nrm, n);
        }

        g_duckBodyNumIdx = static_cast<UINT>(duckI.size());
        g_duckBillStart  = g_duckBodyNumIdx;
        g_duckBillNumIdx = 0;  // STL is all one material — whole duck is yellow

        D3D11_BUFFER_DESC dvbd{};
        dvbd.ByteWidth = static_cast<UINT>(duckV.size() * sizeof(Vertex));
        dvbd.Usage     = D3D11_USAGE_DEFAULT;
        dvbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        D3D11_SUBRESOURCE_DATA dvsd{duckV.data()};
        g_dev->CreateBuffer(&dvbd, &dvsd, &g_duckVB);

        D3D11_BUFFER_DESC dibd{};
        dibd.ByteWidth = static_cast<UINT>(duckI.size() * sizeof(uint32_t));
        dibd.Usage     = D3D11_USAGE_DEFAULT;
        dibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
        D3D11_SUBRESOURCE_DATA disd{duckI.data()};
        g_dev->CreateBuffer(&dibd, &disd, &g_duckIB);

        printf("[OK] Loaded rubber_duck.stl: %u tris, %zu verts (smooth normals)\n",
               numTri, duckV.size());
    }

    // --- Tile floor (flat quad at FLOOR_Y) ---
    float H = GRID * 0.5f;
    Vertex tileV[] = {
        {{-H, FLOOR_Y, -H}, {0,1,0}},
        {{ H, FLOOR_Y, -H}, {0,1,0}},
        {{ H, FLOOR_Y,  H}, {0,1,0}},
        {{-H, FLOOR_Y,  H}, {0,1,0}},
    };
    uint32_t tileI[] = {0, 2, 1, 0, 3, 2};
    g_tileNumIdx = 6;

    D3D11_BUFFER_DESC tvbd{};
    tvbd.ByteWidth = sizeof(tileV);
    tvbd.Usage     = D3D11_USAGE_DEFAULT;
    tvbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA tvsd{tileV};
    g_dev->CreateBuffer(&tvbd, &tvsd, &g_tileVB);

    D3D11_BUFFER_DESC tibd{};
    tibd.ByteWidth = sizeof(tileI);
    tibd.Usage     = D3D11_USAGE_DEFAULT;
    tibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    D3D11_SUBRESOURCE_DATA tisd{tileI};
    g_dev->CreateBuffer(&tibd, &tisd, &g_tileIB);
}

// ---------------------------------------------------------------------------
// Initialize ball pit scene
// ---------------------------------------------------------------------------
static void initBallPit() {
    // --- UV sphere mesh (16 lat x 32 lon) ---
    std::vector<BallVertex> sphereV;
    std::vector<uint32_t> sphereI;
    const int latSegs = 16, lonSegs = 32;
    for (int lat = 0; lat <= latSegs; lat++) {
        float theta = float(lat) / latSegs * XM_PI;
        float sinT = sinf(theta), cosT = cosf(theta);
        for (int lon = 0; lon <= lonSegs; lon++) {
            float phi = float(lon) / lonSegs * XM_2PI;
            XMFLOAT3 p = {sinT * cosf(phi), cosT, sinT * sinf(phi)};
            sphereV.push_back({p, p});
        }
    }
    for (int lat = 0; lat < latSegs; lat++) {
        for (int lon = 0; lon < lonSegs; lon++) {
            int a = lat * (lonSegs + 1) + lon;
            int b = a + lonSegs + 1;
            sphereI.push_back(a); sphereI.push_back(b);     sphereI.push_back(a + 1);
            sphereI.push_back(a + 1); sphereI.push_back(b); sphereI.push_back(b + 1);
        }
    }
    g_ballSphereNumIdx = (UINT)sphereI.size();

    D3D11_BUFFER_DESC svbd{};
    svbd.ByteWidth = (UINT)(sphereV.size() * sizeof(BallVertex));
    svbd.Usage     = D3D11_USAGE_DEFAULT;
    svbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA svsd{sphereV.data()};
    g_dev->CreateBuffer(&svbd, &svsd, &g_ballSphereVB);

    D3D11_BUFFER_DESC sibd{};
    sibd.ByteWidth = (UINT)(sphereI.size() * sizeof(uint32_t));
    sibd.Usage     = D3D11_USAGE_DEFAULT;
    sibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    D3D11_SUBRESOURCE_DATA sisd{sphereI.data()};
    g_dev->CreateBuffer(&sibd, &sisd, &g_ballSphereIB);

    // --- Instance buffer (dynamic, updated each frame) ---
    g_ballInstances.resize(NUM_BALLS);
    D3D11_BUFFER_DESC ibd{};
    ibd.ByteWidth      = NUM_BALLS * sizeof(SphereInstance);
    ibd.Usage          = D3D11_USAGE_DYNAMIC;
    ibd.BindFlags      = D3D11_BIND_VERTEX_BUFFER;
    ibd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    g_dev->CreateBuffer(&ibd, nullptr, &g_ballInstBuf);

    // --- Box mesh (5 quads: floor + 4 walls, normals pointing inward) ---
    std::vector<Vertex> boxV;
    std::vector<uint32_t> boxI;
    auto addQuad = [&](XMFLOAT3 a, XMFLOAT3 b, XMFLOAT3 c, XMFLOAT3 d, XMFLOAT3 n) {
        uint32_t base = (uint32_t)boxV.size();
        boxV.push_back({a, n, 0}); boxV.push_back({b, n, 0});
        boxV.push_back({c, n, 0}); boxV.push_back({d, n, 0});
        boxI.push_back(base); boxI.push_back(base+1); boxI.push_back(base+2);
        boxI.push_back(base); boxI.push_back(base+2); boxI.push_back(base+3);
    };
    float H = BOX_HALF, T = BOX_HEIGHT;
    addQuad({-H,0,-H}, { H,0,-H}, { H,0, H}, {-H,0, H}, {0,1,0});   // floor
    addQuad({-H,0,-H}, { H,0,-H}, { H,T,-H}, {-H,T,-H}, {0,0,1});   // back wall
    addQuad({ H,0, H}, {-H,0, H}, {-H,T, H}, { H,T, H}, {0,0,-1});  // front wall
    addQuad({-H,0, H}, {-H,0,-H}, {-H,T,-H}, {-H,T, H}, {1,0,0});   // left wall
    addQuad({ H,0,-H}, { H,0, H}, { H,T, H}, { H,T,-H}, {-1,0,0});  // right wall
    g_boxNumIdx = (UINT)boxI.size();

    D3D11_BUFFER_DESC bvbd{};
    bvbd.ByteWidth = (UINT)(boxV.size() * sizeof(Vertex));
    bvbd.Usage     = D3D11_USAGE_DEFAULT;
    bvbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA bvsd{boxV.data()};
    g_dev->CreateBuffer(&bvbd, &bvsd, &g_boxVB);

    D3D11_BUFFER_DESC bibd{};
    bibd.ByteWidth = (UINT)(boxI.size() * sizeof(uint32_t));
    bibd.Usage     = D3D11_USAGE_DEFAULT;
    bibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    D3D11_SUBRESOURCE_DATA bisd{boxI.data()};
    g_dev->CreateBuffer(&bibd, &bisd, &g_boxIB);

    // --- No-cull rasterizer state (see inside of box) ---
    D3D11_RASTERIZER_DESC rd{};
    rd.FillMode        = D3D11_FILL_SOLID;
    rd.CullMode        = D3D11_CULL_NONE;
    rd.DepthClipEnable = TRUE;
    g_dev->CreateRasterizerState(&rd, &g_noCullRS);

    // --- Initialize 3000 balls in a grid, slightly above floor ---
    g_balls.resize(NUM_BALLS);
    float spacing = BALL_RADIUS * 2.1f;
    int perRow = (int)((BOX_HALF * 2.0f - BALL_RADIUS * 2.0f) / spacing);
    if (perRow < 1) perRow = 1;
    XMFLOAT3 palette[] = {
        {0.95f, 0.15f, 0.15f}, {0.15f, 0.85f, 0.15f}, {0.15f, 0.35f, 0.95f},
        {0.95f, 0.85f, 0.10f}, {0.95f, 0.50f, 0.05f}, {0.70f, 0.15f, 0.90f},
        {0.05f, 0.85f, 0.85f}, {0.95f, 0.30f, 0.60f},
    };
    std::uniform_int_distribution<int> colDist(0, 7);
    int placed = 0;
    for (int ly = 0; placed < NUM_BALLS; ly++) {
        for (int lz = 0; lz < perRow && placed < NUM_BALLS; lz++) {
            for (int lx = 0; lx < perRow && placed < NUM_BALLS; lx++) {
                float x = -BOX_HALF + BALL_RADIUS + lx * spacing;
                float y = BALL_RADIUS + ly * spacing;
                float z = -BOX_HALF + BALL_RADIUS + lz * spacing;
                g_balls[placed].pos = {x, y, z};
                g_balls[placed].vel = {0, 0, 0};
                g_balls[placed].radius = BALL_RADIUS;
                g_balls[placed].color = palette[colDist(g_rng)];
                placed++;
            }
        }
    }

    // Fill instance data
    for (int i = 0; i < NUM_BALLS; i++) {
        g_ballInstances[i] = {g_balls[i].pos, g_balls[i].radius, g_balls[i].color};
    }

    printf("[OK] Ball pit: %d spheres, %u tri/sphere, box %.0fx%.0fx%.0f\n",
           NUM_BALLS, g_ballSphereNumIdx / 3, BOX_HALF*2, BOX_HEIGHT, BOX_HALF*2);

    // --- Load ball physics NPU model ---
    bool ballModelExists = GetFileAttributesA("ball_physics.onnx") != INVALID_FILE_ATTRIBUTES;
    if (ballModelExists) {
        try {
            // Use same backend type as water model
            bool useOV = g_backend && g_backend->name().find("OpenVINO") != std::string::npos;
            if (useOV) {
#ifdef HAS_OPENVINO
                g_ballBackend = std::make_unique<OpenVinoBackend>("ball_physics.onnx", g_device);
#endif
            } else {
#ifdef HAS_WINDOWSML
                g_ballBackend = std::make_unique<WindowsMLBackend>("ball_physics.onnx", g_device);
#endif
            }
            if (g_ballBackend)
                printf("[OK] Ball physics NPU: %s\n", g_ballBackend->name().c_str());
        } catch (const std::exception& e) {
            printf("WARNING: Ball physics NPU init failed: %s\n", e.what());
            g_ballBackend.reset();
        }
    } else {
        printf("WARNING: ball_physics.onnx not found — run generate_ball_model.py\n");
    }
}

// ---------------------------------------------------------------------------
// Initialize inference backend (OpenVINO or ONNX Runtime + DirectML)
// ---------------------------------------------------------------------------
static void initBackend() {
    try {
        bool useOV = false;

        switch (g_runtimeChoice) {
        case RuntimeChoice::OpenVINO:
            useOV = true;
            break;
        case RuntimeChoice::WindowsML:
            useOV = false;
            break;
        case RuntimeChoice::Auto: {
#ifdef HAS_OPENVINO
            // Check CPU vendor via CPUID — only try OpenVINO on Intel
            int cpuInfo[4] = {};
            __cpuid(cpuInfo, 0);
            char vendor[13] = {};
            memcpy(vendor, &cpuInfo[1], 4);
            memcpy(vendor + 4, &cpuInfo[3], 4);
            memcpy(vendor + 8, &cpuInfo[2], 4);
            if (std::string(vendor) == "GenuineIntel") {
                useOV = OpenVinoBackend::hasIntelNPU() || g_device == "CPU";
            }
#endif
            break;
        }
        }

        if (useOV) {
#ifdef HAS_OPENVINO
            g_backend = std::make_unique<OpenVinoBackend>("water_physics.onnx", g_device);
#else
            fail("OpenVINO support not compiled (HAS_OPENVINO not defined)");
#endif
        } else {
#ifdef HAS_WINDOWSML
            g_backend = std::make_unique<WindowsMLBackend>("water_physics.onnx", g_device);
#else
            fail("Windows ML support not compiled (HAS_WINDOWSML not defined)");
#endif
        }

        // Set conversion functions based on backend precision
        if (g_backend->isBf16()) {
            f32_to_f16 = f32_to_bf16;
            f16_to_f32 = bf16_to_f32;
        } else {
            f32_to_f16 = f32_to_fp16;
            f16_to_f32 = fp16_to_f32;
        }

        g_state.resize(4 * GRID * GRID, uint16_t(0));
        g_wavePhase.resize(N_WAVES, uint16_t(0));
        g_renderBuf.resize(10 * GRID * GRID, uint16_t(0));
        resetState();

        printf("Backend: %s\n", g_backend->name().c_str());
        const char* precStr = g_backend->isBf16() ? "BF16" : (g_backend->isFp32Model() ? "FP32" : "FP16");
        printf("  Grid       : %dx%d %s\n", GRID, GRID, precStr);
        printf("  Ocean      : 32 Gerstner waves (Phillips spectrum)\n");
        printf("  Ripples    : 8 substeps interactive wave equation\n");
        printf("  1 inference call per frame\n\n");

    } catch (const std::exception& e) {
        std::string msg = "Backend initialization failed:\n";
        msg += e.what();
        fail(msg.c_str());
    } catch (...) {
        fail("Backend initialization failed with unknown exception.");
    }
}

// ---------------------------------------------------------------------------
// Run full frame on NPU — 32 Gerstner waves + ripple physics + render
// ---------------------------------------------------------------------------
static void runSimulation() {
    // Input 0: ripple state [1, 4, GRID, GRID]
    memcpy(g_backend->getInput(0), g_state.data(), g_state.size() * sizeof(uint16_t));

    // Input 1: wave phases [1, 32, 1, 1]
    memcpy(g_backend->getInput(1), g_wavePhase.data(), N_WAVES * sizeof(uint16_t));

    // Input 2: camera position [1, 3, 1, 1]
    auto* camIn = g_backend->getInput(2);
    float camAngle = g_time * g_camSpeed + g_camAngle;
    float camDist  = GRID * g_camDist;
    float camH     = GRID * g_camHeight;
    camIn[0] = f32_to_f16(cosf(camAngle) * camDist);
    camIn[1] = f32_to_f16(camH);
    camIn[2] = f32_to_f16(sinf(camAngle) * camDist);

    // Input 3: duck state [1, 7, 1, 1]
    auto* duckIn = g_backend->getInput(3);
    duckIn[0] = f32_to_f16(g_duckX);
    duckIn[1] = f32_to_f16(g_duckZ);
    duckIn[2] = f32_to_f16(g_duckVX);
    duckIn[3] = f32_to_f16(g_duckVZ);
    duckIn[4] = f32_to_f16(g_duckY);
    duckIn[5] = f32_to_f16(g_duckTiltX);
    duckIn[6] = f32_to_f16(g_duckTiltZ);

    // Input 4: delta time [1, 1, 1, 1]
    g_backend->getInput(4)[0] = f32_to_f16(g_dt);

    // Input 5: duck2 state [1, 7, 1, 1]
    auto* duck2In = g_backend->getInput(5);
    duck2In[0] = f32_to_f16(g_duck2X);
    duck2In[1] = f32_to_f16(g_duck2Z);
    duck2In[2] = f32_to_f16(g_duck2VX);
    duck2In[3] = f32_to_f16(g_duck2VZ);
    duck2In[4] = f32_to_f16(g_duck2Y);
    duck2In[5] = f32_to_f16(g_duck2TiltX);
    duck2In[6] = f32_to_f16(g_duck2TiltZ);

    // Input 6: splash [1, 4, 1, 1]
    auto* splashIn = g_backend->getInput(6);
    splashIn[0] = f32_to_f16(g_splashX);
    splashIn[1] = f32_to_f16(g_splashZ);
    splashIn[2] = f32_to_f16(g_splashHeight > 0.0f ? g_splashRadius : 1.0f);
    splashIn[3] = f32_to_f16(g_splashHeight);
    g_splashHeight = 0.0f;

    // Input 7: foam params [1, 4, 1, 1]
    auto* foamIn = g_backend->getInput(7);
    foamIn[0] = f32_to_f16(g_foamThreshold);
    foamIn[1] = f32_to_f16(g_foamCoarseness);
    foamIn[2] = f32_to_f16(g_foamEnabled ? g_foamDecay : 0.0f);
    foamIn[3] = f32_to_f16(g_foamEnabled ? g_foamGeneration : 0.0f);

    // Input 8: render params [1, 3, 1, 1]
    auto* renderIn = g_backend->getInput(8);
    renderIn[0] = f32_to_f16(g_chopScale);
    renderIn[1] = f32_to_f16(g_heightScale);
    renderIn[2] = f32_to_f16(g_normalY);

    // Single inference: all physics on NPU
    g_backend->infer();

    // Output 0: new simulation state
    memcpy(g_state.data(), g_backend->getOutput(0), g_state.size() * sizeof(uint16_t));

    // Output 1: render data [1, 10, GRID, GRID]
    memcpy(g_renderBuf.data(), g_backend->getOutput(1), g_renderBuf.size() * sizeof(uint16_t));

    // Output 2: duck1 state
    auto* duckOut = g_backend->getOutput(2);

    // Output 3: duck2 state
    auto* duck2Out = g_backend->getOutput(3);

    // Output 4: updated wave phases
    memcpy(g_wavePhase.data(), g_backend->getOutput(4), N_WAVES * sizeof(uint16_t));

    // Duck1: if dragged, keep mouse position; otherwise take NPU output
    if (g_draggedDuck == 0) {
        g_duckVX    = (g_duckX - g_duckPrevX) / g_dt;
        g_duckVZ    = (g_duckZ - g_duckPrevZ) / g_dt;
        g_duckPrevX = g_duckX;
        g_duckPrevZ = g_duckZ;
        g_duckY     = f16_to_f32(duckOut[4]);
        g_duckTiltX = f16_to_f32(duckOut[5]);
        g_duckTiltZ = f16_to_f32(duckOut[6]);
    } else {
        g_duckPrevX = g_duckX;
        g_duckPrevZ = g_duckZ;
        g_duckX     = f16_to_f32(duckOut[0]);
        g_duckZ     = f16_to_f32(duckOut[1]);
        g_duckVX    = f16_to_f32(duckOut[2]);
        g_duckVZ    = f16_to_f32(duckOut[3]);
        g_duckY     = f16_to_f32(duckOut[4]);
        g_duckTiltX = f16_to_f32(duckOut[5]);
        g_duckTiltZ = f16_to_f32(duckOut[6]);
    }

    // Duck2: same pattern
    if (g_draggedDuck == 1) {
        g_duck2VX    = (g_duck2X - g_duck2PrevX) / g_dt;
        g_duck2VZ    = (g_duck2Z - g_duck2PrevZ) / g_dt;
        g_duck2PrevX = g_duck2X;
        g_duck2PrevZ = g_duck2Z;
        g_duck2Y     = f16_to_f32(duck2Out[4]);
        g_duck2TiltX = f16_to_f32(duck2Out[5]);
        g_duck2TiltZ = f16_to_f32(duck2Out[6]);
    } else {
        g_duck2PrevX = g_duck2X;
        g_duck2PrevZ = g_duck2Z;
        g_duck2X     = f16_to_f32(duck2Out[0]);
        g_duck2Z     = f16_to_f32(duck2Out[1]);
        g_duck2VX    = f16_to_f32(duck2Out[2]);
        g_duck2VZ    = f16_to_f32(duck2Out[3]);
        g_duck2Y     = f16_to_f32(duck2Out[4]);
        g_duck2TiltX = f16_to_f32(duck2Out[5]);
        g_duck2TiltZ = f16_to_f32(duck2Out[6]);
    }
}

// ---------------------------------------------------------------------------
// Update vertex buffer from NPU render output
// ---------------------------------------------------------------------------
static void updateMesh() {
    const int N  = GRID;
    const int NN = N * N;

    // NPU render output layout: [1, 10, N, N]
    //   Channel 0-2: pos_x, pos_y, pos_z  (choppy displaced + height scaled)
    //   Channel 3-5: nrm_x, nrm_y, nrm_z (surface normals)
    //   Channel 6: caustic
    //   Channel 7-8: refract_x, refract_z
    //   Channel 9: foam density [0..1]
    for (int idx = 0; idx < NN; idx++) {
        g_verts[idx].pos.x = f16_to_f32(g_renderBuf[0 * NN + idx]);
        g_verts[idx].pos.y = f16_to_f32(g_renderBuf[1 * NN + idx]);
        g_verts[idx].pos.z = f16_to_f32(g_renderBuf[2 * NN + idx]);
        g_verts[idx].nrm.x = f16_to_f32(g_renderBuf[3 * NN + idx]);
        g_verts[idx].nrm.y = f16_to_f32(g_renderBuf[4 * NN + idx]);
        g_verts[idx].nrm.z = f16_to_f32(g_renderBuf[5 * NN + idx]);
        g_verts[idx].foam  = g_foamEnabled ? f16_to_f32(g_renderBuf[9 * NN + idx]) : 0.0f;
    }

    D3D11_MAPPED_SUBRESOURCE mapped;
    g_ctx->Map(g_vb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    memcpy(mapped.pData, g_verts.data(), g_verts.size() * sizeof(Vertex));
    g_ctx->Unmap(g_vb, 0);
}

// ---------------------------------------------------------------------------
// Update rubber duck — float on water, drift with waves, inject displacement
// into ripple state so the NPU propagates the duck's water interaction.
// ---------------------------------------------------------------------------
// Duck physics (collision, buoyancy, tilt, slope drift, wall bounce) all on NPU.
// CPU only reads back duck_out = (x, z, vx, vz, y, tiltX, tiltZ) for rendering.
static void updateDuck() {
}

// ---------------------------------------------------------------------------
// Ball pit physics — all on NPU (gravity, walls, N² pairwise collision)
// ---------------------------------------------------------------------------
static bool  g_ballShootEnabled = true;
static float g_ballShootTimer   = 0.0f;
static int   g_ballShootNext    = 0;

static void shootBall() {
    int i = g_ballShootNext;
    g_ballShootNext = (g_ballShootNext + 1) % NUM_BALLS;

    // Random angle around the box edge
    float angle = (float)(rand() % 360) * 3.14159f / 180.0f;
    float edge = BOX_HALF - BALL_RADIUS;
    g_balls[i].pos = { cosf(angle) * edge, BOX_HEIGHT * 0.8f, sinf(angle) * edge };

    // Aim toward center with high speed
    float speed = 400.0f;
    XMFLOAT3 dir = { -g_balls[i].pos.x, -5.0f, -g_balls[i].pos.z };
    float len = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    g_balls[i].vel = { dir.x/len * speed, dir.y/len * speed, dir.z/len * speed };
}

static void updateBalls() {
    if (!g_ballBackend) return;

    // Auto-shoot a ball every second
    if (g_ballShootEnabled) {
        g_ballShootTimer += 1.0f / 30.0f;
        if (g_ballShootTimer >= 1.0f) {
            g_ballShootTimer -= 1.0f;
            for (int s = 0; s < 20; s++) shootBall();
        }
    }

    // Pack positions into [1, 3, N, 1] — channel-first (x,y,z as channels)
    auto* posIn = g_ballBackend->getInput(0);
    auto* velIn = g_ballBackend->getInput(1);
    for (int i = 0; i < NUM_BALLS; i++) {
        posIn[0 * NUM_BALLS + i] = f32_to_fp16(g_balls[i].pos.x);
        posIn[1 * NUM_BALLS + i] = f32_to_fp16(g_balls[i].pos.y);
        posIn[2 * NUM_BALLS + i] = f32_to_fp16(g_balls[i].pos.z);
        velIn[0 * NUM_BALLS + i] = f32_to_fp16(g_balls[i].vel.x);
        velIn[1 * NUM_BALLS + i] = f32_to_fp16(g_balls[i].vel.y);
        velIn[2 * NUM_BALLS + i] = f32_to_fp16(g_balls[i].vel.z);
    }

    // dt [1,1,1,1]
    g_ballBackend->getInput(2)[0] = f32_to_fp16(1.0f / 30.0f);

    // grab_mask [1,1,N,1]: 1.0 = free, 0.0 = grabbed (NPU skips physics for grabbed ball)
    auto* maskIn = g_ballBackend->getInput(3);
    for (int i = 0; i < NUM_BALLS; i++)
        maskIn[i] = f32_to_fp16(i == g_grabbedBall ? 0.0f : 1.0f);

    // Single NPU call: gravity + walls + N² pairwise collision × 4 substeps
    g_ballBackend->infer();

    // Unpack results from [1, 3, N, 1]
    auto* posOut = g_ballBackend->getOutput(0);
    auto* velOut = g_ballBackend->getOutput(1);
    for (int i = 0; i < NUM_BALLS; i++) {
        g_balls[i].pos.x = fp16_to_f32(posOut[0 * NUM_BALLS + i]);
        g_balls[i].pos.y = fp16_to_f32(posOut[1 * NUM_BALLS + i]);
        g_balls[i].pos.z = fp16_to_f32(posOut[2 * NUM_BALLS + i]);
        g_balls[i].vel.x = fp16_to_f32(velOut[0 * NUM_BALLS + i]);
        g_balls[i].vel.y = fp16_to_f32(velOut[1 * NUM_BALLS + i]);
        g_balls[i].vel.z = fp16_to_f32(velOut[2 * NUM_BALLS + i]);
    }

    // Update GPU instance data for rendering
    for (int i = 0; i < NUM_BALLS; i++) {
        g_ballInstances[i].pos = g_balls[i].pos;
        g_ballInstances[i].radius = g_balls[i].radius;
        g_ballInstances[i].color = g_balls[i].color;
    }
}

// ---------------------------------------------------------------------------
// Render one frame
// ---------------------------------------------------------------------------
static void render() {
    float clearColor[] = {0.75f, 0.82f, 0.90f, 1.0f};  // light sky/bathroom background
    g_ctx->ClearRenderTargetView(g_rtv, clearColor);
    g_ctx->ClearDepthStencilView(g_dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);
    g_ctx->OMSetRenderTargets(1, &g_rtv, g_dsv);

    float angle   = g_time * g_camSpeed + g_camAngle;
    float camDist = GRID * g_camDist;
    float camH    = GRID * g_camHeight;
    XMVECTOR eye = XMVectorSet(cosf(angle) * camDist, camH, sinf(angle) * camDist, 0.0f);
    XMVECTOR at  = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
    XMVECTOR up  = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

    XMMATRIX view = XMMatrixLookAtLH(eye, at, up);
    XMMATRIX proj = XMMatrixPerspectiveFovLH(XM_PIDIV4, float(WIN_W) / float(WIN_H), 1.0f, 2000.0f);
    XMMATRIX vp   = view * proj;
    g_lastVP = vp;

    // Shared CB fields
    CB cb{};
    XMFLOAT3 ld3{0.0f, 1.0f, 0.8f};
    XMStoreFloat3(&cb.lightDir, XMVector3Normalize(XMLoadFloat3(&ld3)));
    XMStoreFloat3(&cb.eye, eye);
    cb.time       = g_time;
    cb.fresnelPow = g_fresnelPow;
    cb.fresnelMin = g_fresnelMin;
    cb.fresnelMax = g_fresnelMax;
    cb.specPow    = g_specPow;
    cb.specStr    = g_specStr;
    cb.foamOpacity = g_foamOpacity;

    g_ctx->RSSetState(g_rs);
    g_ctx->IASetInputLayout(g_il);
    g_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    g_ctx->VSSetShader(g_vs, nullptr, 0);
    g_ctx->VSSetConstantBuffers(0, 1, &g_cb);

    UINT stride = sizeof(Vertex), offset = 0;
    D3D11_MAPPED_SUBRESOURCE mapped;

    // Upload NPU caustic + refraction to RGBA16F texture
    g_ctx->Map(g_causticTex, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    const int NN = GRID * GRID;
    auto* dst = reinterpret_cast<uint16_t*>(mapped.pData);
    auto* causticSrc  = &g_renderBuf[6 * NN];
    auto* refractXSrc = &g_renderBuf[7 * NN];
    auto* refractZSrc = &g_renderBuf[8 * NN];
    int dstPitch = mapped.RowPitch / 2;  // in uint16_t units
    for (int row = 0; row < GRID; row++) {
        for (int col = 0; col < GRID; col++) {
            int si = row * GRID + col;
            int di = row * dstPitch + col * 4;
            dst[di + 0] = causticSrc[si];   // R = caustic
            dst[di + 1] = refractXSrc[si];  // G = refract_x
            dst[di + 2] = refractZSrc[si];  // B = refract_z
            dst[di + 3] = 0;                // A = 0
        }
    }
    g_ctx->Unmap(g_causticTex, 0);

    // === PASS 1: Opaque objects (tile floor, duck, ball) ===

    // --- Tile floor (with NPU caustics) ---
    XMStoreFloat4x4(&cb.wvp, XMMatrixTranspose(vp));
    XMStoreFloat4x4(&cb.world, XMMatrixTranspose(XMMatrixIdentity()));
    g_ctx->Map(g_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    memcpy(mapped.pData, &cb, sizeof(cb));
    g_ctx->Unmap(g_cb, 0);

    g_ctx->IASetVertexBuffers(0, 1, &g_tileVB, &stride, &offset);
    g_ctx->IASetIndexBuffer(g_tileIB, DXGI_FORMAT_R32_UINT, 0);
    g_ctx->PSSetShader(g_tilePS, nullptr, 0);
    g_ctx->PSSetConstantBuffers(0, 1, &g_cb);
    g_ctx->PSSetShaderResources(0, 1, &g_causticSRV);
    g_ctx->PSSetSamplers(0, 1, &g_sampler);
    g_ctx->DrawIndexed(g_tileNumIdx, 0, 0);

    // Unbind caustic SRV for other shaders
    ID3D11ShaderResourceView* nullSRV = nullptr;
    g_ctx->PSSetShaderResources(0, 1, &nullSRV);

    // --- Helper lambda to draw a duck at given position/tilt ---
    auto drawDuck = [&](float dx, float dy, float dz, float dtiltX, float dtiltZ) {
        XMVECTOR surfN = XMVector3Normalize(XMVectorSet(-dtiltX, g_normalY, -dtiltZ, 0));
        XMVECTOR upV = XMVectorSet(0, 1, 0, 0);
        XMVECTOR axis = XMVector3Cross(upV, surfN);
        float dot = XMVectorGetY(surfN);
        float ang = acosf(std::max(-1.0f, std::min(1.0f, dot)));
        XMMATRIX tilt = XMMatrixIdentity();
        if (ang > 0.001f && XMVectorGetX(XMVector3Length(axis)) > 0.0001f)
            tilt = XMMatrixRotationAxis(axis, ang);

        XMMATRIX duckWorld = XMMatrixScaling(DUCK_SCALE, DUCK_SCALE, DUCK_SCALE)
                           * tilt
                           * XMMatrixTranslation(dx, dy, dz);
        XMMATRIX duckWvp = duckWorld * vp;
        XMStoreFloat4x4(&cb.wvp, XMMatrixTranspose(duckWvp));
        XMStoreFloat4x4(&cb.world, XMMatrixTranspose(duckWorld));

        g_ctx->Map(g_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
        memcpy(mapped.pData, &cb, sizeof(cb));
        g_ctx->Unmap(g_cb, 0);

        g_ctx->IASetVertexBuffers(0, 1, &g_duckVB, &stride, &offset);
        g_ctx->IASetIndexBuffer(g_duckIB, DXGI_FORMAT_R32_UINT, 0);
        g_ctx->PSSetConstantBuffers(0, 1, &g_cb);

        g_ctx->PSSetShader(g_duckPS, nullptr, 0);
        g_ctx->DrawIndexed(g_duckBodyNumIdx, 0, 0);

        g_ctx->PSSetShader(g_billPS, nullptr, 0);
        g_ctx->DrawIndexed(g_duckBillNumIdx, g_duckBillStart, 0);
    };

    // --- Duck 1 ---
    drawDuck(g_duckX, g_duckY, g_duckZ, g_duckTiltX, g_duckTiltZ);

    // --- Duck 2 ---
    drawDuck(g_duck2X, g_duck2Y, g_duck2Z, g_duck2TiltX, g_duck2TiltZ);

    // === PASS 2: Transparent water with alpha blending ===
    float blendFactor[] = {0, 0, 0, 0};
    g_ctx->OMSetBlendState(g_blend, blendFactor, 0xFFFFFFFF);
    XMStoreFloat4x4(&cb.wvp, XMMatrixTranspose(vp));
    XMStoreFloat4x4(&cb.world, XMMatrixTranspose(XMMatrixIdentity()));
    g_ctx->Map(g_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    memcpy(mapped.pData, &cb, sizeof(cb));
    g_ctx->Unmap(g_cb, 0);

    g_ctx->IASetVertexBuffers(0, 1, &g_vb, &stride, &offset);
    g_ctx->IASetIndexBuffer(g_ib, DXGI_FORMAT_R32_UINT, 0);
    g_ctx->PSSetShader(g_ps, nullptr, 0);
    g_ctx->PSSetConstantBuffers(0, 1, &g_cb);
    g_ctx->DrawIndexed(g_numIdx, 0, 0);

    // Restore no-blend for next frame
    g_ctx->OMSetBlendState(nullptr, blendFactor, 0xFFFFFFFF);

    g_sc->Present(1, 0);
}

// ---------------------------------------------------------------------------
// Render ball pit scene
// ---------------------------------------------------------------------------
static void renderBalls() {
    float clearColor[] = {0.92f, 0.93f, 0.95f, 1.0f};
    g_ctx->ClearRenderTargetView(g_rtv, clearColor);
    g_ctx->ClearDepthStencilView(g_dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);
    g_ctx->OMSetRenderTargets(1, &g_rtv, g_dsv);

    float angle = g_ballCamAngle;
    XMVECTOR target = XMVectorSet(0.0f, BOX_HEIGHT * 0.25f, 0.0f, 0.0f);
    XMVECTOR eye = target + XMVectorSet(
        cosf(angle) * g_ballCamDist, g_ballCamHeight - BOX_HEIGHT * 0.25f,
        sinf(angle) * g_ballCamDist, 0.0f);
    XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

    XMMATRIX view = XMMatrixLookAtLH(eye, target, up);
    XMMATRIX proj = XMMatrixPerspectiveFovLH(XM_PIDIV4, float(WIN_W) / float(WIN_H), 0.1f, 500.0f);
    XMMATRIX vp = view * proj;
    g_lastVP = vp;

    CB cb{};
    XMFLOAT3 ld = {0.3f, 1.0f, 0.5f};
    XMStoreFloat3(&cb.lightDir, XMVector3Normalize(XMLoadFloat3(&ld)));
    XMStoreFloat3(&cb.eye, eye);
    cb.time = g_time;
    XMStoreFloat4x4(&cb.wvp, XMMatrixTranspose(vp));
    XMStoreFloat4x4(&cb.world, XMMatrixTranspose(XMMatrixIdentity()));

    D3D11_MAPPED_SUBRESOURCE mapped;
    g_ctx->Map(g_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    memcpy(mapped.pData, &cb, sizeof(cb));
    g_ctx->Unmap(g_cb, 0);

    // --- Draw spheres (instanced) ---
    g_ctx->Map(g_ballInstBuf, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    memcpy(mapped.pData, g_ballInstances.data(), NUM_BALLS * sizeof(SphereInstance));
    g_ctx->Unmap(g_ballInstBuf, 0);

    g_ctx->RSSetState(g_rs);  // back-face cull is fine for convex spheres
    g_ctx->IASetInputLayout(g_ballIL);
    g_ctx->VSSetShader(g_ballVS, nullptr, 0);
    g_ctx->VSSetConstantBuffers(0, 1, &g_cb);
    g_ctx->PSSetShader(g_ballPS, nullptr, 0);
    g_ctx->PSSetConstantBuffers(0, 1, &g_cb);

    UINT strides[2] = { sizeof(BallVertex), sizeof(SphereInstance) };
    UINT offsets[2] = { 0, 0 };
    ID3D11Buffer* vbs[2] = { g_ballSphereVB, g_ballInstBuf };
    g_ctx->IASetVertexBuffers(0, 2, vbs, strides, offsets);
    g_ctx->IASetIndexBuffer(g_ballSphereIB, DXGI_FORMAT_R32_UINT, 0);
    g_ctx->DrawIndexedInstanced(g_ballSphereNumIdx, NUM_BALLS, 0, 0, 0);

    g_sc->Present(1, 0);
}

// ---------------------------------------------------------------------------
// Window procedure
// ---------------------------------------------------------------------------
static LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_DESTROY:
        g_running = false;
        PostQuitMessage(0);
        return 0;

    case WM_KEYDOWN:
        if (wp == VK_ESCAPE) { g_running = false; PostQuitMessage(0); }
        if (wp == 'B') {
            g_scene = (g_scene == Scene::Water) ? Scene::Balls : Scene::Water;
            g_draggedDuck = -1;
            g_grabbedBall = -1;
        }
        if (g_scene == Scene::Water) {
            if (wp == VK_SPACE) {
                g_splashX = 0.0f; g_splashZ = 0.0f;
                g_splashRadius = 20.0f; g_splashHeight = 0.15f;
            }
            if (wp == 'R') resetState();
            if (wp == 'F') g_foamEnabled = !g_foamEnabled;
        }
        if (g_scene == Scene::Balls && wp == 'S') {
            g_ballShootEnabled = !g_ballShootEnabled;
        }
        if (wp == 'T' && g_panelHwnd) {
            g_showPanel = !g_showPanel;
            ShowWindow(g_panelHwnd, g_showPanel ? SW_SHOW : SW_HIDE);
        }
        return 0;

    case WM_LBUTTONDOWN: {
        float mx = static_cast<float>(LOWORD(lp));
        float my = static_cast<float>(HIWORD(lp));
        if (g_scene == Scene::Water) {
            float wx, wz;
            if (mouseToWaterPlane(mx, my, wx, wz)) {
                float d1 = (wx - g_duckX) * (wx - g_duckX) + (wz - g_duckZ) * (wz - g_duckZ);
                float d2 = (wx - g_duck2X) * (wx - g_duck2X) + (wz - g_duck2Z) * (wz - g_duck2Z);
                float pickR = DUCK_SCALE * DUCK_SCALE;
                if (d1 <= d2 && d1 < pickR) {
                    g_draggedDuck = 0;
                    SetCapture(hwnd);
                } else if (d2 < pickR) {
                    g_draggedDuck = 1;
                    SetCapture(hwnd);
                }
            }
        } else {
            // Ball pit: ray-sphere picking
            float ndcX = 2.0f * mx / WIN_W - 1.0f;
            float ndcY = 1.0f - 2.0f * my / WIN_H;
            XMMATRIX inv = XMMatrixInverse(nullptr, g_lastVP);
            XMVECTOR nearPt = XMVector3TransformCoord(XMVectorSet(ndcX, ndcY, 0.0f, 1.0f), inv);
            XMVECTOR farPt  = XMVector3TransformCoord(XMVectorSet(ndcX, ndcY, 1.0f, 1.0f), inv);
            XMVECTOR rayDir = XMVector3Normalize(farPt - nearPt);
            float bestT = 1e9f;
            int bestIdx = -1;
            for (int i = 0; i < NUM_BALLS; i++) {
                XMVECTOR center = XMVectorSet(g_balls[i].pos.x, g_balls[i].pos.y, g_balls[i].pos.z, 0);
                XMVECTOR oc = nearPt - center;
                float a = XMVectorGetX(XMVector3Dot(rayDir, rayDir));
                float b = 2.0f * XMVectorGetX(XMVector3Dot(oc, rayDir));
                float c = XMVectorGetX(XMVector3Dot(oc, oc)) - g_balls[i].radius * g_balls[i].radius;
                float disc = b * b - 4 * a * c;
                if (disc < 0) continue;
                float t = (-b - sqrtf(disc)) / (2 * a);
                if (t > 0 && t < bestT) { bestT = t; bestIdx = i; }
            }
            if (bestIdx >= 0) {
                g_grabbedBall = bestIdx;
                g_grabY = g_balls[bestIdx].pos.y;
                SetCapture(hwnd);
            }
        }
        return 0;
    }

    case WM_MOUSEMOVE: {
        float mx = static_cast<float>(LOWORD(lp));
        float my = static_cast<float>(HIWORD(lp));
        if (g_scene == Scene::Water) {
            if (g_rDragging) {
                float dx = mx - g_rDragStartX;
                g_camAngle = g_rDragStartAngle - dx * 0.01f;
            }
            if (g_draggedDuck >= 0) {
                float wx, wz;
                if (mouseToWaterPlane(mx, my, wx, wz)) {
                    float half = GRID * 0.5f - DUCK_SCALE * 0.5f;
                    wx = std::max(-half, std::min(half, wx));
                    wz = std::max(-half, std::min(half, wz));
                    if (g_draggedDuck == 0) { g_duckX = wx; g_duckZ = wz; }
                    else                    { g_duck2X = wx; g_duck2Z = wz; }
                }
            }
        } else {
            if (g_ballRDrag) {
                float dx = mx - g_ballRDragStartX;
                g_ballCamAngle = g_ballRDragStartAngle - dx * 0.01f;
            }
            if (g_grabbedBall >= 0) {
                float wx, wz;
                if (mouseToHorizPlane(mx, my, g_grabY, wx, wz)) {
                    Ball& b = g_balls[g_grabbedBall];
                    wx = std::max(-BOX_HALF + b.radius, std::min(BOX_HALF - b.radius, wx));
                    wz = std::max(-BOX_HALF + b.radius, std::min(BOX_HALF - b.radius, wz));
                    b.vel.x = (wx - b.pos.x) * 30.0f;
                    b.vel.z = (wz - b.pos.z) * 30.0f;
                    b.vel.y = 0;
                    b.pos.x = wx;
                    b.pos.z = wz;
                }
            }
        }
        return 0;
    }

    case WM_LBUTTONUP:
        if (g_scene == Scene::Water) {
            if (g_draggedDuck >= 0) {
                g_draggedDuck = -1;
                if (!g_rDragging) ReleaseCapture();
            }
        } else {
            if (g_grabbedBall >= 0) {
                g_grabbedBall = -1;
                if (!g_ballRDrag) ReleaseCapture();
            }
        }
        return 0;

    case WM_RBUTTONDOWN: {
        float rmx = static_cast<float>(LOWORD(lp));
        if (g_scene == Scene::Water) {
            g_rDragging = true;
            g_rDragStartX = rmx;
            g_rDragStartAngle = g_camAngle;
        } else {
            g_ballRDrag = true;
            g_ballRDragStartX = rmx;
            g_ballRDragStartAngle = g_ballCamAngle;
        }
        SetCapture(hwnd);
        return 0;
    }

    case WM_RBUTTONUP:
        if (g_scene == Scene::Water) {
            if (g_rDragging) {
                g_rDragging = false;
                if (g_draggedDuck < 0) ReleaseCapture();
            }
        } else {
            if (g_ballRDrag) {
                g_ballRDrag = false;
                if (g_grabbedBall < 0) ReleaseCapture();
            }
        }
        return 0;

    case WM_MOUSEWHEEL: {
        int delta = GET_WHEEL_DELTA_WPARAM(wp);
        if (g_scene == Scene::Water) {
            g_camDist *= (delta > 0) ? 0.9f : 1.1f;
            g_camDist = std::max(0.3f, std::min(2.0f, g_camDist));
        } else {
            g_ballCamDist *= (delta > 0) ? 0.9f : 1.1f;
            g_ballCamDist = std::max(20.0f, std::min(200.0f, g_ballCamDist));
        }
        return 0;
    }
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Parse flags: --device NPU|CPU (OpenVINO device)  --runtime auto|openvino|ort
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--device") == 0 || strcmp(argv[i], "-d") == 0) && i + 1 < argc) {
            g_device = argv[++i];
            for (auto& c : g_device) c = static_cast<char>(toupper(c));
        }
        if (strcmp(argv[i], "--runtime") == 0 && i + 1 < argc) {
            std::string rt = argv[++i];
            for (auto& c : rt) c = static_cast<char>(tolower(c));
            if (rt == "openvino" || rt == "ov")    g_runtimeChoice = RuntimeChoice::OpenVINO;
            else if (rt == "windowsml" || rt == "wml") g_runtimeChoice = RuntimeChoice::WindowsML;
            else if (rt == "auto")                  g_runtimeChoice = RuntimeChoice::Auto;
            else { printf("Unknown runtime: %s (use auto, openvino, or windowsml)\n", rt.c_str()); return 1; }
        }
    }
    // GPU device only works through Windows ML (DirectML EP)
    if (g_device == "GPU") {
        if (g_runtimeChoice == RuntimeChoice::OpenVINO) {
            printf("GPU not supported with OpenVINO — use --runtime windowsml\n");
            return 1;
        }
        g_runtimeChoice = RuntimeChoice::WindowsML;
    }
    printf("=== NPU Bath ===\n");
    printf("Controls:  Left-drag = grab duck/ball | Right-drag = rotate | Scroll = zoom | Space = splash | B = ball pit | R = reset | T = sliders | F = foam | Esc = quit\n\n");

    initBackend();
    printf("[OK] %s ready\n", g_backend->name().c_str());

    WNDCLASSW wc{};
    wc.lpfnWndProc  = wndProc;
    wc.hInstance     = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"NPUWater";
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);

    DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_VISIBLE;
    RECT r = {0, 0, WIN_W, WIN_H};
    AdjustWindowRect(&r, style, FALSE);

    g_hwnd = CreateWindowW(
        L"NPUWater", L"NPU Bath",
        style, CW_USEDEFAULT, CW_USEDEFAULT,
        r.right - r.left, r.bottom - r.top,
        nullptr, nullptr, wc.hInstance, nullptr);

    if (!g_hwnd) fail("CreateWindowW failed");

    initD3D();
    printf("[OK] D3D11 device created\n");

    initShaders();
    printf("[OK] Shaders compiled\n");

    initMesh();
    printf("[OK] Grid mesh: %d vertices, %u triangles\n", GRID * GRID, g_numIdx / 3);

    initBallPit();

    createSliderPanel(wc.hInstance);
    printf("[OK] Slider panel created (press T to toggle)\n\n");

    auto prevTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float fpsTimer = 0.0f;
    constexpr float SIM_DT = 1.0f / 30.0f;
    float simAccum = 0.0f;
    g_dt = SIM_DT;

    while (g_running) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (!g_running) break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - prevTime).count();
        prevTime = now;
        dt = std::min(dt, 0.1f);

        // Fixed 30Hz tick — simulation + render
        simAccum += dt;
        fpsTimer += dt;
        if (simAccum < SIM_DT) {
            Sleep(1);
            continue;
        }
        {
            simAccum -= SIM_DT;
            g_time += SIM_DT;
            if (g_time >= 50.0f) g_time -= 50.0f;

            if (g_scene == Scene::Water) {
                updateDuck();
                runSimulation();
                updateMesh();
                render();
            } else {
                updateBalls();
                renderBalls();
            }
            frameCount++;
        }

        if (fpsTimer >= 1.0f) {
            char title[256];
            if (g_scene == Scene::Water) {
                const char* prec = g_backend->isBf16() ? "BF16" : (g_backend->isFp32Model() ? "FP32" : "FP16");
                snprintf(title, sizeof(title),
                         "NPU Bath | %s | %d FPS | %dx%d %s",
                         g_backend->name().c_str(), frameCount, GRID, GRID, prec);
            } else {
                snprintf(title, sizeof(title),
                         "Ball Pit | %d balls | %d FPS | B = water",
                         NUM_BALLS, frameCount);
            }
            SetWindowTextA(g_hwnd, title);
            frameCount = 0;
            fpsTimer -= 1.0f;
        }
    }

    // Ball pit cleanup
    if (g_noCullRS)     g_noCullRS->Release();
    if (g_boxIB)        g_boxIB->Release();
    if (g_boxVB)        g_boxVB->Release();
    if (g_boxPS)        g_boxPS->Release();
    if (g_ballPS)       g_ballPS->Release();
    if (g_ballIL)       g_ballIL->Release();
    if (g_ballVS)       g_ballVS->Release();
    if (g_ballInstBuf)  g_ballInstBuf->Release();
    if (g_ballSphereIB) g_ballSphereIB->Release();
    if (g_ballSphereVB) g_ballSphereVB->Release();

    // Water scene cleanup
    if (g_sampler)    g_sampler->Release();
    if (g_causticSRV) g_causticSRV->Release();
    if (g_causticTex) g_causticTex->Release();
    if (g_tilePS) g_tilePS->Release();
    if (g_tileIB) g_tileIB->Release();
    if (g_tileVB) g_tileVB->Release();
    if (g_billPS) g_billPS->Release();
    if (g_duckPS) g_duckPS->Release();
    if (g_duckIB) g_duckIB->Release();
    if (g_duckVB) g_duckVB->Release();
    if (g_blend) g_blend->Release();
    if (g_rs)  g_rs->Release();
    if (g_ps)  g_ps->Release();
    if (g_vs)  g_vs->Release();
    if (g_il)  g_il->Release();
    if (g_cb)  g_cb->Release();
    if (g_ib)  g_ib->Release();
    if (g_vb)  g_vb->Release();
    if (g_dsv) g_dsv->Release();
    if (g_rtv) g_rtv->Release();
    if (g_sc)  g_sc->Release();
    if (g_ctx) g_ctx->Release();
    if (g_dev) g_dev->Release();

    return 0;
}
