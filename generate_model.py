#!/usr/bin/env python3
"""
Generate ONNX model for NPU ocean simulation — Gerstner wave synthesis.

Uses the same approach as AAA game engines:
  - 32 Gerstner waves with proper Phillips spectrum amplitudes
  - Deep-water dispersion: omega(k) = sqrt(g * |k|)
  - Deterministic ocean surface computed from time (no simulation instability)
  - Interactive ripple layer for duck/splash interaction (simple wave equation)

All operations are FP16 tensor ops on Intel NPU via OpenVINO.
  Inputs:  state [1,4,N,N] + time [1,1,1,1] + camera [1,3,1,1] + ...
  Outputs: new_state [1,4,N,N] + render [1,10,N,N] + ...
"""

import sys
import math
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

GRID = 256

# --- Gerstner ocean ---
# Tuned for Sea-of-Thieves-style gentle rolling swells.
# On a 256-cell grid (~256 m), moderate ocean has ~1-2 m significant wave height.
# With 32 waves, RMS sum ~ sqrt(32)*avg_amp, so individual amps must be small.
N_WAVES      = 32
WIND_SPEED   = 10.0          # m/s  (moderate breeze, not a storm)
WIND_ANGLE   = math.pi / 4   # diagonal wind direction
G_ACCEL      = 9.81
HEIGHT_SCALE = 3.0            # vertex displacement multiplier (was 40 — way too high)
AMP_GLOBAL   = 0.02           # global amplitude multiplier (was 0.15 — way too high)

# Time wrapping: omegas are quantized to multiples of 2π/WRAP_PERIOD
# so that time can seamlessly loop without any visible pop.
# Max omega*time = ~1.26 * 50 = 63 → FP16 precision ≈ 0.06, good enough.
WRAP_PERIOD  = 50.0           # seconds — seamless loop period
OMEGA_QUANT  = 2.0 * math.pi / WRAP_PERIOD  # ≈ 0.1257 — omega granularity

# --- Interactive ripple layer ---
RIPPLE_STEPS   = 8
RIPPLE_C       = 20.0        # wave propagation speed (cells/s)
RIPPLE_DT      = 0.02        # c*dt = 0.40 (stable, < 1)
RIPPLE_DAMPING = 0.995        # moderate damping — ripples fade over ~2s
SPONGE_W       = 0            # 0 = no absorption, waves reflect off edges like walls

# --- Foam layer (air bubbles on the water surface) ---
# Two extra state channels: foam density + unused (kept for 4-channel alignment).
# Foam is created only by significant turbulence (collisions, wakes, strong crests).
# Foam is transported by water surface slope (advection → wispy streaks).
# Foam is destroyed by turbulence and natural decay — no isotropic blur.
# All physics runs on NPU as tensor ops.
FOAM_GENERATION   = 2.0       # emission → foam density per frame
FOAM_DECAY        = 0.985     # natural bubble popping per frame (~1.5s half-life at 30fps)
FOAM_DESTROY_RATE = 1.0       # turbulence destruction scale
FOAM_MIN_SURVIVE  = 0.9       # minimum survival fraction (max 10% destroyed per frame)
FOAM_DUCK_SCALE   = 60.0      # amplify duck wake+bow for foam
FOAM_EMIT_CAP     = 0.3       # cap emission per frame
FOAM_BORDER       = 3         # border width (cells) to suppress Conv zero-padding artifacts
FOAM_CREST_THRESH = 0.06      # minimum wave convexity for whitecap foam
FOAM_CREST_SCALE  = 5.0       # wave convexity → foam generation rate
FOAM_MAX          = 1.0       # cap foam density
FOAM_ADVECT_SPEED = 12.0      # advection speed: foam transported by water surface slope
FOAM_ADVECT_STEPS = 4         # advection substeps: particle-like transport
FOAM_HEIGHT       = 0.2       # mesh raise for lighting (must not cause new collisions)
FOAM_DUCK_PUSH    = 8.0       # radial foam push strength from duck collision
FOAM_RIPPLE_BREAK = 6.0      # how much ripples/splashes break apart foam clusters
FOAM_PEAK_GRIP    = 0.85     # how strongly rising wave faces hold foam at the peak (0=none, 1=full)
FOAM_PARTICLE_SHARP = 20.0    # sharpness of particle edges (higher = crisper blobs)
FOAM_BUBBLE_ENHANCE = 4.0     # unsharp mask strength — enhances individual bubble domes


def generate_waves():
    """Generate 32 Gerstner waves with proper Phillips spectrum."""
    rng = np.random.RandomState(42)
    L = WIND_SPEED ** 2 / G_ACCEL   # ~22.9 — largest wave from wind
    small_l = L / 1000.0             # suppression cutoff for tiny waves

    waves = []

    # Logarithmic k sampling from short to long waves
    k_min = 2.0 * math.pi / 300.0   # longest wave: 300 cells (long rolling swells)
    k_max = 2.0 * math.pi / 40.0    # shortest wave: 40 cells
    log_ratio = math.log(k_max / k_min)

    for i in range(N_WAVES):
        t = i / max(N_WAVES - 1, 1)
        k = k_min * math.exp(log_ratio * t)
        wavelength = 2.0 * math.pi / k

        # dk for this sample (logarithmic spacing)
        dk = k * log_ratio / max(N_WAVES - 1, 1)

        # Direction: spread around wind with some randomness
        angle_offset = (rng.random() - 0.5) * math.pi * 0.7
        angle = WIND_ANGLE + angle_offset
        dx, dz = math.cos(angle), math.sin(angle)

        # Phillips spectrum: Ph(k) = A * exp(-1/(kL)^2) / k^4 * cos^2(theta)
        cos_wind = abs(math.cos(angle_offset))
        Ph = math.exp(-1.0 / (k * L) ** 2) / (k ** 4)
        Ph *= cos_wind ** 2
        # Suppress very short waves
        Ph *= math.exp(-(k ** 2) * (small_l ** 2))

        # Amplitude from spectral density: A = sqrt(2 * Ph * dk)
        amp = AMP_GLOBAL * math.sqrt(max(2.0 * abs(Ph) * dk, 0.0))
        # Small random variation
        amp *= (0.8 + 0.4 * rng.random())

        # Deep-water dispersion, quantized for seamless time wrapping
        omega_raw = math.sqrt(G_ACCEL * k)
        omega = round(omega_raw / OMEGA_QUANT) * OMEGA_QUANT

        # Random phase offset
        phi = rng.random() * 2 * math.pi

        waves.append((dx, dz, k, omega, amp, phi))

    return waves


def create_model(output_path: str):
    N = GRID
    P = [1, 1, 1, 1]

    waves = generate_waves()

    # Print wave info
    for wi, (dx, dz, k, omega, amp, phi) in enumerate(waves):
        wl = 2.0 * math.pi / k
        c = omega / k
        print(f"  Wave {wi:2d}: wl={wl:6.1f}  k={k:.4f}  w={omega:.3f}  "
              f"c={c:5.1f}  amp={amp:.4f}")

    # ----------------------------------------------------------------
    # Constant tensors
    # ----------------------------------------------------------------

    # Ripple combined kernel: c^2*dt^2 * Laplacian + 2*Identity
    c2dt2 = RIPPLE_C ** 2 * RIPPLE_DT ** 2
    ripple_k = np.zeros((1, 1, 3, 3), dtype=np.float16)
    ripple_k[0, 0, 1, 1] = np.float16(2.0 - 4.0 * c2dt2)
    ripple_k[0, 0, 0, 1] = np.float16(c2dt2)
    ripple_k[0, 0, 2, 1] = np.float16(c2dt2)
    ripple_k[0, 0, 1, 0] = np.float16(c2dt2)
    ripple_k[0, 0, 1, 2] = np.float16(c2dt2)

    # Derivative kernels for normals
    ddx_w = np.zeros((1, 1, 3, 3), dtype=np.float16)
    ddx_w[0, 0, 1, 0] = np.float16(-0.5)
    ddx_w[0, 0, 1, 2] = np.float16(0.5)

    ddy_w = np.zeros((1, 1, 3, 3), dtype=np.float16)
    ddy_w[0, 0, 0, 1] = np.float16(-0.5)
    ddy_w[0, 0, 2, 1] = np.float16(0.5)

    damping = np.array([[[[RIPPLE_DAMPING]]]], dtype=np.float16)
    h_scale = np.array([[[[HEIGHT_SCALE]]]], dtype=np.float16)

    # Sponge boundary — absorbs reflections without creating a visible dead zone.
    # Per-substep multiplier: 0.88 at edge → 1.0 at interior (smooth cubic ramp).
    # Over 8 substeps: 0.88^8 ≈ 0.36, so reflected energy drops ~64% per frame.
    sponge = np.ones((1, 1, N, N), dtype=np.float32)
    for i in range(SPONGE_W):
        t_val = i / SPONGE_W                          # 0 at edge, 1 at interior
        f = 0.88 + 0.12 * (3 * t_val**2 - 2 * t_val**3)  # smoothstep 0.88 → 1.0
        sponge[:, :, i, :]     = np.minimum(sponge[:, :, i, :], f)
        sponge[:, :, N-1-i, :] = np.minimum(sponge[:, :, N-1-i, :], f)
        sponge[:, :, :, i]     = np.minimum(sponge[:, :, :, i], f)
        sponge[:, :, :, N-1-i] = np.minimum(sponge[:, :, :, N-1-i], f)
    sponge = sponge.astype(np.float16)

    # Gaussian blur kernel (3x3, sigma≈0.85) — applied twice before Laplacian
    # to smooth FP16 quantization noise that second derivatives amplify
    blur_w = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16.0
    blur_w = blur_w.reshape(1, 1, 3, 3).astype(np.float16)

    # Laplacian kernel for NPU-computed caustics
    lap_w = np.zeros((1, 1, 3, 3), dtype=np.float16)
    lap_w[0, 0, 1, 1] = np.float16(-4.0)
    lap_w[0, 0, 0, 1] = np.float16(1.0)
    lap_w[0, 0, 2, 1] = np.float16(1.0)
    lap_w[0, 0, 1, 0] = np.float16(1.0)
    lap_w[0, 0, 1, 2] = np.float16(1.0)
    caustic_scale = np.array([[[[8.0]]]], dtype=np.float16)

    # NPU-computed refraction: base Snell's law offset = slope * HEIGHT_SCALE * depth * (1 - 1/n_water)
    # n_water = 1.33, depth = |FLOOR_Y| = 20
    # Then scaled per-cell by sec(theta_view) for oblique camera angles
    refract_scale = np.array([[[[HEIGHT_SCALE * 20.0 * (1.0 - 1.0 / 1.33)]]]], dtype=np.float16)

    # Coordinate grids for per-cell view angle computation
    coords = np.arange(N, dtype=np.float32) - N * 0.5
    x_grid = np.tile(coords.reshape(1, 1, 1, N), (1, 1, N, 1)).astype(np.float16)
    z_grid = np.tile(coords.reshape(1, 1, N, 1), (1, 1, 1, N)).astype(np.float16)

    # Clamp sec(theta) to prevent extreme distortion at horizon
    sec_clip_min = np.array([1.0], dtype=np.float16)
    sec_clip_max = np.array([3.0], dtype=np.float16)

    # --- Duck hull displacement + wake (NPU, full NxN tensor ops) ---
    DUCK_R = 45.0 * 0.4   # waterline footprint radius (DUCK_SCALE * 0.4)

    # --- Duck physics (NPU integration, CPU supplies slope at duck position) ---
    # All rates are per-second; multiplied by dt input for frame-rate independence
    grid_scale = N / 256.0  # scale physics for larger grids
    duck_force = np.array([[[[-80.0 * grid_scale]]]], dtype=np.float16)  # force per second (scaled for grid size)
    # Drag as exponential decay: drag_per_frame = exp(-decay_rate * dt)
    # At 30fps: 0.92 = exp(-rate * 0.033) → rate = -ln(0.92)/0.033 ≈ 2.53
    duck_decay_rate = np.array([[[[math.log(0.92) / 0.033]]]], dtype=np.float16)  # negative → exp(neg*dt) < 1
    duck_pos_limit = N * 0.5 - DUCK_R
    duck_pos_min = np.array([-duck_pos_limit], dtype=np.float16)
    duck_pos_max = np.array([duck_pos_limit], dtype=np.float16)
    hull_sigma = DUCK_R * 0.35  # tighter Gaussian — falls off faster
    hull_neg_inv_2s2 = np.array([[[[-1.0 / (2.0 * hull_sigma ** 2)]]]], dtype=np.float16)
    hull_depth = np.array([[[[-0.015]]]], dtype=np.float16)  # depression per second (scaled by dt)
    hull_cutoff_thresh = np.array([[[[0.01]]]], dtype=np.float16)  # kill Gaussian tail
    hull_clip_zero = np.array([0.0], dtype=np.float16)
    hull_clip_one  = np.array([1.0], dtype=np.float16)
    wake_epsilon2 = np.array([[[[1e-4]]]], dtype=np.float16)  # safe division
    wake_duckR = np.array([[[[DUCK_R]]]], dtype=np.float16)
    wake_ring_peak = np.array([[[[0.8]]]], dtype=np.float16)
    wake_neg_inv_rw = np.array([[[[-1.0 / 0.12]]]], dtype=np.float16)
    wake_strength = np.array([[[[0.03]]]], dtype=np.float16)  # per second (scaled by dt)

    # Duck bow wave: Gaussian mound pushed ahead of duck, proportional to speed
    bow_sigma2 = np.array([[[[-1.0 / (2.0 * (DUCK_R * 0.5) ** 2)]]]], dtype=np.float16)  # tighter than hull
    bow_strength = np.array([[[[0.015]]]], dtype=np.float16)  # per second
    bow_offset_scale = np.array([[[[DUCK_R * 0.7]]]], dtype=np.float16)  # how far ahead of duck

    # Duck slope sampling: Gaussian (σ²=4, σ=2 cells) — mean(w)≈3.8e-4, FP16-safe
    sample_neg_inv_s2 = np.array([[[[-0.125]]]], dtype=np.float16)  # -1/(2×4)
    slope_prescale = np.array([[[[512.0]]]], dtype=np.float16)       # prevent numerator underflow
    # Duck wall bounce: factor = 1.0 - 1.5 × hit → -0.5 on wall hit
    bounce_1p5 = np.array([[[[1.5]]]], dtype=np.float16)

    # Duck-duck collision constants
    DUCK_SCALE = 45.0
    col_min_dist = np.array([[[[DUCK_SCALE]]]], dtype=np.float16)  # two duck half-sizes
    col_pos_push = np.array([[[[0.5]]]], dtype=np.float16)   # position separation
    col_vel_push = np.array([[[[8.0]]]], dtype=np.float16)   # velocity impulse

    # Duck buoyancy + tilt constants
    duck_y_offset = np.array([[[[DUCK_SCALE * 0.35]]]], dtype=np.float16)  # bob offset
    bob_smooth = np.array([[[[0.4]]]], dtype=np.float16)
    bob_retain = np.array([[[[0.6]]]], dtype=np.float16)
    tilt_smooth = np.array([[[[0.3]]]], dtype=np.float16)
    tilt_retain = np.array([[[[0.7]]]], dtype=np.float16)

    # Ripple amplitude clamp — prevents blowup on FP32 devices (CPU)
    ripple_clip_min = np.array([-10.0], dtype=np.float16)
    ripple_clip_max = np.array([10.0], dtype=np.float16)

    # --- Foam layer constants ---
    foam_generation = np.array([[[[FOAM_GENERATION]]]], dtype=np.float16)
    foam_decay = np.array([[[[FOAM_DECAY]]]], dtype=np.float16)
    foam_destroy_rate = np.array([[[[FOAM_DESTROY_RATE]]]], dtype=np.float16)
    foam_min_survive = np.array([[[[FOAM_MIN_SURVIVE]]]], dtype=np.float16)
    foam_duck_scale = np.array([[[[FOAM_DUCK_SCALE]]]], dtype=np.float16)
    foam_emit_cap = np.array([FOAM_EMIT_CAP], dtype=np.float16)
    foam_crest_thresh = np.array([[[[FOAM_CREST_THRESH]]]], dtype=np.float16)
    foam_crest_scale = np.array([[[[FOAM_CREST_SCALE]]]], dtype=np.float16)
    foam_max = np.array([FOAM_MAX], dtype=np.float16)
    foam_advect_speed = np.array([[[[FOAM_ADVECT_SPEED / FOAM_ADVECT_STEPS]]]], dtype=np.float16)
    foam_height = np.array([[[[FOAM_HEIGHT]]]], dtype=np.float16)
    foam_duck_push = np.array([[[[FOAM_DUCK_PUSH]]]], dtype=np.float16)
    foam_particle_sharp = np.array([[[[FOAM_PARTICLE_SHARP]]]], dtype=np.float16)
    foam_bubble_enhance = np.array([[[[FOAM_BUBBLE_ENHANCE]]]], dtype=np.float16)
    foam_ripple_break = np.array([[[[FOAM_RIPPLE_BREAK]]]], dtype=np.float16)
    foam_peak_grip = np.array([[[[FOAM_PEAK_GRIP]]]], dtype=np.float16)
    foam_grip_scale = np.array([[[[5.0]]]], dtype=np.float16)  # sharpness of rising detection
    # Border mask: 0 in FOAM_BORDER-cell border, 1 interior
    B = FOAM_BORDER
    spl_border = np.ones((1, 1, N, N), dtype=np.float16)
    spl_border[:, :, :B, :] = 0
    spl_border[:, :, -B:, :] = 0
    spl_border[:, :, :, :B] = 0
    spl_border[:, :, :, -B:] = 0

    # Splash constants — matches CPU addSplash: height * exp(-d2 / (r2 * 0.25))
    splash_neg4 = np.array([[[[-4.0]]]], dtype=np.float16)

    initializers = [
        numpy_helper.from_array(ripple_k, "ripple_kernel"),
        numpy_helper.from_array(ddx_w,    "ddx_w"),
        numpy_helper.from_array(ddy_w,    "ddy_w"),
        numpy_helper.from_array(damping,  "damping"),
        numpy_helper.from_array(sponge,   "sponge_mask"),
        numpy_helper.from_array(h_scale,  "h_scale"),
        numpy_helper.from_array(blur_w,   "blur_w"),
        numpy_helper.from_array(lap_w,    "lap_w"),
        numpy_helper.from_array(caustic_scale, "caustic_scale"),
        numpy_helper.from_array(refract_scale, "refract_scale"),
        numpy_helper.from_array(x_grid,   "x_grid"),
        numpy_helper.from_array(z_grid,   "z_grid"),
        numpy_helper.from_array(sec_clip_min, "sec_clip_min"),
        numpy_helper.from_array(sec_clip_max, "sec_clip_max"),
        numpy_helper.from_array(duck_force, "duck_force"),
        numpy_helper.from_array(duck_decay_rate, "duck_decay_rate"),
        numpy_helper.from_array(duck_pos_min, "duck_pos_min"),
        numpy_helper.from_array(duck_pos_max, "duck_pos_max"),
        numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "split2"),
        numpy_helper.from_array(np.array([1, 1, 1], dtype=np.int64), "split3"),
        numpy_helper.from_array(np.array([1, 1, N, N], dtype=np.int64), "grid_shape"),
        numpy_helper.from_array(np.array([1, 1, 1, 1], dtype=np.int64), "split4"),
        numpy_helper.from_array(np.array([1, 1, 1, 1, 1, 1], dtype=np.int64), "split6"),
        numpy_helper.from_array(np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.int64), "split7"),
        numpy_helper.from_array(hull_neg_inv_2s2, "hull_neg_inv_2s2"),
        numpy_helper.from_array(hull_depth, "hull_depth"),
        numpy_helper.from_array(hull_cutoff_thresh, "hull_cutoff_thresh"),
        numpy_helper.from_array(hull_clip_zero, "hull_clip_zero"),
        numpy_helper.from_array(hull_clip_one, "hull_clip_one"),
        numpy_helper.from_array(wake_epsilon2, "wake_epsilon2"),
        numpy_helper.from_array(wake_duckR, "wake_duckR"),
        numpy_helper.from_array(wake_ring_peak, "wake_ring_peak"),
        numpy_helper.from_array(wake_neg_inv_rw, "wake_neg_inv_rw"),
        numpy_helper.from_array(wake_strength, "wake_strength"),
        numpy_helper.from_array(bow_sigma2, "bow_sigma2"),
        numpy_helper.from_array(bow_strength, "bow_strength"),
        numpy_helper.from_array(bow_offset_scale, "bow_offset_scale"),
        numpy_helper.from_array(sample_neg_inv_s2, "sample_neg_inv_s2"),
        numpy_helper.from_array(slope_prescale, "slope_prescale"),
        numpy_helper.from_array(bounce_1p5, "bounce_1p5"),
        numpy_helper.from_array(ripple_clip_min, "ripple_clip_min"),
        numpy_helper.from_array(ripple_clip_max, "ripple_clip_max"),
        numpy_helper.from_array(splash_neg4, "splash_neg4"),
        numpy_helper.from_array(col_min_dist, "col_min_dist"),
        numpy_helper.from_array(col_pos_push, "col_pos_push"),
        numpy_helper.from_array(col_vel_push, "col_vel_push"),
        numpy_helper.from_array(duck_y_offset, "duck_y_offset"),
        numpy_helper.from_array(bob_smooth, "bob_smooth"),
        numpy_helper.from_array(bob_retain, "bob_retain"),
        numpy_helper.from_array(tilt_smooth, "tilt_smooth"),
        numpy_helper.from_array(tilt_retain, "tilt_retain"),
        numpy_helper.from_array(foam_generation, "foam_generation"),
        numpy_helper.from_array(foam_decay, "foam_decay"),
        numpy_helper.from_array(foam_destroy_rate, "foam_destroy_rate"),
        numpy_helper.from_array(foam_min_survive, "foam_min_survive"),
        numpy_helper.from_array(foam_duck_scale, "foam_duck_scale"),
        numpy_helper.from_array(foam_emit_cap, "foam_emit_cap"),
        numpy_helper.from_array(foam_crest_thresh, "foam_crest_thresh"),
        numpy_helper.from_array(foam_crest_scale, "foam_crest_scale"),
        numpy_helper.from_array(foam_max, "foam_max"),
        numpy_helper.from_array(foam_advect_speed, "foam_advect_speed"),
        numpy_helper.from_array(foam_height, "foam_height"),
        numpy_helper.from_array(spl_border, "spl_border_mask"),
        numpy_helper.from_array(foam_duck_push, "foam_duck_push"),
        numpy_helper.from_array(foam_particle_sharp, "foam_particle_sharp"),
        numpy_helper.from_array(foam_bubble_enhance, "foam_bubble_enhance"),
        numpy_helper.from_array(foam_ripple_break, "foam_ripple_break"),
        numpy_helper.from_array(foam_peak_grip, "foam_peak_grip"),
        numpy_helper.from_array(foam_grip_scale, "foam_grip_scale"),
    ]

    # Precompute per-wave constants — packed into batch tensors
    # Phase grids wrapped to [-π, π) for FP16 precision (computed in float64)
    all_phase_grids = np.zeros((1, N_WAVES, N, N), dtype=np.float64)
    omega_all = np.zeros((1, N_WAVES, 1, 1), dtype=np.float16)
    amp_kernel = np.zeros((1, N_WAVES, 1, 1), dtype=np.float16)

    for wi, (dx, dz, k, omega, amp, phi) in enumerate(waves):
        for j in range(N):
            z = j - N * 0.5
            for ic in range(N):
                x = ic - N * 0.5
                all_phase_grids[0, wi, j, ic] = k * (x * dx + z * dz) + phi
        omega_all[0, wi, 0, 0] = np.float16(omega)
        amp_kernel[0, wi, 0, 0] = np.float16(amp)

    # Wrap phase grids to [-π, π) — eliminates large FP16 phase values
    all_phase_grids = all_phase_grids % (2.0 * np.pi)
    all_phase_grids[all_phase_grids > np.pi] -= 2.0 * np.pi
    all_phase_grids = all_phase_grids.astype(np.float16)

    # Phase wrapping constants (for incremental omega*time tracking on NPU)
    pi_val = np.array([[[[np.pi]]]], dtype=np.float16)
    two_pi_val = np.array([[[[2.0 * np.pi]]]], dtype=np.float16)
    wrap_scale = np.array([[[[10000.0]]]], dtype=np.float16)

    initializers.extend([
        numpy_helper.from_array(all_phase_grids, "all_phase_grids"),
        numpy_helper.from_array(omega_all, "omega_all"),
        numpy_helper.from_array(amp_kernel, "amp_kernel"),
        numpy_helper.from_array(pi_val, "pi_val"),
        numpy_helper.from_array(two_pi_val, "two_pi_val"),
        numpy_helper.from_array(wrap_scale, "wrap_scale"),
    ])

    # ----------------------------------------------------------------
    # Graph
    # ----------------------------------------------------------------
    state_in = helper.make_tensor_value_info(
        "state", TensorProto.FLOAT16, [1, 4, N, N])
    wave_phase_in = helper.make_tensor_value_info(
        "wave_phase", TensorProto.FLOAT16, [1, N_WAVES, 1, 1])
    camera_in = helper.make_tensor_value_info(
        "camera", TensorProto.FLOAT16, [1, 3, 1, 1])
    duck_in = helper.make_tensor_value_info(
        "duck_in", TensorProto.FLOAT16, [1, 7, 1, 1])
    dt_in = helper.make_tensor_value_info(
        "dt", TensorProto.FLOAT16, [1, 1, 1, 1])
    duck2_in = helper.make_tensor_value_info(
        "duck2_in", TensorProto.FLOAT16, [1, 7, 1, 1])
    splash_in = helper.make_tensor_value_info(
        "splash_in", TensorProto.FLOAT16, [1, 4, 1, 1])
    foam_params_in = helper.make_tensor_value_info(
        "foam_params", TensorProto.FLOAT16, [1, 4, 1, 1])
    render_params_in = helper.make_tensor_value_info(
        "render_params", TensorProto.FLOAT16, [1, 3, 1, 1])

    nodes = []

    # ================================================================
    # GERSTNER OCEAN — 32 waves, incremental phase tracking (FP16-safe)
    #
    # Instead of computing Sin(phase_grid + omega*time) where the argument
    # can reach ~70 (FP16 precision 0.06), we maintain per-wave wrapped
    # phases in [-π, π) and advance by omega*dt each frame.
    # Sin argument never exceeds ~2π — consistent precision on all NPUs.
    # ================================================================

    # Advance phases: wave_phase += omega * dt
    nodes.append(helper.make_node("Mul", ["omega_all", "dt"], ["omega_dt_all"]))
    nodes.append(helper.make_node("Add", ["wave_phase", "omega_dt_all"], ["wp_advanced"]))

    # Wrap to [-π, π) using soft step: if phase > π, subtract 2π
    nodes.append(helper.make_node("Sub", ["wp_advanced", "pi_val"], ["wp_diff"]))
    nodes.append(helper.make_node("Mul", ["wp_diff", "wrap_scale"], ["wp_scaled"]))
    nodes.append(helper.make_node("Clip", ["wp_scaled", "hull_clip_zero", "hull_clip_one"], ["wp_step"]))
    nodes.append(helper.make_node("Mul", ["wp_step", "two_pi_val"], ["wp_correction"]))
    nodes.append(helper.make_node("Sub", ["wp_advanced", "wp_correction"], ["wp_wrapped"]))

    # Total phase = wrapped_grid_phase + wrapped_omega_time  (both in [-π,π))
    # → argument to Sin always in [-2π, 2π], FP16 precision ~0.004
    nodes.append(helper.make_node("Add", ["all_phase_grids", "wp_wrapped"], ["gw_total_phase"]))
    nodes.append(helper.make_node("Sin", ["gw_total_phase"], ["gw_sin_all"]))

    # Amplitude-weighted sum: 1x1 Conv over 32 wave channels → single height
    nodes.append(helper.make_node("Conv", ["gw_sin_all", "amp_kernel"], ["ocean_h"]))
    ocean_h = "ocean_h"

    # Save updated phases for next frame
    nodes.append(helper.make_node("Identity", ["wp_wrapped"], ["wave_phase_out"]))

    # ================================================================
    # DUCK INPUT SPLIT — needed before ripple loop for hull displacement
    # ================================================================
    nodes.append(helper.make_node(
        "Split", ["duck_in", "split7"],
        ["duck_x", "duck_z", "duck_vx", "duck_vz", "duck_y", "duck_tiltx", "duck_tiltz"],
        axis=1))

    nodes.append(helper.make_node(
        "Split", ["duck2_in", "split7"],
        ["duck2_x", "duck2_z", "duck2_vx", "duck2_vz", "duck2_y", "duck2_tiltx", "duck2_tiltz"],
        axis=1))

    # ================================================================
    # INTERACTIVE RIPPLE LAYER — simple damped wave equation
    # ================================================================
    nodes.append(helper.make_node(
        "Split", ["state", "split4"], ["rh_in", "rhp_in", "spl_h_in", "spl_v_in"], axis=1))

    # ================================================================
    # NPU HULL DISPLACEMENT — Gaussian depression under duck1, hard-cutoff tail
    # ================================================================
    nodes.append(helper.make_node("Sub", ["x_grid", "duck_x"], ["hull_dx"]))
    nodes.append(helper.make_node("Sub", ["z_grid", "duck_z"], ["hull_dz"]))
    nodes.append(helper.make_node("Mul", ["hull_dx", "hull_dx"], ["hull_dx2"]))
    nodes.append(helper.make_node("Mul", ["hull_dz", "hull_dz"], ["hull_dz2"]))
    nodes.append(helper.make_node("Add", ["hull_dx2", "hull_dz2"], ["hull_dist2"]))
    nodes.append(helper.make_node("Mul", ["hull_dist2", "hull_neg_inv_2s2"], ["hull_exp_arg"]))
    nodes.append(helper.make_node("Exp", ["hull_exp_arg"], ["hull_gauss_raw"]))
    # Hard cutoff: subtract threshold, clip to [0,1] — kills the long tail
    nodes.append(helper.make_node("Sub", ["hull_gauss_raw", "hull_cutoff_thresh"], ["hull_gauss_shifted"]))
    nodes.append(helper.make_node("Clip", ["hull_gauss_shifted", "hull_clip_zero", "hull_clip_one"], ["hull_gauss"]))
    nodes.append(helper.make_node("Mul", ["hull_gauss", "hull_depth"], ["hull_depress_ps"]))
    nodes.append(helper.make_node("Mul", ["hull_depress_ps", "dt"], ["hull_depress"]))
    nodes.append(helper.make_node("Add", ["rh_in", "hull_depress"], ["rh_hulled"]))

    # ================================================================
    # NPU WAKE INJECTION — directional ring impulse from duck1 motion
    # ================================================================
    nodes.append(helper.make_node("Add", ["hull_dist2", "wake_epsilon2"], ["wake_dist2s"]))
    nodes.append(helper.make_node("Sqrt", ["wake_dist2s"], ["wake_dist"]))
    nodes.append(helper.make_node("Div", ["wake_dist", "wake_duckR"], ["wake_tn"]))
    nodes.append(helper.make_node("Sub", ["wake_tn", "wake_ring_peak"], ["wake_tn_off"]))
    nodes.append(helper.make_node("Mul", ["wake_tn_off", "wake_tn_off"], ["wake_tn_off2"]))
    nodes.append(helper.make_node("Mul", ["wake_tn_off2", "wake_neg_inv_rw"], ["wake_ring_arg"]))
    nodes.append(helper.make_node("Exp", ["wake_ring_arg"], ["wake_ring_raw"]))
    nodes.append(helper.make_node("Mul", ["wake_ring_raw", "hull_gauss"], ["wake_ring"]))
    nodes.append(helper.make_node("Div", ["hull_dx", "wake_dist"], ["wake_nx"]))
    nodes.append(helper.make_node("Div", ["hull_dz", "wake_dist"], ["wake_nz"]))
    nodes.append(helper.make_node("Mul", ["duck_vx", "duck_vx"], ["wake_vx2"]))
    nodes.append(helper.make_node("Mul", ["duck_vz", "duck_vz"], ["wake_vz2"]))
    nodes.append(helper.make_node("Add", ["wake_vx2", "wake_vz2"], ["wake_spd2"]))
    nodes.append(helper.make_node("Add", ["wake_spd2", "wake_epsilon2"], ["wake_spd2s"]))
    nodes.append(helper.make_node("Sqrt", ["wake_spd2s"], ["wake_speed"]))
    nodes.append(helper.make_node("Div", ["duck_vx", "wake_speed"], ["wake_mdx"]))
    nodes.append(helper.make_node("Div", ["duck_vz", "wake_speed"], ["wake_mdz"]))
    nodes.append(helper.make_node("Mul", ["wake_nx", "wake_mdx"], ["wake_fx"]))
    nodes.append(helper.make_node("Mul", ["wake_nz", "wake_mdz"], ["wake_fz"]))
    nodes.append(helper.make_node("Add", ["wake_fx", "wake_fz"], ["wake_facing"]))
    nodes.append(helper.make_node("Mul", ["wake_facing", "wake_ring"], ["wake_fr"]))
    nodes.append(helper.make_node("Mul", ["wake_fr", "wake_speed"], ["wake_frs"]))
    nodes.append(helper.make_node("Mul", ["wake_frs", "wake_strength"], ["wake_impulse_ps"]))
    nodes.append(helper.make_node("Mul", ["wake_impulse_ps", "dt"], ["wake_impulse"]))
    nodes.append(helper.make_node("Add", ["rh_hulled", "wake_impulse"], ["rh_wake_done"]))

    # ================================================================
    # NPU BOW WAVE — Gaussian mound ahead of duck1, proportional to speed
    # ================================================================
    nodes.append(helper.make_node("Mul", ["wake_mdx", "bow_offset_scale"], ["bow_off_x"]))
    nodes.append(helper.make_node("Mul", ["wake_mdz", "bow_offset_scale"], ["bow_off_z"]))
    nodes.append(helper.make_node("Add", ["duck_x", "bow_off_x"], ["bow_cx"]))
    nodes.append(helper.make_node("Add", ["duck_z", "bow_off_z"], ["bow_cz"]))
    nodes.append(helper.make_node("Sub", ["x_grid", "bow_cx"], ["bow_dx"]))
    nodes.append(helper.make_node("Sub", ["z_grid", "bow_cz"], ["bow_dz"]))
    nodes.append(helper.make_node("Mul", ["bow_dx", "bow_dx"], ["bow_dx2"]))
    nodes.append(helper.make_node("Mul", ["bow_dz", "bow_dz"], ["bow_dz2"]))
    nodes.append(helper.make_node("Add", ["bow_dx2", "bow_dz2"], ["bow_dist2"]))
    nodes.append(helper.make_node("Mul", ["bow_dist2", "bow_sigma2"], ["bow_exp_arg"]))
    nodes.append(helper.make_node("Exp", ["bow_exp_arg"], ["bow_gauss"]))
    nodes.append(helper.make_node("Mul", ["bow_gauss", "wake_speed"], ["bow_gs"]))
    nodes.append(helper.make_node("Mul", ["bow_gs", "bow_strength"], ["bow_impulse_ps"]))
    nodes.append(helper.make_node("Mul", ["bow_impulse_ps", "dt"], ["bow_impulse"]))
    nodes.append(helper.make_node("Add", ["rh_wake_done", "bow_impulse"], ["rh_duck_ready"]))

    # ================================================================
    # DUCK2 HULL DISPLACEMENT — Gaussian depression under duck2
    # ================================================================
    nodes.append(helper.make_node("Sub", ["x_grid", "duck2_x"], ["hull2_dx"]))
    nodes.append(helper.make_node("Sub", ["z_grid", "duck2_z"], ["hull2_dz"]))
    nodes.append(helper.make_node("Mul", ["hull2_dx", "hull2_dx"], ["hull2_dx2"]))
    nodes.append(helper.make_node("Mul", ["hull2_dz", "hull2_dz"], ["hull2_dz2"]))
    nodes.append(helper.make_node("Add", ["hull2_dx2", "hull2_dz2"], ["hull2_dist2"]))
    nodes.append(helper.make_node("Mul", ["hull2_dist2", "hull_neg_inv_2s2"], ["hull2_exp_arg"]))
    nodes.append(helper.make_node("Exp", ["hull2_exp_arg"], ["hull2_gauss_raw"]))
    nodes.append(helper.make_node("Sub", ["hull2_gauss_raw", "hull_cutoff_thresh"], ["hull2_gauss_shifted"]))
    nodes.append(helper.make_node("Clip", ["hull2_gauss_shifted", "hull_clip_zero", "hull_clip_one"], ["hull2_gauss"]))
    nodes.append(helper.make_node("Mul", ["hull2_gauss", "hull_depth"], ["hull2_depress_ps"]))
    nodes.append(helper.make_node("Mul", ["hull2_depress_ps", "dt"], ["hull2_depress"]))
    nodes.append(helper.make_node("Add", ["rh_duck_ready", "hull2_depress"], ["rh_hulled2"]))

    # ================================================================
    # DUCK2 WAKE INJECTION — directional ring impulse from duck2 motion
    # ================================================================
    nodes.append(helper.make_node("Add", ["hull2_dist2", "wake_epsilon2"], ["wake2_dist2s"]))
    nodes.append(helper.make_node("Sqrt", ["wake2_dist2s"], ["wake2_dist"]))
    nodes.append(helper.make_node("Div", ["wake2_dist", "wake_duckR"], ["wake2_tn"]))
    nodes.append(helper.make_node("Sub", ["wake2_tn", "wake_ring_peak"], ["wake2_tn_off"]))
    nodes.append(helper.make_node("Mul", ["wake2_tn_off", "wake2_tn_off"], ["wake2_tn_off2"]))
    nodes.append(helper.make_node("Mul", ["wake2_tn_off2", "wake_neg_inv_rw"], ["wake2_ring_arg"]))
    nodes.append(helper.make_node("Exp", ["wake2_ring_arg"], ["wake2_ring_raw"]))
    nodes.append(helper.make_node("Mul", ["wake2_ring_raw", "hull2_gauss"], ["wake2_ring"]))
    nodes.append(helper.make_node("Div", ["hull2_dx", "wake2_dist"], ["wake2_nx"]))
    nodes.append(helper.make_node("Div", ["hull2_dz", "wake2_dist"], ["wake2_nz"]))
    nodes.append(helper.make_node("Mul", ["duck2_vx", "duck2_vx"], ["wake2_vx2"]))
    nodes.append(helper.make_node("Mul", ["duck2_vz", "duck2_vz"], ["wake2_vz2"]))
    nodes.append(helper.make_node("Add", ["wake2_vx2", "wake2_vz2"], ["wake2_spd2"]))
    nodes.append(helper.make_node("Add", ["wake2_spd2", "wake_epsilon2"], ["wake2_spd2s"]))
    nodes.append(helper.make_node("Sqrt", ["wake2_spd2s"], ["wake2_speed"]))
    nodes.append(helper.make_node("Div", ["duck2_vx", "wake2_speed"], ["wake2_mdx"]))
    nodes.append(helper.make_node("Div", ["duck2_vz", "wake2_speed"], ["wake2_mdz"]))
    nodes.append(helper.make_node("Mul", ["wake2_nx", "wake2_mdx"], ["wake2_fx"]))
    nodes.append(helper.make_node("Mul", ["wake2_nz", "wake2_mdz"], ["wake2_fz"]))
    nodes.append(helper.make_node("Add", ["wake2_fx", "wake2_fz"], ["wake2_facing"]))
    nodes.append(helper.make_node("Mul", ["wake2_facing", "wake2_ring"], ["wake2_fr"]))
    nodes.append(helper.make_node("Mul", ["wake2_fr", "wake2_speed"], ["wake2_frs"]))
    nodes.append(helper.make_node("Mul", ["wake2_frs", "wake_strength"], ["wake2_impulse_ps"]))
    nodes.append(helper.make_node("Mul", ["wake2_impulse_ps", "dt"], ["wake2_impulse"]))
    nodes.append(helper.make_node("Add", ["rh_hulled2", "wake2_impulse"], ["rh_wake2_done"]))

    # ================================================================
    # DUCK2 BOW WAVE — Gaussian mound ahead of duck2
    # ================================================================
    nodes.append(helper.make_node("Mul", ["wake2_mdx", "bow_offset_scale"], ["bow2_off_x"]))
    nodes.append(helper.make_node("Mul", ["wake2_mdz", "bow_offset_scale"], ["bow2_off_z"]))
    nodes.append(helper.make_node("Add", ["duck2_x", "bow2_off_x"], ["bow2_cx"]))
    nodes.append(helper.make_node("Add", ["duck2_z", "bow2_off_z"], ["bow2_cz"]))
    nodes.append(helper.make_node("Sub", ["x_grid", "bow2_cx"], ["bow2_dx"]))
    nodes.append(helper.make_node("Sub", ["z_grid", "bow2_cz"], ["bow2_dz"]))
    nodes.append(helper.make_node("Mul", ["bow2_dx", "bow2_dx"], ["bow2_dx2"]))
    nodes.append(helper.make_node("Mul", ["bow2_dz", "bow2_dz"], ["bow2_dz2"]))
    nodes.append(helper.make_node("Add", ["bow2_dx2", "bow2_dz2"], ["bow2_dist2"]))
    nodes.append(helper.make_node("Mul", ["bow2_dist2", "bow_sigma2"], ["bow2_exp_arg"]))
    nodes.append(helper.make_node("Exp", ["bow2_exp_arg"], ["bow2_gauss"]))
    nodes.append(helper.make_node("Mul", ["bow2_gauss", "wake2_speed"], ["bow2_gs"]))
    nodes.append(helper.make_node("Mul", ["bow2_gs", "bow_strength"], ["bow2_impulse_ps"]))
    nodes.append(helper.make_node("Mul", ["bow2_impulse_ps", "dt"], ["bow2_impulse"]))
    nodes.append(helper.make_node("Add", ["rh_wake2_done", "bow2_impulse"], ["rh_duck2_ready"]))

    # ================================================================
    # NPU SPLASH — Gaussian impulse at arbitrary position
    # splash_in = (x, z, radius, height) — height=0 when no splash
    # ================================================================
    nodes.append(helper.make_node(
        "Split", ["splash_in", "split4"],
        ["splash_x", "splash_z", "splash_radius", "splash_height"],
        axis=1))
    nodes.append(helper.make_node("Sub", ["x_grid", "splash_x"], ["sp_dx"]))
    nodes.append(helper.make_node("Sub", ["z_grid", "splash_z"], ["sp_dz"]))
    nodes.append(helper.make_node("Mul", ["sp_dx", "sp_dx"], ["sp_dx2"]))
    nodes.append(helper.make_node("Mul", ["sp_dz", "sp_dz"], ["sp_dz2"]))
    nodes.append(helper.make_node("Add", ["sp_dx2", "sp_dz2"], ["sp_dist2"]))
    nodes.append(helper.make_node("Mul", ["splash_radius", "splash_radius"], ["sp_r2"]))
    nodes.append(helper.make_node("Div", ["sp_dist2", "sp_r2"], ["sp_ratio"]))
    nodes.append(helper.make_node("Mul", ["sp_ratio", "splash_neg4"], ["sp_exp_arg"]))
    nodes.append(helper.make_node("Exp", ["sp_exp_arg"], ["sp_gauss"]))
    nodes.append(helper.make_node("Sub", ["sp_r2", "sp_dist2"], ["sp_cut_diff"]))
    nodes.append(helper.make_node("Mul", ["sp_cut_diff", "wrap_scale"], ["sp_cut_sc"]))
    nodes.append(helper.make_node("Clip", ["sp_cut_sc", "hull_clip_zero", "hull_clip_one"], ["sp_mask"]))
    nodes.append(helper.make_node("Mul", ["sp_gauss", "sp_mask"], ["sp_masked"]))
    nodes.append(helper.make_node("Mul", ["sp_masked", "splash_height"], ["sp_impulse"]))
    nodes.append(helper.make_node("Add", ["rh_duck2_ready", "sp_impulse"], ["rh_ready"]))

    rh, rhp = "rh_ready", "rhp_in"
    for step in range(RIPPLE_STEPS):
        s = f"_r{step}"
        nodes.append(helper.make_node(
            "Conv", [rh, "ripple_kernel"], [f"rprop{s}"], pads=P))
        nodes.append(helper.make_node(
            "Sub", [f"rprop{s}", rhp], [f"rraw{s}"]))
        nodes.append(helper.make_node(
            "Mul", [f"rraw{s}", "damping"], [f"rdamp{s}"]))
        nodes.append(helper.make_node(
            "Mul", [f"rdamp{s}", "sponge_mask"], [f"rsponge{s}"]))
        rh_out = f"rh_{step + 1}"
        nodes.append(helper.make_node(
            "Clip", [f"rsponge{s}", "ripple_clip_min", "ripple_clip_max"], [rh_out]))
        rhp_out = f"rhp_{step + 1}"
        nodes.append(helper.make_node("Identity", [rh], [rhp_out]))
        rh, rhp = rh_out, rhp_out

    # ================================================================
    # COMBINE + RENDER
    # ================================================================
    nodes.append(helper.make_node("Add", [ocean_h, rh], ["total_h"]))

    nodes.append(helper.make_node("Mul",  ["total_h", "h_scale"], ["h_sc"]))
    nodes.append(helper.make_node("Conv", ["total_h", "ddx_w"],   ["r_dhdx"], pads=P))
    nodes.append(helper.make_node("Conv", ["total_h", "ddy_w"],   ["r_dhdy"], pads=P))
    nodes.append(helper.make_node("Mul",  ["r_dhdx", "h_scale"],  ["dhdx_sc"]))
    nodes.append(helper.make_node("Mul",  ["r_dhdy", "h_scale"],  ["dhdy_sc"]))

    # NPU caustics: blur → blur → Laplacian → Abs → scale
    nodes.append(helper.make_node("Conv", ["total_h", "blur_w"],  ["blur1"],  pads=P))
    nodes.append(helper.make_node("Conv", ["blur1",   "blur_w"],  ["blur2"],  pads=P))
    nodes.append(helper.make_node("Conv", ["blur2",   "lap_w"],   ["lap_h"],  pads=P))
    nodes.append(helper.make_node("Abs",  ["lap_h"],              ["abs_lap"]))
    nodes.append(helper.make_node("Mul",  ["abs_lap", "caustic_scale"], ["caustic"]))

    # ================================================================
    # VIEW-DEPENDENT REFRACTION
    # ================================================================
    nodes.append(helper.make_node(
        "Split", ["camera", "split3"], ["eye_x", "eye_y", "eye_z"], axis=1))
    nodes.append(helper.make_node("Sub", ["x_grid", "eye_x"], ["vdx"]))
    nodes.append(helper.make_node("Sub", ["z_grid", "eye_z"], ["vdz"]))
    nodes.append(helper.make_node("Mul", ["vdx", "vdx"], ["vdx2"]))
    nodes.append(helper.make_node("Mul", ["vdz", "vdz"], ["vdz2"]))
    nodes.append(helper.make_node("Mul", ["eye_y", "eye_y"], ["vdy2"]))
    nodes.append(helper.make_node("Add", ["vdx2", "vdz2"], ["vdxz2"]))
    nodes.append(helper.make_node("Add", ["vdxz2", "vdy2"], ["vdist2"]))
    nodes.append(helper.make_node("Sqrt", ["vdist2"], ["vdist"]))
    nodes.append(helper.make_node("Div", ["vdist", "eye_y"], ["sec_raw"]))
    nodes.append(helper.make_node("Clip", ["sec_raw", "sec_clip_min", "sec_clip_max"], ["sec_theta"]))
    nodes.append(helper.make_node("Mul", ["r_dhdx", "refract_scale"], ["refract_base_x"]))
    nodes.append(helper.make_node("Mul", ["r_dhdy", "refract_scale"], ["refract_base_z"]))
    nodes.append(helper.make_node("Mul", ["refract_base_x", "sec_theta"], ["refract_x"]))
    nodes.append(helper.make_node("Mul", ["refract_base_z", "sec_theta"], ["refract_z"]))

    # ================================================================
    # DUCK1 SLOPE SAMPLING — Gaussian weighted average
    # ================================================================
    nodes.append(helper.make_node("Mul", ["hull_dist2", "sample_neg_inv_s2"], ["sample_exp_arg"]))
    nodes.append(helper.make_node("Exp", ["sample_exp_arg"], ["sample_w"]))
    nodes.append(helper.make_node("Mul", ["dhdx_sc", "slope_prescale"], ["dhdx_big"]))
    nodes.append(helper.make_node("Mul", ["dhdy_sc", "slope_prescale"], ["dhdy_big"]))
    nodes.append(helper.make_node("Mul", ["sample_w", "dhdx_big"], ["sample_wx"]))
    nodes.append(helper.make_node("Mul", ["sample_w", "dhdy_big"], ["sample_wz"]))
    nodes.append(helper.make_node("GlobalAveragePool", ["sample_wx"], ["slope_x_mean"]))
    nodes.append(helper.make_node("GlobalAveragePool", ["sample_wz"], ["slope_z_mean"]))
    nodes.append(helper.make_node("GlobalAveragePool", ["sample_w"], ["w_mean"]))
    nodes.append(helper.make_node("Add", ["w_mean", "wake_epsilon2"], ["w_mean_safe"]))
    nodes.append(helper.make_node("Div", ["slope_x_mean", "w_mean_safe"], ["dk_slope_x_big"]))
    nodes.append(helper.make_node("Div", ["slope_z_mean", "w_mean_safe"], ["dk_slope_z_big"]))
    nodes.append(helper.make_node("Div", ["dk_slope_x_big", "slope_prescale"], ["dk_slope_x"]))
    nodes.append(helper.make_node("Div", ["dk_slope_z_big", "slope_prescale"], ["dk_slope_z"]))

    # ================================================================
    # DUCK2 SLOPE SAMPLING — same pattern, uses hull2_dist2
    # ================================================================
    nodes.append(helper.make_node("Mul", ["hull2_dist2", "sample_neg_inv_s2"], ["sample2_exp_arg"]))
    nodes.append(helper.make_node("Exp", ["sample2_exp_arg"], ["sample2_w"]))
    nodes.append(helper.make_node("Mul", ["sample2_w", "dhdx_big"], ["sample2_wx"]))
    nodes.append(helper.make_node("Mul", ["sample2_w", "dhdy_big"], ["sample2_wz"]))
    nodes.append(helper.make_node("GlobalAveragePool", ["sample2_wx"], ["slope2_x_mean"]))
    nodes.append(helper.make_node("GlobalAveragePool", ["sample2_wz"], ["slope2_z_mean"]))
    nodes.append(helper.make_node("GlobalAveragePool", ["sample2_w"], ["w2_mean"]))
    nodes.append(helper.make_node("Add", ["w2_mean", "wake_epsilon2"], ["w2_mean_safe"]))
    nodes.append(helper.make_node("Div", ["slope2_x_mean", "w2_mean_safe"], ["dk2_slope_x_big"]))
    nodes.append(helper.make_node("Div", ["slope2_z_mean", "w2_mean_safe"], ["dk2_slope_z_big"]))
    nodes.append(helper.make_node("Div", ["dk2_slope_x_big", "slope_prescale"], ["dk2_slope_x"]))
    nodes.append(helper.make_node("Div", ["dk2_slope_z_big", "slope_prescale"], ["dk2_slope_z"]))

    # ================================================================
    # DUCK-DUCK COLLISION — push both ducks apart when overlapping
    # ================================================================
    nodes.append(helper.make_node("Sub", ["duck_x", "duck2_x"], ["col_dx"]))
    nodes.append(helper.make_node("Sub", ["duck_z", "duck2_z"], ["col_dz"]))
    nodes.append(helper.make_node("Mul", ["col_dx", "col_dx"], ["col_dx2"]))
    nodes.append(helper.make_node("Mul", ["col_dz", "col_dz"], ["col_dz2"]))
    nodes.append(helper.make_node("Add", ["col_dx2", "col_dz2"], ["col_dist2"]))
    nodes.append(helper.make_node("Add", ["col_dist2", "wake_epsilon2"], ["col_dist2s"]))
    nodes.append(helper.make_node("Sqrt", ["col_dist2s"], ["col_dist"]))
    # overlap = max(0, minDist - dist)
    nodes.append(helper.make_node("Sub", ["col_min_dist", "col_dist"], ["col_overlap_raw"]))
    nodes.append(helper.make_node("Clip", ["col_overlap_raw", "hull_clip_zero", "col_min_dist"], ["col_overlap"]))
    # Push direction (duck1→duck2 normalized)
    nodes.append(helper.make_node("Div", ["col_dx", "col_dist"], ["col_nx"]))
    nodes.append(helper.make_node("Div", ["col_dz", "col_dist"], ["col_nz"]))
    # Velocity impulse
    nodes.append(helper.make_node("Mul", ["col_overlap", "col_vel_push"], ["col_vscale"]))
    nodes.append(helper.make_node("Mul", ["col_nx", "col_vscale"], ["col_dvx"]))
    nodes.append(helper.make_node("Mul", ["col_nz", "col_vscale"], ["col_dvz"]))
    # Duck1: push in +direction (away from duck2)
    nodes.append(helper.make_node("Add", ["duck_vx", "col_dvx"], ["dk_vx_col"]))
    nodes.append(helper.make_node("Add", ["duck_vz", "col_dvz"], ["dk_vz_col"]))
    # Duck2: push in -direction (away from duck1)
    nodes.append(helper.make_node("Sub", ["duck2_vx", "col_dvx"], ["dk2_vx_col"]))
    nodes.append(helper.make_node("Sub", ["duck2_vz", "col_dvz"], ["dk2_vz_col"]))
    # Position separation
    nodes.append(helper.make_node("Mul", ["col_overlap", "col_pos_push"], ["col_pscale"]))
    nodes.append(helper.make_node("Mul", ["col_nx", "col_pscale"], ["col_dpx"]))
    nodes.append(helper.make_node("Mul", ["col_nz", "col_pscale"], ["col_dpz"]))
    nodes.append(helper.make_node("Add", ["duck_x", "col_dpx"], ["dk_x_col"]))
    nodes.append(helper.make_node("Add", ["duck_z", "col_dpz"], ["dk_z_col"]))
    nodes.append(helper.make_node("Sub", ["duck2_x", "col_dpx"], ["dk2_x_col"]))
    nodes.append(helper.make_node("Sub", ["duck2_z", "col_dpz"], ["dk2_z_col"]))

    # ================================================================
    # DUCK1 HEIGHT SAMPLING — Gaussian weighted average for buoyancy
    # ================================================================
    nodes.append(helper.make_node("Mul", ["sample_w", "h_sc"], ["sample_wh"]))
    nodes.append(helper.make_node("GlobalAveragePool", ["sample_wh"], ["h_mean"]))
    nodes.append(helper.make_node("Div", ["h_mean", "w_mean_safe"], ["dk_h_at_duck"]))
    nodes.append(helper.make_node("Add", ["dk_h_at_duck", "duck_y_offset"], ["dk_target_y"]))
    nodes.append(helper.make_node("Mul", ["duck_y", "bob_retain"], ["dk_y_old"]))
    nodes.append(helper.make_node("Mul", ["dk_target_y", "bob_smooth"], ["dk_y_new"]))
    nodes.append(helper.make_node("Add", ["dk_y_old", "dk_y_new"], ["dk_y_out"]))

    # ================================================================
    # DUCK2 HEIGHT SAMPLING
    # ================================================================
    nodes.append(helper.make_node("Mul", ["sample2_w", "h_sc"], ["sample2_wh"]))
    nodes.append(helper.make_node("GlobalAveragePool", ["sample2_wh"], ["h2_mean"]))
    nodes.append(helper.make_node("Div", ["h2_mean", "w2_mean_safe"], ["dk2_h_at_duck"]))
    nodes.append(helper.make_node("Add", ["dk2_h_at_duck", "duck_y_offset"], ["dk2_target_y"]))
    nodes.append(helper.make_node("Mul", ["duck2_y", "bob_retain"], ["dk2_y_old"]))
    nodes.append(helper.make_node("Mul", ["dk2_target_y", "bob_smooth"], ["dk2_y_new"]))
    nodes.append(helper.make_node("Add", ["dk2_y_old", "dk2_y_new"], ["dk2_y_out"]))

    # ================================================================
    # DUCK1 TILT SMOOTHING
    # ================================================================
    nodes.append(helper.make_node("Mul", ["duck_tiltx", "tilt_retain"], ["dk_tiltx_old"]))
    nodes.append(helper.make_node("Mul", ["dk_slope_x", "tilt_smooth"], ["dk_tiltx_new"]))
    nodes.append(helper.make_node("Add", ["dk_tiltx_old", "dk_tiltx_new"], ["dk_tiltx_out"]))
    nodes.append(helper.make_node("Mul", ["duck_tiltz", "tilt_retain"], ["dk_tiltz_old"]))
    nodes.append(helper.make_node("Mul", ["dk_slope_z", "tilt_smooth"], ["dk_tiltz_new"]))
    nodes.append(helper.make_node("Add", ["dk_tiltz_old", "dk_tiltz_new"], ["dk_tiltz_out"]))

    # ================================================================
    # DUCK2 TILT SMOOTHING
    # ================================================================
    nodes.append(helper.make_node("Mul", ["duck2_tiltx", "tilt_retain"], ["dk2_tiltx_old"]))
    nodes.append(helper.make_node("Mul", ["dk2_slope_x", "tilt_smooth"], ["dk2_tiltx_new"]))
    nodes.append(helper.make_node("Add", ["dk2_tiltx_old", "dk2_tiltx_new"], ["dk2_tiltx_out"]))
    nodes.append(helper.make_node("Mul", ["duck2_tiltz", "tilt_retain"], ["dk2_tiltz_old"]))
    nodes.append(helper.make_node("Mul", ["dk2_slope_z", "tilt_smooth"], ["dk2_tiltz_new"]))
    nodes.append(helper.make_node("Add", ["dk2_tiltz_old", "dk2_tiltz_new"], ["dk2_tiltz_out"]))

    # ================================================================
    # DUCK1 PHYSICS — free-floating rubber duck on NPU (dt-scaled)
    # ================================================================
    nodes.append(helper.make_node("Mul", ["dk_slope_x", "duck_force"], ["dk_fx_ps"]))
    nodes.append(helper.make_node("Mul", ["dk_slope_z", "duck_force"], ["dk_fz_ps"]))
    nodes.append(helper.make_node("Mul", ["dk_fx_ps", "dt"], ["dk_fx"]))
    nodes.append(helper.make_node("Mul", ["dk_fz_ps", "dt"], ["dk_fz"]))
    nodes.append(helper.make_node("Add", ["dk_vx_col", "dk_fx"], ["dk_vx_push"]))
    nodes.append(helper.make_node("Add", ["dk_vz_col", "dk_fz"], ["dk_vz_push"]))
    nodes.append(helper.make_node("Mul", ["duck_decay_rate", "dt"], ["dk_decay_dt"]))
    nodes.append(helper.make_node("Exp", ["dk_decay_dt"], ["dk_drag"]))
    nodes.append(helper.make_node("Mul", ["dk_vx_push", "dk_drag"], ["dk_vx_new"]))
    nodes.append(helper.make_node("Mul", ["dk_vz_push", "dk_drag"], ["dk_vz_new"]))
    nodes.append(helper.make_node("Mul", ["dk_vx_new", "dt"], ["dk_dx_step"]))
    nodes.append(helper.make_node("Mul", ["dk_vz_new", "dt"], ["dk_dz_step"]))
    nodes.append(helper.make_node("Add", ["dk_x_col", "dk_dx_step"], ["dk_x_raw"]))
    nodes.append(helper.make_node("Add", ["dk_z_col", "dk_dz_step"], ["dk_z_raw"]))
    nodes.append(helper.make_node(
        "Clip", ["dk_x_raw", "duck_pos_min", "duck_pos_max"], ["dk_x_clamp"]))
    nodes.append(helper.make_node(
        "Clip", ["dk_z_raw", "duck_pos_min", "duck_pos_max"], ["dk_z_clamp"]))
    # Wall bounce
    nodes.append(helper.make_node("Sub", ["dk_x_raw", "dk_x_clamp"], ["bounce_dx"]))
    nodes.append(helper.make_node("Abs", ["bounce_dx"], ["bounce_dx_abs"]))
    nodes.append(helper.make_node("Mul", ["bounce_dx_abs", "wrap_scale"], ["bounce_dx_sc"]))
    nodes.append(helper.make_node("Clip", ["bounce_dx_sc", "hull_clip_zero", "hull_clip_one"], ["bounce_x_hit"]))
    nodes.append(helper.make_node("Mul", ["bounce_x_hit", "bounce_1p5"], ["bounce_x_off"]))
    nodes.append(helper.make_node("Sub", ["hull_clip_one", "bounce_x_off"], ["bounce_x_fac"]))
    nodes.append(helper.make_node("Mul", ["dk_vx_new", "bounce_x_fac"], ["dk_vx_bounced"]))
    nodes.append(helper.make_node("Sub", ["dk_z_raw", "dk_z_clamp"], ["bounce_dz"]))
    nodes.append(helper.make_node("Abs", ["bounce_dz"], ["bounce_dz_abs"]))
    nodes.append(helper.make_node("Mul", ["bounce_dz_abs", "wrap_scale"], ["bounce_dz_sc"]))
    nodes.append(helper.make_node("Clip", ["bounce_dz_sc", "hull_clip_zero", "hull_clip_one"], ["bounce_z_hit"]))
    nodes.append(helper.make_node("Mul", ["bounce_z_hit", "bounce_1p5"], ["bounce_z_off"]))
    nodes.append(helper.make_node("Sub", ["hull_clip_one", "bounce_z_off"], ["bounce_z_fac"]))
    nodes.append(helper.make_node("Mul", ["dk_vz_new", "bounce_z_fac"], ["dk_vz_bounced"]))
    nodes.append(helper.make_node("Concat",
        ["dk_x_clamp", "dk_z_clamp", "dk_vx_bounced", "dk_vz_bounced",
         "dk_y_out", "dk_tiltx_out", "dk_tiltz_out"],
        ["duck_out"], axis=1))

    # ================================================================
    # DUCK2 PHYSICS — identical pattern, reuses dk_drag
    # ================================================================
    nodes.append(helper.make_node("Mul", ["dk2_slope_x", "duck_force"], ["dk2_fx_ps"]))
    nodes.append(helper.make_node("Mul", ["dk2_slope_z", "duck_force"], ["dk2_fz_ps"]))
    nodes.append(helper.make_node("Mul", ["dk2_fx_ps", "dt"], ["dk2_fx"]))
    nodes.append(helper.make_node("Mul", ["dk2_fz_ps", "dt"], ["dk2_fz"]))
    nodes.append(helper.make_node("Add", ["dk2_vx_col", "dk2_fx"], ["dk2_vx_push"]))
    nodes.append(helper.make_node("Add", ["dk2_vz_col", "dk2_fz"], ["dk2_vz_push"]))
    nodes.append(helper.make_node("Mul", ["dk2_vx_push", "dk_drag"], ["dk2_vx_new"]))
    nodes.append(helper.make_node("Mul", ["dk2_vz_push", "dk_drag"], ["dk2_vz_new"]))
    nodes.append(helper.make_node("Mul", ["dk2_vx_new", "dt"], ["dk2_dx_step"]))
    nodes.append(helper.make_node("Mul", ["dk2_vz_new", "dt"], ["dk2_dz_step"]))
    nodes.append(helper.make_node("Add", ["dk2_x_col", "dk2_dx_step"], ["dk2_x_raw"]))
    nodes.append(helper.make_node("Add", ["dk2_z_col", "dk2_dz_step"], ["dk2_z_raw"]))
    nodes.append(helper.make_node(
        "Clip", ["dk2_x_raw", "duck_pos_min", "duck_pos_max"], ["dk2_x_clamp"]))
    nodes.append(helper.make_node(
        "Clip", ["dk2_z_raw", "duck_pos_min", "duck_pos_max"], ["dk2_z_clamp"]))
    # Wall bounce
    nodes.append(helper.make_node("Sub", ["dk2_x_raw", "dk2_x_clamp"], ["bounce2_dx"]))
    nodes.append(helper.make_node("Abs", ["bounce2_dx"], ["bounce2_dx_abs"]))
    nodes.append(helper.make_node("Mul", ["bounce2_dx_abs", "wrap_scale"], ["bounce2_dx_sc"]))
    nodes.append(helper.make_node("Clip", ["bounce2_dx_sc", "hull_clip_zero", "hull_clip_one"], ["bounce2_x_hit"]))
    nodes.append(helper.make_node("Mul", ["bounce2_x_hit", "bounce_1p5"], ["bounce2_x_off"]))
    nodes.append(helper.make_node("Sub", ["hull_clip_one", "bounce2_x_off"], ["bounce2_x_fac"]))
    nodes.append(helper.make_node("Mul", ["dk2_vx_new", "bounce2_x_fac"], ["dk2_vx_bounced"]))
    nodes.append(helper.make_node("Sub", ["dk2_z_raw", "dk2_z_clamp"], ["bounce2_dz"]))
    nodes.append(helper.make_node("Abs", ["bounce2_dz"], ["bounce2_dz_abs"]))
    nodes.append(helper.make_node("Mul", ["bounce2_dz_abs", "wrap_scale"], ["bounce2_dz_sc"]))
    nodes.append(helper.make_node("Clip", ["bounce2_dz_sc", "hull_clip_zero", "hull_clip_one"], ["bounce2_z_hit"]))
    nodes.append(helper.make_node("Mul", ["bounce2_z_hit", "bounce_1p5"], ["bounce2_z_off"]))
    nodes.append(helper.make_node("Sub", ["hull_clip_one", "bounce2_z_off"], ["bounce2_z_fac"]))
    nodes.append(helper.make_node("Mul", ["dk2_vz_new", "bounce2_z_fac"], ["dk2_vz_bounced"]))
    nodes.append(helper.make_node("Concat",
        ["dk2_x_clamp", "dk2_z_clamp", "dk2_vx_bounced", "dk2_vz_bounced",
         "dk2_y_out", "dk2_tiltx_out", "dk2_tiltz_out"],
        ["duck2_out"], axis=1))

    # ================================================================
    # FOAM LAYER
    # ================================================================

    # Split foam_params input: (threshold, coarseness, decay, generation)
    nodes.append(helper.make_node(
        "Split", ["foam_params", "split4"],
        ["foam_thresh_in", "foam_coarse_in", "foam_decay_in", "foam_gen_in"], axis=1))

    # --- Foam emission: bubbles created by high velocity + height + collisions ---
    # Bubbles form when waves are tall AND fast-moving, or when waves collide
    # with themselves or objects. Three signals combined:
    #   1. |velocity| × |height| — tall fast waves break into foam
    #   2. Laplacian magnitude — wave collisions/convergence (curvature)
    #   3. All thresholded by user-controllable foam_thresh_in

    # Signal 1: velocity × height (breaking waves)
    nodes.append(helper.make_node("Sub", [rh, rhp], ["ripple_vel"]))
    nodes.append(helper.make_node("Abs", ["ripple_vel"], ["ripple_vel_abs"]))
    nodes.append(helper.make_node("Abs", [rh], ["ripple_h_abs"]))
    nodes.append(helper.make_node("Mul", ["ripple_vel_abs", "ripple_h_abs"], ["vel_x_height"]))

    # Signal 2: wave collision/convergence (Laplacian curvature)
    # Negative Laplacian = concave up = wave crests colliding
    nodes.append(helper.make_node("Neg", ["lap_h"], ["neg_lap_h"]))
    nodes.append(helper.make_node("Clip", ["neg_lap_h", "hull_clip_zero", "ripple_clip_max"], ["collision_raw"]))
    nodes.append(helper.make_node("Mul", ["collision_raw", "foam_crest_scale"], ["collision_foam"]))

    # Combine: breaking waves + collisions
    nodes.append(helper.make_node("Add", ["vel_x_height", "collision_foam"], ["em_combined"]))
    nodes.append(helper.make_node("Clip", ["em_combined", "hull_clip_zero", "foam_emit_cap"], ["em_capped"]))

    # Threshold: only create bubbles above user-controlled threshold
    nodes.append(helper.make_node("Sub", ["em_capped", "foam_thresh_in"], ["em_over"]))
    nodes.append(helper.make_node("Clip", ["em_over", "hull_clip_zero", "ripple_clip_max"], ["em_thresh"]))
    nodes.append(helper.make_node("Mul", ["em_thresh", "spl_border_mask"], ["em_masked"]))
    nodes.append(helper.make_node("Mul", ["em_masked", "foam_gen_in"], ["foam_gen"]))

    # --- Foam destruction: water turbulence pops bubbles ---
    nodes.append(helper.make_node("Mul", ["abs_lap", "foam_destroy_rate"], ["foam_turb"]))
    nodes.append(helper.make_node("Sub", ["hull_clip_one", "foam_turb"], ["foam_surv_raw"]))
    nodes.append(helper.make_node("Clip", ["foam_surv_raw", "foam_min_survive", "hull_clip_one"], ["foam_survive"]))

    # --- Wave peak grip: rising surface holds foam, falling surface lets it slide ---
    # ripple_vel > 0 means surface rising (front face) → grip foam
    # ripple_vel < 0 means surface falling (back face) → let slide
    nodes.append(helper.make_node("Mul", ["ripple_vel", "foam_grip_scale"], ["grip_scaled"]))
    nodes.append(helper.make_node("Clip", ["grip_scaled", "hull_clip_zero", "hull_clip_one"], ["rising_amt"]))
    nodes.append(helper.make_node("Mul", ["rising_amt", "foam_peak_grip"], ["grip_reduce"]))
    nodes.append(helper.make_node("Sub", ["hull_clip_one", "grip_reduce"], ["foam_friction"]))
    # foam_friction: ~0.15 on rising faces (foam sticks), ~1.0 on falling faces (foam slides)

    # --- Foam advection: multi-substep transport by water surface slope ---
    foam_cur = "spl_h_in"
    for step in range(FOAM_ADVECT_STEPS):
        a = f"_a{step}"
        nodes.append(helper.make_node("Conv", [foam_cur, "ddx_w"], [f"foam_dx{a}"], pads=P))
        nodes.append(helper.make_node("Conv", [foam_cur, "ddy_w"], [f"foam_dz{a}"], pads=P))
        nodes.append(helper.make_node("Mul", ["r_dhdx", f"foam_dx{a}"], [f"advect_x{a}"]))
        nodes.append(helper.make_node("Mul", ["r_dhdy", f"foam_dz{a}"], [f"advect_z{a}"]))
        nodes.append(helper.make_node("Add", [f"advect_x{a}", f"advect_z{a}"], [f"advect_dot{a}"]))
        nodes.append(helper.make_node("Mul", [f"advect_dot{a}", "foam_advect_speed"], [f"advect_raw{a}"]))
        nodes.append(helper.make_node("Mul", [f"advect_raw{a}", "foam_friction"], [f"advect_term{a}"]))
        nodes.append(helper.make_node("Add", [foam_cur, f"advect_term{a}"], [f"foam_adv_raw{a}"]))
        foam_next = f"foam_adv_{step}"
        nodes.append(helper.make_node("Clip", [f"foam_adv_raw{a}", "hull_clip_zero", "foam_max"], [foam_next]))
        foam_cur = foam_next
    nodes.append(helper.make_node("Identity", [foam_cur], ["foam_adv_clamp"]))

    # --- Ripple/splash breakup: turbulence diffuses foam clusters apart ---
    # Where waves are active, foam spreads out. Where calm, foam stays together.
    nodes.append(helper.make_node("Conv", ["foam_adv_clamp", "blur_w"], ["foam_diff_blur"], pads=P))
    nodes.append(helper.make_node("Sub", ["foam_diff_blur", "foam_adv_clamp"], ["foam_diff_term"]))
    nodes.append(helper.make_node("Mul", ["foam_diff_term", "ripple_vel_abs"], ["foam_diff_turb"]))
    nodes.append(helper.make_node("Mul", ["foam_diff_turb", "foam_ripple_break"], ["foam_diff_scaled"]))
    nodes.append(helper.make_node("Add", ["foam_adv_clamp", "foam_diff_scaled"], ["foam_broken"]))
    nodes.append(helper.make_node("Clip", ["foam_broken", "hull_clip_zero", "foam_max"], ["foam_broken_clamp"]))

    # --- Duck-foam collision: ducks radially push foam outward ---
    # Advection equation: df/dt = -v·∇f where v points radially from duck
    # Uses existing hull_gauss (proximity), wake_nx/nz (direction), wake_speed

    # Duck1: compute foam gradient, project onto radial direction, push
    nodes.append(helper.make_node("Conv", ["foam_broken_clamp", "ddx_w"], ["fpush_dx1"], pads=P))
    nodes.append(helper.make_node("Conv", ["foam_broken_clamp", "ddy_w"], ["fpush_dz1"], pads=P))
    nodes.append(helper.make_node("Mul", ["wake_nx", "fpush_dx1"], ["fpush_rx1"]))
    nodes.append(helper.make_node("Mul", ["wake_nz", "fpush_dz1"], ["fpush_rz1"]))
    nodes.append(helper.make_node("Add", ["fpush_rx1", "fpush_rz1"], ["fpush_radial1"]))
    nodes.append(helper.make_node("Mul", ["hull_gauss", "wake_speed"], ["fpush_gs1"]))
    nodes.append(helper.make_node("Mul", ["fpush_gs1", "foam_duck_push"], ["fpush_str1"]))
    nodes.append(helper.make_node("Mul", ["fpush_str1", "fpush_radial1"], ["fpush_term1"]))
    nodes.append(helper.make_node("Sub", ["foam_broken_clamp", "fpush_term1"], ["foam_dk1_raw"]))
    nodes.append(helper.make_node("Clip", ["foam_dk1_raw", "hull_clip_zero", "foam_max"], ["foam_dk1_done"]))

    # Duck2: same radial push
    nodes.append(helper.make_node("Conv", ["foam_dk1_done", "ddx_w"], ["fpush_dx2"], pads=P))
    nodes.append(helper.make_node("Conv", ["foam_dk1_done", "ddy_w"], ["fpush_dz2"], pads=P))
    nodes.append(helper.make_node("Mul", ["wake2_nx", "fpush_dx2"], ["fpush_rx2"]))
    nodes.append(helper.make_node("Mul", ["wake2_nz", "fpush_dz2"], ["fpush_rz2"]))
    nodes.append(helper.make_node("Add", ["fpush_rx2", "fpush_rz2"], ["fpush_radial2"]))
    nodes.append(helper.make_node("Mul", ["hull2_gauss", "wake2_speed"], ["fpush_gs2"]))
    nodes.append(helper.make_node("Mul", ["fpush_gs2", "foam_duck_push"], ["fpush_str2"]))
    nodes.append(helper.make_node("Mul", ["fpush_str2", "fpush_radial2"], ["fpush_term2"]))
    nodes.append(helper.make_node("Sub", ["foam_dk1_done", "fpush_term2"], ["foam_dk2_raw"]))
    nodes.append(helper.make_node("Clip", ["foam_dk2_raw", "hull_clip_zero", "foam_max"], ["foam_dk_done"]))

    # --- Foam update: decay × survive_turbulence + new_generation ---
    nodes.append(helper.make_node("Mul", ["foam_dk_done", "foam_decay_in"], ["foam_decayed"]))
    nodes.append(helper.make_node("Mul", ["foam_decayed", "foam_survive"], ["foam_survived"]))
    nodes.append(helper.make_node("Add", ["foam_survived", "foam_gen"], ["foam_updated"]))
    nodes.append(helper.make_node("Clip", ["foam_updated", "hull_clip_zero", "foam_max"], ["foam_clamped"]))
    nodes.append(helper.make_node("Mul", ["foam_clamped", "spl_border_mask"], ["foam_out"]))

    # --- Particle shaping: coarseness controls particle size ---
    # Blend raw foam (small particles) ↔ blurred foam (large round blobs)
    # coarseness=0 → raw, coarseness=1 → fully blurred
    # 4 blur passes = wider smoothing → larger round particles
    nodes.append(helper.make_node("Conv", ["foam_out", "blur_w"], ["foam_blur1"], pads=P))
    nodes.append(helper.make_node("Conv", ["foam_blur1", "blur_w"], ["foam_blur2"], pads=P))
    nodes.append(helper.make_node("Conv", ["foam_blur2", "blur_w"], ["foam_blur3"], pads=P))
    nodes.append(helper.make_node("Conv", ["foam_blur3", "blur_w"], ["foam_blur4"], pads=P))
    nodes.append(helper.make_node("Sub", ["foam_blur4", "foam_out"], ["foam_blur_diff"]))
    nodes.append(helper.make_node("Mul", ["foam_blur_diff", "foam_coarse_in"], ["foam_blend_term"]))
    nodes.append(helper.make_node("Add", ["foam_out", "foam_blend_term"], ["foam_sized"]))
    # Age-dependent threshold: fresh foam → crisp 3D, old foam → flat 2D texture
    nodes.append(helper.make_node("Mul", ["foam_thresh_in", "foam_out"], ["foam_age_thresh"]))
    nodes.append(helper.make_node("Sub", ["foam_sized", "foam_age_thresh"], ["foam_over_thresh"]))
    nodes.append(helper.make_node("Mul", ["foam_over_thresh", "foam_particle_sharp"], ["foam_sharp"]))
    nodes.append(helper.make_node("Clip", ["foam_sharp", "hull_clip_zero", "hull_clip_one"], ["foam_render"]))

    # --- Render: 3D bubble domes via unsharp mask ---
    # Extract per-cell detail: foam_render - blur(foam_render) = high-freq bumps
    # Each local peak in the foam field becomes a distinct raised bubble dome
    nodes.append(helper.make_node("Conv", ["foam_render", "blur_w"], ["foam_render_blur"], pads=P))
    nodes.append(helper.make_node("Sub", ["foam_render", "foam_render_blur"], ["foam_detail"]))
    nodes.append(helper.make_node("Mul", ["foam_detail", "foam_bubble_enhance"], ["foam_bump"]))
    nodes.append(helper.make_node("Add", ["foam_render", "foam_bump"], ["foam_3d"]))
    nodes.append(helper.make_node("Clip", ["foam_3d", "hull_clip_zero", "hull_clip_one"], ["foam_3d_clamp"]))
    nodes.append(helper.make_node("Mul", ["foam_3d_clamp", "foam_height"], ["foam_h"]))
    nodes.append(helper.make_node("Add", ["h_sc", "foam_h"], ["h_with_foam"]))

    nodes.append(helper.make_node("Conv", ["h_with_foam", "ddx_w"], ["dhdx_foam"], pads=P))
    nodes.append(helper.make_node("Conv", ["h_with_foam", "ddy_w"], ["dhdy_foam"], pads=P))

    # Split render_params [1,3,1,1] → chopScale, heightScale, normalY
    nodes.append(helper.make_node(
        "Split", ["render_params", "split3"], ["rp_chop", "rp_height", "rp_normalY"], axis=1))

    # Vertex Y = h_with_foam * heightScale (was CPU: renderBuf[0] * g_heightScale)
    nodes.append(helper.make_node("Mul", ["h_with_foam", "rp_height"], ["pos_y"]))

    # Choppy displacement: pos_x = x_grid - chopScale * dhdx_foam
    #                      pos_z = z_grid - chopScale * dhdy_foam
    nodes.append(helper.make_node("Mul", ["dhdx_foam", "rp_chop"], ["chop_dx"]))
    nodes.append(helper.make_node("Mul", ["dhdy_foam", "rp_chop"], ["chop_dz"]))
    nodes.append(helper.make_node("Sub", ["x_grid", "chop_dx"], ["pos_x"]))
    nodes.append(helper.make_node("Sub", ["z_grid", "chop_dz"], ["pos_z"]))

    # Normals: nrm = (-dhdx_foam, normalY, -dhdz_foam)
    nodes.append(helper.make_node("Neg", ["dhdx_foam"], ["nrm_x"]))
    nodes.append(helper.make_node("Neg", ["dhdy_foam"], ["nrm_z"]))
    # Expand normalY scalar to full grid
    nodes.append(helper.make_node("Expand", ["rp_normalY", "grid_shape"], ["nrm_y"]))

    nodes.append(helper.make_node("Concat",
        ["pos_x", "pos_y", "pos_z", "nrm_x", "nrm_y", "nrm_z", "caustic", "refract_x", "refract_z", "foam_render"],
        ["render_out"], axis=1))

    # State output [1, 4, N, N] = (rh, rhp, foam, foam_dup)
    nodes.append(helper.make_node("Concat",
        [rh, rhp, "foam_out", "foam_out"],
        ["new_state"], axis=1))

    # ----------------------------------------------------------------
    # Build model
    # ----------------------------------------------------------------
    state_out = helper.make_tensor_value_info(
        "new_state", TensorProto.FLOAT16, [1, 4, N, N])
    render_out = helper.make_tensor_value_info(
        "render_out", TensorProto.FLOAT16, [1, 10, N, N])
    duck_out = helper.make_tensor_value_info(
        "duck_out", TensorProto.FLOAT16, [1, 7, 1, 1])
    duck2_out_info = helper.make_tensor_value_info(
        "duck2_out", TensorProto.FLOAT16, [1, 7, 1, 1])
    wave_phase_out_info = helper.make_tensor_value_info(
        "wave_phase_out", TensorProto.FLOAT16, [1, N_WAVES, 1, 1])

    graph = helper.make_graph(
        nodes, "ocean_npu",
        [state_in, wave_phase_in, camera_in, duck_in, dt_in, duck2_in, splash_in, foam_params_in, render_params_in],
        [state_out, render_out, duck_out, duck2_out_info, wave_phase_out_info],
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    total = len(nodes)
    amp_min = min(w[4] for w in waves)
    amp_max = max(w[4] for w in waves)
    print(f"\nSaved: {output_path}")
    print(f"  Grid          : {N}x{N} FP16")
    print(f"  Ocean         : {len(waves)} Gerstner waves (Phillips spectrum, batch)")
    print(f"  Amp range     : {amp_min:.4f} .. {amp_max:.4f} (ratio {amp_max/amp_min:.1f}:1)")
    print(f"  Phase tracking: incremental omega*dt, wrapped [-pi,pi) (FP16-safe)")
    print(f"  Dispersion    : omega = sqrt(g * |k|)  (deep water)")
    print(f"  Ripple layer  : {RIPPLE_STEPS} substeps, c={RIPPLE_C}")
    print(f"  Caustics      : Laplacian convolution on NPU")
    print(f"  Refraction    : Snell's law + view-angle correction on NPU")
    print(f"  Duck physics  : 2 ducks, hull + wake + collision + drift on NPU")
    print(f"  Splash layer  : 256x256 ballistic heightfield, Weber emission + re-entry")
    print(f"  Total nodes   : {total}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "water_physics.onnx"
    create_model(out)
