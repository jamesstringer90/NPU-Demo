#!/usr/bin/env python3
"""
Generate ONNX model for NPU ocean simulation — Gerstner wave synthesis.

Uses the same approach as AAA game engines:
  - 32 Gerstner waves with proper Phillips spectrum amplitudes
  - Deep-water dispersion: omega(k) = sqrt(g * |k|)
  - Deterministic ocean surface computed from time (no simulation instability)
  - Interactive ripple layer for ball/splash interaction (simple wave equation)

All operations are FP16 tensor ops on Intel NPU via OpenVINO.
  Inputs:  state [1,2,256,256] + time [1,1,1,1] + camera [1,3,1,1]
  Outputs: new_state [1,2,256,256] + render [1,6,256,256]
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


def generate_waves():
    """Generate 32 Gerstner waves with proper Phillips spectrum."""
    rng = np.random.RandomState(42)
    L = WIND_SPEED ** 2 / G_ACCEL   # ~22.9 — largest wave from wind
    small_l = L / 1000.0             # suppression cutoff for tiny waves

    waves = []

    # Logarithmic k sampling from short to long waves
    k_min = 2.0 * math.pi / 300.0   # longest wave: 300 cells (long rolling swells)
    k_max = 2.0 * math.pi / 40.0    # shortest wave: 40 cells (was 20 — too spiky)
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

    # --- Duck physics (NPU integration, CPU supplies slope at duck position) ---
    # All rates are per-second; multiplied by dt input for frame-rate independence
    duck_force = np.array([[[[-80.0]]]], dtype=np.float16)  # force per second
    # Drag as exponential decay: drag_per_frame = exp(-decay_rate * dt)
    # At 30fps: 0.92 = exp(-rate * 0.033) → rate = -ln(0.92)/0.033 ≈ 2.53
    duck_decay_rate = np.array([[[[math.log(0.92) / 0.033]]]], dtype=np.float16)  # negative → exp(neg*dt) < 1
    duck_pos_min = np.array([-83.0], dtype=np.float16)
    duck_pos_max = np.array([83.0], dtype=np.float16)

    # --- Duck hull displacement + wake (NPU, full 256x256 tensor ops) ---
    DUCK_R = 45.0 * 0.4   # waterline footprint radius (DUCK_SCALE * 0.4)
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

    # Duck slope sampling: wide Gaussian (σ²=64) — mean(w)≈0.006, FP16-safe
    sample_neg_inv_s2 = np.array([[[[-1.0 / 128.0]]]], dtype=np.float16)  # -1/(2×64)
    # Duck wall bounce: factor = 1.0 - 1.5 × hit → -0.5 on wall hit
    bounce_1p5 = np.array([[[[1.5]]]], dtype=np.float16)

    # Ball splash constants
    ball_cutoff_15 = np.array([[[[1.5]]]], dtype=np.float16)   # matches CPU `dist2 > R*R*1.5f`
    ball_pt4 = np.array([[[[0.4]]]], dtype=np.float16)          # matches CPU `speed / (radius * 0.4)`

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
        numpy_helper.from_array(np.array([1, 1, 1, 1], dtype=np.int64), "split4"),
        numpy_helper.from_array(np.array([1, 1, 1, 1, 1, 1], dtype=np.int64), "split6"),
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
        numpy_helper.from_array(sample_neg_inv_s2, "sample_neg_inv_s2"),
        numpy_helper.from_array(bounce_1p5, "bounce_1p5"),
        numpy_helper.from_array(ball_cutoff_15, "ball_cutoff_15"),
        numpy_helper.from_array(ball_pt4, "ball_pt4"),
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
        "state", TensorProto.FLOAT16, [1, 2, N, N])
    wave_phase_in = helper.make_tensor_value_info(
        "wave_phase", TensorProto.FLOAT16, [1, N_WAVES, 1, 1])
    camera_in = helper.make_tensor_value_info(
        "camera", TensorProto.FLOAT16, [1, 3, 1, 1])
    duck_in = helper.make_tensor_value_info(
        "duck_in", TensorProto.FLOAT16, [1, 4, 1, 1])
    dt_in = helper.make_tensor_value_info(
        "dt", TensorProto.FLOAT16, [1, 1, 1, 1])
    ball_in = helper.make_tensor_value_info(
        "ball_in", TensorProto.FLOAT16, [1, 6, 1, 1])

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
        "Split", ["duck_in", "split4"],
        ["duck_x", "duck_z", "duck_vx", "duck_vz"],
        axis=1))

    # ================================================================
    # INTERACTIVE RIPPLE LAYER — simple damped wave equation
    # ================================================================
    nodes.append(helper.make_node(
        "Split", ["state", "split2"], ["rh_in", "rhp_in"], axis=1))

    # ================================================================
    # NPU HULL DISPLACEMENT — Gaussian depression under duck, hard-cutoff tail
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
    # NPU WAKE INJECTION — directional ring impulse from duck motion
    # Uses same hull_gauss as spatial mask to confine wake to duck area
    # ================================================================
    nodes.append(helper.make_node("Add", ["hull_dist2", "wake_epsilon2"], ["wake_dist2s"]))
    nodes.append(helper.make_node("Sqrt", ["wake_dist2s"], ["wake_dist"]))
    nodes.append(helper.make_node("Div", ["wake_dist", "wake_duckR"], ["wake_tn"]))
    # Ring profile
    nodes.append(helper.make_node("Sub", ["wake_tn", "wake_ring_peak"], ["wake_tn_off"]))
    nodes.append(helper.make_node("Mul", ["wake_tn_off", "wake_tn_off"], ["wake_tn_off2"]))
    nodes.append(helper.make_node("Mul", ["wake_tn_off2", "wake_neg_inv_rw"], ["wake_ring_arg"]))
    nodes.append(helper.make_node("Exp", ["wake_ring_arg"], ["wake_ring_raw"]))
    # Mask wake by hull Gaussian cutoff — confines ring to duck neighborhood
    nodes.append(helper.make_node("Mul", ["wake_ring_raw", "hull_gauss"], ["wake_ring"]))
    # Cell direction
    nodes.append(helper.make_node("Div", ["hull_dx", "wake_dist"], ["wake_nx"]))
    nodes.append(helper.make_node("Div", ["hull_dz", "wake_dist"], ["wake_nz"]))
    # Duck speed + direction
    nodes.append(helper.make_node("Mul", ["duck_vx", "duck_vx"], ["wake_vx2"]))
    nodes.append(helper.make_node("Mul", ["duck_vz", "duck_vz"], ["wake_vz2"]))
    nodes.append(helper.make_node("Add", ["wake_vx2", "wake_vz2"], ["wake_spd2"]))
    nodes.append(helper.make_node("Add", ["wake_spd2", "wake_epsilon2"], ["wake_spd2s"]))
    nodes.append(helper.make_node("Sqrt", ["wake_spd2s"], ["wake_speed"]))
    nodes.append(helper.make_node("Div", ["duck_vx", "wake_speed"], ["wake_mdx"]))
    nodes.append(helper.make_node("Div", ["duck_vz", "wake_speed"], ["wake_mdz"]))
    # facing = dot(cell_dir, velocity_dir)
    nodes.append(helper.make_node("Mul", ["wake_nx", "wake_mdx"], ["wake_fx"]))
    nodes.append(helper.make_node("Mul", ["wake_nz", "wake_mdz"], ["wake_fz"]))
    nodes.append(helper.make_node("Add", ["wake_fx", "wake_fz"], ["wake_facing"]))
    # impulse = facing * ring * speed * strength
    nodes.append(helper.make_node("Mul", ["wake_facing", "wake_ring"], ["wake_fr"]))
    nodes.append(helper.make_node("Mul", ["wake_fr", "wake_speed"], ["wake_frs"]))
    nodes.append(helper.make_node("Mul", ["wake_frs", "wake_strength"], ["wake_impulse_ps"]))
    nodes.append(helper.make_node("Mul", ["wake_impulse_ps", "dt"], ["wake_impulse"]))
    nodes.append(helper.make_node("Add", ["rh_hulled", "wake_impulse"], ["rh_duck_ready"]))

    # ================================================================
    # NPU BALL SPLASH — exact port of CPU applyBall()
    # Gaussian ring at ~0.8*R, facing dot, hard cutoff at 1.5*R² and < 1.0
    # CPU packs ball_in = (x, z, vx, vz, radius, push) — push=0 when idle
    # ================================================================
    nodes.append(helper.make_node(
        "Split", ["ball_in", "split6"],
        ["ball_x", "ball_z", "ball_vx", "ball_vz", "ball_radius", "ball_push"],
        axis=1))
    # Distance from ball to each grid cell
    nodes.append(helper.make_node("Sub", ["x_grid", "ball_x"], ["ball_dx"]))
    nodes.append(helper.make_node("Sub", ["z_grid", "ball_z"], ["ball_dz"]))
    nodes.append(helper.make_node("Mul", ["ball_dx", "ball_dx"], ["ball_dx2"]))
    nodes.append(helper.make_node("Mul", ["ball_dz", "ball_dz"], ["ball_dz2"]))
    nodes.append(helper.make_node("Add", ["ball_dx2", "ball_dz2"], ["ball_dist2"]))
    nodes.append(helper.make_node("Add", ["ball_dist2", "wake_epsilon2"], ["ball_dist2s"]))
    nodes.append(helper.make_node("Sqrt", ["ball_dist2s"], ["ball_dist"]))

    # Cutoff mask: 1.0 where 1.0 <= dist2 <= R*R*1.5, 0.0 elsewhere
    # This matches the CPU's `if (dist2 > R*R*1.5f || dist2 < 1.0f) continue;`
    # outer_limit = R * R * 1.5
    nodes.append(helper.make_node("Mul", ["ball_radius", "ball_radius"], ["ball_R2"]))
    nodes.append(helper.make_node("Mul", ["ball_R2", "ball_cutoff_15"], ["ball_outer_lim"]))
    # outer_ok = clip((outer_lim - dist2) * big, 0, 1) → 1 inside, 0 outside
    nodes.append(helper.make_node("Sub", ["ball_outer_lim", "ball_dist2"], ["ball_outer_diff"]))
    nodes.append(helper.make_node("Mul", ["ball_outer_diff", "wrap_scale"], ["ball_outer_sc"]))
    nodes.append(helper.make_node("Clip", ["ball_outer_sc", "hull_clip_zero", "hull_clip_one"], ["ball_outer_ok"]))
    # inner_ok = clip((dist2 - 1.0) * big, 0, 1) → 0 at center, 1 outside
    nodes.append(helper.make_node("Sub", ["ball_dist2", "hull_clip_one"], ["ball_inner_diff"]))
    nodes.append(helper.make_node("Mul", ["ball_inner_diff", "wrap_scale"], ["ball_inner_sc"]))
    nodes.append(helper.make_node("Clip", ["ball_inner_sc", "hull_clip_zero", "hull_clip_one"], ["ball_inner_ok"]))
    nodes.append(helper.make_node("Mul", ["ball_outer_ok", "ball_inner_ok"], ["ball_cutoff_mask"]))

    # Normalized distance: tn = dist / radius
    nodes.append(helper.make_node("Div", ["ball_dist", "ball_radius"], ["ball_tn"]))
    # Gaussian ring profile peaked at ~0.8*radius: exp(-((tn - 0.8)² / 0.12))
    nodes.append(helper.make_node("Sub", ["ball_tn", "wake_ring_peak"], ["ball_tn_off"]))
    nodes.append(helper.make_node("Mul", ["ball_tn_off", "ball_tn_off"], ["ball_tn_off2"]))
    nodes.append(helper.make_node("Mul", ["ball_tn_off2", "wake_neg_inv_rw"], ["ball_ring_arg"]))
    nodes.append(helper.make_node("Exp", ["ball_ring_arg"], ["ball_ring_raw"]))
    # Apply cutoff mask
    nodes.append(helper.make_node("Mul", ["ball_ring_raw", "ball_cutoff_mask"], ["ball_ring"]))

    # Cell direction (normalized)
    nodes.append(helper.make_node("Div", ["ball_dx", "ball_dist"], ["ball_nx"]))
    nodes.append(helper.make_node("Div", ["ball_dz", "ball_dist"], ["ball_nz"]))
    # Ball speed + normalized direction
    nodes.append(helper.make_node("Mul", ["ball_vx", "ball_vx"], ["ball_svx2"]))
    nodes.append(helper.make_node("Mul", ["ball_vz", "ball_vz"], ["ball_svz2"]))
    nodes.append(helper.make_node("Add", ["ball_svx2", "ball_svz2"], ["ball_spd2"]))
    nodes.append(helper.make_node("Add", ["ball_spd2", "wake_epsilon2"], ["ball_spd2s"]))
    nodes.append(helper.make_node("Sqrt", ["ball_spd2s"], ["ball_speed"]))
    nodes.append(helper.make_node("Div", ["ball_vx", "ball_speed"], ["ball_mdx"]))
    nodes.append(helper.make_node("Div", ["ball_vz", "ball_speed"], ["ball_mdz"]))
    # facing = dot(cell_dir, velocity_dir)
    nodes.append(helper.make_node("Mul", ["ball_nx", "ball_mdx"], ["ball_fx"]))
    nodes.append(helper.make_node("Mul", ["ball_nz", "ball_mdz"], ["ball_fz"]))
    nodes.append(helper.make_node("Add", ["ball_fx", "ball_fz"], ["ball_facing"]))
    # invSteps: matches CPU numSteps = max(1, floor(speed / (radius * 0.4)) + 1)
    nodes.append(helper.make_node("Mul", ["ball_radius", "ball_pt4"], ["ball_step_denom"]))
    nodes.append(helper.make_node("Div", ["ball_speed", "ball_step_denom"], ["ball_step_raw"]))
    nodes.append(helper.make_node("Floor", ["ball_step_raw"], ["ball_step_floor"]))
    nodes.append(helper.make_node("Add", ["ball_step_floor", "hull_clip_one"], ["ball_numsteps"]))
    nodes.append(helper.make_node("Reciprocal", ["ball_numsteps"], ["ball_invsteps"]))
    # impulse = facing * ring * push * speed * invSteps (push=0 when idle → zero)
    nodes.append(helper.make_node("Mul", ["ball_facing", "ball_ring"], ["ball_fr"]))
    nodes.append(helper.make_node("Mul", ["ball_fr", "ball_speed"], ["ball_frs"]))
    nodes.append(helper.make_node("Mul", ["ball_frs", "ball_push"], ["ball_frsp"]))
    nodes.append(helper.make_node("Mul", ["ball_frsp", "ball_invsteps"], ["ball_impulse"]))
    nodes.append(helper.make_node("Add", ["rh_duck_ready", "ball_impulse"], ["rh_ready"]))

    rh, rhp = "rh_ready", "rhp_in"
    for step in range(RIPPLE_STEPS):
        s = f"_r{step}"
        # Combined kernel: 2*h + c^2*dt^2 * Laplacian(h)
        nodes.append(helper.make_node(
            "Conv", [rh, "ripple_kernel"], [f"rprop{s}"], pads=P))
        # Verlet: h_next = conv_result - h_prev
        nodes.append(helper.make_node(
            "Sub", [f"rprop{s}", rhp], [f"rraw{s}"]))
        # Damping
        nodes.append(helper.make_node(
            "Mul", [f"rraw{s}", "damping"], [f"rdamp{s}"]))
        # Sponge boundary
        rh_out = f"rh_{step + 1}"
        nodes.append(helper.make_node(
            "Mul", [f"rdamp{s}", "sponge_mask"], [rh_out]))
        # Shift history
        rhp_out = f"rhp_{step + 1}"
        nodes.append(helper.make_node("Identity", [rh], [rhp_out]))
        rh, rhp = rh_out, rhp_out

    # ================================================================
    # COMBINE + RENDER
    # ================================================================
    # Total height = ocean waves + interactive ripples
    nodes.append(helper.make_node("Add", [ocean_h, rh], ["total_h"]))

    # Render: (h*scale, dh/dx*scale, dh/dy*scale, caustics)
    nodes.append(helper.make_node("Mul",  ["total_h", "h_scale"], ["h_sc"]))
    nodes.append(helper.make_node("Conv", ["total_h", "ddx_w"],   ["r_dhdx"], pads=P))
    nodes.append(helper.make_node("Conv", ["total_h", "ddy_w"],   ["r_dhdy"], pads=P))
    nodes.append(helper.make_node("Mul",  ["r_dhdx", "h_scale"],  ["dhdx_sc"]))
    nodes.append(helper.make_node("Mul",  ["r_dhdy", "h_scale"],  ["dhdy_sc"]))

    # NPU caustics: blur → blur → Laplacian → Abs → scale
    # Two blur passes smooth FP16 noise before the second-derivative amplifies it
    nodes.append(helper.make_node("Conv", ["total_h", "blur_w"],  ["blur1"],  pads=P))
    nodes.append(helper.make_node("Conv", ["blur1",   "blur_w"],  ["blur2"],  pads=P))
    nodes.append(helper.make_node("Conv", ["blur2",   "lap_w"],   ["lap_h"],  pads=P))
    nodes.append(helper.make_node("Abs",  ["lap_h"],              ["abs_lap"]))
    nodes.append(helper.make_node("Mul",  ["abs_lap", "caustic_scale"], ["caustic"]))

    # ================================================================
    # VIEW-DEPENDENT REFRACTION — Snell's law with oblique angle correction
    # NPU computes per-cell: refract = slope * base_scale * sec(theta_view)
    # where theta_view = angle from vertical at each grid cell
    # ================================================================
    # Split camera [1,3,1,1] → eye_x, eye_y, eye_z [each 1,1,1,1]
    nodes.append(helper.make_node(
        "Split", ["camera", "split3"], ["eye_x", "eye_y", "eye_z"], axis=1))

    # Per-cell view direction from camera to grid cell
    nodes.append(helper.make_node("Sub", ["x_grid", "eye_x"], ["vdx"]))
    nodes.append(helper.make_node("Sub", ["z_grid", "eye_z"], ["vdz"]))
    nodes.append(helper.make_node("Mul", ["vdx", "vdx"], ["vdx2"]))
    nodes.append(helper.make_node("Mul", ["vdz", "vdz"], ["vdz2"]))
    nodes.append(helper.make_node("Mul", ["eye_y", "eye_y"], ["vdy2"]))
    nodes.append(helper.make_node("Add", ["vdx2", "vdz2"], ["vdxz2"]))
    nodes.append(helper.make_node("Add", ["vdxz2", "vdy2"], ["vdist2"]))
    nodes.append(helper.make_node("Sqrt", ["vdist2"], ["vdist"]))

    # sec(theta) = dist / eye_y = 1/cos(view angle from vertical)
    nodes.append(helper.make_node("Div", ["vdist", "eye_y"], ["sec_raw"]))
    # Clamp to [1, 3] — prevents blowup near horizon
    nodes.append(helper.make_node("Clip", ["sec_raw", "sec_clip_min", "sec_clip_max"], ["sec_theta"]))

    # Base Snell's law offset × view angle correction
    nodes.append(helper.make_node("Mul", ["r_dhdx", "refract_scale"], ["refract_base_x"]))
    nodes.append(helper.make_node("Mul", ["r_dhdy", "refract_scale"], ["refract_base_z"]))
    nodes.append(helper.make_node("Mul", ["refract_base_x", "sec_theta"], ["refract_x"]))
    nodes.append(helper.make_node("Mul", ["refract_base_z", "sec_theta"], ["refract_z"]))

    # ================================================================
    # DUCK SLOPE SAMPLING — wide Gaussian weighted average (σ²=64, FP16-safe)
    # mean(w)≈0.006 — well above FP16 min normal (6e-5)
    # Reuses hull_dist2 from hull displacement section
    # ================================================================
    nodes.append(helper.make_node("Mul", ["hull_dist2", "sample_neg_inv_s2"], ["sample_exp_arg"]))
    nodes.append(helper.make_node("Exp", ["sample_exp_arg"], ["sample_w"]))
    nodes.append(helper.make_node("Mul", ["sample_w", "dhdx_sc"], ["sample_wx"]))
    nodes.append(helper.make_node("Mul", ["sample_w", "dhdy_sc"], ["sample_wz"]))
    nodes.append(helper.make_node("GlobalAveragePool", ["sample_wx"], ["slope_x_mean"]))
    nodes.append(helper.make_node("GlobalAveragePool", ["sample_wz"], ["slope_z_mean"]))
    nodes.append(helper.make_node("GlobalAveragePool", ["sample_w"], ["w_mean"]))
    nodes.append(helper.make_node("Add", ["w_mean", "wake_epsilon2"], ["w_mean_safe"]))
    nodes.append(helper.make_node("Div", ["slope_x_mean", "w_mean_safe"], ["dk_slope_x"]))
    nodes.append(helper.make_node("Div", ["slope_z_mean", "w_mean_safe"], ["dk_slope_z"]))

    # ================================================================
    # DUCK PHYSICS — free-floating rubber duck on NPU (dt-scaled)
    # Slope sampled on NPU via Gaussian-weighted GlobalAveragePool
    # ================================================================
    # force_impulse = slope * force * dt
    nodes.append(helper.make_node("Mul", ["dk_slope_x", "duck_force"], ["dk_fx_ps"]))
    nodes.append(helper.make_node("Mul", ["dk_slope_z", "duck_force"], ["dk_fz_ps"]))
    nodes.append(helper.make_node("Mul", ["dk_fx_ps", "dt"], ["dk_fx"]))
    nodes.append(helper.make_node("Mul", ["dk_fz_ps", "dt"], ["dk_fz"]))
    nodes.append(helper.make_node("Add", ["duck_vx", "dk_fx"], ["dk_vx_push"]))
    nodes.append(helper.make_node("Add", ["duck_vz", "dk_fz"], ["dk_vz_push"]))
    # drag = exp(-decay_rate * dt)  — frame-rate independent exponential decay
    nodes.append(helper.make_node("Mul", ["duck_decay_rate", "dt"], ["dk_decay_dt"]))
    nodes.append(helper.make_node("Exp", ["dk_decay_dt"], ["dk_drag"]))
    nodes.append(helper.make_node("Mul", ["dk_vx_push", "dk_drag"], ["dk_vx_new"]))
    nodes.append(helper.make_node("Mul", ["dk_vz_push", "dk_drag"], ["dk_vz_new"]))

    # pos_new = pos + v_new * dt
    nodes.append(helper.make_node("Mul", ["dk_vx_new", "dt"], ["dk_dx_step"]))
    nodes.append(helper.make_node("Mul", ["dk_vz_new", "dt"], ["dk_dz_step"]))
    nodes.append(helper.make_node("Add", ["duck_x", "dk_dx_step"], ["dk_x_raw"]))
    nodes.append(helper.make_node("Add", ["duck_z", "dk_dz_step"], ["dk_z_raw"]))

    # Clamp position to grid bounds
    nodes.append(helper.make_node(
        "Clip", ["dk_x_raw", "duck_pos_min", "duck_pos_max"], ["dk_x_clamp"]))
    nodes.append(helper.make_node(
        "Clip", ["dk_z_raw", "duck_pos_min", "duck_pos_max"], ["dk_z_clamp"]))

    # Wall bounce: if position was clamped, flip velocity × -0.5
    # bounce_factor = 1.0 - 1.5 × was_clamped → 1.0 (free) or -0.5 (wall hit)
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

    # Duck output: [1, 4, 1, 1] = (x, z, vx, vz) with wall bounce applied
    nodes.append(helper.make_node("Concat",
        ["dk_x_clamp", "dk_z_clamp", "dk_vx_bounced", "dk_vz_bounced"],
        ["duck_out"], axis=1))

    nodes.append(helper.make_node("Concat",
        ["h_sc", "dhdx_sc", "dhdy_sc", "caustic", "refract_x", "refract_z"],
        ["render_out"], axis=1))

    # Ripple state output
    nodes.append(helper.make_node("Concat", [rh, rhp], ["new_state"], axis=1))

    # ----------------------------------------------------------------
    # Build model
    # ----------------------------------------------------------------
    state_out = helper.make_tensor_value_info(
        "new_state", TensorProto.FLOAT16, [1, 2, N, N])
    render_out = helper.make_tensor_value_info(
        "render_out", TensorProto.FLOAT16, [1, 6, N, N])
    duck_out = helper.make_tensor_value_info(
        "duck_out", TensorProto.FLOAT16, [1, 4, 1, 1])
    wave_phase_out_info = helper.make_tensor_value_info(
        "wave_phase_out", TensorProto.FLOAT16, [1, N_WAVES, 1, 1])

    graph = helper.make_graph(
        nodes, "ocean_npu",
        [state_in, wave_phase_in, camera_in, duck_in, dt_in, ball_in],
        [state_out, render_out, duck_out, wave_phase_out_info],
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
    print(f"  Duck physics  : hull displacement + wake + Newtonian drift on NPU")
    print(f"  Ball splash   : directional ring impulse on NPU")
    print(f"  Total nodes   : {total}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "water_physics.onnx"
    create_model(out)
