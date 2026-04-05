#!/usr/bin/env python3
"""
Generate ball_physics.onnx — NPU-accelerated ball pit physics.

All 3000-ball collision physics run as tensor operations on the NPU:
  - Gravity integration
  - Wall/floor collision with velocity reflection
  - O(N²) pairwise sphere-sphere collision detection and response
  - Multiple substeps baked into the graph to prevent tunneling

Tensor layout: [1, C, N, 1] (NCHW with H=N balls, W=1)
  C=3 for positions/velocities (x, y, z channels)
  C=1 for scalar per-ball quantities (grab mask)

Pairwise collision uses broadcasting:
  pos_i [1,3,N,1] - pos_j [1,3,1,N] → diff [1,3,N,N]
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

N = 1024
BALL_RADIUS = 2.0
BOX_HALF = 30.0
GRAVITY = -49.05
RESTITUTION = 0.4
DAMPING = 0.995
SUBSTEPS = 4


def build_model():
    initializers = []
    all_nodes = []

    def add_fp16(name, val):
        initializers.append(numpy_helper.from_array(
            np.array(val, dtype=np.float16), name))

    def add_i64(name, val):
        initializers.append(numpy_helper.from_array(
            np.array(val, dtype=np.int64), name))

    # ---------- Constants ----------
    add_fp16('gravity',   [[[[0]], [[GRAVITY]], [[0]]]])                           # [1,3,1,1]
    add_fp16('box_min',   [[[[-BOX_HALF+BALL_RADIUS]], [[BALL_RADIUS]],
                            [[-BOX_HALF+BALL_RADIUS]]]])                           # [1,3,1,1]
    add_fp16('box_max',   [[[[BOX_HALF-BALL_RADIUS]], [[10000.0]],
                            [[BOX_HALF-BALL_RADIUS]]]])                            # [1,3,1,1]
    add_fp16('two_r',     [[[[2 * BALL_RADIUS]]]])
    add_fp16('rest',      [[[[RESTITUTION]]]])
    add_fp16('rest_p1',   [[[[-(1 + RESTITUTION)]]]])
    add_fp16('half',      [[[[0.5]]]])
    add_fp16('eps',       [[[[0.0001]]]])
    add_fp16('fzero',     [[[[0.0]]]])
    add_fp16('damping',   [[[[DAMPING]]]])
    add_fp16('sub_dt_sc', [[[[1.0 / SUBSTEPS]]]])

    add_i64('axes_1', [1])
    add_i64('axes_3', [3])

    # Eye-mask indices — [1,1,N,1] and [1,1,1,N] for Equal → [1,1,N,N] bool
    add_i64('ri', np.arange(N).reshape(1, 1, N, 1))
    add_i64('rj', np.arange(N).reshape(1, 1, 1, N))

    # ---------- One-time setup ----------
    all_nodes.append(helper.make_node('Equal', ['ri', 'rj'], ['eye']))
    all_nodes.append(helper.make_node('Not', ['eye'], ['not_eye']))
    all_nodes.append(helper.make_node('Mul', ['dt', 'sub_dt_sc'], ['sub_dt']))
    all_nodes.append(helper.make_node('Mul', ['gravity', 'sub_dt'], ['grav_dt']))

    # ---------- Substeps ----------
    pos_cur, vel_cur = 'pos', 'vel'

    for s in range(SUBSTEPS):
        p = f's{s}'

        # Gravity + damping + integrate
        all_nodes.append(helper.make_node('Add', [vel_cur, 'grav_dt'], [f'{p}_v1']))
        all_nodes.append(helper.make_node('Mul', [f'{p}_v1', 'damping'], [f'{p}_v1d']))
        all_nodes.append(helper.make_node('Mul', [f'{p}_v1d', 'sub_dt'], [f'{p}_vdt']))
        all_nodes.append(helper.make_node('Add', [pos_cur, f'{p}_vdt'], [f'{p}_p1']))

        # Wall collision — clamp pos, reflect vel
        all_nodes.append(helper.make_node('Less',    [f'{p}_p1', 'box_min'], [f'{p}_hlo']))
        all_nodes.append(helper.make_node('Greater', [f'{p}_p1', 'box_max'], [f'{p}_hhi']))
        all_nodes.append(helper.make_node('Clip',    [f'{p}_p1', 'box_min', 'box_max'], [f'{p}_p2']))
        all_nodes.append(helper.make_node('Abs',     [f'{p}_v1d'], [f'{p}_av']))
        all_nodes.append(helper.make_node('Mul',     [f'{p}_av', 'rest'], [f'{p}_vb']))
        all_nodes.append(helper.make_node('Neg',     [f'{p}_vb'], [f'{p}_vbn']))
        all_nodes.append(helper.make_node('Where',   [f'{p}_hlo', f'{p}_vb',  f'{p}_v1d'], [f'{p}_v2a']))
        all_nodes.append(helper.make_node('Where',   [f'{p}_hhi', f'{p}_vbn', f'{p}_v2a'], [f'{p}_v2']))

        # Pairwise diff: [1,3,N,1] - [1,3,1,N] → [1,3,N,N]
        all_nodes.append(helper.make_node('Transpose', [f'{p}_p2'], [f'{p}_pj'], perm=[0,1,3,2]))
        all_nodes.append(helper.make_node('Sub', [f'{p}_p2', f'{p}_pj'], [f'{p}_diff']))

        # dist² = sum(diff², axis=1) → [1,1,N,N]
        all_nodes.append(helper.make_node('Mul', [f'{p}_diff', f'{p}_diff'], [f'{p}_d2']))
        all_nodes.append(helper.make_node('ReduceSum', [f'{p}_d2', 'axes_1'], [f'{p}_ds2'], keepdims=1))

        # dist = sqrt(dist² + eps)
        all_nodes.append(helper.make_node('Add',  [f'{p}_ds2', 'eps'], [f'{p}_ds2e']))
        all_nodes.append(helper.make_node('Sqrt', [f'{p}_ds2e'], [f'{p}_dst']))

        # overlap = 2r - dist; colliding mask
        all_nodes.append(helper.make_node('Sub',     ['two_r', f'{p}_dst'], [f'{p}_ov']))
        all_nodes.append(helper.make_node('Greater', [f'{p}_ov', 'fzero'],  [f'{p}_ovb']))
        all_nodes.append(helper.make_node('And',     [f'{p}_ovb', 'not_eye'], [f'{p}_cb']))
        all_nodes.append(helper.make_node('Cast',    [f'{p}_cb'], [f'{p}_c'], to=TensorProto.FLOAT16))

        # normal = diff / dist
        all_nodes.append(helper.make_node('Div', [f'{p}_diff', f'{p}_dst'], [f'{p}_n']))

        # Position correction: sum_j(overlap * 0.5 * coll * normal) → [1,3,N,1]
        all_nodes.append(helper.make_node('Mul', [f'{p}_ov', f'{p}_c'],    [f'{p}_oc']))
        all_nodes.append(helper.make_node('Mul', [f'{p}_oc', 'half'],      [f'{p}_och']))
        all_nodes.append(helper.make_node('Mul', [f'{p}_och', f'{p}_n'],   [f'{p}_corr']))
        all_nodes.append(helper.make_node('ReduceSum', [f'{p}_corr', 'axes_3'], [f'{p}_pd'], keepdims=1))
        all_nodes.append(helper.make_node('Mul', [f'{p}_pd', 'grab_mask'], [f'{p}_pdm']))
        all_nodes.append(helper.make_node('Add', [f'{p}_p2', f'{p}_pdm'],  [f'{p}_p3']))

        # Velocity impulse
        all_nodes.append(helper.make_node('Transpose', [f'{p}_v2'], [f'{p}_vj'], perm=[0,1,3,2]))
        all_nodes.append(helper.make_node('Sub', [f'{p}_v2', f'{p}_vj'], [f'{p}_rv']))

        # rel_vel_n = sum(relv * normal, axis=1) → [1,1,N,N]
        all_nodes.append(helper.make_node('Mul', [f'{p}_rv', f'{p}_n'], [f'{p}_rvnp']))
        all_nodes.append(helper.make_node('ReduceSum', [f'{p}_rvnp', 'axes_1'], [f'{p}_rvn'], keepdims=1))

        # approaching mask (rvn < 0)
        all_nodes.append(helper.make_node('Less', [f'{p}_rvn', 'fzero'], [f'{p}_ab']))
        all_nodes.append(helper.make_node('Cast', [f'{p}_ab'], [f'{p}_a'], to=TensorProto.FLOAT16))

        # impulse = -(1+e) * rvn * 0.5 * coll * approaching
        all_nodes.append(helper.make_node('Mul', ['rest_p1', f'{p}_rvn'], [f'{p}_i1']))
        all_nodes.append(helper.make_node('Mul', [f'{p}_i1', 'half'],     [f'{p}_i2']))
        all_nodes.append(helper.make_node('Mul', [f'{p}_i2', f'{p}_c'],   [f'{p}_i3']))
        all_nodes.append(helper.make_node('Mul', [f'{p}_i3', f'{p}_a'],   [f'{p}_im']))
        all_nodes.append(helper.make_node('Mul', [f'{p}_im', f'{p}_n'],   [f'{p}_iv']))
        all_nodes.append(helper.make_node('ReduceSum', [f'{p}_iv', 'axes_3'], [f'{p}_vd'], keepdims=1))
        all_nodes.append(helper.make_node('Mul', [f'{p}_vd', 'grab_mask'], [f'{p}_vdm']))
        all_nodes.append(helper.make_node('Add', [f'{p}_v2', f'{p}_vdm'],  [f'{p}_v3']))

        pos_cur, vel_cur = f'{p}_p3', f'{p}_v3'

    # ---------- Outputs ----------
    all_nodes.append(helper.make_node('Identity', [pos_cur], ['out_pos']))
    all_nodes.append(helper.make_node('Identity', [vel_cur], ['out_vel']))

    # ---------- Graph ----------
    inputs = [
        helper.make_tensor_value_info('pos',       TensorProto.FLOAT16, [1, 3, N, 1]),
        helper.make_tensor_value_info('vel',       TensorProto.FLOAT16, [1, 3, N, 1]),
        helper.make_tensor_value_info('dt',        TensorProto.FLOAT16, [1, 1, 1, 1]),
        helper.make_tensor_value_info('grab_mask', TensorProto.FLOAT16, [1, 1, N, 1]),
    ]
    outputs = [
        helper.make_tensor_value_info('out_pos', TensorProto.FLOAT16, [1, 3, N, 1]),
        helper.make_tensor_value_info('out_vel', TensorProto.FLOAT16, [1, 3, N, 1]),
    ]

    graph = helper.make_graph(all_nodes, 'ball_physics', inputs, outputs,
                              initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, 'ball_physics.onnx')

    print(f'Saved ball_physics.onnx')
    print(f'  Balls:       {N}')
    print(f'  Substeps:    {SUBSTEPS} (baked into graph)')
    print(f'  Pairwise:    {N}x{N} = {N*N:,} pairs/substep')
    print(f'  Peak tensor: [1, 3, {N}, {N}] = {3*N*N*2/1024/1024:.1f} MB (FP16)')
    print(f'  Physics:     gravity={GRAVITY}, restitution={RESTITUTION}, damping={DAMPING}')
    print(f'  Box:         {BOX_HALF*2}x{BOX_HALF*2}, radius={BALL_RADIUS}')


if __name__ == '__main__':
    build_model()
