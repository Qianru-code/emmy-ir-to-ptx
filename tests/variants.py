"""Independent GPU checks for unseen sizes/ops and reordered kernel arguments."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compile import Compiler, kind
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.elementwise import ElementwiseImpl
from submit import post, RUNNER

ROOT = Path(__file__).resolve().parents[1]


def execute(graph, node, launch, inputs, expected):
    args = []
    for name in launch['arg_order']:
        args.append({'name': name, 'dtype': 'f32', 'shape': list(expected.shape if name == node.outputs[0].name else inputs[name].shape),
                     **({'output': True} if name == node.outputs[0].name else {'data': inputs[name].ravel().tolist()})})
    ptx = Compiler(graph, node, launch).compile()
    result = post(RUNNER, '/run', {'ptx': ptx, 'entry': node.op.name,
        'grid': launch['grid'], 'block': launch['block'], 'smem_bytes': launch['smem_bytes'],
        'args': args, 'warmup': 0, 'iters': 1, 'max_return': max(4096, expected.size)})
    assert result.get('ok'), result
    got = np.array(result['outputs'][node.outputs[0].name]['values'], dtype=np.float32).reshape(expected.shape)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
    return float(np.max(np.abs(got - expected)))


def pointwise():
    p = ROOT / 'problems/p01_vector_add'
    raw = json.loads(next(p.glob('*.ir.json')).read_text())
    for node in raw['nodes'].values():
        for t in node['outputs']:
            t['shape'] = [74]
    g = Graph.from_dict(raw)
    node = next(n for n in g.nodes.values() if kind(n.op) == 'KernelOp')
    tile = node.op.body[0]
    body = tuple(replace(s, op=ElementwiseImpl('subtract')) if kind(s)=='Assign' else s for s in tile.body)
    node.op = replace(node.op, body=(replace(tile, axes=(replace(tile.axes[0], extent=Dim(37)),), body=body),))
    launch = dict(json.loads((p/'meta.json').read_text())['kernels'][0])
    launch.update(arg_order=['add','x1','x0'],grid=[1,1,1],block=[64,1,1])
    rng = np.random.default_rng(123)
    x,y = (rng.standard_normal(74).astype('float32') for _ in range(2))
    return execute(g,node,launch,{'x0':x,'x1':y},x-y)


def row_reduction():
    p = ROOT / 'problems/r04_row_sum'
    raw = json.loads(next(p.glob('*.ir.json')).read_text())
    raw['nodes']['x']['outputs'][0]['shape'] = [7,1003]
    raw['nodes']['sum_1']['outputs'][0]['shape'] = [7]
    g = Graph.from_dict(raw);node=g.nodes['sum_1'];tile=node.op.body[0]
    body=tuple(replace(s,axis=replace(s.axis,extent=Dim(1003))) if kind(s)=='StridedLoop' else s for s in tile.body)
    node.op=replace(node.op,body=(replace(tile,axes=(replace(tile.axes[0],extent=Dim(7)),tile.axes[1]),body=body),))
    launch=dict(json.loads((p/'meta.json').read_text())['kernels'][0])
    launch.update(arg_order=['sum_1','x'],grid=[7,1,1])
    x=np.random.default_rng(321).standard_normal((7,1003)).astype('float32')
    return execute(g,node,launch,{'x':x},x.sum(axis=-1))


if __name__=='__main__':
    for name,fn in [('37-cell subtract, excess threads, output-first args',pointwise),('7x1003 row sums, strided tail, output-first args',row_reduction)]:
        print(f'{name}: PASS (max error {fn():.3e})',flush=True)
