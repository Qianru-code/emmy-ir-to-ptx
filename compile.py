#!/usr/bin/env python3
"""Translate typed Emmy Kernel IR directly to PTX. No CUDA/compiler subprocesses."""
from __future__ import annotations
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path
from emmy.compiler.graph import Graph
from emmy.compiler.dim import as_static


def kind(x):
    return type(x).__name__


def f32(x):
    return '0f' + struct.pack('>f', float(x)).hex().upper()


def walk(body):
    for s in body:
        yield s
        for field in ('body', 'else_body'):
            child = getattr(s, field, ())
            if child:
                yield from walk(child)


class Compiler:
    def __init__(self, graph, node, launch):
        self.node, self.launch = node, launch
        self.code, self.shared = [], []
        self.count = defaultdict(int)
        self.values, self.ptrs, self.shapes, self.smem = {}, {}, {}, set()
        self.label_count = 0
        self.shared_offsets = {}
        total = 0
        max_align = 4
        for stmt in walk(node.op.body):
            if kind(stmt) == "Smem":
                align = max(4, stmt.align)
                max_align = max(max_align, align)
                total = (total + align - 1) // align * align
                self.shared_offsets[stmt.name] = total
                total += math.prod(stmt.extents) * 4
        self.dynamic_shared = total > 48 * 1024
        if self.dynamic_shared:
            if launch["smem_bytes"] < total:
                raise ValueError(f"launch reserves {launch['smem_bytes']} shared bytes; need {total}")
            self.shared.append(f"    .extern .shared .align {max_align} .b8 dynamic_smem[];")
        for n in graph.nodes.values():
            for t in n.outputs:
                self.shapes[t.name] = tuple(as_static(d) for d in t.shape)
                if t.dtype.name != 'f32':
                    raise NotImplementedError(f'buffer dtype {t.dtype.name}; currently f32 only')

    def emit(self, line):
        self.code.append('    ' + line)

    def reg(self, ty='f'):
        self.count[ty] += 1
        return f'%{ty}{self.count[ty]}'

    def var(self, name, ty='f'):
        if name not in self.values:
            self.values[name] = self.reg(ty)
        return self.values[name]

    def label(self):
        self.label_count += 1
        return f'L{self.label_count}'

    def integer(self, op, a, b):
        operands = []
        for v in (a, b):
            if v.startswith(('%tid.', '%ntid.', '%ctaid.')):
                tmp = self.reg('r')
                self.emit(f'mov.u32 {tmp}, {v};')
                v = tmp
            operands.append(v)
        a, b = operands
        r = self.reg('r')
        inst = {'+':'add.s32','-':'sub.s32','*':'mul.lo.s32','/':'div.s32','//':'div.s32','%':'rem.s32','&':'and.b32','|':'or.b32','>>':'shr.s32','<<':'shl.b32'}[op]
        self.emit(f'{inst} {r}, {a}, {b};')
        return r

    def expr(self, x):
        k = kind(x)
        if isinstance(x, (int, float)):
            return str(x)
        if k == 'Var':
            return self.values[x.name]
        if k == 'Literal':
            return f32(x.value) if x.dtype in ('float', 'f32') else str(int(x.value))
        if k == 'BinaryExpr':
            a,b = self.expr(x.left), self.expr(x.right)
            if x.op in ('==','!=','<','<=','>','>='):
                p = self.reg('p')
                cmp = {'==':'eq','!=':'ne','<':'lt','<=':'le','>':'gt','>=':'ge'}[x.op]
                self.emit(f'setp.{cmp}.s32 {p}, {a}, {b};')
                return p
            if x.op in ('and','or','&&','||'):
                p = self.reg('p')
                op = 'and' if x.op in ('and','&&') else 'or'
                self.emit(f'{op}.pred {p}, {a}, {b};')
                return p
            return self.integer(x.op, a, b)
        if k == 'TernaryExpr':
            p,a,b = self.expr(x.cond), self.expr(x.if_true), self.expr(x.if_false)
            r = self.reg('r')
            self.emit(f'selp.s32 {r}, {a}, {b}, {p};')
            return r
        raise NotImplementedError(f'expression {x!r}')

    def address(self, name, index):
        shape = self.shapes[name]
        offset = '0'
        for i, x in enumerate(index):
            term = self.expr(x)
            stride = math.prod(shape[i+1:])
            if stride != 1:
                term = self.integer('*', term, str(stride))
            offset = self.integer('+', offset, term)
        if name in self.smem:
            byte = self.integer('*', offset, '4')
            ptr = self.reg('r')
            self.emit(f'add.u32 {ptr}, {self.ptrs[name]}, {byte};')
        else:
            byte,ptr = self.reg('d'),self.reg('d')
            self.emit(f'mul.wide.s32 {byte}, {offset}, 4;')
            self.emit(f'add.u64 {ptr}, {self.ptrs[name]}, {byte};')
        return ptr

    def load(self, name, index, targets):
        ptr = self.address(name, index)
        space = 'shared' if name in self.smem else 'global'
        # Scalarized vectors preserve consecutive element semantics without alignment assumptions.
        for i, target in enumerate(targets):
            self.emit(f'ld.{space}.f32 {self.var(target)}, [{ptr}+{i*4}];')

    def store(self, name, index, values, atomic=False):
        ptr = self.address(name, index)
        space = 'shared' if name in self.smem else 'global'
        for i, value in enumerate(values):
            inst = f'red.{space}.add.f32' if atomic else f'st.{space}.f32'
            self.emit(f'{inst} [{ptr}+{i*4}], {self.values[value]};')

    def arithmetic(self, dest, op, args):
        binary = {'add':'add.f32','sum':'add.f32','subtract':'sub.f32','multiply':'mul.f32','prod':'mul.f32','divide':'div.rn.f32','maximum':'max.f32','amax':'max.f32','minimum':'min.f32','power':None}
        unary = {'copy':'mov.f32','negative':'neg.f32','abs':'abs.f32','sqrt':'sqrt.rn.f32','reciprocal':'rcp.rn.f32','rsqrt':'rsqrt.approx.f32'}
        if op in binary and binary[op]:
            self.emit(f'{binary[op]} {dest}, {", ".join(args)};')
        elif op in unary:
            self.emit(f'{unary[op]} {dest}, {args[0]};')
        elif op == 'relu':
            self.emit(f'max.f32 {dest}, {args[0]}, {f32(0)};')
        elif op == 'exp':
            t = self.reg()
            self.emit(f'mul.f32 {t}, {args[0]}, {f32(math.log2(math.e))};')
            self.emit(f'ex2.approx.f32 {dest}, {t};')
        elif op == 'log':
            t = self.reg()
            self.emit(f'lg2.approx.f32 {t}, {args[0]};')
            self.emit(f'mul.f32 {dest}, {t}, {f32(math.log(2))};')
        else:
            raise NotImplementedError(f'elementwise op {op}')

    def seed_loop(self, s):
        if not s.seed:
            return
        seen = set()
        for a in walk(s.body):
            if kind(a) == 'Accum' and s.axis.name in a.axes and a.name not in seen:
                self.emit(f'mov.f32 {self.var(a.name)}, {f32(a.op.identity)};')
                seen.add(a.name)

    def body(self, body):
        for s in body:
            k = kind(s)
            if k == 'Tile':
                if s.aux_threads:
                    raise NotImplementedError('grouped raster/auxiliary threads')
                extents = [as_static(a.extent) for a in s.axes]
                gid = self.integer('+', self.integer('*','%ctaid.x','%ntid.x'),'%tid.x')
                p = self.reg('p')
                self.emit(f'setp.ge.u32 {p}, {gid}, {math.prod(extents)};')
                self.emit(f'@{p} bra EXIT;')
                for i,a in enumerate(s.axes):
                    v = gid
                    stride = math.prod(extents[i+1:])
                    if stride != 1:
                        v = self.integer('/',v,str(stride))
                    if i:
                        v = self.integer('%',v,str(extents[i]))
                    self.values[a.name] = v
                if s.raster_group is not None:
                    names = list(s.raster_axes or ())
                    if len(names) != 2:
                        raise NotImplementedError('raster grouping requires two axes')
                    if s.raster_orient == 'n':
                        names.reverse()
                    axes = {a.name: as_static(a.extent) for a in s.axes}
                    rows, cols = (axes[n] for n in names)
                    group = min(rows, int(s.raster_group))
                    inner = math.prod(axes[a.name] for a in s.axes if a.name not in names)
                    block = self.integer('/', gid, str(inner))
                    group_id = self.integer('/', block, str(group * cols))
                    base = self.integer('*', group_id, str(group))
                    remaining = self.integer('-', str(rows), base)
                    width = self.reg('r')
                    self.emit(f'min.s32 {width}, {remaining}, {group};')
                    local = self.integer('%', block, str(group * cols))
                    self.values[names[0]] = self.integer('+', base, self.integer('%', local, width))
                    self.values[names[1]] = self.integer('/', local, width)
                self.body(s.body)
            elif k == 'Load':
                self.load(s.input,s.index,s.names)
            elif k == 'Write':
                if s.swizzle != 'NONE':
                    raise NotImplementedError(f'swizzle {s.swizzle}')
                self.store(s.output,s.index,s.values,s.atomic)
            elif k == 'Assign':
                args = [self.values[a] for a in s.args]
                self.arithmetic(self.var(s.name),s.op.name,args)
            elif k == 'Accum':
                dest = self.var(s.name)
                self.arithmetic(dest,s.op.name,[self.values[s.base] if s.base else dest,self.values[s.value]])
            elif k == 'Init':
                self.emit(f'mov.f32 {self.var(s.name)}, {f32(s.identity)};')
            elif k == 'Let':
                ty = 'f' if s.dtype.name == 'f32' else 'r'
                self.emit(f'mov.{"f32" if ty=="f" else "s32"} {self.var(s.name,ty)}, {self.expr(s.value)};')
            elif k in ('Loop','StridedLoop'):
                self.seed_loop(s)
                start,step,end = '0','1',str(as_static(s.axis.extent))
                if k == 'StridedLoop':
                    start,step = self.expr(s.start),self.expr(s.step)
                    if s.end is not None:
                        end = self.expr(s.end)
                v = self.reg('r'); self.values[s.axis.name] = v
                self.emit(f'mov.s32 {v}, {start};')
                top,done = self.label(),self.label()
                self.emit(top+':')
                p = self.reg('p');self.emit(f'setp.ge.s32 {p}, {v}, {end};')
                self.emit(f'@{p} bra {done};')
                self.body(s.body)
                self.emit(f'add.s32 {v}, {v}, {step};')
                self.emit(f'bra {top};'); self.emit(done+':')
            elif k == 'Cond':
                p = self.expr(s.cond); other,done = self.label(),self.label()
                self.emit(f'@!{p} bra {other};'); self.body(s.body)
                self.emit(f'bra {done};'); self.emit(other+':')
                self.body(s.else_body);self.emit(done+':')
            elif k == 'Smem':
                self.smem.add(s.name); self.shapes[s.name] = tuple(int(x) for x in s.extents)
                self.ptrs[s.name] = self.reg('r')
                if self.dynamic_shared:
                    self.emit(f'mov.u32 {self.ptrs[s.name]}, dynamic_smem;')
                    self.emit(f'add.u32 {self.ptrs[s.name]}, {self.ptrs[s.name]}, {self.shared_offsets[s.name]};')
                else:
                    self.shared.append(f'    .shared .align {max(4,s.align)} .b8 S_{s.name}[{math.prod(self.shapes[s.name])*4}];')
                    self.emit(f'mov.u32 {self.ptrs[s.name]}, S_{s.name};')
            elif k == 'Sync':
                self.barrier(s.barrier_id,s.count,s.warp)
            elif k == 'WarpShuffle':
                mask = self.reg('r'); self.emit(f'activemask.b32 {mask};')
                delta = int(s.length)//2
                while delta:
                    for a,b in zip(s.state,s.state_b):
                        self.emit(f'shfl.sync.bfly.b32 {self.var(b)}, {self.values[a]}, {delta}, 31, {mask};')
                    self.body(s.combine_states); delta //= 2
            elif k == 'TreeHalve':
                if s.inner:
                    raise NotImplementedError('inner TreeHalve')
                delta = int(s.length)//2
                while delta:
                    p = self.reg('p');done = self.label()
                    self.emit(f'setp.ge.s32 {p}, {self.values[s.tid_var]}, {delta};')
                    self.emit(f'@{p} bra {done};')
                    from emmy.compiler.ir.expr import Var, Literal, BinaryExpr
                    index = Var(s.tid_var)
                    for buf,a,b in zip(s.bufs,s.state,s.state_b):
                        self.load(buf,(index,),(a,))
                        self.load(buf,(BinaryExpr('+',index,Literal(delta,'int')),),(b,))
                    self.body(s.combine_states)
                    for buf,a in zip(s.bufs,s.state):
                        self.store(buf,(index,),(a,))
                    self.emit(done+':');self.barrier(s.barrier_id,s.barrier_count)
                    delta //= 2
                from emmy.compiler.ir.expr import Literal
                for buf,a in zip(s.bufs,s.state):
                    self.load(buf,(Literal(0,'int'),),(a,))
            elif k == 'CpAsyncCopy':
                if s.swizzle != 'NONE':
                    raise NotImplementedError('cp.async swizzle')
                index = s.smem_index
                if s.lane_index is not None:
                    from emmy.compiler.ir.expr import BinaryExpr
                    index = tuple(BinaryExpr('+', a, b) for a, b in zip(index, s.lane_index))
                dst,src = self.address(s.smem,index),self.address(s.src,s.src_index)
                self.emit(f'cp.async.ca.shared.global [{dst}], [{src}], {s.nbytes};')
            elif k == 'CpAsyncCommit':
                self.emit('cp.async.commit_group;')
            elif k == 'CpAsyncWait':
                self.emit(f'cp.async.wait_group {s.group};')
            else:
                raise NotImplementedError(f'statement {k}')

    def barrier(self, bid=0, count=None, warp=False):
        if warp:
            mask = self.reg('r');self.emit(f'activemask.b32 {mask};')
            self.emit(f'bar.warp.sync {mask};')
        else:
            self.emit(f'bar.sync {bid}' + (f', {count}' if count else '') + ';')

    def compile(self):
        for i,name in enumerate(self.launch['arg_order']):
            self.ptrs[name] = self.reg('d')
            self.emit(f'ld.param.u64 {self.ptrs[name]}, [param_{i}];')
        self.values['lane'] = self.integer('&','%tid.x','31')
        self.values['warp'] = self.integer('>>','%tid.x','5')
        self.body(self.node.op.body)
        signature = ',\n'.join(f'    .param .u64 param_{i}' for i in range(len(self.launch['arg_order'])))
        decl = []
        for r,t in [('f','f32'),('r','b32'),('d','b64'),('p','pred')]:
            if self.count[r]:
                decl.append(f'    .reg .{t} %{r}<{self.count[r]+1}>;')
        return '\n'.join(['.version 8.0','.target sm_89','.address_size 64','',*[d for d in self.shared if '.extern' in d],f'.visible .entry {self.node.op.name}(',signature,')','{',*decl,*[d for d in self.shared if '.extern' not in d],*self.code,'EXIT:','    ret;','}',''])


def main():
    path = Path(sys.argv[1])
    graph = Graph.from_dict(json.loads(path.read_text()))
    nodes = [n for n in graph.nodes.values() if kind(n.op)=='KernelOp']
    if len(nodes) != 1:
        raise ValueError('expected one KernelOp per IR file')
    meta = json.loads(path.with_name('meta.json').read_text())
    launch = next(k for k in meta['kernels'] if k['entry']==nodes[0].op.name)
    print(Compiler(graph,nodes[0],launch).compile(),end='')


if __name__ == '__main__':
    main()
