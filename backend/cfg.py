# -*- coding: utf-8 -*-
"""
控制流图（Control-Flow Graph）构建。

输入是代码生成器产出、已经过窥孔优化的 :class:`~backend.bytecode.ProgramCode`，
因此图中的每条边都对应 VM 真实会执行的跳转，天然与「字节码 / 中间代码」页的
偏移、行号严格对齐（偏移采用与字节码页一致的 1-based 展示，跳转操作数仍为
0-based 索引，二者换算关系为 ``显示偏移 = 索引 + 1``）。

对每个函数（含顶层 <main>）产出：
  * blocks —— 基本块列表。线性扫描指令序列，在"跳转目标"与"跳转后一条"处切分；
  * edges  —— 块间控制流边，细分为 fallthrough（顺序）、branch（条件真假）、
    jump（break/continue/回边/汇合等无条件跳转）、return（返回退出）；
  * loops  —— 通过支配关系找出的自然循环（头、回边、体、嵌套深度），
    多层嵌套循环因此可以被准确框出而不会错连；
  * calls  —— 块内的函数调用点，经栈平衡回溯解析出静态被调函数（含递归自调用）。

程序级另产出 call_graph（函数间调用关系 + 强连通分量，用于识别递归 / 互递归）。

正确性要点：
  * 提前 return 后出现的不可达指令仍保留为不可达基本块（虚线展示），不丢字节码；
  * 跳转目标等于指令条数（跳到函数末尾）统一连到函数的 EXIT 出口块；
  * 多层 break/continue 依赖 codegen 在 JUMP 上打的 note 标记，经跳转折叠后
    仍可由"回边/结构位置"兜底分类。
"""

from collections import defaultdict, deque
import sys
from typing import Dict, List, Optional, Set, Tuple

from . import bytecode as bc

TERMINATORS = {bc.OP_JUMP, bc.OP_JUMP_IF_FALSE, bc.OP_JUMP_IF_TRUE,
               bc.OP_RETURN, bc.OP_RETURN_NONE}

COND_OPS = {bc.OP_JUMP_IF_FALSE, bc.OP_JUMP_IF_TRUE}

# 跳转语义标记（由 codegen 打在 JUMP 指令上） -> 边的分类
NOTE_KINDS = {
    "while_back": "back",
    "for_back": "back",
    "break": "break",
    "continue": "continue",
}


# ---------------------------------------------------------------------------
# 单条指令的抽象栈效果（压栈为 +1，弹栈为 -1），用于回溯解析 CALL 的被调函数
# ---------------------------------------------------------------------------
def _stack_delta(ins) -> int:
    op = ins.op
    if op in (bc.OP_LOAD_CONST, bc.OP_LOAD_VAR, bc.OP_LOAD_GLOBAL,
              bc.OP_LOAD_BUILTIN, bc.OP_LOAD_FUNC, bc.OP_DUP):
        return 1
    if op in (bc.OP_STORE_VAR, bc.OP_STORE_GLOBAL, bc.OP_POP,
              bc.OP_RETURN_NONE, bc.OP_JUMP, bc.OP_NOP, bc.OP_LINE):
        return 0
    if op == bc.OP_BINARY:
        return -1
    if op == bc.OP_UNARY:
        return 0  # 语义上 -1+1（弹一个压一个）
    if op in (bc.OP_JUMP_IF_FALSE, bc.OP_JUMP_IF_TRUE):
        return -1
    if op == bc.OP_RETURN:
        return -1
    if op == bc.OP_DUP2:
        return 2
    if op == bc.OP_CALL:
        # 被调对象 + argc 个实参 -> 1 个结果
        return 1 - (1 + (ins.operand or 0))
    if op == bc.OP_MAKE_LIST:
        return 1 - (ins.operand or 0)
    if op == bc.OP_INDEX_LOAD:
        return -1
    if op == bc.OP_INDEX_STORE:
        return -3
    return 0


def _resolve_callee(instructions: List[bc.Instruction], call_idx: int):
    """从 CALL 指令向前回溯，按栈平衡找到压入被调对象的指令。

    返回 (kind, name)：kind ∈ {"user","builtin","indirect"}。
    """
    argc = instructions[call_idx].operand or 0
    depth = argc  # 先跳过 argc 个实参表达式
    i = call_idx - 1
    while i >= 0:
        ins = instructions[i]
        delta = _stack_delta(ins)
        depth -= delta
        if depth < 0:
            # 该指令压入的就是被调对象
            if ins.op == bc.OP_LOAD_FUNC:
                return "user", ins.operand
            if ins.op == bc.OP_LOAD_BUILTIN:
                return "builtin", ins.operand
            return "indirect", None
        i -= 1
    return "indirect", None


# ---------------------------------------------------------------------------
# 支配关系 + 自然循环
# ---------------------------------------------------------------------------
def _dominators(pred: Dict[int, Set[int]], entry: int, nodes: List[int]):
    """迭代求支配集合（经典 Cooper/简单集合交算法）。返回 dom: node -> set。"""
    node_set = set(nodes)
    dom: Dict[int, Set[int]] = {n: set(nodes) for n in nodes}
    dom[entry] = {entry}
    order = [n for n in nodes if n != entry]  # 节点按块顺序近似 RPO
    changed = True
    while changed:
        changed = False
        for n in order:
            ps = [p for p in pred.get(n, ()) if p in dom]
            if not ps:
                new = {n}
            else:
                new = set.intersection(*(dom[p] for p in ps))
                new.add(n)
            if new != dom[n]:
                dom[n] = new
                changed = True
    return dom


def _natural_loops(succ: Dict[int, Set[int]], pred: Dict[int, Set[int]],
                   dom: Dict[int, Set[int]], nodes: List[int]):
    """识别自然循环。返回 (loops, back_edges)。

    back edge: n -> h 且 h 支配 n。循环体 = 所有能到达 n 而不经过 h 的节点 + h。
    """
    loops = []
    back_edges: Set[Tuple[int, int]] = set()
    for n in nodes:
        for h in succ.get(n, ()):
            if h in dom.get(n, ()) and h != n:
                back_edges.add((n, h))
    # 同一循环头可能有多个回边（while + continue），按头合并
    by_header: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for (n, h) in back_edges:
        by_header[h].append((n, h))
    for h, edges in by_header.items():
        body = {h}
        for (n, _h) in edges:
            body.add(n)
            dq = deque([n])
            while dq:
                x = dq.popleft()
                for p in pred.get(x, ()):
                    if p not in body and p != h:
                        body.add(p)
                        dq.append(p)
        loops.append({"header": h, "tails": sorted({n for n, _ in edges}),
                      "body": sorted(body), "back_edges": sorted(edges)})
    loops.sort(key=lambda lp: (len(lp["body"]), lp["header"]))
    return loops, back_edges


# ---------------------------------------------------------------------------
# 基本块
# ---------------------------------------------------------------------------
class _Block:
    __slots__ = ("id", "start", "end", "instructions", "reachable", "kinds",
                 "loop_depth", "loops", "calls")

    def __init__(self, bid, start, instructions):
        self.id = bid
        self.start = start
        self.end = start + len(instructions) - 1
        self.instructions = instructions
        self.reachable = True
        self.kinds: Set[str] = set()        # entry / exit / return
        self.loop_depth = 0
        self.loops: List[int] = []          # 所属循环序号（外 -> 内）
        self.calls: List[dict] = []

    @property
    def is_entry(self):
        return "entry" in self.kinds


def _split_blocks(fc: bc.FunctionCode) -> List[_Block]:
    """线性扫描切分基本块。leaders: 入口 0、跳转目标、终止指令的下一条。"""
    ins = fc.instructions
    n = len(ins)
    if n == 0:
        return []
    leaders = {0}
    for idx, i in enumerate(ins):
        if i.op in COND_OPS or i.op == bc.OP_JUMP:
            t = i.operand
            if isinstance(t, int):
                if 0 <= t < n:
                    leaders.add(t)
                elif t >= n:
                    leaders.add(n)  # 跳到函数末尾（EXIT）
            if idx + 1 < n:
                leaders.add(idx + 1)
        elif i.op in (bc.OP_RETURN, bc.OP_RETURN_NONE):
            if idx + 1 < n:
                leaders.add(idx + 1)
    sorted_leaders = sorted(leaders)
    blocks: List[_Block] = []
    for k, start in enumerate(sorted_leaders):
        end = sorted_leaders[k + 1] if k + 1 < len(sorted_leaders) else n
        if start >= n:
            continue  # n 仅作为 EXIT 目标标记，不是真实块
        blocks.append(_Block(k, start, ins[start:end]))
    # 重新按顺序编号（过滤越界 leader 后保持 0..m-1）
    for bid, b in enumerate(blocks):
        b.id = bid
    blocks[0].kinds.add("entry")
    return blocks


def build_function_cfg(fc: bc.FunctionCode):
    """构建单个函数的 CFG（dict，供 JSON 序列化）。空函数返回 None。"""
    blocks = _split_blocks(fc)
    if not blocks:
        return None
    ins = fc.instructions
    n = len(ins)
    by_start: Dict[int, _Block] = {b.start: b for b in blocks}

    # 0-based 指令索引 -> 所属块
    owner: Dict[int, _Block] = {}
    for b in blocks:
        for k in range(b.start, b.end + 1):
            owner[k] = b

    # ---- 可达性（从入口 BFS，顺序/跳转两边都走） ----
    def target_block(t):
        if not isinstance(t, int):
            return None
        if 0 <= t < n:
            return owner.get(t)
        return None  # t == n -> EXIT

    succ: Dict[int, Set[int]] = defaultdict(set)
    pred: Dict[int, Set[int]] = defaultdict(set)
    edges: List[dict] = []

    def add_edge(src: _Block, dst: Optional[_Block], kind: str, label: str,
                 dst_exit=False, note=""):
        e = {"src": src.id, "kind": kind, "label": label}
        if dst_exit:
            e["dst"] = None
            e["to_exit"] = True
        else:
            e["dst"] = dst.id
            succ[src.id].add(dst.id)
            pred[dst.id].add(src.id)
        if note:
            e["note"] = note
        edges.append(e)

    for b in blocks:
        last = ins[b.end]
        op = last.op
        if op == bc.OP_JUMP:
            t = last.operand
            dst = target_block(t)
            note = last.note or ""
            if note in NOTE_KINDS:
                kind = "jump"
            else:
                kind = "jump"
            if isinstance(t, int) and t >= n:
                add_edge(b, None, "return", "退出", dst_exit=True, note=note)
            elif dst is not None:
                label, sub = _classify_jump(b, dst, note, blocks)
                add_edge(b, dst, "jump", label, note=sub or note)
        elif op in COND_OPS:
            t = last.operand
            dst = target_block(t)
            # JUMP_IF_FALSE: 弹出栈顶，为假跳转 -> 跳转边=假，顺序边=真
            # JUMP_IF_TRUE : 为真跳转 -> 跳转边=真，顺序边=假
            jump_label = "假" if op == bc.OP_JUMP_IF_FALSE else "真"
            fall_label = "真" if op == bc.OP_JUMP_IF_FALSE else "假"
            # 紧邻 DUP 的条件跳转来自 &&/|| 短路求值（见 codegen._logical）
            short_circuit = b.end >= 1 and ins[b.end - 1].op == bc.OP_DUP
            sc_note = "short_circuit" if short_circuit else ""
            sc_tag = "·短路" if short_circuit else ""
            if isinstance(t, int) and t >= n:
                add_edge(b, None, "branch", jump_label + sc_tag + "·退出", dst_exit=True,
                         note=sc_note)
            elif dst is not None:
                add_edge(b, dst, "branch", jump_label + sc_tag, note=sc_note)
            # 条件不成立（顺序执行）
            nxt = by_start.get(b.end + 1)
            if nxt is not None:
                add_edge(b, nxt, "branch", fall_label + sc_tag, note=sc_note)
        elif op in (bc.OP_RETURN, bc.OP_RETURN_NONE):
            b.kinds.add("return")
            add_edge(b, None, "return",
                     "返回值" if op == bc.OP_RETURN else "返回",
                     dst_exit=True)
        else:
            # 末尾不是终止指令：顺序进入下一块；若已是最后一块则隐式退出
            nxt = by_start.get(b.end + 1)
            if nxt is not None:
                add_edge(b, nxt, "fall", "")
            else:
                add_edge(b, None, "return", "隐式返回", dst_exit=True)

    # 可达性
    dq = deque([blocks[0]])
    seen = {blocks[0].id}
    while dq:
        x = dq.popleft()
        for y in succ.get(x.id, ()):
            if y not in seen:
                seen.add(y)
                dq.append(blocks[y])
    for b in blocks:
        b.reachable = b.id in seen

    # ---- 支配关系 + 自然循环（仅在可达块上做，避免不可达块污染） ----
    reachable = [b for b in blocks if b.reachable]
    rids = [b.id for b in reachable]
    dom = _dominators(pred, blocks[0].id, rids)
    loops, back_edges = _natural_loops(succ, pred, dom, rids)

    def _back_notes(u, v):
        """收集回边的语义标记。

        优化器的跳转折叠会让多条 JUMP 收敛到同一目标：源块末尾 JUMP 的操作数
        若被折叠，其它携带 ``*_back`` / break / continue 标记、且沿折叠链最终
        也指向同一目标的 JUMP，其语义都属于这条边。
        """
        end_idx = blocks[u].end
        target = ins[end_idx].operand
        notes = set()
        if ins[end_idx].note:
            notes.add(ins[end_idx].note)
        for k, x in enumerate(ins):
            if x.op == bc.OP_JUMP and x.note and k != end_idx:
                t = x.operand
                seen = set()
                while isinstance(t, int) and 0 <= t < n and ins[t].op == bc.OP_JUMP \
                        and t not in seen:
                    seen.add(t)
                    t = ins[t].operand
                if isinstance(t, int) and t == target:
                    notes.add(x.note)
        return notes

    # 循环嵌套深度：直接按"头支配 + 体包含"计算每个块属于多少层循环
    for li, lp in enumerate(loops):
        lp["depth"] = 1 + sum(1 for other in loops
                              if other is not lp and lp["header"] in other["body"])
        for bid in lp["body"]:
            blk = blocks[bid]
            blk.loops.append(li)
        # 回边分类校正：仅作用于无条件 JUMP 边（break/continue/回边）。
        # 条件分支（branch，如 if 条件为假跳回循环头）保留"真/假"语义标签。
        for (u, v) in lp["back_edges"]:
            notes = _back_notes(u, v)
            for e in edges:
                if e["src"] == u and e.get("dst") == v and e["kind"] != "branch":
                    e["back"] = True
                    if "break" in notes:
                        e["label"] = "break"
                        e["note"] = "break"
                    elif "continue" in notes:
                        e["label"] = "continue"
                        e["note"] = "continue"
                    else:
                        e["label"] = "循环回边"
                        e["note"] = "back"
                        if "for_back" in notes:
                            e["back_kind"] = "for"
                        elif "while_back" in notes:
                            e["back_kind"] = "while"

    # 块深度 / 循环头标记
    for b in blocks:
        b.loop_depth = len(b.loops)
    # 再由循环体包含关系求真实嵌套深度（头在别的循环体里则 +1）
    for li, lp in enumerate(loops):
        depth = 1
        for oj, other in enumerate(loops):
            if oj != li and lp["header"] in other["body"]:
                depth += 1
        lp["depth"] = depth

    # ---- 调用点解析（块内 CALL 指令） ----
    call_edges: List[dict] = []
    for b in blocks:
        for k in range(b.start, b.end + 1):
            i = ins[k]
            if i.op != bc.OP_CALL:
                continue
            kind, name = _resolve_callee(ins, k)
            rec = {"offset": k + 1, "line": i.line, "argc": i.operand or 0,
                   "callee_kind": kind, "callee": name,
                   "label": (name + "()") if name else "间接调用"}
            b.calls.append(rec)
            if kind in ("user", "builtin") and name:
                call_edges.append({"src_block": b.id, "offset": k + 1,
                                   "line": i.line, "kind": kind, "callee": name})

    # ---- 循环展示元信息（头块行号 / while|for 判定） ----
    loop_infos = []
    for li, lp in enumerate(loops):
        hblk = blocks[lp["header"]]
        # for/while 判定：回边上携带的 *_back 标记（可能经跳转折叠，需反查）
        back_notes = set()
        for (u, v) in lp["back_edges"]:
            back_notes |= _back_notes(u, v)
        if "for_back" in back_notes:
            lkind, lname = "for", "for 循环"
        elif "while_back" in back_notes:
            lkind, lname = "while", "while 循环"
        else:
            # 兜底：头部块跨行（含 for 的 init）则视为 for
            hlines = {x.line for x in hblk.instructions if x.line >= 1}
            if len(hlines) >= 2:
                lkind, lname = "for", "for 循环"
            else:
                lkind, lname = "loop", "循环"
        body_lines = [ins[k].line for bid in lp["body"] for k in
                      range(blocks[bid].start, blocks[bid].end + 1) if ins[k].line >= 1]
        loop_infos.append({
            "id": li, "kind": lkind, "name": lname,
            "header_block": lp["header"],
            "tails": lp["tails"],
            "blocks": lp["body"],
            "depth": lp["depth"],
            "line_start": min(body_lines) if body_lines else hblk.instructions[0].line,
            "line_end": max(body_lines) if body_lines else hblk.instructions[0].line,
        })

    # ---- 序列化 ----
    blk_dicts = []
    for b in blocks:
        blk_dicts.append({
            "id": b.id,
            "start": b.start + 1,           # 与字节码页一致的 1-based 偏移
            "end": b.end + 1,
            "start_index": b.start,         # 0-based 索引（边目标换算用）
            "line_start": b.instructions[0].line,
            "line_end": b.instructions[-1].line,
            "reachable": b.reachable,
            "is_entry": b.id == blocks[0].id,
            "has_return": "return" in b.kinds,
            "loop_depth": b.loop_depth,
            "loops": b.loops,
            "is_loop_header": any(lp["header"] == b.id for lp in loops),
            "calls": b.calls,
            "instructions": [i.to_dict() for i in b.instructions],
        })

    return {
        "name": fc.name,
        "arity": fc.arity,
        "params": fc.params,
        "is_main": fc.is_main,
        "entry": 0,
        "instruction_count": n,
        "blocks": blk_dicts,
        "edges": edges,
        "loops": loop_infos,
        "call_sites": call_edges,
    }


def _classify_jump(src: _Block, dst: _Block, note: str, blocks: List[_Block]):
    """给无条件 JUMP 边定标签。返回 (label, sub_note)。"""
    if note in NOTE_KINDS:
        sub = NOTE_KINDS[note]
        if sub == "back":
            return "循环回边", "back"
        if sub == "break":
            return "break", "break"
        if sub == "continue":
            return "continue", "continue"
    # 向后跳 = 回边（优化折叠后 note 可能丢失的兜底）
    if dst.id < src.id:
        return "循环回边", "back"
    # 紧邻向下的跳转：if/elif 分支汇合
    if dst.id == src.id + 1:
        return "汇合", "merge"
    return "跳转", "jump"


# ---------------------------------------------------------------------------
# 程序级：调用图 + 递归识别（Tarjan SCC）
# ---------------------------------------------------------------------------
def build_program_cfg(program: bc.ProgramCode) -> dict:
    funcs: List[bc.FunctionCode] = []
    if program.main is not None:
        funcs.append(program.main)
    funcs.extend(program.functions.values())

    cfgs = []
    user_calls: Dict[str, Set[str]] = defaultdict(set)
    builtin_calls: Dict[str, Set[str]] = defaultdict(set)
    call_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for fc in funcs:
        cfg = build_function_cfg(fc)
        if cfg is None:
            continue
        cfgs.append(cfg)
        for cs in cfg["call_sites"]:
            if cs["kind"] == "user":
                user_calls[fc.name].add(cs["callee"])
                call_counts[(fc.name, cs["callee"])] += 1
            else:
                builtin_calls[fc.name].add(cs["callee"])

    names = [c["name"] for c in cfgs]
    name_set = set(names)

    # ---- Tarjan 强连通分量（识别直接/互相递归） ----
    index = {}
    low = {}
    stack = []
    on_stack = set()
    counter = [0]
    sccs = []

    def strong(v):
        index[v] = counter[0]
        low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in user_calls.get(v, ()):
            if w not in name_set:
                continue
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    sys.setrecursionlimit(10000)
    for v in names:
        if v not in index:
            strong(v)

    recursive_groups = []
    recursion_kind: Dict[str, str] = {}
    for comp in sccs:
        if len(comp) > 1:
            recursive_groups.append(sorted(comp))
            for v in comp:
                recursion_kind[v] = "mutual"
        elif comp[0] in user_calls.get(comp[0], ()):
            recursive_groups.append([comp[0]])
            recursion_kind[comp[0]] = "self"

    call_graph = {
        "nodes": [{"name": n,
                   "is_main": next((c["is_main"] for c in cfgs if c["name"] == n), False),
                   "recursion": recursion_kind.get(n, "")}
                  for n in names],
        "edges": [{"caller": a, "callee": b, "count": call_counts[(a, b)]}
                  for (a, b) in sorted(call_counts)],
        "builtin_used": {k: sorted(v) for k, v in builtin_calls.items()},
        "recursive_groups": recursive_groups,
    }

    return {"functions": cfgs, "call_graph": call_graph}
