/* ============================================================
   MiniLang 平台 —— 控制流图（CFG）引擎与渲染器
   纯前端实现，不依赖任何第三方库。
   职责：
     1. ML.CFG.build(bytecode)
        对每个函数代码段做「前导指令（leader）」分析切分基本块，
        并由真实的字节码跳转 / 返回指令推导块间控制流边——分支、
        循环回边（含 continue）、break 跳转、提前返回全部来自实发
        指令，因此递归、多层循环、return 提前退出都不会漏边或错连。
        CALL 指令结合块内抽象栈模拟识别被调用户函数 / 内置函数
        （自调用形成递归自环，互递归形成跨函数调用边）。
     2. ML.CFG.layout(graph)
        纯几何的分层（ranked）布局：每个函数一条泳道，自上而下
        排布；前向边走三次贝塞尔，循环回边走左侧直角通道，
        跨函数调用走右侧虚线汇流通道。
     3. ML.CFG.Renderer
        「一个 SVG 画边 + 绝对定位 HTML 画块」的混合渲染。
   ============================================================ */
(function () {
  "use strict";

  const ML = (window.ML = window.ML || {});

  const TERMINATORS = new Set(["JUMP", "JUMP_IF_FALSE", "JUMP_IF_TRUE", "RETURN", "RETURN_NONE"]);

  // 边类型 -> 语义（着色 / 图例 / 过滤共用）
  // fall    顺序执行 / 无条件跳转（含 break 跳出循环）
  // branch  条件分支（真 / 假）
  // back    循环回边（while/for 回跳）
  // return  提前返回 / 函数出口
  // call    用户函数调用（跨函数；自调用为递归自环）
  const EDGE_KINDS = ["fall", "branch", "back", "return", "call"];

  /* ============================================================
     1. 构建：字节码 -> 每个函数的基本块与边
     ============================================================ */
  function buildProgram(bytecode) {
    const fns = (bytecode.functions || []).map((fc) => buildFunction(fc));
    return { functions: fns };
  }

  function buildFunction(fc) {
    const ins = fc.instructions || [];
    const n = ins.length;

    // ---------- 1.1 找前导指令（leaders） ----------
    const leader = new Set();
    if (n > 0) leader.add(0);
    for (let i = 0; i < n; i++) {
      const op = ins[i].op;
      if (op === "JUMP" || op === "JUMP_IF_FALSE" || op === "JUMP_IF_TRUE") {
        const t = Number(ins[i].operand);
        if (Number.isInteger(t) && t >= 0 && t < n) leader.add(t);
        if (i + 1 < n) leader.add(i + 1);  // 条件跳转的假出口；JUMP 后另起新块
      } else if (op === "RETURN" || op === "RETURN_NONE") {
        if (i + 1 < n) leader.add(i + 1);  // 返回之后的代码（可能不可达）另起一块
      }
    }
    const starts = Array.from(leader).sort((a, b) => a - b);

    // ---------- 1.2 切基本块 ----------
    const blocks = starts.map((start, k) => {
      const end = k + 1 < starts.length ? starts[k + 1] - 1 : n - 1;
      const list = ins.slice(start, end + 1);
      return {
        id: fc.name + "#b" + k,
        fnName: fc.name,
        index: k,
        start,
        end,
        instructions: list,
        terminator: (list.length && TERMINATORS.has(list[list.length - 1].op))
          ? list[list.length - 1] : null,
        calls: [],
        reachable: false,
      };
    });
    const blockIndex = new Map();  // 块 id -> 块下标
    blocks.forEach((b, bi) => blockIndex.set(b.id, bi));
    const owner = new Int32Array(n).fill(-1);  // 指令下标 -> 块下标
    blocks.forEach((b, bi) => { for (let j = b.start; j <= b.end; j++) owner[j] = bi; });

    // ---------- 1.3 块内 CALL 识别（抽象栈模拟，正确处理嵌套调用） ----------
    // 栈元素：{kind:'builtin'|'func'|'unknown', name}
    blocks.forEach((b) => {
      const stack = [];
      const pop = () => (stack.length ? stack.pop() : { kind: "unknown" });
      for (const inx of b.instructions) {
        switch (inx.op) {
          case "LOAD_BUILTIN": stack.push({ kind: "builtin", name: inx.operand }); break;
          case "LOAD_FUNC":    stack.push({ kind: "func", name: inx.operand }); break;
          case "LOAD_CONST": case "LOAD_VAR": case "LOAD_GLOBAL": case "DUP":
            stack.push({ kind: "unknown" }); break;
          case "STORE_VAR": case "STORE_GLOBAL": case "POP": case "UNARY":
            pop(); break;
          case "BINARY": case "INDEX_LOAD":
            pop(); pop(); stack.push({ kind: "unknown" }); break;
          case "INDEX_STORE":
            pop(); pop(); pop(); break;
          case "DUP2":
            stack.push({ kind: "unknown" }, { kind: "unknown" }); break;
          case "MAKE_LIST": {
            const cnt = Number(inx.operand) || 0;
            for (let q = 0; q < cnt; q++) pop();
            stack.push({ kind: "unknown" });
            break;
          }
          case "CALL": {
            const argc = Number(inx.operand) || 0;
            for (let q = 0; q < argc; q++) pop();
            const callee = pop();
            b.calls.push({
              offset: inx.offset, line: inx.line,
              builtin: callee.kind === "builtin",
              indirect: callee.kind === "unknown",
              name: callee.name != null ? callee.name : null,
            });
            stack.push({ kind: "unknown" });  // 返回值
            break;
          }
          default:
            break;  // 其余指令不影响抽象栈
        }
      }
    });

    // ---------- 1.4 控制流边 ----------
    const edges = [];
    const addEdge = (kind, fromBi, toBi, label, line) => {
      const e = {
        id: "e" + edges.length + ":" + fc.name,
        kind,
        from: blocks[fromBi].id,
        to: toBi >= 0 ? blocks[toBi].id : null,  // return 边先留空，1.6 收口到 exit
        label: label || "", line: line || 0,
        fnName: fc.name, intra: true, unreachable: false,
      };
      edges.push(e);
      return e;
    };

    blocks.forEach((b, bi) => {
      const t = b.terminator;
      if (!t) {
        if (b.end + 1 < n && owner[b.end + 1] >= 0) {
          addEdge("fall", bi, owner[b.end + 1], "", b.instructions[b.instructions.length - 1].line);
        }
        return;
      }
      if (t.op === "JUMP") {
        const target = Number(t.operand);
        if (Number.isInteger(target) && owner[target] >= 0) {
          addEdge("fall", bi, owner[target], "跳转", t.line);
        }
      } else if (t.op === "JUMP_IF_FALSE" || t.op === "JUMP_IF_TRUE") {
        const target = Number(t.operand);
        const isFalse = t.op === "JUMP_IF_FALSE";
        if (Number.isInteger(target) && owner[target] >= 0) {
          addEdge("branch", bi, owner[target], isFalse ? "假" : "真", t.line);
        }
        if (bi + 1 < blocks.length) {
          addEdge("branch", bi, bi + 1, isFalse ? "真" : "假", t.line);
        }
      } else {  // RETURN / RETURN_NONE
        addEdge("return", bi, -1, "return", t.line);
      }
    });

    // ---------- 1.5 DFS：可达性 + 回边识别 ----------
    // 代码生成器只产出可归约（reducible）控制流：循环头必支配其回边源点，
    // 因此凡指向 DFS 栈中祖先的边都是回边，与其原始种类（JUMP /
    // JUMP_IF_* 的真假出口）无关。漏标任何一条都会让分层松弛在环上发散。
    const adj = blocks.map(() => []);
    edges.forEach((e) => {
      if (e.to && blockIndex.has(e.to)) adj[blockIndex.get(e.from)].push(blockIndex.get(e.to));
    });
    const reachable = new Set();
    const backSet = new Set();
    if (blocks.length) {
      reachable.add(0);
      const color = new Array(blocks.length).fill(0);
      const stack = [0];
      color[0] = 1;
      while (stack.length) {
        const u = stack[stack.length - 1];
        const kids = adj[u].slice().sort((a, b) => a - b);
        let descended = false;
        for (const v of kids) {
          if (color[v] === 0) {
            color[v] = 1; reachable.add(v);
            stack.push(v); descended = true;
            break;
          } else if (color[v] === 1) {
            // 同端点可能有多条边（理论上），全部标记为回边
            edges.forEach((x) => {
              if (x.from === blocks[u].id && x.to === blocks[v].id) backSet.add(x.id);
            });
          }
        }
        if (!descended) { color[u] = 2; stack.pop(); }
      }
    }
    blocks.forEach((b, bi) => { b.reachable = reachable.has(bi); });
    edges.forEach((e) => {
      if (backSet.has(e.id)) e.kind = "back";
      const f = blockIndex.get(e.from);
      const t = e.to ? blockIndex.get(e.to) : -1;
      if (f != null && !blocks[f].reachable) e.unreachable = true;
      if (t != null && t >= 0 && !blocks[t].reachable) e.unreachable = true;
    });

    // ---------- 1.6 虚拟入口 / 出口，返回边统一收口 ----------
    const entry = { id: fc.name + "#entry", virtual: "entry", fnName: fc.name };
    const exit = { id: fc.name + "#exit", virtual: "exit", fnName: fc.name };
    if (blocks.length) {
      edges.push({
        id: "e-entry:" + fc.name, kind: "fall", from: entry.id, to: blocks[0].id,
        label: "", line: 0, fnName: fc.name, intra: true, unreachable: !blocks[0].reachable,
      });
    }
    edges.forEach((e) => { if (e.kind === "return" && e.to == null) e.to = exit.id; });

    // ---------- 1.7 调用边（用户函数：跨函数；自调用：递归自环） ----------
    const callEdges = [];
    blocks.forEach((b) => {
      b.calls.forEach((c) => {
        if (c.builtin || c.indirect || c.name == null) return;
        callEdges.push({
          id: `e-call:${fc.name}:${b.index}:${c.offset}`,
          kind: "call",
          from: b.id, to: null,
          callee: c.name,
          label: "调用 " + c.name + (c.name === fc.name ? "（递归）" : ""),
          line: c.line, fnName: fc.name, intra: false,
          recursive: c.name === fc.name, unreachable: !b.reachable,
        });
      });
    });

    return {
      name: fc.name,
      isMain: !!fc.is_main,
      arity: fc.arity,
      params: fc.params || [],
      constants: fc.constants || [],
      blocks,
      entry,
      exit,
      nodes: [entry].concat(blocks).concat([exit]),
      edges: edges.concat(callEdges),
    };
  }

  /* ============================================================
     2. 布局：分层 + 泳道
     ============================================================ */
  const LAYOUT = {
    NODE_W: 300,
    ROW_GAP: 38,
    COL_GAP: 56,
    PAD_TOP: 48,
    PAD_LEFT: 58,
    PAD_RIGHT: 110,
    TERM_W: 72,
    TERM_H: 30,
    LINE_H: 19,
    HEAD_H: 26,
    CALL_H: 20,
    PAD_BOTTOM: 12,
    LANE_GAP: 64,
  };

  function blockHeight(b) {
    const calls = b.calls.length ? LAYOUT.CALL_H + 2 : 0;
    return LAYOUT.HEAD_H + b.instructions.length * LAYOUT.LINE_H + calls + LAYOUT.PAD_BOTTOM;
  }

  function layoutProgram(graph, visibleKinds) {
    const visible = new Set(visibleKinds || EDGE_KINDS);
    const lanes = graph.functions.map((fn) => layoutFunction(fn, visible));

    let totalH = 0;
    lanes.forEach((L, i) => {
      L.y = totalH;
      totalH += L.height + (i < lanes.length - 1 ? LAYOUT.LANE_GAP : 28);
    });
    const contentW = Math.max.apply(null, lanes.map((L) => L.contentW).concat([420]));
    const width = LAYOUT.PAD_LEFT + contentW + LAYOUT.PAD_RIGHT;
    const busX = LAYOUT.PAD_LEFT + contentW + 24;

    return { width, height: totalH, lanes, busX, visibleKinds: visible };
  }

  function layoutFunction(fn, visible) {
    // 几何分层始终使用完整的函数内控制流（块的位置不随「边过滤」抖动）；
    // 哪些边参与绘制由渲染阶段按 visible 决定。
    void visible;
    const nodeById = new Map(fn.nodes.map((nd) => [nd.id, nd]));

    // ---------- 2.1 邻接（全部函数内边） ----------
    const succ = new Map(fn.nodes.map((nd) => [nd.id, []]));
    const pred = new Map(fn.nodes.map((nd) => [nd.id, []]));
    const rank = new Map(fn.nodes.map((nd) => [nd.id, 0]));
    const intraEdges = fn.edges.filter((e) => e.intra && nodeById.has(e.to));
    intraEdges.forEach((e) => { succ.get(e.from).push(e.to); pred.get(e.to).push(e); });
    const backIds = new Set(intraEdges.filter((e) => e.kind === "back").map((e) => e.id));

    // ---------- 2.2 从入口可达（沿可见边） ----------
    const reachIds = new Set([fn.entry.id]);
    {
      const q = [fn.entry.id];
      while (q.length) {
        const u = q.shift();
        succ.get(u).forEach((v) => { if (!reachIds.has(v)) { reachIds.add(v); q.push(v); } });
      }
    }

    // ---------- 2.3 最长路径分层（回边不参与，保证 DAG） ----------
    for (let iter = 0; iter < 300; iter++) {
      let changed = false;
      fn.blocks.forEach((b) => {
        if (!reachIds.has(b.id)) return;
        let best = 1;
        pred.get(b.id).forEach((e) => {
          if (backIds.has(e.id) || !reachIds.has(e.from)) return;
          best = Math.max(best, rank.get(e.from) + 1);
        });
        if (rank.get(b.id) !== best) { rank.set(b.id, best); changed = true; }
      });
      if (!changed) break;
    }
    let exitRank = 1;
    intraEdges.forEach((e) => {
      if (e.kind === "return") exitRank = Math.max(exitRank, rank.get(e.from) + 1);
    });
    fn.blocks.forEach((b) => { if (reachIds.has(b.id)) exitRank = Math.max(exitRank, rank.get(b.id) + 1); });
    rank.set(fn.exit.id, exitRank);
    rank.set(fn.entry.id, 0);

    // 不可达块排在出口之后（按指令顺序逐层）
    let dr = exitRank + 1;
    fn.blocks.forEach((b) => { if (!reachIds.has(b.id)) rank.set(b.id, dr++); });
    const maxRank = Math.max.apply(null, Array.from(rank.values()));

    // ---------- 2.4 同层排序：初始按起始偏移，再做重心扫描 ----------
    const byRank = new Map();
    for (let r = 0; r <= maxRank; r++) byRank.set(r, []);
    fn.nodes.forEach((nd) => byRank.get(rank.get(nd.id)).push(nd));
    byRank.forEach((arr) => arr.sort((a, b) => {
      const sa = a.virtual ? (a.virtual === "entry" ? -1 : 1e9) : a.start;
      const sb = b.virtual ? (b.virtual === "entry" ? -1 : 1e9) : b.start;
      return sa - sb;
    }));
    const rankOf = (id) => rank.get(id);
    for (let sweep = 0; sweep < 10; sweep++) {
      for (let r = 1; r <= maxRank; r++) {
        const arr = byRank.get(r);
        arr.forEach((nd) => {
          if (nd.virtual === "entry") { nd._w = -1e6; return; }
          if (nd.virtual === "exit") { nd._w = 1e6; return; }
          const ps = pred.get(nd.id)
            .filter((e) => rankOf(e.from) === r - 1)
            .map((e) => byRank.get(r - 1).findIndex((x) => x.id === e.from));
          nd._w = ps.length ? ps.reduce((a, b2) => a + b2, 0) / ps.length : nd.start;
        });
        arr.sort((a, b) => a._w - b._w);
      }
    }

    // ---------- 2.5 几何 ----------
    const nodeGeo = new Map();
    const rankHeights = [];
    let contentW = 0;
    for (let r = 0; r <= maxRank; r++) {
      const arr = byRank.get(r);
      let h = 0;
      arr.forEach((nd) => { h = Math.max(h, nd.virtual ? LAYOUT.TERM_H : blockHeight(nd)); });
      rankHeights.push(h);
      contentW = Math.max(contentW, arr.length * LAYOUT.NODE_W + (arr.length - 1) * LAYOUT.COL_GAP);
    }
    let yCursor = LAYOUT.PAD_TOP;
    for (let r = 0; r <= maxRank; r++) {
      const arr = byRank.get(r);
      arr.forEach((nd, col) => {
        const x = LAYOUT.PAD_LEFT + col * (LAYOUT.NODE_W + LAYOUT.COL_GAP);
        const h = nd.virtual ? LAYOUT.TERM_H : blockHeight(nd);
        nodeGeo.set(nd.id, { x, y: yCursor + (rankHeights[r] - h) / 2, w: LAYOUT.NODE_W, h, rank: r, col });
      });
      yCursor += rankHeights[r] + LAYOUT.ROW_GAP;
    }
    const height = yCursor - LAYOUT.ROW_GAP + LAYOUT.PAD_BOTTOM;

    // ---------- 2.6 回边左侧通道编号（跨度大的更靠外） ----------
    const backEdges = intraEdges.filter((e) => e.kind === "back");
    backEdges.sort((a, b) => {
      const da = nodeGeo.get(a.from).rank - nodeGeo.get(a.to).rank;
      const db = nodeGeo.get(b.from).rank - nodeGeo.get(b.to).rank;
      return db - da;
    });
    backEdges.forEach((e, i) => { e._channel = 1 + (i % 3); });

    return { fn, width: LAYOUT.PAD_LEFT + contentW + LAYOUT.PAD_RIGHT, contentW, height, nodeGeo, maxRank };
  }

  /* ============================================================
     3. 渲染器
     ============================================================ */
  const Renderer = class {
    constructor(container) {
      this.container = container;
      this.sourceLines = [];
      this.showSource = true;
      this._buildDom();
    }

    _buildDom() {
      this.container.innerHTML = "";
      this.stageWrap = document.createElement("div");
      this.stageWrap.className = "cfg-stage-wrap";
      this.stage = document.createElement("div");
      this.stage.className = "cfg-stage";
      this.svgNS = "http://www.w3.org/2000/svg";
      this.svg = document.createElementNS(this.svgNS, "svg");
      this.svg.classList.add("cfg-svg");
      this.defs = document.createElementNS(this.svgNS, "defs");
      this.svg.appendChild(this.defs);
      this.edgeLayer = document.createElementNS(this.svgNS, "g");
      this.svg.appendChild(this.edgeLayer);
      this.nodeLayer = document.createElement("div");
      this.nodeLayer.className = "cfg-nodes";
      this.labelLayer = document.createElement("div");
      this.labelLayer.className = "cfg-edge-labels";
      this.stage.appendChild(this.svg);
      this.stage.appendChild(this.nodeLayer);
      this.stage.appendChild(this.labelLayer);
      this.stageWrap.appendChild(this.stage);
      this.container.appendChild(this.stageWrap);
      this._buildMarkers();
    }

    _buildMarkers() {
      const colors = {
        fall: "var(--text-faint)", branch: "var(--warning)", back: "var(--info)",
        return: "var(--danger)", call: "var(--accent)",
      };
      Object.keys(colors).forEach((kind) => {
        const m = document.createElementNS(this.svgNS, "marker");
        m.setAttribute("id", "cfg-arrow-" + kind);
        m.setAttribute("viewBox", "0 0 10 10");
        m.setAttribute("refX", "8.5");
        m.setAttribute("refY", "5");
        m.setAttribute("markerWidth", "7");
        m.setAttribute("markerHeight", "7");
        m.setAttribute("orient", "auto-start-reverse");
        const p = document.createElementNS(this.svgNS, "path");
        p.setAttribute("d", "M0,0 L10,5 L0,10 z");
        p.setAttribute("class", "cfg-marker cfg-marker-" + kind);
        // 注意：CSS 变量在 presentation 属性里不会解析，必须走 style
        p.style.fill = colors[kind];
        m.appendChild(p);
        this.defs.appendChild(m);
      });
    }

    setSource(lines) { this.sourceLines = (lines || []).slice(); }

    render(graph, layout, opts) {
      opts = opts || {};
      this.graph = graph;
      this.layout = layout;
      this.showSource = opts.showSource !== false;

      this.nodeLayer.innerHTML = "";
      this.labelLayer.innerHTML = "";
      this.edgeLayer.textContent = "";

      this.laneByFn = new Map();
      layout.lanes.forEach((L) => this.laneByFn.set(L.fn.name, L));

      this.stage.style.width = layout.width + "px";
      this.stage.style.height = layout.height + "px";
      this.svg.setAttribute("width", layout.width);
      this.svg.setAttribute("height", layout.height);

      layout.lanes.forEach((L) => this._renderLaneHead(L));
      layout.lanes.forEach((L) => L.fn.nodes.forEach((nd) => this._renderNode(L, nd)));
      graph.functions.forEach((fn) => fn.edges.forEach((e) => this._renderEdge(e)));

      this._bindHover();
    }

    _renderLaneHead(L) {
      const head = document.createElement("div");
      head.className = "cfg-lane-head";
      head.style.left = "12px";
      head.style.top = L.y + 8 + "px";
      const name = L.fn.isMain
        ? "&lt;main&gt;（顶层代码）"
        : ML.escapeHtml(L.fn.name) + "(" + L.fn.params.map(ML.escapeHtml).join(", ") + ")";
      const deadCount = L.fn.blocks.filter((b) => !b.reachable).length;
      const rec = L.fn.edges.some((e) => e.recursive);
      head.innerHTML =
        `<span class="cfg-lane-title">${name}</span>` +
        `<span class="badge gray">${L.fn.blocks.length} 个基本块</span>` +
        (rec ? `<span class="badge purple">递归</span>` : "") +
        (deadCount ? `<span class="badge red">${deadCount} 块不可达</span>` : "");
      this.nodeLayer.appendChild(head);
    }

    _geo(id) {
      const h = id.indexOf("#");
      const fnName = id.slice(0, h);
      const L = this.laneByFn.get(fnName);
      if (!L) return null;
      const g = L.nodeGeo.get(id);
      return g ? { g, laneY: L.y } : null;
    }

    _centerTop(id) { const z = this._geo(id); return z && { x: z.g.x + z.g.w / 2, y: z.laneY + z.g.y }; }
    _centerBottom(id) { const z = this._geo(id); return z && { x: z.g.x + z.g.w / 2, y: z.laneY + z.g.y + z.g.h }; }

    _renderNode(L, nd) {
      const g = L.nodeGeo.get(nd.id);
      const el = document.createElement("div");
      if (nd.virtual) {
        el.className = "cfg-term cfg-term-" + nd.virtual;
        el.textContent = nd.virtual === "entry" ? "入口" : "出口";
        el.style.left = g.x + (g.w - LAYOUT.TERM_W) / 2 + "px";
        el.style.top = L.y + g.y + "px";
        el.style.width = LAYOUT.TERM_W + "px";
        el.style.height = LAYOUT.TERM_H + "px";
      } else {
        el.className = "cfg-block" + (nd.reachable ? "" : " unreachable");
        el.style.left = g.x + "px";
        el.style.top = L.y + g.y + "px";
        el.style.width = g.w + "px";
        el.dataset.blockId = nd.id;
        el.innerHTML = this._blockHtml(nd);
      }
      this.nodeLayer.appendChild(el);
    }

    _constText(fnName, idx) {
      const fn = this.graph.functions.find((f) => f.name === fnName);
      if (!fn) return "";
      const c = fn.constants[idx];
      if (c == null) return c === null ? "null" : "";
      if (typeof c === "string") return '"' + (c.length > 12 ? c.slice(0, 12) + "…" : c) + '"';
      return String(c);
    }

    _blockHtml(b) {
      const esc = ML.escapeHtml;
      const rows = b.instructions.map((i) => {
        const term = TERMINATORS.has(i.op);
        let operand = "";
        if (i.op === "JUMP" || i.op === "JUMP_IF_FALSE" || i.op === "JUMP_IF_TRUE") {
          operand = "→ " + (Number(i.operand) + 1);
        } else if (i.op === "LOAD_CONST") {
          operand = "#" + i.operand + "  " + this._constText(b.fnName, i.operand);
        } else if (i.operand != null && i.operand !== "") {
          operand = String(i.operand);
        }
        const src = this.sourceLines[i.line - 1];
        let right;
        if (this.showSource && src != null && !term) {
          const t = src.trim();
          right = `<span class="cfg-src" title="${esc(t)}">${esc(t)}</span>`;
        } else {
          right = `<span class="cfg-arg">${esc(operand)}</span>`;
        }
        return `<div class="cfg-instr${term ? " term" : ""}" title="${esc(i.op + " " + operand)}  @L${i.line}">` +
          `<span class="cfg-i-off">${i.offset}</span><span class="cfg-i-op">${esc(i.op)}</span>${right}</div>`;
      });
      const chips = b.calls.map((c) => {
        if (c.builtin) return `<span class="cfg-chip bi" title="内置函数调用 @L${c.line}">⚙ ${esc(c.name)}()</span>`;
        if (c.indirect) return `<span class="cfg-chip dyn" title="动态调用，被调目标无法静态确定">⚡ 动态调用</span>`;
        return `<span class="cfg-chip fn" title="用户函数调用 @L${c.line}">↪ ${esc(c.name)}()${c.name === b.fnName ? " · 递归" : ""}</span>`;
      }).join("");
      const first = b.instructions[0], last = b.instructions[b.instructions.length - 1];
      const rng = first.offset === last.offset ? `${first.offset}` : `${first.offset}–${last.offset}`;
      const lr = first.line === last.line ? `L${first.line}` : `L${first.line}–L${last.line}`;
      return `<div class="cfg-block-head">` +
        `<span class="cfg-bid">B${b.index}</span>` +
        `<span class="cfg-brng">偏移 ${rng}</span>` +
        `<span class="cfg-blr">${lr}</span>` +
        (b.reachable ? "" : `<span class="badge red" style="font-size:10px;padding:0 6px">不可达</span>`) +
        `</div><div class="cfg-instrs">${rows.join("")}</div>` +
        (chips ? `<div class="cfg-calls">${chips}</div>` : "");
    }

    /* ---------------- 边 ---------------- */
    _renderEdge(e) {
      if (!this.layout.visibleKinds.has(e.kind)) return;
      // 源块不可达：该路径在真实执行中永不发生。完整连到目标会画出跨整张
      // 图的「回头长线」，故改为块底的短虚线桩 + 目标标注（边仍然保留、不漏）。
      // 目标不可达而源可达时仍画完整边（可达代码确实能进入死块之后的区域）。
      if (e.intra && this._isSourceDead(e.from)) {
        this._drawDeadStub(e);
        return;
      }
      let path = null;
      if (e.intra) path = e.kind === "back" ? this._drawBackEdge(e) : this._drawForwardEdge(e);
      else path = this._drawCallEdge(e);
      if (path) {
        path.dataset.from = e.from;
        path.dataset.to = e.to || (e.callee ? e.callee + "#b0" : "");
        path.dataset.kind = e.kind;
      }
    }

    _isSourceDead(blockId) {
      const h = blockId.indexOf("#");
      const fnName = blockId.slice(0, h);
      const fn = this.graph.functions.find((f) => f.name === fnName);
      if (!fn) return false;
      const b = fn.blocks.find((x) => x.id === blockId);
      return b ? !b.reachable : false;
    }

    _targetLabel(id) {
      if (id.endsWith("#entry")) return "入口";
      if (id.endsWith("#exit")) return "出口";
      const h = id.indexOf("#");
      const tail = id.slice(h + 1);
      return tail[0] === "b" ? "B" + tail.slice(1) : tail;
    }

    _drawDeadStub(e) {
      const from = this._centerBottom(e.from);
      if (!from) return null;
      const x1 = from.x - 14, x2 = from.x + 14;
      const d = `M${x1},${from.y} L${x2},${from.y + 11} L${x1},${from.y + 22}`;
      const path = this._mkPath(d, "cfg-e-" + (e.kind === "return" ? "return" : "fall") + " dead", null, true);
      path.dataset.from = e.from;
      path.dataset.to = e.to || "";
      const target = e.kind === "return" ? "出口（不可达）" : this._targetLabel(e.to) + "（不可达）";
      this._mkLabel(from.x + 18, from.y + 12, "✕ " + target, "cfg-lb-dead", "start");
      return path;
    }

    _mkPath(d, cls, marker, dashed) {
      const p = document.createElementNS(this.svgNS, "path");
      p.setAttribute("d", d);
      p.setAttribute("fill", "none");
      p.setAttribute("class", "cfg-edge " + cls + (dashed ? " dashed" : ""));
      if (marker) p.setAttribute("marker-end", "url(#" + marker + ")");
      this.edgeLayer.appendChild(p);
      return p;
    }

    _mkLabel(x, y, text, cls, anchor) {
      if (!text) return null;
      const d = document.createElement("div");
      d.className = "cfg-elabel " + cls;
      d.textContent = text;
      d.style.left = x + "px";
      d.style.top = y + "px";
      const tx = anchor === "end" ? "-100%" : anchor === "middle" ? "-50%" : "0";
      d.style.transform = `translate(${tx}, -50%)`;
      this.labelLayer.appendChild(d);
      return d;
    }

    _drawForwardEdge(e) {
      const from = this._centerBottom(e.from);
      const to = this._centerTop(e.to);
      if (!from || !to) return null;
      const cls = "cfg-e-" + e.kind + (e.unreachable ? " dead" : "");
      const straight = Math.abs(from.x - to.x) < 2;
      let d, lx, ly, anchor;
      if (straight) {
        d = `M${from.x},${from.y} L${to.x},${to.y - 2}`;
        lx = from.x + 7; ly = (from.y + to.y) / 2; anchor = "start";
      } else {
        const dy = Math.max(26, (to.y - from.y) * 0.5);
        d = `M${from.x},${from.y} C${from.x},${from.y + dy} ${to.x},${to.y - dy} ${to.x},${to.y - 2}`;
        lx = (from.x + to.x) / 2; ly = (from.y + to.y) / 2 - 3; anchor = "middle";
      }
      const path = this._mkPath(d, cls, "cfg-arrow-" + e.kind, e.kind === "return");
      this._mkLabel(lx, ly, e.label, "cfg-lb-" + e.kind + (e.unreachable ? " dead" : ""), anchor);
      return path;
    }

    _drawBackEdge(e) {
      // 左侧直角通道：源块底 -> 短下行 -> 水平到通道 -> 上行到目标顶前一层间距 -> 进目标
      const from = this._centerBottom(e.from);
      const to = this._centerTop(e.to);
      if (!from || !to) return null;
      const ch = e._channel || 1;
      const xOut = to.x - 12 - (ch - 1) * 13;
      const yDip = from.y + 14;
      const yRise = to.y - 9;
      const d = `M${from.x},${from.y} ` +
        `L${from.x},${yDip} L${xOut},${yDip} ` +
        `L${xOut},${yRise} L${to.x},${yRise} L${to.x},${to.y - 2}`;
      const path = this._mkPath(d, "cfg-e-back" + (e.unreachable ? " dead" : ""), "cfg-arrow-back", false);
      this._mkLabel(xOut + 5, (yDip + yRise) / 2, e.label || "回边", "cfg-lb-back", "start");
      return path;
    }

    _drawCallEdge(e) {
      const z = this._geo(e.from);
      if (!z) return null;
      const block = this.nodeBlock(e.from);
      const userCalls = block ? block.calls.filter((c) => !c.builtin && !c.indirect && c.name != null) : [];
      const k = Math.max(0, userCalls.findIndex((c) => c.name === e.callee));
      // 起点对齐块底的调用芯片行（与 blockHeight 中 CALL_H 的预留一致）
      const callsTop = z.g.h - LAYOUT.PAD_BOTTOM - (block && block.calls.length ? LAYOUT.CALL_H + 2 : 0);
      const offY = callsTop + LAYOUT.CALL_H / 2;
      const from = { x: z.g.x + z.g.w, y: z.laneY + z.g.y + offY + k * 22 };
      const toZ = this._geo(e.callee + "#b0");
      const to = toZ && { x: toZ.g.x, y: toZ.laneY + toZ.g.y + 8 };

      if (e.recursive || !to) {
        // 递归自环：源块右侧回弧，同块多个递归调用纵向错开
        const d = `M${from.x},${from.y} ` +
          `C${from.x + 54},${from.y - 26} ${from.x + 54},${from.y + 34} ${from.x},${from.y + 24}`;
        const path = this._mkPath(d, "cfg-e-call self", "cfg-arrow-call", true);
        const lb = this._mkLabel(from.x + 58, from.y + 4, "递归调用", "cfg-lb-call", "start");
        if (lb) lb.style.whiteSpace = "nowrap";
        return path;
      }

      // 跨函数调用：右侧汇流通道
      const busX = this.layout.busX;
      const d = `M${from.x},${from.y} L${busX},${from.y} L${busX},${to.y} L${to.x - 2},${to.y}`;
      const path = this._mkPath(d, "cfg-e-call xfn" + (e.unreachable ? " dead" : ""), "cfg-arrow-call", true);
      const lb = this._mkLabel(busX + 4, (from.y + to.y) / 2, "↔ " + e.callee, "cfg-lb-call xfn", "start");
      if (lb) lb.style.whiteSpace = "nowrap";
      return path;
    }

    nodeBlock(id) {
      if (!this.graph) return null;
      const h = id.indexOf("#");
      const fn = this.graph.functions.find((f) => f.name === id.slice(0, h));
      return fn ? fn.blocks.find((b) => b.id === id) || null : null;
    }

    _bindHover() {
      this.nodeLayer.querySelectorAll(".cfg-block").forEach((el) => {
        el.addEventListener("mouseenter", () => {
          const id = el.dataset.blockId;
          this.edgeLayer.querySelectorAll(".cfg-edge").forEach((p) => {
            p.classList.toggle("dim", p.dataset.from !== id && p.dataset.to !== id);
          });
        });
        el.addEventListener("mouseleave", () => {
          this.edgeLayer.querySelectorAll(".cfg-edge.dim").forEach((p) => p.classList.remove("dim"));
        });
      });
      this.edgeLayer.querySelectorAll(".cfg-edge").forEach((p) => {
        p.addEventListener("mouseenter", () => p.classList.add("hot"));
        p.addEventListener("mouseleave", () => p.classList.remove("hot"));
      });
    }
  };

  /* ============================================================
     4. 统计
     ============================================================ */
  function summarize(graph) {
    const s = {
      functions: graph.functions.length, blocks: 0, dead: 0,
      edges: { fall: 0, branch: 0, back: 0, return: 0, call: 0 },
      recursiveFns: [],
    };
    const rec = new Set();
    graph.functions.forEach((fn) => {
      s.blocks += fn.blocks.length;
      fn.blocks.forEach((b) => { if (!b.reachable) s.dead++; });
      fn.edges.forEach((e) => {
        if (s.edges[e.kind] != null) s.edges[e.kind]++;
        if (e.recursive) rec.add(fn.name);
      });
    });
    s.recursiveFns = Array.from(rec);
    return s;
  }

  ML.CFG = { build: buildProgram, layout: layoutProgram, Renderer, summarize, EDGE_KINDS, LAYOUT };
})();
