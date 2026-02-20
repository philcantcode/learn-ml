import { useState, useEffect, useCallback, useRef } from "react";

// -- Computation graph node data for autograd demo --
const GRAPH_STEPS = [
  {
    title: "Forward Pass: Build the Graph",
    desc: "Every operation creates a node. PyTorch records the entire computation.",
    nodes: [
      { id: "a", label: "a = 2.0", x: 80, y: 180, type: "input", active: true },
      { id: "b", label: "b = 3.0", x: 80, y: 320, type: "input", active: true },
      { id: "c", label: "c = a×b = 6", x: 280, y: 250, type: "op", active: true, op: "×" },
      { id: "d", label: "d = c+a = 8", x: 460, y: 200, type: "op", active: true, op: "+" },
      { id: "e", label: "e = d² = 64", x: 640, y: 200, type: "output", active: true, op: "²" },
    ],
    edges: [
      { from: "a", to: "c" }, { from: "b", to: "c" },
      { from: "c", to: "d" }, { from: "a", to: "d" },
      { from: "d", to: "e" },
    ],
    highlightEdges: [],
    gradients: {},
  },
  {
    title: "Backward: Gradient from e to d",
    desc: "e = d², so ∂e/∂d = 2d = 2×8 = 16. Start at the output, flow backward.",
    nodes: [
      { id: "a", label: "a = 2.0", x: 80, y: 180, type: "input", active: false },
      { id: "b", label: "b = 3.0", x: 80, y: 320, type: "input", active: false },
      { id: "c", label: "c = 6", x: 280, y: 250, type: "op", active: false },
      { id: "d", label: "d = 8", x: 460, y: 200, type: "op", active: true },
      { id: "e", label: "e = 64", x: 640, y: 200, type: "output", active: true },
    ],
    edges: [
      { from: "a", to: "c" }, { from: "b", to: "c" },
      { from: "c", to: "d" }, { from: "a", to: "d" },
      { from: "d", to: "e" },
    ],
    highlightEdges: [{ from: "d", to: "e" }],
    gradients: { e: "1", d: "16" },
  },
  {
    title: "Backward: Gradient from d to c and a",
    desc: "d = c + a, so ∂d/∂c = 1 and ∂d/∂a = 1. Multiply by upstream gradient (16). Chain rule!",
    nodes: [
      { id: "a", label: "a = 2.0", x: 80, y: 180, type: "input", active: true },
      { id: "b", label: "b = 3.0", x: 80, y: 320, type: "input", active: false },
      { id: "c", label: "c = 6", x: 280, y: 250, type: "op", active: true },
      { id: "d", label: "d = 8", x: 460, y: 200, type: "op", active: true },
      { id: "e", label: "e = 64", x: 640, y: 200, type: "output", active: false },
    ],
    edges: [
      { from: "a", to: "c" }, { from: "b", to: "c" },
      { from: "c", to: "d" }, { from: "a", to: "d" },
      { from: "d", to: "e" },
    ],
    highlightEdges: [{ from: "c", to: "d" }, { from: "a", to: "d" }],
    gradients: { d: "16", c: "16", a: "16 (partial)" },
  },
  {
    title: "Backward: Gradient from c to a and b",
    desc: "c = a×b, so ∂c/∂a = b = 3, ∂c/∂b = a = 2. Multiply by upstream (16). Then a accumulates both paths!",
    nodes: [
      { id: "a", label: "a = 2.0", x: 80, y: 180, type: "input", active: true },
      { id: "b", label: "b = 3.0", x: 80, y: 320, type: "input", active: true },
      { id: "c", label: "c = 6", x: 280, y: 250, type: "op", active: true },
      { id: "d", label: "d = 8", x: 460, y: 200, type: "op", active: false },
      { id: "e", label: "e = 64", x: 640, y: 200, type: "output", active: false },
    ],
    edges: [
      { from: "a", to: "c" }, { from: "b", to: "c" },
      { from: "c", to: "d" }, { from: "a", to: "d" },
      { from: "d", to: "e" },
    ],
    highlightEdges: [{ from: "a", to: "c" }, { from: "b", to: "c" }],
    gradients: { c: "16", a: "16 + 48 = 64", b: "32" },
  },
  {
    title: "Done! All gradients computed.",
    desc: "∂e/∂a = 64, ∂e/∂b = 32. This is what loss.backward() does for your entire network — automatically.",
    nodes: [
      { id: "a", label: "∂e/∂a = 64", x: 80, y: 180, type: "input", active: true },
      { id: "b", label: "∂e/∂b = 32", x: 80, y: 320, type: "input", active: true },
      { id: "c", label: "c = 6", x: 280, y: 250, type: "op", active: true },
      { id: "d", label: "d = 8", x: 460, y: 200, type: "op", active: true },
      { id: "e", label: "e = 64", x: 640, y: 200, type: "output", active: true },
    ],
    edges: [
      { from: "a", to: "c" }, { from: "b", to: "c" },
      { from: "c", to: "d" }, { from: "a", to: "d" },
      { from: "d", to: "e" },
    ],
    highlightEdges: [
      { from: "a", to: "c" }, { from: "b", to: "c" },
      { from: "c", to: "d" }, { from: "a", to: "d" },
      { from: "d", to: "e" },
    ],
    gradients: { a: "64 ✓", b: "32 ✓" },
  },
];

// -- Comparison data: from-scratch vs pytorch --
const COMPARISON = [
  { concept: "Data", scratch: "np.array", pytorch: "torch.tensor", note: "Tensors track gradients" },
  { concept: "Layer", scratch: "W = he_init(16,2)\nb = np.zeros((16,1))", pytorch: "nn.Linear(2, 16)", note: "Weights + bias + init" },
  { concept: "Forward", scratch: "z1 = W1 @ X + b1\na1 = relu(z1)\nz2 = W2 @ a1 + b2\na2 = relu(z2)\nz3 = W3 @ a2 + b3\na3 = sigmoid(z3)", pytorch: "x = self.relu(self.layer1(x))\nx = self.relu(self.layer2(x))\nx = self.sigmoid(self.layer3(x))", note: "Same structure, cleaner" },
  { concept: "Loss", scratch: "-np.mean(\n  y*np.log(a3+eps) +\n  (1-y)*np.log(1-a3+eps)\n)", pytorch: "nn.BCELoss()(pred, y)", note: "Built-in, numerically stable" },
  { concept: "Backward", scratch: "dz3 = a3 - yb\ndW3 = (1/m) * dz3 @ a2.T\ndb3 = (1/m) * np.sum(dz3)\nda2 = W3.T @ dz3\ndz2 = da2 * relu_deriv(z2)\ndW2 = (1/m) * dz2 @ a1.T\ndb2 = (1/m) * np.sum(dz2)\nda1 = W2.T @ dz2\ndz1 = da1 * relu_deriv(z1)\ndW1 = (1/m) * dz1 @ Xb.T\ndb1 = (1/m) * np.sum(dz1)", pytorch: "loss.backward()", note: "ONE LINE replaces all that chain rule" },
  { concept: "Update", scratch: "W3 -= lr * dW3\nb3 -= lr * db3\nW2 -= lr * dW2\nb2 -= lr * db2\nW1 -= lr * dW1\nb1 -= lr * db1", pytorch: "optimizer.step()\noptimizer.zero_grad()", note: "Handles all params + Adam/etc" },
];

const COLORS = {
  bg: "#0a0f1a",
  surface: "#111827",
  surfaceLight: "#1a2234",
  border: "#2a3650",
  borderLight: "#3b4d6b",
  text: "#e2e8f0",
  textDim: "#8899b4",
  accent: "#f59e0b",
  accentDim: "#b45309",
  green: "#34d399",
  greenDim: "#065f46",
  red: "#f87171",
  blue: "#60a5fa",
  purple: "#a78bfa",
  orange: "#fb923c",
};

function ComputationGraph({ step }) {
  const s = GRAPH_STEPS[step];
  const canvasW = 760;
  const canvasH = 440;

  const isHighlighted = (from, to) =>
    s.highlightEdges.some(e => e.from === from && e.to === to);

  const getNodePos = (id) => {
    const n = s.nodes.find(n => n.id === id);
    return n ? { x: n.x, y: n.y } : { x: 0, y: 0 };
  };

  return (
    <div style={{ position: "relative", width: canvasW, height: canvasH, margin: "0 auto" }}>
      <svg width={canvasW} height={canvasH} style={{ position: "absolute", top: 0, left: 0 }}>
        {s.edges.map((e, i) => {
          const from = getNodePos(e.from);
          const to = getNodePos(e.to);
          const hl = isHighlighted(e.from, e.to);
          return (
            <line
              key={i}
              x1={from.x} y1={from.y} x2={to.x} y2={to.y}
              stroke={hl ? COLORS.accent : COLORS.borderLight}
              strokeWidth={hl ? 3 : 1.5}
              strokeDasharray={hl ? "none" : "6 4"}
              style={{ transition: "all 0.4s ease" }}
            />
          );
        })}
        {s.edges.filter(e => isHighlighted(e.from, e.to)).map((e, i) => {
          const from = getNodePos(e.from);
          const to = getNodePos(e.to);
          const mx = (from.x + to.x) / 2;
          const my = (from.y + to.y) / 2;
          const angle = Math.atan2(from.y - to.y, from.x - to.x);
          return (
            <polygon
              key={`arrow-${i}`}
              points="-8,-5 0,0 -8,5"
              fill={COLORS.accent}
              transform={`translate(${mx},${my}) rotate(${angle * 180 / Math.PI})`}
            />
          );
        })}
      </svg>
      {s.nodes.map(n => {
        const colors = {
          input: { bg: COLORS.blue, border: "#93c5fd" },
          op: { bg: COLORS.purple, border: "#c4b5fd" },
          output: { bg: COLORS.accent, border: "#fbbf24" },
        };
        const c = colors[n.type];
        return (
          <div
            key={n.id}
            style={{
              position: "absolute",
              left: n.x - 52,
              top: n.y - 22,
              width: 104,
              padding: "6px 4px",
              background: n.active ? c.bg + "22" : COLORS.surface,
              border: `2px solid ${n.active ? c.bg : COLORS.border}`,
              borderRadius: 10,
              textAlign: "center",
              fontSize: 12,
              fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
              color: n.active ? COLORS.text : COLORS.textDim,
              transition: "all 0.4s ease",
              zIndex: 2,
            }}
          >
            {n.label}
            {s.gradients[n.id] && (
              <div style={{
                position: "absolute",
                top: -28,
                left: "50%",
                transform: "translateX(-50%)",
                background: COLORS.accent,
                color: "#000",
                padding: "2px 8px",
                borderRadius: 6,
                fontSize: 10,
                fontWeight: 700,
                whiteSpace: "nowrap",
              }}>
                grad: {s.gradients[n.id]}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function CodeComparison({ item, isExpanded, onToggle }) {
  return (
    <div
      onClick={onToggle}
      style={{
        background: COLORS.surfaceLight,
        borderRadius: 12,
        border: `1px solid ${isExpanded ? COLORS.accent : COLORS.border}`,
        overflow: "hidden",
        cursor: "pointer",
        transition: "border-color 0.3s ease",
        marginBottom: 8,
      }}
    >
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "14px 18px",
      }}>
        <span style={{ color: COLORS.text, fontWeight: 600, fontSize: 15 }}>{item.concept}</span>
        <span style={{
          color: COLORS.textDim, fontSize: 12,
          transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)",
          transition: "transform 0.2s ease",
        }}>▼</span>
      </div>
      {isExpanded && (
        <div style={{ padding: "0 18px 16px" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 10 }}>
            <div>
              <div style={{ fontSize: 10, color: COLORS.red, fontWeight: 700, marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>
                From Scratch
              </div>
              <pre style={{
                background: COLORS.bg,
                padding: 12,
                borderRadius: 8,
                fontSize: 11,
                lineHeight: 1.5,
                color: COLORS.textDim,
                margin: 0,
                overflowX: "auto",
                fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                whiteSpace: "pre-wrap",
                border: `1px solid ${COLORS.border}`,
              }}>
                {item.scratch}
              </pre>
            </div>
            <div>
              <div style={{ fontSize: 10, color: COLORS.green, fontWeight: 700, marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>
                PyTorch
              </div>
              <pre style={{
                background: COLORS.bg,
                padding: 12,
                borderRadius: 8,
                fontSize: 11,
                lineHeight: 1.5,
                color: COLORS.green,
                margin: 0,
                overflowX: "auto",
                fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                whiteSpace: "pre-wrap",
                border: `1px solid ${COLORS.greenDim}`,
              }}>
                {item.pytorch}
              </pre>
            </div>
          </div>
          <div style={{
            fontSize: 12,
            color: COLORS.accent,
            fontStyle: "italic",
            paddingLeft: 4,
          }}>
            → {item.note}
          </div>
        </div>
      )}
    </div>
  );
}

function TrainingLoopAnim() {
  const [phase, setPhase] = useState(0);
  const phases = [
    { label: "Forward", code: "predictions = model(X_batch)", color: COLORS.blue, icon: "→", desc: "Data flows through layers" },
    { label: "Loss", code: "loss = loss_fn(predictions, y)", color: COLORS.red, icon: "△", desc: "Measure how wrong" },
    { label: "Backward", code: "loss.backward()", color: COLORS.accent, icon: "←", desc: "Compute ALL gradients" },
    { label: "Update", code: "optimizer.step()", color: COLORS.green, icon: "↻", desc: "Adjust weights" },
    { label: "Zero Grad", code: "optimizer.zero_grad()", color: COLORS.purple, icon: "⊘", desc: "Reset for next batch" },
  ];

  useEffect(() => {
    const timer = setInterval(() => setPhase(p => (p + 1) % 5), 2000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div style={{ display: "flex", gap: 6, alignItems: "stretch" }}>
      {phases.map((p, i) => (
        <div
          key={i}
          onClick={() => setPhase(i)}
          style={{
            flex: 1,
            padding: "14px 8px",
            background: phase === i ? p.color + "20" : COLORS.surface,
            border: `2px solid ${phase === i ? p.color : COLORS.border}`,
            borderRadius: 10,
            textAlign: "center",
            cursor: "pointer",
            transition: "all 0.3s ease",
            display: "flex",
            flexDirection: "column",
            gap: 6,
          }}
        >
          <div style={{ fontSize: 24, filter: phase === i ? "none" : "grayscale(1) opacity(0.4)" }}>
            {p.icon}
          </div>
          <div style={{
            fontSize: 12,
            fontWeight: 700,
            color: phase === i ? p.color : COLORS.textDim,
          }}>
            {p.label}
          </div>
          {phase === i && (
            <>
              <code style={{
                fontSize: 10,
                color: p.color,
                fontFamily: "'JetBrains Mono', monospace",
                wordBreak: "break-all",
              }}>
                {p.code}
              </code>
              <div style={{ fontSize: 10, color: COLORS.textDim }}>{p.desc}</div>
            </>
          )}
        </div>
      ))}
    </div>
  );
}

export default function PyTorchLesson() {
  const [graphStep, setGraphStep] = useState(0);
  const [expandedItem, setExpandedItem] = useState(4); // Start with backward expanded
  const [tab, setTab] = useState("autograd");

  const tabs = [
    { id: "autograd", label: "Autograd", icon: "⟲" },
    { id: "compare", label: "Code Comparison", icon: "⟺" },
    { id: "loop", label: "Training Loop", icon: "↻" },
  ];

  return (
    <div style={{
      background: COLORS.bg,
      minHeight: "100vh",
      color: COLORS.text,
      fontFamily: "'Instrument Sans', 'DM Sans', system-ui, sans-serif",
    }}>
      {/* Header */}
      <div style={{
        background: `linear-gradient(135deg, ${COLORS.surface} 0%, ${COLORS.bg} 100%)`,
        borderBottom: `1px solid ${COLORS.border}`,
        padding: "32px 24px 24px",
      }}>
        <div style={{ maxWidth: 800, margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
            <span style={{
              background: COLORS.accent,
              color: "#000",
              fontSize: 10,
              fontWeight: 800,
              padding: "3px 10px",
              borderRadius: 20,
              letterSpacing: 1,
            }}>LESSON 5</span>
            <span style={{ color: COLORS.textDim, fontSize: 13 }}>Stage 2: Deep Learning Fundamentals</span>
          </div>
          <h1 style={{
            fontSize: 28,
            fontWeight: 800,
            margin: "8px 0 6px",
            background: `linear-gradient(90deg, ${COLORS.text}, ${COLORS.accent})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            letterSpacing: "-0.5px",
          }}>
            From Scratch to PyTorch
          </h1>
          <p style={{ color: COLORS.textDim, fontSize: 14, margin: 0, lineHeight: 1.6 }}>
            You built backpropagation by hand. Now see how PyTorch does it in <code style={{
              background: COLORS.accent + "22",
              color: COLORS.accent,
              padding: "2px 6px",
              borderRadius: 4,
              fontSize: 13,
            }}>one line</code>.
          </p>
        </div>
      </div>

      {/* Key Insight Banner */}
      <div style={{ maxWidth: 800, margin: "20px auto 0", padding: "0 24px" }}>
        <div style={{
          background: COLORS.accent + "10",
          border: `1px solid ${COLORS.accent}40`,
          borderRadius: 12,
          padding: "16px 20px",
          display: "flex",
          gap: 14,
          alignItems: "flex-start",
        }}>
          <span style={{ fontSize: 22, marginTop: 2 }}>💡</span>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.accent, marginBottom: 4 }}>
              The Key Insight
            </div>
            <div style={{ fontSize: 13, color: COLORS.textDim, lineHeight: 1.6 }}>
              PyTorch records every operation in your forward pass as a <strong style={{ color: COLORS.text }}>computation graph</strong>. 
              When you call <code style={{ color: COLORS.accent }}>loss.backward()</code>, it walks this graph in reverse, 
              applying the chain rule at each node — exactly like you did manually, but for <em>any</em> computation, no matter how complex.
            </div>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div style={{ maxWidth: 800, margin: "20px auto 0", padding: "0 24px" }}>
        <div style={{ display: "flex", gap: 4, background: COLORS.surface, borderRadius: 10, padding: 4 }}>
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                flex: 1,
                padding: "10px 12px",
                background: tab === t.id ? COLORS.surfaceLight : "transparent",
                border: tab === t.id ? `1px solid ${COLORS.borderLight}` : "1px solid transparent",
                borderRadius: 8,
                color: tab === t.id ? COLORS.text : COLORS.textDim,
                fontSize: 13,
                fontWeight: tab === t.id ? 700 : 500,
                cursor: "pointer",
                transition: "all 0.2s ease",
                fontFamily: "inherit",
              }}
            >
              {t.icon} {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div style={{ maxWidth: 800, margin: "16px auto 0", padding: "0 24px 60px" }}>
        
        {/* AUTOGRAD TAB */}
        {tab === "autograd" && (
          <div>
            <div style={{
              background: COLORS.surface,
              borderRadius: 14,
              border: `1px solid ${COLORS.border}`,
              overflow: "hidden",
            }}>
              <div style={{ padding: "18px 20px 10px" }}>
                <h3 style={{ margin: "0 0 4px", fontSize: 16, color: COLORS.accent }}>
                  {GRAPH_STEPS[graphStep].title}
                </h3>
                <p style={{ margin: 0, fontSize: 13, color: COLORS.textDim, lineHeight: 1.5 }}>
                  {GRAPH_STEPS[graphStep].desc}
                </p>
              </div>

              <div style={{
                background: COLORS.bg,
                margin: "0 12px",
                borderRadius: 10,
                padding: "8px 0",
                overflow: "hidden",
              }}>
                <ComputationGraph step={graphStep} />
              </div>

              {/* Step controls */}
              <div style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "14px 20px 16px",
              }}>
                <button
                  onClick={() => setGraphStep(Math.max(0, graphStep - 1))}
                  disabled={graphStep === 0}
                  style={{
                    background: graphStep === 0 ? COLORS.surface : COLORS.surfaceLight,
                    border: `1px solid ${COLORS.border}`,
                    color: graphStep === 0 ? COLORS.textDim : COLORS.text,
                    padding: "8px 18px",
                    borderRadius: 8,
                    cursor: graphStep === 0 ? "default" : "pointer",
                    fontSize: 13,
                    fontFamily: "inherit",
                  }}
                >
                  ← Back
                </button>
                <div style={{ display: "flex", gap: 6 }}>
                  {GRAPH_STEPS.map((_, i) => (
                    <div
                      key={i}
                      onClick={() => setGraphStep(i)}
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: "50%",
                        background: i === graphStep ? COLORS.accent : i < graphStep ? COLORS.accentDim : COLORS.border,
                        cursor: "pointer",
                        transition: "all 0.2s ease",
                      }}
                    />
                  ))}
                </div>
                <button
                  onClick={() => setGraphStep(Math.min(GRAPH_STEPS.length - 1, graphStep + 1))}
                  disabled={graphStep === GRAPH_STEPS.length - 1}
                  style={{
                    background: graphStep === GRAPH_STEPS.length - 1 ? COLORS.surface : COLORS.accent,
                    border: "none",
                    color: graphStep === GRAPH_STEPS.length - 1 ? COLORS.textDim : "#000",
                    padding: "8px 18px",
                    borderRadius: 8,
                    cursor: graphStep === GRAPH_STEPS.length - 1 ? "default" : "pointer",
                    fontSize: 13,
                    fontWeight: 700,
                    fontFamily: "inherit",
                  }}
                >
                  Next →
                </button>
              </div>
            </div>

            <div style={{
              marginTop: 16,
              background: COLORS.surface,
              borderRadius: 12,
              border: `1px solid ${COLORS.border}`,
              padding: "16px 20px",
            }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.green, marginBottom: 8 }}>
                What this means for your neural network:
              </div>
              <div style={{ fontSize: 13, color: COLORS.textDim, lineHeight: 1.7 }}>
                Your from-scratch backward pass had <strong style={{ color: COLORS.red }}>11 gradient variables</strong> (dz3, dW3, db3, da2, dz2, dW2, db2, da1, dz1, dW1, db1) — 
                each computed manually with the chain rule. With PyTorch, you define the forward pass and call <code style={{ color: COLORS.accent }}>loss.backward()</code>. 
                The framework traces through every matmul, ReLU, and sigmoid, and computes all those gradients automatically. 
                Add a layer? Change an activation? The gradients update themselves.
              </div>
            </div>
          </div>
        )}

        {/* COMPARE TAB */}
        {tab === "compare" && (
          <div>
            <div style={{
              fontSize: 13,
              color: COLORS.textDim,
              marginBottom: 14,
              lineHeight: 1.6,
            }}>
              Click each row to see the from-scratch code next to its PyTorch equivalent. 
              The <strong style={{ color: COLORS.accent }}>Backward</strong> row is the most dramatic — expand it to see why people use frameworks.
            </div>
            {COMPARISON.map((item, i) => (
              <CodeComparison
                key={i}
                item={item}
                isExpanded={expandedItem === i}
                onToggle={() => setExpandedItem(expandedItem === i ? -1 : i)}
              />
            ))}
            <div style={{
              marginTop: 16,
              background: COLORS.greenDim + "40",
              border: `1px solid ${COLORS.green}40`,
              borderRadius: 12,
              padding: "16px 20px",
              fontSize: 13,
              lineHeight: 1.6,
            }}>
              <strong style={{ color: COLORS.green }}>The core training loop in PyTorch is 5 lines.</strong>{" "}
              <span style={{ color: COLORS.textDim }}>
                But because you implemented it from scratch first, you know exactly what each line is doing under the hood. 
                That understanding is what separates someone who can debug training issues from someone who just copies boilerplate.
              </span>
            </div>
          </div>
        )}

        {/* TRAINING LOOP TAB */}
        {tab === "loop" && (
          <div>
            <div style={{
              background: COLORS.surface,
              borderRadius: 14,
              border: `1px solid ${COLORS.border}`,
              padding: 20,
              marginBottom: 16,
            }}>
              <h3 style={{ margin: "0 0 14px", fontSize: 16, color: COLORS.text }}>
                The PyTorch Training Loop
              </h3>
              <TrainingLoopAnim />
              <div style={{
                marginTop: 14,
                fontSize: 12,
                color: COLORS.textDim,
                textAlign: "center",
              }}>
                Click any phase or watch it cycle. This same 5-step pattern trains everything from MNIST to GPT.
              </div>
            </div>

            {/* The complete code */}
            <div style={{
              background: COLORS.surface,
              borderRadius: 14,
              border: `1px solid ${COLORS.border}`,
              padding: 20,
            }}>
              <h3 style={{ margin: "0 0 12px", fontSize: 15, color: COLORS.text }}>
                Complete PyTorch Training Loop
              </h3>
              <pre style={{
                background: COLORS.bg,
                padding: 16,
                borderRadius: 10,
                fontSize: 12,
                lineHeight: 1.7,
                overflowX: "auto",
                fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                border: `1px solid ${COLORS.border}`,
                margin: 0,
              }}>
                <span style={{ color: COLORS.textDim }}>{"# Define model, loss, optimizer\n"}</span>
                <span style={{ color: COLORS.purple }}>model</span>{" = CircleNet()\n"}
                <span style={{ color: COLORS.purple }}>loss_fn</span>{" = nn.BCELoss()\n"}
                <span style={{ color: COLORS.purple }}>optimizer</span>{" = torch.optim.Adam(model.parameters(), lr="}<span style={{ color: COLORS.accent }}>0.01</span>{")\n\n"}
                <span style={{ color: COLORS.textDim }}>{"# Training loop\n"}</span>
                <span style={{ color: COLORS.orange }}>for</span>{" epoch "}<span style={{ color: COLORS.orange }}>in</span>{" range(num_epochs):\n"}
                {"    "}<span style={{ color: COLORS.orange }}>for</span>{" X_batch, y_batch "}<span style={{ color: COLORS.orange }}>in</span>{" loader:\n"}
                {"        "}
                <span style={{ color: COLORS.blue }}>predictions = model(X_batch)</span>{"     "}<span style={{ color: COLORS.textDim }}>{"# forward\n"}</span>
                {"        "}
                <span style={{ color: COLORS.red }}>loss = loss_fn(predictions, y_batch)</span><span style={{ color: COLORS.textDim }}>{"  # loss\n"}</span>
                {"        "}
                <span style={{ color: COLORS.accent }}>loss.backward()</span>{"                     "}<span style={{ color: COLORS.textDim }}>{"# backward\n"}</span>
                {"        "}
                <span style={{ color: COLORS.green }}>optimizer.step()</span>{"                    "}<span style={{ color: COLORS.textDim }}>{"# update\n"}</span>
                {"        "}
                <span style={{ color: COLORS.purple }}>optimizer.zero_grad()</span>{"               "}<span style={{ color: COLORS.textDim }}>{"# reset"}</span>
              </pre>
              <div style={{
                marginTop: 14,
                padding: "12px 16px",
                background: COLORS.accent + "10",
                borderRadius: 8,
                border: `1px solid ${COLORS.accent}30`,
              }}>
                <div style={{ fontSize: 13, color: COLORS.text, lineHeight: 1.6 }}>
                  <strong style={{ color: COLORS.accent }}>This exact pattern scales to any model.</strong>{" "}
                  Swap <code>CircleNet</code> for a ResNet, a transformer, or GPT — the loop stays the same. 
                  Only the model architecture and data loading change.
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
