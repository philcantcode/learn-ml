import { useState, useEffect, useRef } from "react";

const C = {
  bg: "#080c14",
  surface: "#101729",
  surfaceLight: "#182040",
  border: "#253058",
  borderLight: "#3a4d7a",
  text: "#e4e8f4",
  textDim: "#7888b0",
  accent: "#f76e3c",
  accentDim: "#a84a1e",
  green: "#2ee8a8",
  greenDim: "#0d6b4a",
  red: "#ef5f5f",
  redDim: "#7a2020",
  blue: "#5ba0f8",
  purple: "#a07afa",
  cyan: "#38ddf8",
  yellow: "#f5c842",
};

// Simulate gradient magnitude through layers
function computeGradients(numLayers, hasSkip, gradDecay = 0.72) {
  const grads = [];
  let g = 1.0;
  for (let i = 0; i < numLayers; i++) {
    grads.push(g);
    if (hasSkip) {
      // Skip connection adds a direct path: gradient = decay*g + skip_g
      // The skip path preserves a portion of the gradient
      g = gradDecay * g + 0.35;
      g = Math.min(g, 1.0);
    } else {
      g *= gradDecay;
    }
  }
  return grads.reverse(); // reverse so index 0 = first layer (farthest from loss)
}

function GradientFlowViz() {
  const [numLayers, setNumLayers] = useState(12);
  const [showSkip, setShowSkip] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animFrame, setAnimFrame] = useState(-1);
  const timerRef = useRef(null);

  const gradsNoSkip = computeGradients(numLayers, false);
  const gradsSkip = computeGradients(numLayers, true);
  const grads = showSkip ? gradsSkip : gradsNoSkip;

  useEffect(() => {
    if (isAnimating) {
      setAnimFrame(numLayers - 1);
      let frame = numLayers - 1;
      timerRef.current = setInterval(() => {
        frame--;
        if (frame < 0) {
          setIsAnimating(false);
          clearInterval(timerRef.current);
          return;
        }
        setAnimFrame(frame);
      }, 200);
    }
    return () => clearInterval(timerRef.current);
  }, [isAnimating, numLayers]);

  const startAnim = () => {
    setAnimFrame(numLayers);
    setIsAnimating(true);
  };

  const barWidth = Math.min(42, 600 / numLayers);

  return (
    <div>
      {/* Controls */}
      <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap", alignItems: "center" }}>
        <div style={{ display: "flex", gap: 6 }}>
          <button
            onClick={() => setShowSkip(false)}
            style={{
              padding: "8px 16px",
              background: !showSkip ? C.red + "20" : C.surface,
              border: `1.5px solid ${!showSkip ? C.red : C.border}`,
              borderRadius: 8,
              color: !showSkip ? C.red : C.textDim,
              fontSize: 12, fontWeight: !showSkip ? 700 : 500,
              cursor: "pointer", fontFamily: "inherit",
            }}
          >
            No Skip Connections
          </button>
          <button
            onClick={() => setShowSkip(true)}
            style={{
              padding: "8px 16px",
              background: showSkip ? C.green + "20" : C.surface,
              border: `1.5px solid ${showSkip ? C.green : C.border}`,
              borderRadius: 8,
              color: showSkip ? C.green : C.textDim,
              fontSize: 12, fontWeight: showSkip ? 700 : 500,
              cursor: "pointer", fontFamily: "inherit",
            }}
          >
            With Skip Connections
          </button>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 11, color: C.textDim }}>Layers:</span>
          <input
            type="range" min="4" max="30" value={numLayers}
            onChange={e => setNumLayers(parseInt(e.target.value))}
            style={{ width: 100, accentColor: C.accent }}
          />
          <span style={{ fontSize: 12, color: C.accent, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>{numLayers}</span>
        </div>

        <button
          onClick={startAnim}
          style={{
            padding: "8px 16px",
            background: C.accent,
            border: "none",
            borderRadius: 8,
            color: "#000",
            fontSize: 12, fontWeight: 700,
            cursor: "pointer", fontFamily: "inherit",
          }}
        >
          ▶ Animate Backward Pass
        </button>
      </div>

      {/* Gradient bars */}
      <div style={{
        background: C.surface,
        borderRadius: 12,
        border: `1px solid ${C.border}`,
        padding: "16px 16px 12px",
      }}>
        <div style={{ fontSize: 10, color: C.textDim, marginBottom: 8, display: "flex", justifyContent: "space-between" }}>
          <span>Layer 1 (first layer — farthest from loss)</span>
          <span>Layer {numLayers} (last layer — closest to loss)</span>
        </div>

        <div style={{ display: "flex", gap: 2, alignItems: "flex-end", height: 140, marginBottom: 8 }}>
          {grads.map((g, i) => {
            const isReached = animFrame < 0 || i >= animFrame;
            const color = showSkip ? C.green : C.red;
            const height = Math.max(g * 130, 2);
            const opacity = isReached ? 1 : 0.15;

            return (
              <div key={i} style={{
                width: barWidth,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 2,
              }}>
                <div style={{
                  fontSize: 8,
                  color: C.textDim,
                  opacity: (i === 0 || i === numLayers - 1 || g < 0.05) ? 1 : 0,
                }}>
                  {g < 0.01 ? "≈0" : g.toFixed(2)}
                </div>
                <div style={{
                  width: barWidth - 4,
                  height: height,
                  background: `linear-gradient(to top, ${color}, ${color}88)`,
                  borderRadius: "4px 4px 0 0",
                  opacity,
                  transition: "opacity 0.15s ease, height 0.3s ease",
                }} />
              </div>
            );
          })}
        </div>

        <div style={{ fontSize: 10, color: C.textDim, textAlign: "center" }}>
          ← Gradient magnitude at each layer (taller = stronger gradient = faster learning) →
        </div>
      </div>

      {/* Diagnosis */}
      <div style={{
        marginTop: 12,
        padding: "12px 16px",
        borderRadius: 10,
        background: showSkip ? C.greenDim + "30" : C.redDim + "30",
        border: `1px solid ${showSkip ? C.green : C.red}30`,
      }}>
        {showSkip ? (
          <div style={{ fontSize: 13, color: C.textDim, lineHeight: 1.6 }}>
            <strong style={{ color: C.green }}>Gradients stay healthy.</strong> The skip connections provide an
            alternative path for gradients to flow — even through {numLayers} layers, the first layer
            gets a gradient of <strong style={{ color: C.green }}>{grads[0].toFixed(2)}</strong>.
            Every layer can learn effectively.
          </div>
        ) : (
          <div style={{ fontSize: 13, color: C.textDim, lineHeight: 1.6 }}>
            <strong style={{ color: C.red }}>Vanishing gradients.</strong> By layer 1, the gradient has decayed to{" "}
            <strong style={{ color: C.red }}>{grads[0] < 0.01 ? "nearly zero" : grads[0].toFixed(3)}</strong>.
            {numLayers > 10 && " The first layers barely update — the network can't learn deep features."}
            {numLayers > 20 && " This is why networks deeper than ~20 layers were impossible before ResNets."}
          </div>
        )}
      </div>
    </div>
  );
}

function ResidualBlockDiagram() {
  const [showMath, setShowMath] = useState(false);

  return (
    <div>
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start" }}>
        {/* Visual diagram */}
        <svg width={400} height={260} style={{ flex: "0 0 400px" }}>
          {/* Main path */}
          <line x1={60} y1={40} x2={60} y2={220} stroke={C.blue} strokeWidth={2} />
          <rect x={20} y={60} width={80} height={34} rx={8} fill={C.blue + "20"} stroke={C.blue} strokeWidth={1.5} />
          <text x={60} y={82} textAnchor="middle" fill={C.blue} fontSize={11} fontWeight={600} fontFamily="'JetBrains Mono', monospace">Conv</text>
          <rect x={20} y={105} width={80} height={26} rx={8} fill={C.purple + "15"} stroke={C.purple} strokeWidth={1} />
          <text x={60} y={122} textAnchor="middle" fill={C.purple} fontSize={10} fontFamily="'JetBrains Mono', monospace">BatchNorm</text>
          <rect x={20} y={142} width={80} height={26} rx={8} fill={C.accent + "15"} stroke={C.accent} strokeWidth={1} />
          <text x={60} y={159} textAnchor="middle" fill={C.accent} fontSize={10} fontFamily="'JetBrains Mono', monospace">ReLU</text>
          <rect x={20} y={179} width={80} height={34} rx={8} fill={C.blue + "20"} stroke={C.blue} strokeWidth={1.5} />
          <text x={60} y={201} textAnchor="middle" fill={C.blue} fontSize={11} fontWeight={600} fontFamily="'JetBrains Mono', monospace">Conv</text>

          {/* Skip connection */}
          <path d={`M 100 40 Q 200 40 200 130 Q 200 220 100 220`} fill="none" stroke={C.green} strokeWidth={2.5} strokeDasharray="6 3" />
          <text x={215} y={130} fill={C.green} fontSize={12} fontWeight={700} fontFamily="'JetBrains Mono', monospace">skip</text>
          <text x={215} y={146} fill={C.green} fontSize={10} fontFamily="'JetBrains Mono', monospace">identity</text>

          {/* Addition */}
          <circle cx={60} cy={220} r={14} fill={C.green + "20"} stroke={C.green} strokeWidth={2} />
          <text x={60} y={225} textAnchor="middle" fill={C.green} fontSize={16} fontWeight={700}>+</text>

          {/* Labels */}
          <text x={60} y={30} textAnchor="middle" fill={C.text} fontSize={12} fontWeight={600}>input x</text>
          <text x={60} y={255} textAnchor="middle" fill={C.text} fontSize={12} fontWeight={600}>output = F(x) + x</text>

          {/* Arrow heads */}
          <polygon points="55,48 60,58 65,48" fill={C.blue} />
          <polygon points="55,235 60,245 65,235" fill={C.text} />
        </svg>

        {/* Explanation */}
        <div style={{ flex: "1 1 280px", minWidth: 250 }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: C.text, marginBottom: 10 }}>
            The Residual Block
          </div>
          <div style={{ fontSize: 13, color: C.textDim, lineHeight: 1.8, marginBottom: 12 }}>
            Instead of learning the output directly, the layers learn the <strong style={{ color: C.green }}>residual</strong> — the <em>difference</em> between input and desired output.
          </div>
          <div style={{ fontSize: 13, color: C.textDim, lineHeight: 1.8, marginBottom: 12 }}>
            <strong style={{ color: C.blue }}>Without skip:</strong>{" "}
            output = F(x) — the layers must learn the entire transformation.
          </div>
          <div style={{ fontSize: 13, color: C.textDim, lineHeight: 1.8, marginBottom: 16 }}>
            <strong style={{ color: C.green }}>With skip:</strong>{" "}
            output = F(x) + x — the layers only need to learn what to <em>change</em>. If a layer has nothing to contribute, it can just output zeros and the input passes through.
          </div>

          <button
            onClick={() => setShowMath(!showMath)}
            style={{
              padding: "8px 14px",
              background: showMath ? C.surfaceLight : C.surface,
              border: `1px solid ${C.border}`,
              borderRadius: 8,
              color: C.textDim,
              fontSize: 12,
              cursor: "pointer",
              fontFamily: "inherit",
            }}
          >
            {showMath ? "Hide" : "Show"} gradient math
          </button>

          {showMath && (
            <div style={{
              marginTop: 10,
              padding: "12px 14px",
              background: C.bg,
              borderRadius: 8,
              border: `1px solid ${C.border}`,
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 12,
              lineHeight: 2,
            }}>
              <div style={{ color: C.textDim }}>
                <span style={{ color: C.red }}>Without skip:</span>
              </div>
              <div style={{ color: C.text, marginLeft: 8 }}>
                ∂L/∂x = ∂L/∂F · ∂F/∂x
              </div>
              <div style={{ color: C.textDim, marginLeft: 8, fontSize: 10 }}>
                (gradient must flow through F — can vanish)
              </div>
              <div style={{ color: C.textDim, marginTop: 8 }}>
                <span style={{ color: C.green }}>With skip:</span>
              </div>
              <div style={{ color: C.text, marginLeft: 8 }}>
                ∂L/∂x = ∂L/∂F · ∂F/∂x + <span style={{ color: C.green, fontWeight: 700 }}>∂L/∂x</span>
              </div>
              <div style={{ color: C.green, marginLeft: 8, fontSize: 10 }}>
                (the +1 gradient from identity ALWAYS flows through!)
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function HistoryTimeline() {
  const milestones = [
    { year: "2012", name: "AlexNet", layers: 8, color: C.blue, note: "Won ImageNet by a huge margin. Started the deep learning revolution.", err: "16.4%" },
    { year: "2014", name: "VGG", layers: 19, color: C.purple, note: "Showed deeper = better... up to a point. Training 19 layers was already painful.", err: "7.3%" },
    { year: "2014", name: "GoogLeNet", layers: 22, color: C.cyan, note: "Used 'inception modules' — clever parallel paths. Needed auxiliary classifiers to train.", err: "6.7%" },
    { year: "2015", name: "ResNet", layers: 152, color: C.green, note: "Skip connections made 152 layers trainable. Won ImageNet. Changed everything.", err: "3.6%" },
    { year: "2017", name: "Transformer", layers: null, color: C.accent, note: "Same residual connections, applied to ATTENTION layers. Foundation of GPT, Claude, etc.", err: null },
  ];

  const [hovered, setHovered] = useState(null);

  return (
    <div>
      <div style={{ display: "flex", gap: 3, alignItems: "flex-end" }}>
        {milestones.map((m, i) => (
          <div
            key={i}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            style={{
              flex: 1,
              cursor: "default",
              transition: "all 0.2s",
            }}
          >
            {/* Depth bar */}
            {m.layers && (
              <div style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                marginBottom: 4,
              }}>
                <div style={{ fontSize: 10, color: m.color, fontWeight: 700, marginBottom: 2 }}>
                  {m.layers}L
                </div>
                <div style={{
                  width: "70%",
                  height: Math.min(m.layers * 0.9, 140),
                  background: `linear-gradient(to top, ${m.color}, ${m.color}44)`,
                  borderRadius: "6px 6px 0 0",
                  transition: "all 0.3s",
                  opacity: hovered === i ? 1 : 0.7,
                }} />
              </div>
            )}
            {!m.layers && (
              <div style={{ height: 50, display: "flex", alignItems: "flex-end", justifyContent: "center" }}>
                <span style={{ fontSize: 20 }}>🔥</span>
              </div>
            )}

            {/* Label */}
            <div style={{
              textAlign: "center",
              padding: "8px 4px",
              background: hovered === i ? m.color + "15" : "transparent",
              borderTop: `2px solid ${m.color}`,
              borderRadius: "0 0 6px 6px",
              transition: "all 0.2s",
            }}>
              <div style={{ fontSize: 10, color: C.textDim }}>{m.year}</div>
              <div style={{ fontSize: 12, fontWeight: 700, color: m.color }}>{m.name}</div>
              {m.err && <div style={{ fontSize: 9, color: C.textDim }}>Error: {m.err}</div>}
            </div>
          </div>
        ))}
      </div>

      {/* Hover info */}
      {hovered !== null && (
        <div style={{
          marginTop: 12,
          padding: "10px 14px",
          background: milestones[hovered].color + "10",
          border: `1px solid ${milestones[hovered].color}30`,
          borderRadius: 10,
          fontSize: 13,
          color: C.textDim,
          lineHeight: 1.5,
        }}>
          <strong style={{ color: milestones[hovered].color }}>{milestones[hovered].name}:</strong>{" "}
          {milestones[hovered].note}
        </div>
      )}
    </div>
  );
}

function TransformerConnection() {
  return (
    <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
      {/* Transformer block diagram */}
      <div style={{ flex: "0 0 240px" }}>
        <div style={{ fontSize: 10, color: C.textDim, marginBottom: 6, textAlign: "center", fontWeight: 600, letterSpacing: 0.5, textTransform: "uppercase" }}>
          Transformer Block
        </div>
        <svg width={240} height={340}>
          {/* Input */}
          <text x={70} y={20} fill={C.text} fontSize={11} fontWeight={600}>input x</text>
          <line x1={90} y1={28} x2={90} y2={55} stroke={C.blue} strokeWidth={2} />

          {/* Attention block */}
          <rect x={30} y={55} width={120} height={36} rx={8} fill={C.purple + "18"} stroke={C.purple} strokeWidth={1.5} />
          <text x={90} y={78} textAnchor="middle" fill={C.purple} fontSize={11} fontWeight={600}>Attention</text>

          {/* Skip 1 */}
          <path d="M 150 40 Q 190 40 190 90 Q 190 108 150 108" fill="none" stroke={C.green} strokeWidth={2.5} strokeDasharray="5 3" />

          {/* Add + Norm 1 */}
          <line x1={90} y1={91} x2={90} y2={100} stroke={C.blue} strokeWidth={2} />
          <circle cx={90} cy={108} r={10} fill={C.green + "20"} stroke={C.green} strokeWidth={2} />
          <text x={90} y={112} textAnchor="middle" fill={C.green} fontSize={13} fontWeight={700}>+</text>
          <rect x={40} y={124} width={100} height={22} rx={6} fill={C.cyan + "12"} stroke={C.cyan} strokeWidth={1} />
          <text x={90} y={139} textAnchor="middle" fill={C.cyan} fontSize={9}>LayerNorm</text>

          {/* FFN block */}
          <line x1={90} y1={146} x2={90} y2={165} stroke={C.blue} strokeWidth={2} />
          <rect x={30} y={165} width={120} height={36} rx={8} fill={C.blue + "18"} stroke={C.blue} strokeWidth={1.5} />
          <text x={90} y={188} textAnchor="middle" fill={C.blue} fontSize={11} fontWeight={600}>FFN (Linear)</text>

          {/* Skip 2 */}
          <path d="M 150 150 Q 190 150 190 200 Q 190 218 150 218" fill="none" stroke={C.green} strokeWidth={2.5} strokeDasharray="5 3" />

          {/* Add + Norm 2 */}
          <line x1={90} y1={201} x2={90} y2={210} stroke={C.blue} strokeWidth={2} />
          <circle cx={90} cy={218} r={10} fill={C.green + "20"} stroke={C.green} strokeWidth={2} />
          <text x={90} y={222} textAnchor="middle" fill={C.green} fontSize={13} fontWeight={700}>+</text>
          <rect x={40} y={234} width={100} height={22} rx={6} fill={C.cyan + "12"} stroke={C.cyan} strokeWidth={1} />
          <text x={90} y={249} textAnchor="middle" fill={C.cyan} fontSize={9}>LayerNorm</text>

          {/* Output */}
          <line x1={90} y1={256} x2={90} y2={280} stroke={C.blue} strokeWidth={2} />
          <text x={70} y={295} fill={C.text} fontSize={11} fontWeight={600}>output</text>

          {/* Labels */}
          <text x={206} y={76} fill={C.green} fontSize={9} fontWeight={600}>skip</text>
          <text x={206} y={186} fill={C.green} fontSize={9} fontWeight={600}>skip</text>
        </svg>
      </div>

      {/* Explanation */}
      <div style={{ flex: "1 1 300px", minWidth: 260 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: C.text, marginBottom: 10 }}>
          Every transformer block is a residual block
        </div>
        <div style={{ fontSize: 13, color: C.textDim, lineHeight: 1.8, marginBottom: 14 }}>
          Look at the pattern: the attention output gets <strong style={{ color: C.green }}>added back to the input</strong>, then normalized.
          Same for the feed-forward network. It's the exact same skip connection idea from ResNets.
        </div>
        <div style={{ fontSize: 13, color: C.textDim, lineHeight: 1.8, marginBottom: 14 }}>
          GPT-3 has <strong style={{ color: C.accent }}>96 transformer blocks</strong> stacked on top of each other.
          Without residual connections, training 96 blocks deep would be impossible — gradients would vanish before reaching the early layers.
        </div>

        <div style={{
          padding: "12px 14px",
          background: C.accent + "10",
          border: `1px solid ${C.accent}30`,
          borderRadius: 10,
          fontSize: 13,
          color: C.text,
          lineHeight: 1.6,
        }}>
          <strong style={{ color: C.accent }}>The concept you're learning now is the same one that makes GPT, Claude, and every LLM possible.</strong>{" "}
          <span style={{ color: C.textDim }}>
            This isn't background theory — residual connections are load-bearing infrastructure in every frontier model.
          </span>
        </div>

        <div style={{
          marginTop: 14,
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 8,
        }}>
          {[
            { label: "ResNet (2015)", detail: "Conv → BN → ReLU → Conv + skip", color: C.green },
            { label: "Transformer (2017)", detail: "Attn + skip → Norm → FFN + skip → Norm", color: C.purple },
            { label: "GPT / Claude", detail: "96+ stacked transformer blocks with residual connections", color: C.accent },
            { label: "Your CNN (last lesson)", detail: "Exercise 5 was: add skip connections!", color: C.blue },
          ].map((item, i) => (
            <div key={i} style={{
              padding: "10px 12px",
              background: C.surface,
              border: `1px solid ${C.border}`,
              borderRadius: 8,
            }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: item.color, marginBottom: 3 }}>{item.label}</div>
              <div style={{ fontSize: 10, color: C.textDim, lineHeight: 1.4 }}>{item.detail}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function DegradationDemo() {
  // Simulated training curves
  const [hoveredEpoch, setHoveredEpoch] = useState(null);
  const svgW = 520;
  const svgH = 200;
  const pad = { l: 50, r: 20, t: 20, b: 30 };
  const plotW = svgW - pad.l - pad.r;
  const plotH = svgH - pad.t - pad.b;

  // Generate curves
  const epochs = 40;
  const curves = {
    shallow: { label: "20-layer plain", color: C.blue, data: [] },
    deep: { label: "56-layer plain", color: C.red, data: [] },
    deepRes: { label: "56-layer ResNet", color: C.green, data: [] },
  };

  for (let i = 0; i <= epochs; i++) {
    const t = i / epochs;
    curves.shallow.data.push(38 - 30 * (1 - Math.exp(-3.5 * t)));
    curves.deep.data.push(42 - 28 * (1 - Math.exp(-2 * t)));
    curves.deepRes.data.push(38 - 33 * (1 - Math.exp(-4 * t)));
  }

  const yMin = 4;
  const yMax = 44;
  const xScale = (i) => pad.l + (i / epochs) * plotW;
  const yScale = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * plotH;

  const makePath = (data) =>
    data.map((v, i) => `${i === 0 ? "M" : "L"} ${xScale(i)} ${yScale(v)}`).join(" ");

  return (
    <div>
      <svg
        width={svgW}
        height={svgH}
        style={{ display: "block", margin: "0 auto" }}
        onMouseMove={e => {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left - pad.l;
          const epoch = Math.round((x / plotW) * epochs);
          setHoveredEpoch(Math.max(0, Math.min(epochs, epoch)));
        }}
        onMouseLeave={() => setHoveredEpoch(null)}
      >
        {/* Grid lines */}
        {[10, 20, 30, 40].map(v => (
          <g key={v}>
            <line x1={pad.l} y1={yScale(v)} x2={svgW - pad.r} y2={yScale(v)} stroke={C.border} strokeWidth={0.5} />
            <text x={pad.l - 6} y={yScale(v) + 3} textAnchor="end" fill={C.textDim} fontSize={9}>{v}%</text>
          </g>
        ))}

        {/* Axes */}
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={svgH - pad.b} stroke={C.border} strokeWidth={1} />
        <line x1={pad.l} y1={svgH - pad.b} x2={svgW - pad.r} y2={svgH - pad.b} stroke={C.border} strokeWidth={1} />
        <text x={svgW / 2} y={svgH - 4} textAnchor="middle" fill={C.textDim} fontSize={10}>Epoch</text>
        <text x={12} y={svgH / 2} textAnchor="middle" fill={C.textDim} fontSize={10} transform={`rotate(-90, 12, ${svgH / 2})`}>Error %</text>

        {/* Curves */}
        {Object.values(curves).map(c => (
          <path key={c.label} d={makePath(c.data)} fill="none" stroke={c.color} strokeWidth={2.5} />
        ))}

        {/* Hover line */}
        {hoveredEpoch !== null && (
          <line x1={xScale(hoveredEpoch)} y1={pad.t} x2={xScale(hoveredEpoch)} y2={svgH - pad.b} stroke={C.textDim} strokeWidth={1} strokeDasharray="3 3" />
        )}
      </svg>

      {/* Legend */}
      <div style={{ display: "flex", justifyContent: "center", gap: 20, marginTop: 8 }}>
        {Object.values(curves).map(c => (
          <div key={c.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 16, height: 3, background: c.color, borderRadius: 2 }} />
            <span style={{ fontSize: 11, color: c.color, fontWeight: 600 }}>{c.label}</span>
            {hoveredEpoch !== null && (
              <span style={{ fontSize: 10, color: C.textDim, fontFamily: "'JetBrains Mono', monospace" }}>
                {c.data[hoveredEpoch].toFixed(1)}%
              </span>
            )}
          </div>
        ))}
      </div>

      <div style={{
        marginTop: 12,
        padding: "10px 14px",
        background: C.surface,
        border: `1px solid ${C.border}`,
        borderRadius: 10,
        fontSize: 12,
        color: C.textDim,
        lineHeight: 1.6,
      }}>
        <strong style={{ color: C.red }}>The degradation problem:</strong> The 56-layer plain network performs
        <em> worse</em> than the 20-layer one — even on training data! It's not overfitting; the deeper network
        simply can't learn the identity mapping. ResNet fixes this: the 56-layer ResNet outperforms both.
      </div>
    </div>
  );
}


export default function ResNetLesson() {
  const [tab, setTab] = useState("problem");

  const tabs = [
    { id: "problem", label: "The Problem", icon: "⚠" },
    { id: "solution", label: "Skip Connections", icon: "⤴" },
    { id: "gradients", label: "Gradient Flow", icon: "↕" },
    { id: "transformer", label: "→ Transformers", icon: "🔗" },
  ];

  return (
    <div style={{
      background: C.bg,
      minHeight: "100vh",
      color: C.text,
      fontFamily: "'Instrument Sans', 'DM Sans', system-ui, sans-serif",
    }}>
      {/* Header */}
      <div style={{
        background: `linear-gradient(135deg, ${C.surface} 0%, ${C.bg} 100%)`,
        borderBottom: `1px solid ${C.border}`,
        padding: "32px 24px 24px",
      }}>
        <div style={{ maxWidth: 860, margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
            <span style={{
              background: C.accent,
              color: "#000",
              fontSize: 10,
              fontWeight: 800,
              padding: "3px 10px",
              borderRadius: 20,
              letterSpacing: 1,
            }}>LESSON 7</span>
            <span style={{ color: C.textDim, fontSize: 13 }}>Stage 2: Deep Learning → Stage 3 Bridge</span>
          </div>
          <h1 style={{
            fontSize: 28,
            fontWeight: 800,
            margin: "8px 0 6px",
            background: `linear-gradient(90deg, ${C.text}, ${C.green})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}>
            Residual Connections
          </h1>
          <p style={{ color: C.textDim, fontSize: 14, margin: 0, lineHeight: 1.6 }}>
            The idea that made deep networks possible — and the same idea inside every <code style={{
              background: C.accent + "22",
              color: C.accent,
              padding: "2px 6px",
              borderRadius: 4,
              fontSize: 13,
            }}>transformer</code>.
          </p>
        </div>
      </div>

      {/* Key insight */}
      <div style={{ maxWidth: 860, margin: "20px auto 0", padding: "0 24px" }}>
        <div style={{
          background: C.accent + "10",
          border: `1px solid ${C.accent}40`,
          borderRadius: 12,
          padding: "16px 20px",
          display: "flex",
          gap: 14,
          alignItems: "flex-start",
        }}>
          <span style={{ fontSize: 22, marginTop: 2 }}>💡</span>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: C.accent, marginBottom: 4 }}>
              The Key Insight
            </div>
            <div style={{ fontSize: 13, color: C.textDim, lineHeight: 1.6 }}>
              Instead of asking a layer to learn the full transformation, let it learn the <strong style={{ color: C.green }}>difference</strong> from the input.
              Output = F(x) <strong style={{ color: C.green }}>+ x</strong>. If a layer has nothing useful to add, F(x) can be zero and the input passes through unchanged.
              This simple addition creates a <strong style={{ color: C.text }}>gradient highway</strong> that lets signals flow through hundreds of layers.
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ maxWidth: 860, margin: "20px auto 0", padding: "0 24px" }}>
        <div style={{ display: "flex", gap: 4, background: C.surface, borderRadius: 10, padding: 4 }}>
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                flex: 1,
                padding: "10px 8px",
                background: tab === t.id ? C.surfaceLight : "transparent",
                border: tab === t.id ? `1px solid ${C.borderLight}` : "1px solid transparent",
                borderRadius: 8,
                color: tab === t.id ? C.text : C.textDim,
                fontSize: 13,
                fontWeight: tab === t.id ? 700 : 500,
                cursor: "pointer",
                fontFamily: "inherit",
                transition: "all 0.2s",
              }}
            >
              {t.icon} {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div style={{ maxWidth: 860, margin: "16px auto 0", padding: "0 24px 60px" }}>

        {tab === "problem" && (
          <div>
            <div style={{
              background: C.surface,
              borderRadius: 14,
              border: `1px solid ${C.border}`,
              padding: 20,
              marginBottom: 16,
            }}>
              <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>
                The Degradation Problem
              </h3>
              <p style={{ margin: "0 0 14px", fontSize: 13, color: C.textDim, lineHeight: 1.5 }}>
                More layers should mean a more powerful network. But in practice, plain networks get <em>worse</em> past ~20 layers — even on training data. Hover over the chart to compare.
              </p>
              <DegradationDemo />
            </div>

            <div style={{
              background: C.surface,
              borderRadius: 14,
              border: `1px solid ${C.border}`,
              padding: 20,
            }}>
              <h3 style={{ margin: "0 0 10px", fontSize: 15, color: C.text }}>
                The ImageNet Race
              </h3>
              <p style={{ margin: "0 0 14px", fontSize: 13, color: C.textDim }}>
                Hover over each architecture to see how depth evolved — and hit a wall until ResNet.
              </p>
              <HistoryTimeline />
            </div>
          </div>
        )}

        {tab === "solution" && (
          <div style={{
            background: C.surface,
            borderRadius: 14,
            border: `1px solid ${C.border}`,
            padding: 20,
          }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>
              The Residual Block
            </h3>
            <p style={{ margin: "0 0 16px", fontSize: 13, color: C.textDim }}>
              A deceptively simple idea from He et al. (2015) that unlocked networks 10× deeper.
            </p>
            <ResidualBlockDiagram />
          </div>
        )}

        {tab === "gradients" && (
          <div style={{
            background: C.surface,
            borderRadius: 14,
            border: `1px solid ${C.border}`,
            padding: 20,
          }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>
              Gradient Flow Comparison
            </h3>
            <p style={{ margin: "0 0 16px", fontSize: 13, color: C.textDim }}>
              Watch gradients flow backward from loss to the first layer. Toggle skip connections and try different depths.
            </p>
            <GradientFlowViz />
          </div>
        )}

        {tab === "transformer" && (
          <div style={{
            background: C.surface,
            borderRadius: 14,
            border: `1px solid ${C.border}`,
            padding: 20,
          }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>
              From ResNets to Transformers
            </h3>
            <p style={{ margin: "0 0 16px", fontSize: 13, color: C.textDim }}>
              The transformer block is a residual block. This is the bridge to everything that follows.
            </p>
            <TransformerConnection />
          </div>
        )}
      </div>
    </div>
  );
}
