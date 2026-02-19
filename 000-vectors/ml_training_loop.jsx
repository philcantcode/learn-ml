import { useState, useEffect, useRef, useCallback } from "react";

const STEPS = [
  {
    id: 0,
    label: "Input Vector",
    short: "INPUT",
    description: "Data enters the model as a vector — an array of numbers representing features.",
  },
  {
    id: 1,
    label: "Matrix Multiply",
    short: "MULTIPLY",
    description: "The layer multiplies the input by its weight matrix. This is the model's 'thinking' — a linear transformation.",
  },
  {
    id: 2,
    label: "Measure Loss",
    short: "LOSS",
    description: "We compare the prediction to the true answer. The loss function measures how wrong the model is.",
  },
  {
    id: 3,
    label: "Compute Gradient",
    short: "GRADIENT",
    description: "Calculus tells us: for each weight, which direction would reduce the loss? That's the gradient.",
  },
  {
    id: 4,
    label: "Update Weights",
    short: "UPDATE",
    description: "We nudge every weight in the opposite direction of its gradient. Small step downhill.",
  },
  {
    id: 5,
    label: "Repeat",
    short: "REPEAT",
    description: "Do it again. And again. Each pass makes the model slightly less wrong, until it converges.",
  },
];

// Simple linear model simulation
function runModel(weights, input) {
  return weights.reduce((sum, w, i) => sum + w * input[i], 0) + weights[weights.length - 1];
}

function computeGradients(weights, input, target) {
  const pred = runModel(weights, input);
  const error = pred - target;
  return input.map((x) => 2 * error * x).concat([2 * error]);
}

// Easing
function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

// Animated number
function AnimatedNumber({ value, duration = 600, decimals = 1 }) {
  const [display, setDisplay] = useState(value);
  const prev = useRef(value);

  useEffect(() => {
    const from = prev.current;
    const to = value;
    if (Math.abs(from - to) < 0.001) return;
    const start = performance.now();
    const animate = (now) => {
      const t = Math.min((now - start) / duration, 1);
      setDisplay(from + (to - from) * easeOutCubic(t));
      if (t < 1) requestAnimationFrame(animate);
      else prev.current = to;
    };
    requestAnimationFrame(animate);
    return () => { prev.current = to; };
  }, [value, duration]);

  return <span>{display.toFixed(decimals)}</span>;
}

// --- Vector/Matrix visual components ---

function VectorDisplay({ values, label, color, highlight, glow, compact }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
      {label && (
        <span style={{
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          fontSize: compact ? 10 : 11,
          color: color || "#94a3b8",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          fontWeight: 600,
        }}>{label}</span>
      )}
      <div style={{
        display: "flex",
        flexDirection: "column",
        gap: 3,
        padding: compact ? "6px 4px" : "8px 6px",
        borderRadius: 8,
        background: glow ? `${color}11` : "rgba(15,23,42,0.6)",
        border: `1.5px solid ${highlight ? color || "#3b82f6" : "rgba(100,116,139,0.2)"}`,
        boxShadow: glow ? `0 0 20px ${color}22, inset 0 0 12px ${color}08` : "none",
        transition: "all 0.5s ease",
        backdropFilter: "blur(8px)",
      }}>
        {values.map((v, i) => (
          <div key={i} style={{
            fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
            fontSize: compact ? 13 : 15,
            fontWeight: 500,
            color: highlight ? (color || "#3b82f6") : "#e2e8f0",
            padding: compact ? "3px 8px" : "4px 12px",
            borderRadius: 4,
            background: highlight ? `${color || "#3b82f6"}15` : "transparent",
            transition: "all 0.4s ease",
            textAlign: "right",
            minWidth: compact ? 48 : 56,
          }}>
            <AnimatedNumber value={v} decimals={compact ? 1 : 2} />
          </div>
        ))}
      </div>
    </div>
  );
}

function MatrixDisplay({ values, label, highlight, gradients, compact }) {
  const rows = values.length;
  const cols = values[0]?.length || 0;
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
      {label && (
        <span style={{
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          fontSize: compact ? 10 : 11,
          color: "#f59e0b",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          fontWeight: 600,
        }}>{label}</span>
      )}
      <div style={{
        display: "flex",
        gap: 2,
        padding: compact ? "6px 4px" : "8px 6px",
        borderRadius: 8,
        background: highlight ? "rgba(245,158,11,0.06)" : "rgba(15,23,42,0.6)",
        border: `1.5px solid ${highlight ? "#f59e0b" : "rgba(100,116,139,0.2)"}`,
        boxShadow: highlight ? "0 0 24px rgba(245,158,11,0.12), inset 0 0 12px rgba(245,158,11,0.04)" : "none",
        transition: "all 0.5s ease",
        backdropFilter: "blur(8px)",
      }}>
        {Array.from({ length: cols }).map((_, c) => (
          <div key={c} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            {Array.from({ length: rows }).map((_, r) => {
              const grad = gradients ? gradients[r]?.[c] : null;
              const gradColor = grad ? (grad > 0 ? "#ef4444" : "#22c55e") : null;
              return (
                <div key={r} style={{
                  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                  fontSize: compact ? 12 : 14,
                  fontWeight: 500,
                  color: gradColor || (highlight ? "#f59e0b" : "#e2e8f0"),
                  padding: compact ? "3px 6px" : "4px 10px",
                  borderRadius: 4,
                  background: gradColor ? `${gradColor}15` : (highlight ? "rgba(245,158,11,0.1)" : "transparent"),
                  transition: "all 0.4s ease",
                  textAlign: "right",
                  minWidth: compact ? 44 : 52,
                  position: "relative",
                }}>
                  <AnimatedNumber value={values[r][c]} decimals={compact ? 1 : 2} />
                  {grad !== null && (
                    <span style={{
                      position: "absolute",
                      top: -2,
                      right: -2,
                      fontSize: 8,
                      color: gradColor,
                      fontWeight: 700,
                    }}>
                      {grad > 0 ? "↓" : "↑"}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

// Arrow connector
function Arrow({ direction = "right", color = "#475569", active, label, length }) {
  const isVertical = direction === "down" || direction === "up";
  const c = active ? "#60a5fa" : color;
  const len = length || (isVertical ? 32 : 40);

  return (
    <div style={{
      display: "flex",
      flexDirection: isVertical ? "column" : "row",
      alignItems: "center",
      gap: 2,
      opacity: active ? 1 : 0.4,
      transition: "opacity 0.5s ease",
    }}>
      {label && (
        <span style={{
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          fontSize: 9,
          color: c,
          letterSpacing: "0.05em",
          whiteSpace: "nowrap",
        }}>{label}</span>
      )}
      <svg
        width={isVertical ? 16 : len}
        height={isVertical ? len : 16}
        style={{ display: "block" }}
      >
        {isVertical ? (
          <>
            <line x1="8" y1="2" x2="8" y2={len - 6} stroke={c} strokeWidth="1.5" strokeDasharray={active ? "none" : "3,3"} />
            <polygon
              points={direction === "down" ? `4,${len - 6} 12,${len - 6} 8,${len - 1}` : `4,6 12,6 8,1`}
              fill={c}
            />
          </>
        ) : (
          <>
            <line x1="2" y1="8" x2={len - 6} y2="8" stroke={c} strokeWidth="1.5" strokeDasharray={active ? "none" : "3,3"} />
            <polygon
              points={direction === "right" ? `${len - 6},4 ${len - 6},12 ${len - 1},8` : `6,4 6,12 1,8`}
              fill={c}
            />
          </>
        )}
      </svg>
    </div>
  );
}

// Loss bar
function LossBar({ loss, maxLoss, active }) {
  const pct = Math.min(loss / maxLoss, 1) * 100;
  const color = loss > maxLoss * 0.6 ? "#ef4444" : loss > maxLoss * 0.3 ? "#f59e0b" : "#22c55e";
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 4,
      opacity: active ? 1 : 0.3,
      transition: "opacity 0.5s ease",
    }}>
      <span style={{
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 11,
        color: "#94a3b8",
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        fontWeight: 600,
      }}>LOSS</span>
      <div style={{
        width: 36,
        height: 80,
        borderRadius: 6,
        background: "rgba(15,23,42,0.6)",
        border: `1.5px solid ${active ? color : "rgba(100,116,139,0.2)"}`,
        overflow: "hidden",
        position: "relative",
        backdropFilter: "blur(8px)",
      }}>
        <div style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          height: `${pct}%`,
          background: `linear-gradient(to top, ${color}, ${color}88)`,
          transition: "height 0.8s ease, background 0.5s ease",
          boxShadow: active ? `0 0 12px ${color}44` : "none",
        }} />
      </div>
      <span style={{
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 13,
        fontWeight: 600,
        color: active ? color : "#64748b",
        transition: "color 0.5s ease",
      }}>
        <AnimatedNumber value={loss} decimals={1} />
      </span>
    </div>
  );
}

// Step indicator
function StepIndicator({ steps, current }) {
  return (
    <div style={{
      display: "flex",
      gap: 0,
      alignItems: "center",
      justifyContent: "center",
    }}>
      {steps.map((s, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center" }}>
          <div style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 4,
            cursor: "pointer",
          }}>
            <div style={{
              width: i === current ? 32 : 10,
              height: 10,
              borderRadius: 5,
              background: i === current ? "#60a5fa" : i < current ? "#334155" : "rgba(100,116,139,0.2)",
              transition: "all 0.4s ease",
              boxShadow: i === current ? "0 0 12px rgba(96,165,250,0.4)" : "none",
            }} />
            <span style={{
              fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
              fontSize: 9,
              color: i === current ? "#60a5fa" : "#475569",
              letterSpacing: "0.05em",
              transition: "color 0.3s ease",
              fontWeight: i === current ? 700 : 500,
            }}>{s.short}</span>
          </div>
          {i < steps.length - 1 && <div style={{ width: 16 }} />}
        </div>
      ))}
    </div>
  );
}

// ==================== MAIN COMPONENT ====================

export default function MLTrainingLoop() {
  const [step, setStep] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const timerRef = useRef(null);

  // Model state
  const input = [2.0, 3.0, 1.5];
  const target = 14.5;
  const lr = 0.02;

  const [weights, setWeights] = useState([
    [0.5, -0.3, 0.8],
    [0.2, 0.6, -0.4],
  ]);
  const [bias, setBias] = useState([0.1, -0.2]);
  const [lossHistory, setLossHistory] = useState([]);

  const output = weights.map((row, i) =>
    row.reduce((s, w, j) => s + w * input[j], 0) + bias[i]
  );
  const loss = output.reduce((s, o) => s + (o - target / 2) ** 2, 0);

  const gradients = weights.map((row, i) => {
    const error = 2 * (output[i] - target / 2);
    return row.map((_, j) => error * input[j]);
  });

  const doUpdate = useCallback(() => {
    setWeights((prev) =>
      prev.map((row, i) => {
        const error = 2 * (output[i] - target / 2);
        return row.map((w, j) => w - lr * error * input[j]);
      })
    );
    setBias((prev) =>
      prev.map((b, i) => {
        const error = 2 * (output[i] - target / 2);
        return b - lr * error;
      })
    );
    setLossHistory((prev) => [...prev.slice(-29), loss]);
    setEpoch((e) => e + 1);
  }, [output, loss]);

  const nextStep = useCallback(() => {
    setStep((s) => {
      if (s === 5) {
        doUpdate();
        return 0;
      }
      return s + 1;
    });
  }, [doUpdate]);

  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(nextStep, 1200);
    }
    return () => clearInterval(timerRef.current);
  }, [isPlaying, nextStep]);

  const reset = () => {
    setWeights([[0.5, -0.3, 0.8], [0.2, 0.6, -0.4]]);
    setBias([0.1, -0.2]);
    setStep(0);
    setEpoch(0);
    setLossHistory([]);
    setIsPlaying(false);
  };

  const currentStep = STEPS[step];

  // Loss chart
  const chartW = 240;
  const chartH = 60;
  const allLoss = [...lossHistory, loss];
  const maxL = Math.max(...allLoss, 1);

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0b1120",
      color: "#e2e8f0",
      fontFamily: "'Inter', 'SF Pro Display', -apple-system, sans-serif",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      padding: "24px 16px",
      overflow: "hidden",
    }}>
      {/* Title */}
      <div style={{ textAlign: "center", marginBottom: 20 }}>
        <h1 style={{
          fontSize: 20,
          fontWeight: 700,
          color: "#f8fafc",
          margin: 0,
          letterSpacing: "-0.02em",
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        }}>
          THE TRAINING LOOP
        </h1>
        <p style={{
          fontSize: 12,
          color: "#64748b",
          margin: "4px 0 0",
          fontFamily: "'JetBrains Mono', monospace",
          letterSpacing: "0.06em",
        }}>
          HOW EVERY ML MODEL LEARNS — EPOCH {epoch}
        </p>
      </div>

      {/* Step indicator */}
      <StepIndicator steps={STEPS} current={step} />

      {/* Step description */}
      <div style={{
        margin: "16px 0",
        padding: "12px 20px",
        borderRadius: 8,
        background: "rgba(96,165,250,0.06)",
        border: "1px solid rgba(96,165,250,0.15)",
        maxWidth: 520,
        textAlign: "center",
        minHeight: 52,
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
      }}>
        <div style={{
          fontSize: 14,
          fontWeight: 600,
          color: "#60a5fa",
          marginBottom: 2,
          fontFamily: "'JetBrains Mono', monospace",
        }}>
          {currentStep.label}
        </div>
        <div style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.4 }}>
          {currentStep.description}
        </div>
      </div>

      {/* Main visualization */}
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 12,
        flexWrap: "wrap",
        padding: "8px 0",
        maxWidth: 700,
      }}>
        {/* Input vector */}
        <VectorDisplay
          values={input}
          label="input x"
          color="#3b82f6"
          highlight={step === 0}
          glow={step === 0}
          compact
        />

        <Arrow direction="right" active={step <= 1} label="×" />

        {/* Weight matrix */}
        <MatrixDisplay
          values={weights}
          label="weights W"
          highlight={step === 1 || step === 4}
          gradients={step === 3 || step === 4 ? gradients : null}
          compact
        />

        <Arrow direction="right" active={step >= 1 && step <= 2} label="=" />

        {/* Output vector */}
        <VectorDisplay
          values={output}
          label="output ŷ"
          color="#8b5cf6"
          highlight={step >= 1 && step <= 2}
          glow={step === 2}
          compact
        />

        {/* Loss */}
        <div style={{ marginLeft: 8 }}>
          <LossBar
            loss={loss}
            maxLoss={Math.max(maxL, 50)}
            active={step >= 2}
          />
        </div>
      </div>

      {/* Equation display */}
      <div style={{
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 12,
        color: "#475569",
        margin: "8px 0",
        padding: "8px 16px",
        borderRadius: 6,
        background: "rgba(15,23,42,0.4)",
        border: "1px solid rgba(100,116,139,0.1)",
        textAlign: "center",
      }}>
        {step === 0 && (
          <span><span style={{ color: "#3b82f6" }}>x</span> = [{input.map((v) => v.toFixed(1)).join(", ")}]</span>
        )}
        {step === 1 && (
          <span><span style={{ color: "#8b5cf6" }}>ŷ</span> = <span style={{ color: "#f59e0b" }}>W</span> · <span style={{ color: "#3b82f6" }}>x</span> + <span style={{ color: "#f59e0b" }}>b</span></span>
        )}
        {step === 2 && (
          <span>L = Σ(<span style={{ color: "#8b5cf6" }}>ŷ</span> - y)² = <span style={{ color: loss > 10 ? "#ef4444" : "#22c55e" }}>{loss.toFixed(2)}</span></span>
        )}
        {step === 3 && (
          <span>∂L/∂<span style={{ color: "#f59e0b" }}>W</span> → <span style={{ color: "#ef4444" }}>which direction reduces loss?</span></span>
        )}
        {step === 4 && (
          <span><span style={{ color: "#f59e0b" }}>W</span> = <span style={{ color: "#f59e0b" }}>W</span> - α · ∇L &nbsp; <span style={{ color: "#64748b" }}>(α = {lr})</span></span>
        )}
        {step === 5 && (
          <span style={{ color: "#22c55e" }}>epoch {epoch} → {epoch + 1} &nbsp; <span style={{color:"#64748b"}}>keep going until loss → 0</span></span>
        )}
      </div>

      {/* Loss history chart */}
      {allLoss.length > 1 && (
        <div style={{
          margin: "8px 0",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 4,
        }}>
          <span style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            color: "#475569",
            letterSpacing: "0.08em",
          }}>LOSS OVER TIME</span>
          <svg width={chartW} height={chartH} style={{ display: "block" }}>
            <rect x={0} y={0} width={chartW} height={chartH} rx={6} fill="rgba(15,23,42,0.5)" stroke="rgba(100,116,139,0.15)" strokeWidth={1} />
            {allLoss.length > 1 && (
              <polyline
                fill="none"
                stroke="#60a5fa"
                strokeWidth={1.5}
                strokeLinecap="round"
                strokeLinejoin="round"
                points={allLoss
                  .map((l, i) => {
                    const x = 8 + (i / (allLoss.length - 1)) * (chartW - 16);
                    const y = 6 + (1 - l / maxL) * (chartH - 12);
                    return `${x},${y}`;
                  })
                  .join(" ")}
              />
            )}
            {allLoss.length > 0 && (() => {
              const lastI = allLoss.length - 1;
              const cx = 8 + (lastI / Math.max(allLoss.length - 1, 1)) * (chartW - 16);
              const cy = 6 + (1 - allLoss[lastI] / maxL) * (chartH - 12);
              return <circle cx={cx} cy={cy} r={3} fill="#60a5fa" />;
            })()}
          </svg>
        </div>
      )}

      {/* Controls */}
      <div style={{
        display: "flex",
        gap: 10,
        marginTop: 12,
        alignItems: "center",
      }}>
        <button
          onClick={() => { setIsPlaying(false); nextStep(); }}
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            fontWeight: 600,
            padding: "8px 20px",
            borderRadius: 6,
            border: "1px solid rgba(96,165,250,0.3)",
            background: "rgba(96,165,250,0.1)",
            color: "#60a5fa",
            cursor: "pointer",
            letterSpacing: "0.05em",
            transition: "all 0.2s ease",
          }}
          onMouseEnter={(e) => { e.target.style.background = "rgba(96,165,250,0.2)"; }}
          onMouseLeave={(e) => { e.target.style.background = "rgba(96,165,250,0.1)"; }}
        >
          STEP →
        </button>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            fontWeight: 600,
            padding: "8px 20px",
            borderRadius: 6,
            border: `1px solid ${isPlaying ? "rgba(239,68,68,0.3)" : "rgba(34,197,94,0.3)"}`,
            background: isPlaying ? "rgba(239,68,68,0.1)" : "rgba(34,197,94,0.1)",
            color: isPlaying ? "#ef4444" : "#22c55e",
            cursor: "pointer",
            letterSpacing: "0.05em",
            transition: "all 0.2s ease",
          }}
        >
          {isPlaying ? "⏸ PAUSE" : "▶ AUTO"}
        </button>
        <button
          onClick={reset}
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            fontWeight: 600,
            padding: "8px 16px",
            borderRadius: 6,
            border: "1px solid rgba(100,116,139,0.2)",
            background: "rgba(100,116,139,0.08)",
            color: "#64748b",
            cursor: "pointer",
            letterSpacing: "0.05em",
          }}
        >
          RESET
        </button>
      </div>

      {/* Gradient legend for step 3/4 */}
      {(step === 3 || step === 4) && (
        <div style={{
          marginTop: 10,
          display: "flex",
          gap: 16,
          fontSize: 11,
          fontFamily: "'JetBrains Mono', monospace",
          color: "#64748b",
        }}>
          <span><span style={{ color: "#ef4444" }}>↓ red</span> = weight too high</span>
          <span><span style={{ color: "#22c55e" }}>↑ green</span> = weight too low</span>
        </div>
      )}
    </div>
  );
}
