import { useState, useEffect, useRef, useCallback } from "react";

// --- Math helpers ---
function lerp(a, b, t) { return a + (b - a) * t; }

// Compute loss for unnormalized: very elongated ellipse
// sqft weight on x-axis (tiny scale ~0.001-0.5), bedrooms weight on y-axis (large scale ~1000-20000)
function lossUnnorm(wx, wy) {
  // Simulates MSE loss landscape where sqft range >> bedrooms range
  // This creates elongated contours along the sqft-weight axis
  const dx = wx - 0.2;   // optimal sqft weight ~200 but we scale the viz
  const dy = wy - 0.5;   // optimal bedroom weight
  return 0.3 * dx * dx + 40 * dy * dy; // 133:1 ratio = elongated
}

// Compute loss for normalized: nice circular bowl
function lossNorm(wx, wy) {
  const dx = wx - 0.5;
  const dy = wy - 0.5;
  return 8 * dx * dx + 8 * dy * dy; // equal = circular
}

// Gradient
function gradUnnorm(wx, wy) {
  return [0.6 * (wx - 0.2), 80 * (wy - 0.5)];
}
function gradNorm(wx, wy) {
  return [16 * (wx - 0.5), 16 * (wy - 0.5)];
}

// Generate contour data as a grid of loss values
function genContours(lossFn, res = 60) {
  const grid = [];
  for (let j = 0; j < res; j++) {
    const row = [];
    for (let i = 0; i < res; i++) {
      const x = i / (res - 1);
      const y = j / (res - 1);
      row.push(lossFn(x, y));
    }
    grid.push(row);
  }
  return grid;
}

// Run gradient descent and return path
function runGD(gradFn, startX, startY, lr, steps) {
  const path = [{ x: startX, y: startY }];
  let wx = startX, wy = startY;
  for (let i = 0; i < steps; i++) {
    const [gx, gy] = gradFn(wx, wy);
    wx -= lr * gx;
    wy -= lr * gy;
    wx = Math.max(0, Math.min(1, wx));
    wy = Math.max(0, Math.min(1, wy));
    path.push({ x: wx, y: wy });
  }
  return path;
}

// --- Contour rendering ---
function ContourPlot({ grid, path, animIdx, optimum, width, height, colors, title, subtitle, axisLabels, isActive }) {
  const res = grid.length;
  const pad = { top: 36, right: 12, bottom: 32, left: 36 };
  const pw = width - pad.left - pad.right;
  const ph = height - pad.top - pad.bottom;

  // Find min/max for color mapping
  let minV = Infinity, maxV = -Infinity;
  for (const row of grid) for (const v of row) {
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
  }

  // Draw contour levels as SVG paths using marching squares (simplified: just filled rects)
  const cellW = pw / (res - 1);
  const cellH = ph / (res - 1);

  // Color interpolation
  function valToColor(v) {
    const t = Math.pow((v - minV) / (maxV - minV + 1e-9), 0.45); // gamma for better spread
    const r = Math.round(lerp(8, 50, t));
    const g = Math.round(lerp(20, 20, t));
    const b = Math.round(lerp(50, 70, t));
    const r2 = Math.round(lerp(colors[0], colors[3], t));
    const g2 = Math.round(lerp(colors[1], colors[4], t));
    const b2 = Math.round(lerp(colors[2], colors[5], t));
    return `rgb(${r2},${g2},${b2})`;
  }

  // Contour lines at fixed levels
  const numLevels = 10;
  const levels = Array.from({ length: numLevels }, (_, i) => minV + ((maxV - minV) * (i + 1)) / (numLevels + 1));

  // Build contour line segments using simple threshold
  function getContourPaths(level) {
    const segments = [];
    for (let j = 0; j < res - 1; j++) {
      for (let i = 0; i < res - 1; i++) {
        const corners = [grid[j][i], grid[j][i+1], grid[j+1][i+1], grid[j+1][i]];
        const above = corners.map(c => c >= level);
        // Simple: if mixed, draw line through cell center
        const count = above.filter(Boolean).length;
        if (count > 0 && count < 4) {
          const cx = pad.left + (i + 0.5) * cellW;
          const cy = pad.top + (j + 0.5) * cellH;
          segments.push({ cx, cy });
        }
      }
    }
    return segments;
  }

  // Render heatmap cells
  const cells = [];
  for (let j = 0; j < res - 1; j++) {
    for (let i = 0; i < res - 1; i++) {
      const avg = (grid[j][i] + grid[j][i+1] + grid[j+1][i] + grid[j+1][i+1]) / 4;
      cells.push(
        <rect
          key={`${i}-${j}`}
          x={pad.left + i * cellW}
          y={pad.top + j * cellH}
          width={cellW + 0.5}
          height={cellH + 0.5}
          fill={valToColor(avg)}
        />
      );
    }
  }

  // Contour lines
  const contourElements = levels.map((level, li) => {
    const pts = getContourPaths(level);
    if (pts.length < 3) return null;
    // Sort points to approximate contour - group nearby points
    return pts.map((p, pi) => (
      <circle
        key={`${li}-${pi}`}
        cx={p.cx}
        cy={p.cy}
        r={0.6}
        fill="rgba(255,255,255,0.12)"
      />
    ));
  });

  // Animated path
  const visiblePath = path.slice(0, animIdx + 1);

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 0,
      opacity: isActive ? 1 : 0.5,
      transition: "opacity 0.5s ease",
    }}>
      <svg width={width} height={height} style={{ display: "block" }}>
        {/* Background */}
        <rect x={pad.left} y={pad.top} width={pw} height={ph} rx={4} fill="#0a0f1e" />

        {/* Heatmap */}
        <g clipPath="url(#clip)">
          <clipPath id="clip">
            <rect x={pad.left} y={pad.top} width={pw} height={ph} rx={4} />
          </clipPath>
          {cells}
          {contourElements}
        </g>

        {/* Border */}
        <rect x={pad.left} y={pad.top} width={pw} height={ph} rx={4} fill="none" stroke="rgba(100,116,139,0.25)" strokeWidth={1} />

        {/* Optimum marker */}
        <circle
          cx={pad.left + optimum.x * pw}
          cy={pad.top + optimum.y * ph}
          r={5}
          fill="none"
          stroke="#22c55e"
          strokeWidth={1.5}
          opacity={0.8}
        />
        <circle
          cx={pad.left + optimum.x * pw}
          cy={pad.top + optimum.y * ph}
          r={2}
          fill="#22c55e"
          opacity={0.9}
        />

        {/* GD Path */}
        {visiblePath.length > 1 && (
          <polyline
            points={visiblePath.map(p => `${pad.left + p.x * pw},${pad.top + p.y * ph}`).join(" ")}
            fill="none"
            stroke="rgba(251,191,36,0.8)"
            strokeWidth={1.5}
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeDasharray="4,3"
          />
        )}

        {/* Path points */}
        {visiblePath.map((p, i) => (
          <circle
            key={i}
            cx={pad.left + p.x * pw}
            cy={pad.top + p.y * ph}
            r={i === visiblePath.length - 1 ? 4 : 2}
            fill={i === visiblePath.length - 1 ? "#fbbf24" : "rgba(251,191,36,0.5)"}
            stroke={i === visiblePath.length - 1 ? "#fbbf24" : "none"}
            strokeWidth={i === visiblePath.length - 1 ? 2 : 0}
          >
            {i === visiblePath.length - 1 && (
              <animate attributeName="r" values="4;6;4" dur="1.5s" repeatCount="indefinite" />
            )}
          </circle>
        ))}

        {/* Start label */}
        {visiblePath.length > 0 && (
          <text
            x={pad.left + visiblePath[0].x * pw + 7}
            y={pad.top + visiblePath[0].y * ph - 6}
            fontSize={9}
            fill="#fbbf24"
            fontFamily="'JetBrains Mono', monospace"
            fontWeight={600}
          >START</text>
        )}

        {/* Optimum label */}
        <text
          x={pad.left + optimum.x * pw + 7}
          y={pad.top + optimum.y * ph - 6}
          fontSize={9}
          fill="#22c55e"
          fontFamily="'JetBrains Mono', monospace"
          fontWeight={600}
        >MIN</text>

        {/* Title */}
        <text x={width / 2} y={16} textAnchor="middle" fontSize={13} fill="#e2e8f0" fontFamily="'JetBrains Mono', monospace" fontWeight={700} letterSpacing="0.04em">
          {title}
        </text>
        <text x={width / 2} y={28} textAnchor="middle" fontSize={10} fill="#64748b" fontFamily="'JetBrains Mono', monospace">
          {subtitle}
        </text>

        {/* Axis labels */}
        <text x={pad.left + pw / 2} y={height - 4} textAnchor="middle" fontSize={10} fill="#64748b" fontFamily="'JetBrains Mono', monospace">
          {axisLabels[0]}
        </text>
        <text x={10} y={pad.top + ph / 2} textAnchor="middle" fontSize={10} fill="#64748b" fontFamily="'JetBrains Mono', monospace" transform={`rotate(-90, 10, ${pad.top + ph / 2})`}>
          {axisLabels[1]}
        </text>
      </svg>
    </div>
  );
}

// Sidebar scale bars showing feature ranges
function ScaleBars({ normalized }) {
  const features = normalized
    ? [
        { name: "sqft", min: "-1.7", max: "1.7", range: "mean=0, std=1", color: "#3b82f6" },
        { name: "beds", min: "-1.4", max: "1.4", range: "mean=0, std=1", color: "#8b5cf6" },
      ]
    : [
        { name: "sqft", min: "800", max: "3000", range: "range: 2200", color: "#3b82f6" },
        { name: "beds", min: "1", max: "5", range: "range: 4", color: "#8b5cf6" },
      ];

  return (
    <div style={{
      display: "flex",
      gap: 20,
      justifyContent: "center",
      alignItems: "flex-end",
    }}>
      {features.map((f, i) => {
        const barWidth = normalized ? 80 : (i === 0 ? 140 : 30);
        return (
          <div key={f.name} style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 4,
          }}>
            <span style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 10,
              color: f.color,
              fontWeight: 600,
              letterSpacing: "0.06em",
            }}>{f.name.toUpperCase()}</span>
            <div style={{
              width: barWidth,
              height: 10,
              borderRadius: 5,
              background: `linear-gradient(90deg, ${f.color}33, ${f.color})`,
              border: `1px solid ${f.color}44`,
              transition: "width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)",
            }} />
            <div style={{
              display: "flex",
              justifyContent: "space-between",
              width: barWidth,
              transition: "width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)",
            }}>
              <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#64748b" }}>{f.min}</span>
              <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#64748b" }}>{f.max}</span>
            </div>
            <span style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 9,
              color: "#475569",
            }}>{f.range}</span>
          </div>
        );
      })}
    </div>
  );
}

// Step counter
function StepCounter({ current, total, steps, label }) {
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 2,
    }}>
      <span style={{
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 10,
        color: "#475569",
        letterSpacing: "0.08em",
      }}>{label}</span>
      <span style={{
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 22,
        color: "#fbbf24",
        fontWeight: 700,
      }}>{current}</span>
      <span style={{
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 9,
        color: "#475569",
      }}>of {total} steps</span>
    </div>
  );
}

// ==================== MAIN ====================

export default function NormalizationViz() {
  const [animIdx, setAnimIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showNorm, setShowNorm] = useState(false);
  const timerRef = useRef(null);

  // Precompute
  const gridUnnorm = useRef(genContours(lossUnnorm, 50)).current;
  const gridNorm = useRef(genContours(lossNorm, 50)).current;

  const startX = 0.85, startY = 0.15;
  const pathUnnorm = useRef(runGD(gradUnnorm, startX, startY, 0.009, 80)).current;
  const pathNorm = useRef(runGD(gradNorm, startX, startY, 0.045, 80)).current;

  const maxSteps = Math.max(pathUnnorm.length, pathNorm.length) - 1;

  useEffect(() => {
    if (isPlaying && animIdx < maxSteps) {
      timerRef.current = setTimeout(() => setAnimIdx(i => Math.min(i + 1, maxSteps)), 100);
    } else if (animIdx >= maxSteps) {
      setIsPlaying(false);
    }
    return () => clearTimeout(timerRef.current);
  }, [isPlaying, animIdx, maxSteps]);

  const reset = () => {
    setAnimIdx(0);
    setIsPlaying(false);
  };

  const play = () => {
    if (animIdx >= maxSteps) setAnimIdx(0);
    setIsPlaying(true);
  };

  // Did normalized path converge already?
  const normConverged = animIdx > 15;

  const plotW = 280;
  const plotH = 260;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0b1120",
      color: "#e2e8f0",
      fontFamily: "'Inter', -apple-system, sans-serif",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      padding: "24px 12px",
      gap: 16,
    }}>
      {/* Header */}
      <div style={{ textAlign: "center" }}>
        <h1 style={{
          fontSize: 18,
          fontWeight: 700,
          color: "#f8fafc",
          margin: 0,
          fontFamily: "'JetBrains Mono', monospace",
          letterSpacing: "-0.01em",
        }}>
          WHY NORMALIZATION MATTERS
        </h1>
        <p style={{
          fontSize: 11,
          color: "#64748b",
          margin: "4px 0 0",
          fontFamily: "'JetBrains Mono', monospace",
          letterSpacing: "0.04em",
        }}>
          SAME START POINT · SAME LEARNING RATE · VERY DIFFERENT PATHS
        </p>
      </div>

      {/* Feature scale comparison */}
      <div style={{
        display: "flex",
        gap: 40,
        padding: "12px 24px",
        borderRadius: 8,
        background: "rgba(15,23,42,0.5)",
        border: "1px solid rgba(100,116,139,0.12)",
        alignItems: "center",
        flexWrap: "wrap",
        justifyContent: "center",
      }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#ef4444", fontWeight: 600, marginBottom: 6, letterSpacing: "0.08em" }}>
            RAW FEATURES
          </div>
          <ScaleBars normalized={false} />
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#ef4444", marginTop: 6, opacity: 0.7 }}>
            550× difference in scale!
          </div>
        </div>
        <div style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 20,
          color: "#334155",
        }}>→</div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#22c55e", fontWeight: 600, marginBottom: 6, letterSpacing: "0.08em" }}>
            AFTER STANDARDIZATION
          </div>
          <ScaleBars normalized={true} />
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#22c55e", marginTop: 6, opacity: 0.7 }}>
            both features ~same scale
          </div>
        </div>
      </div>

      {/* Contour plots side by side */}
      <div style={{
        display: "flex",
        gap: 16,
        alignItems: "flex-start",
        flexWrap: "wrap",
        justifyContent: "center",
      }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
          <ContourPlot
            grid={gridUnnorm}
            path={pathUnnorm}
            animIdx={animIdx}
            optimum={{ x: 0.2, y: 0.5 }}
            width={plotW}
            height={plotH}
            colors={[30, 10, 60, 100, 20, 30]}
            title="WITHOUT NORMALIZATION"
            subtitle="elongated loss landscape"
            axisLabels={["w_sqft", "w_beds"]}
            isActive={true}
          />
          <div style={{
            padding: "8px 14px",
            borderRadius: 6,
            background: "rgba(239,68,68,0.06)",
            border: "1px solid rgba(239,68,68,0.15)",
            maxWidth: plotW,
            textAlign: "center",
          }}>
            <span style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 11,
              color: "#ef4444",
              fontWeight: 600,
            }}>Zigzag path</span>
            <span style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 10,
              color: "#94a3b8",
              display: "block",
              marginTop: 2,
            }}>
              Gradient overshoots on bedrooms axis,<br />
              barely moves on sqft axis
            </span>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
          <ContourPlot
            grid={gridNorm}
            path={pathNorm}
            animIdx={animIdx}
            optimum={{ x: 0.5, y: 0.5 }}
            width={plotW}
            height={plotH}
            colors={[10, 25, 60, 15, 60, 40]}
            title="WITH NORMALIZATION"
            subtitle="circular loss landscape"
            axisLabels={["w_sqft (norm)", "w_beds (norm)"]}
            isActive={true}
          />
          <div style={{
            padding: "8px 14px",
            borderRadius: 6,
            background: "rgba(34,197,94,0.06)",
            border: "1px solid rgba(34,197,94,0.15)",
            maxWidth: plotW,
            textAlign: "center",
          }}>
            <span style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 11,
              color: "#22c55e",
              fontWeight: 600,
            }}>Direct path{normConverged && animIdx > 0 ? " ✓ converged!" : ""}</span>
            <span style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 10,
              color: "#94a3b8",
              display: "block",
              marginTop: 2,
            }}>
              Equal gradient scale on both axes,<br />
              straight line to minimum
            </span>
          </div>
        </div>
      </div>

      {/* Step counter */}
      <StepCounter
        current={animIdx}
        total={maxSteps}
        label="GRADIENT DESCENT STEPS"
      />

      {/* Controls */}
      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
        <button
          onClick={play}
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
          }}
          onMouseEnter={e => { if(!isPlaying) e.target.style.background = "rgba(34,197,94,0.18)"; }}
          onMouseLeave={e => { if(!isPlaying) e.target.style.background = "rgba(34,197,94,0.1)"; }}
        >
          {isPlaying ? "⏸ PAUSE" : "▶ RUN DESCENT"}
        </button>
        <button
          onClick={() => { setIsPlaying(false); setAnimIdx(i => Math.min(i + 1, maxSteps)); }}
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            fontWeight: 600,
            padding: "8px 16px",
            borderRadius: 6,
            border: "1px solid rgba(96,165,250,0.3)",
            background: "rgba(96,165,250,0.1)",
            color: "#60a5fa",
            cursor: "pointer",
            letterSpacing: "0.05em",
          }}
        >
          STEP →
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

      {/* Explanation */}
      <div style={{
        maxWidth: 580,
        padding: "14px 20px",
        borderRadius: 8,
        background: "rgba(15,23,42,0.5)",
        border: "1px solid rgba(100,116,139,0.12)",
        textAlign: "center",
      }}>
        <p style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 11,
          color: "#94a3b8",
          lineHeight: 1.7,
          margin: 0,
        }}>
          Each <span style={{ color: "#fbbf24" }}>yellow dot</span> is one gradient descent step.
          The <span style={{ color: "#22c55e" }}>green dot</span> is the minimum (optimal weights).
          <br/><br/>
          <strong style={{ color: "#e2e8f0" }}>Without normalization:</strong> sqft (800–3000) dominates the loss landscape,
          creating a narrow valley. The gradient bounces between the walls instead of heading straight to the minimum.
          <br/><br/>
          <strong style={{ color: "#e2e8f0" }}>With normalization:</strong> both features have the same scale,
          creating circular contours. The gradient points directly at the minimum. Same algorithm, dramatically faster convergence.
        </p>
      </div>
    </div>
  );
}
