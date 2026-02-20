import { useState, useEffect, useCallback, useRef } from "react";

const C = {
  bg: "#0b0e17",
  surface: "#12172a",
  surfaceLight: "#1b2240",
  border: "#283055",
  borderLight: "#3a4878",
  text: "#e4e8f4",
  textDim: "#7d8ab4",
  accent: "#ee6b2f",
  accentDim: "#a84a1e",
  green: "#2dd4a0",
  greenDim: "#0d5c47",
  red: "#ef6b6b",
  blue: "#5b9cf5",
  purple: "#9d7afa",
  cyan: "#38d9f5",
  yellow: "#f5c842",
};

// 8x8 simplified "image" (a vertical edge pattern)
const SAMPLE_IMAGE = [
  [0, 0, 0, 1, 1, 1, 1, 1],
  [0, 0, 0, 1, 1, 1, 1, 1],
  [0, 0, 0, 1, 1, 1, 1, 1],
  [0, 0, 0, 1, 1, 1, 1, 1],
  [0, 0, 0, 1, 1, 1, 1, 1],
  [0, 0, 0, 1, 1, 1, 1, 1],
  [0, 0, 0, 1, 1, 1, 1, 1],
  [0, 0, 0, 1, 1, 1, 1, 1],
];

const SAMPLE_IMAGE_2 = [
  [1, 1, 1, 1, 1, 0, 0, 0],
  [1, 1, 1, 1, 0, 0, 0, 0],
  [1, 1, 1, 0, 0, 0, 0, 0],
  [1, 1, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 1, 1],
  [0, 0, 0, 0, 0, 1, 1, 1],
  [0, 0, 0, 0, 1, 1, 1, 1],
];

const FILTERS = [
  {
    name: "Vertical Edge",
    kernel: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
    color: C.blue,
    desc: "Bright where vertical edges are",
  },
  {
    name: "Horizontal Edge",
    kernel: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
    color: C.green,
    desc: "Bright where horizontal edges are",
  },
  {
    name: "Diagonal Edge",
    kernel: [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]],
    color: C.purple,
    desc: "Detects diagonal boundaries",
  },
  {
    name: "Sharpen",
    kernel: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    color: C.yellow,
    desc: "Enhances local contrast",
  },
];

function applyConvolution(image, kernel) {
  const rows = image.length;
  const cols = image[0].length;
  const kSize = kernel.length;
  const pad = Math.floor(kSize / 2);
  const outRows = rows - kSize + 1;
  const outCols = cols - kSize + 1;
  const result = [];

  for (let r = 0; r < outRows; r++) {
    const row = [];
    for (let c = 0; c < outCols; c++) {
      let sum = 0;
      for (let kr = 0; kr < kSize; kr++) {
        for (let kc = 0; kc < kSize; kc++) {
          sum += image[r + kr][c + kc] * kernel[kr][kc];
        }
      }
      row.push(sum);
    }
    result.push(row);
  }
  return result;
}

function maxPool2x2(featureMap) {
  const rows = featureMap.length;
  const cols = featureMap[0].length;
  const result = [];
  for (let r = 0; r < rows - 1; r += 2) {
    const row = [];
    for (let c = 0; c < cols - 1; c += 2) {
      row.push(Math.max(
        featureMap[r][c], featureMap[r][c + 1],
        featureMap[r + 1][c], featureMap[r + 1][c + 1]
      ));
    }
    result.push(row);
  }
  return result;
}

function Grid({ data, cellSize = 36, highlightRegion, filterColor, label, showValues = true, valueScale = 1 }) {
  const maxAbs = Math.max(...data.flat().map(Math.abs), 1);

  return (
    <div style={{ display: "inline-block" }}>
      {label && (
        <div style={{ fontSize: 10, color: C.textDim, marginBottom: 4, textAlign: "center", fontWeight: 600, letterSpacing: 0.5, textTransform: "uppercase" }}>
          {label}
        </div>
      )}
      <div style={{
        display: "grid",
        gridTemplateColumns: `repeat(${data[0].length}, ${cellSize}px)`,
        gap: 1,
        background: C.border,
        border: `2px solid ${filterColor || C.border}`,
        borderRadius: 6,
        padding: 1,
        width: "fit-content",
      }}>
        {data.map((row, r) =>
          row.map((val, c) => {
            const isHighlighted = highlightRegion &&
              r >= highlightRegion.r && r < highlightRegion.r + highlightRegion.h &&
              c >= highlightRegion.c && c < highlightRegion.c + highlightRegion.w;

            const normVal = val / maxAbs;
            let bgColor;
            if (normVal > 0) {
              const intensity = Math.min(normVal * valueScale, 1);
              bgColor = `rgba(93, 160, 245, ${intensity * 0.7 + 0.05})`;
            } else if (normVal < 0) {
              const intensity = Math.min(-normVal * valueScale, 1);
              bgColor = `rgba(239, 107, 107, ${intensity * 0.7 + 0.05})`;
            } else {
              bgColor = C.surface;
            }

            return (
              <div
                key={`${r}-${c}`}
                style={{
                  width: cellSize,
                  height: cellSize,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: bgColor,
                  border: isHighlighted ? `2px solid ${C.accent}` : "none",
                  boxSizing: "border-box",
                  fontSize: cellSize < 30 ? 8 : 10,
                  fontFamily: "'JetBrains Mono', monospace",
                  color: isHighlighted ? C.accent : (Math.abs(normVal) > 0.5 ? "#fff" : C.textDim),
                  fontWeight: isHighlighted ? 800 : 500,
                  transition: "all 0.15s ease",
                }}
              >
                {showValues ? (Number.isInteger(val) ? val : val.toFixed(1)) : ""}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

function ConvolutionDemo() {
  const [filterIdx, setFilterIdx] = useState(0);
  const [posR, setPosR] = useState(0);
  const [posC, setPosC] = useState(0);
  const [imageIdx, setImageIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const timerRef = useRef(null);

  const images = [SAMPLE_IMAGE, SAMPLE_IMAGE_2];
  const image = images[imageIdx];
  const filter = FILTERS[filterIdx];
  const featureMap = applyConvolution(image, filter.kernel);
  const outRows = featureMap.length;
  const outCols = featureMap[0].length;

  const safeR = Math.min(posR, outRows - 1);
  const safeC = Math.min(posC, outCols - 1);

  // Compute the single convolution value at current position
  let currentVal = 0;
  const multiplications = [];
  for (let kr = 0; kr < 3; kr++) {
    for (let kc = 0; kc < 3; kc++) {
      const imgVal = image[safeR + kr][safeC + kc];
      const kVal = filter.kernel[kr][kc];
      const product = imgVal * kVal;
      currentVal += product;
      multiplications.push({ imgVal, kVal, product });
    }
  }

  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        setPosC(c => {
          if (c >= outCols - 1) {
            setPosR(r => {
              if (r >= outRows - 1) {
                setIsPlaying(false);
                return 0;
              }
              return r + 1;
            });
            return 0;
          }
          return c + 1;
        });
      }, 300);
    }
    return () => clearInterval(timerRef.current);
  }, [isPlaying, outCols, outRows]);

  return (
    <div>
      {/* Filter selector */}
      <div style={{ display: "flex", gap: 6, marginBottom: 16, flexWrap: "wrap" }}>
        {FILTERS.map((f, i) => (
          <button
            key={i}
            onClick={() => { setFilterIdx(i); setPosR(0); setPosC(0); }}
            style={{
              padding: "8px 14px",
              background: filterIdx === i ? f.color + "22" : C.surface,
              border: `1.5px solid ${filterIdx === i ? f.color : C.border}`,
              borderRadius: 8,
              color: filterIdx === i ? f.color : C.textDim,
              fontSize: 12,
              fontWeight: filterIdx === i ? 700 : 500,
              cursor: "pointer",
              fontFamily: "inherit",
              transition: "all 0.2s",
            }}
          >
            {f.name}
          </button>
        ))}
        <div style={{ flex: 1 }} />
        <button
          onClick={() => { setImageIdx(i => (i + 1) % images.length); setPosR(0); setPosC(0); }}
          style={{
            padding: "8px 14px",
            background: C.surfaceLight,
            border: `1.5px solid ${C.border}`,
            borderRadius: 8,
            color: C.textDim,
            fontSize: 12,
            cursor: "pointer",
            fontFamily: "inherit",
          }}
        >
          Switch Image
        </button>
      </div>

      <div style={{ display: "flex", gap: 20, alignItems: "flex-start", flexWrap: "wrap" }}>
        {/* Input image */}
        <div>
          <Grid
            data={image}
            cellSize={38}
            highlightRegion={{ r: safeR, c: safeC, h: 3, w: 3 }}
            label="Input Image (8×8)"
          />
        </div>

        {/* Multiplication symbol */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", paddingTop: 50 }}>
          <span style={{ fontSize: 28, color: C.accent }}>×</span>
        </div>

        {/* Filter/kernel */}
        <div>
          <Grid data={filter.kernel} cellSize={38} filterColor={filter.color} label={`Filter (3×3)`} />
          <div style={{ fontSize: 11, color: filter.color, marginTop: 6, textAlign: "center" }}>
            {filter.desc}
          </div>
        </div>

        {/* Arrow */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", paddingTop: 50 }}>
          <span style={{ fontSize: 28, color: C.accent }}>→</span>
        </div>

        {/* Output feature map */}
        <div>
          <Grid
            data={featureMap}
            cellSize={38}
            filterColor={filter.color}
            highlightRegion={{ r: safeR, c: safeC, h: 1, w: 1 }}
            label={`Feature Map (${outRows}×${outCols})`}
            valueScale={0.3}
          />
        </div>
      </div>

      {/* Current computation */}
      <div style={{
        marginTop: 16,
        background: C.surface,
        borderRadius: 10,
        border: `1px solid ${C.border}`,
        padding: "12px 16px",
      }}>
        <div style={{ fontSize: 11, color: C.textDim, marginBottom: 6 }}>
          Position ({safeR}, {safeC}): Sum of element-wise products:
        </div>
        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: C.text, lineHeight: 1.8 }}>
          {multiplications.map((m, i) => (
            <span key={i}>
              <span style={{ color: C.blue }}>{m.imgVal}</span>
              <span style={{ color: C.textDim }}>×</span>
              <span style={{ color: filter.color }}>{m.kVal}</span>
              {i < 8 ? <span style={{ color: C.textDim }}> + </span> : ""}
            </span>
          ))}
          <span style={{ color: C.accent }}> = {currentVal}</span>
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 8, marginTop: 12, alignItems: "center" }}>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          style={{
            padding: "8px 20px",
            background: isPlaying ? C.red + "22" : C.accent,
            border: isPlaying ? `1.5px solid ${C.red}` : "none",
            borderRadius: 8,
            color: isPlaying ? C.red : "#000",
            fontSize: 12,
            fontWeight: 700,
            cursor: "pointer",
            fontFamily: "inherit",
          }}
        >
          {isPlaying ? "⏸ Pause" : "▶ Slide Filter"}
        </button>
        <button
          onClick={() => { setPosR(0); setPosC(0); setIsPlaying(false); }}
          style={{
            padding: "8px 14px",
            background: C.surfaceLight,
            border: `1.5px solid ${C.border}`,
            borderRadius: 8,
            color: C.textDim,
            fontSize: 12,
            cursor: "pointer",
            fontFamily: "inherit",
          }}
        >
          Reset
        </button>
        <div style={{ marginLeft: "auto", fontSize: 11, color: C.textDim }}>
          Or click the feature map to jump to a position ↑
        </div>
      </div>
    </div>
  );
}

function PoolingDemo() {
  const filter = FILTERS[0];
  const featureMap = applyConvolution(SAMPLE_IMAGE, filter.kernel);
  const pooled = maxPool2x2(featureMap);
  const [highlightPool, setHighlightPool] = useState(null);

  return (
    <div style={{ display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap" }}>
      <div>
        <Grid data={featureMap} cellSize={38} filterColor={C.blue} label="Feature Map (6×6)" valueScale={0.3}
          highlightRegion={highlightPool ? { r: highlightPool.r * 2, c: highlightPool.c * 2, h: 2, w: 2 } : null}
        />
      </div>
      <div style={{ paddingTop: 50, display: "flex", flexDirection: "column", alignItems: "center" }}>
        <span style={{ fontSize: 12, color: C.cyan, fontWeight: 700, marginBottom: 4 }}>Max Pool</span>
        <span style={{ fontSize: 12, color: C.cyan }}>2×2</span>
        <span style={{ fontSize: 22, color: C.cyan, marginTop: 4 }}>→</span>
      </div>
      <div>
        <Grid data={pooled} cellSize={46} filterColor={C.cyan} label="Pooled (3×3)" valueScale={0.3} />
        <div style={{ fontSize: 11, color: C.cyan, marginTop: 6, textAlign: "center" }}>
          Each cell = max of 2×2 block
        </div>
      </div>
      <div style={{ flex: "1 1 200px", minWidth: 200, paddingTop: 10 }}>
        <div style={{ fontSize: 13, color: C.text, fontWeight: 600, marginBottom: 6 }}>Why pool?</div>
        <div style={{ fontSize: 12, color: C.textDim, lineHeight: 1.7 }}>
          <strong style={{ color: C.cyan }}>Reduce size</strong> — 6×6 becomes 3×3 (75% fewer values).
          With real images, this is the difference between tractable and impossible.
        </div>
        <div style={{ fontSize: 12, color: C.textDim, lineHeight: 1.7, marginTop: 8 }}>
          <strong style={{ color: C.cyan }}>Translation invariance</strong> — if the edge shifts by 1 pixel, the max-pooled result barely changes. The network becomes robust to small shifts.
        </div>
      </div>
    </div>
  );
}

function HierarchyDemo() {
  const layers = [
    { name: "Conv Layer 1", size: "32×32", filters: 16, detects: "Edges, gradients, colors", icon: "╱╲│─", color: C.blue },
    { name: "Pool 1", size: "16×16", filters: 16, detects: "Downsampled edges", icon: "↓", color: C.cyan },
    { name: "Conv Layer 2", size: "16×16", filters: 32, detects: "Corners, textures, simple shapes", icon: "◢◣∟○", color: C.green },
    { name: "Pool 2", size: "8×8", filters: 32, detects: "Downsampled shapes", icon: "↓", color: C.cyan },
    { name: "Conv Layer 3", size: "8×8", filters: 64, detects: "Eyes, wheels, windows, fur", icon: "👁🔵◻", color: C.purple },
    { name: "Pool 3", size: "4×4", filters: 64, detects: "Downsampled parts", icon: "↓", color: C.cyan },
    { name: "Flatten + FC", size: "1024→10", filters: null, detects: "Cat? Dog? Car? → Classification", icon: "🎯", color: C.accent },
  ];

  const [hoveredLayer, setHoveredLayer] = useState(null);

  return (
    <div>
      <div style={{ display: "flex", gap: 2, alignItems: "stretch" }}>
        {layers.map((layer, i) => (
          <div
            key={i}
            onMouseEnter={() => setHoveredLayer(i)}
            onMouseLeave={() => setHoveredLayer(null)}
            style={{
              flex: 1,
              padding: "12px 6px",
              background: hoveredLayer === i ? layer.color + "18" : C.surface,
              border: `1.5px solid ${hoveredLayer === i ? layer.color : C.border}`,
              borderRadius: 8,
              textAlign: "center",
              cursor: "default",
              transition: "all 0.2s",
              display: "flex",
              flexDirection: "column",
              gap: 6,
              minWidth: 0,
            }}
          >
            <div style={{ fontSize: 16 }}>{layer.icon}</div>
            <div style={{ fontSize: 10, fontWeight: 700, color: layer.color, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {layer.name}
            </div>
            <div style={{ fontSize: 9, color: C.textDim }}>{layer.size}</div>
            {hoveredLayer === i && (
              <div style={{ fontSize: 10, color: C.text, lineHeight: 1.4, marginTop: 2 }}>
                {layer.detects}
              </div>
            )}
          </div>
        ))}
      </div>
      <div style={{
        marginTop: 12,
        display: "flex",
        justifyContent: "space-between",
        fontSize: 11,
      }}>
        <span style={{ color: C.blue }}>← Simple features</span>
        <span style={{ color: C.textDim }}>Each layer combines features from the previous one</span>
        <span style={{ color: C.purple }}>Complex features →</span>
      </div>
    </div>
  );
}

function WhyNotFCDemo() {
  const [mode, setMode] = useState("fc");

  return (
    <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
      <div style={{ flex: "1 1 320px" }}>
        <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
          {[
            { id: "fc", label: "Fully Connected", color: C.red },
            { id: "conv", label: "Convolutional", color: C.green },
          ].map(m => (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              style={{
                flex: 1,
                padding: "10px 12px",
                background: mode === m.id ? m.color + "18" : C.surface,
                border: `1.5px solid ${mode === m.id ? m.color : C.border}`,
                borderRadius: 8,
                color: mode === m.id ? m.color : C.textDim,
                fontSize: 13,
                fontWeight: mode === m.id ? 700 : 500,
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              {m.label}
            </button>
          ))}
        </div>
        {mode === "fc" ? (
          <div style={{ fontSize: 13, color: C.textDim, lineHeight: 1.8 }}>
            <p style={{ margin: "0 0 10px" }}>
              A <strong style={{ color: C.red }}>fully connected</strong> network treats an image as a flat list of pixels. A tiny 32×32 color image = <strong style={{ color: C.red }}>3,072 inputs</strong>.
            </p>
            <p style={{ margin: "0 0 10px" }}>
              With 512 neurons in the first hidden layer, that's <strong style={{ color: C.red }}>3,072 × 512 = 1,572,864 parameters</strong> — just for the first layer!
            </p>
            <p style={{ margin: "0 0 10px" }}>
              Worse: it has <strong style={{ color: C.red }}>no concept of spatial structure</strong>. It doesn't know that pixel (0,0) is next to pixel (0,1). If you shift the image by 1 pixel, the network sees entirely different inputs.
            </p>
            <div style={{
              background: C.red + "12",
              border: `1px solid ${C.red}30`,
              borderRadius: 8,
              padding: "10px 14px",
              fontSize: 12,
            }}>
              <strong style={{ color: C.red }}>Problems:</strong> Too many parameters, no spatial awareness, not shift-invariant, doesn't scale to real images (224×224 = 150,528 inputs!)
            </div>
          </div>
        ) : (
          <div style={{ fontSize: 13, color: C.textDim, lineHeight: 1.8 }}>
            <p style={{ margin: "0 0 10px" }}>
              A <strong style={{ color: C.green }}>convolutional</strong> layer slides a small filter (e.g., 3×3) across the image. One filter = <strong style={{ color: C.green }}>3×3×3 + 1 = 28 parameters</strong> (for 3 color channels + bias).
            </p>
            <p style={{ margin: "0 0 10px" }}>
              With 16 filters, that's <strong style={{ color: C.green }}>16 × 28 = 448 parameters</strong> — vs 1.5 million! And it works on <em>any</em> image size.
            </p>
            <p style={{ margin: "0 0 10px" }}>
              The filter <strong style={{ color: C.green }}>shares weights</strong> across all positions — so it detects the same edge whether it's in the top-left or bottom-right. Spatial structure is baked into the design.
            </p>
            <div style={{
              background: C.green + "12",
              border: `1px solid ${C.green}30`,
              borderRadius: 8,
              padding: "10px 14px",
              fontSize: 12,
            }}>
              <strong style={{ color: C.green }}>Advantages:</strong> Far fewer parameters, built-in spatial awareness, weight sharing = shift invariance, scales to any image size
            </div>
          </div>
        )}
      </div>

      {/* Parameter comparison visual */}
      <div style={{ flex: "0 0 180px", display: "flex", flexDirection: "column", justifyContent: "center", gap: 12 }}>
        <div style={{ textAlign: "center", fontSize: 10, color: C.textDim, fontWeight: 600, letterSpacing: 0.5, textTransform: "uppercase" }}>
          First Layer Parameters
        </div>
        <div style={{ display: "flex", gap: 12, justifyContent: "center", alignItems: "flex-end", height: 120 }}>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
            <div style={{
              width: 50,
              height: 110,
              background: `linear-gradient(to top, ${C.red}, ${C.red}88)`,
              borderRadius: "6px 6px 0 0",
            }} />
            <span style={{ fontSize: 9, color: C.red, fontWeight: 700 }}>1.5M</span>
            <span style={{ fontSize: 9, color: C.textDim }}>FC</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
            <div style={{
              width: 50,
              height: 4,
              background: `linear-gradient(to top, ${C.green}, ${C.green}88)`,
              borderRadius: "6px 6px 0 0",
            }} />
            <span style={{ fontSize: 9, color: C.green, fontWeight: 700 }}>448</span>
            <span style={{ fontSize: 9, color: C.textDim }}>Conv</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function CNNLesson() {
  const [tab, setTab] = useState("why");

  const tabs = [
    { id: "why", label: "Why CNNs?", icon: "?" },
    { id: "conv", label: "Convolution", icon: "⊞" },
    { id: "pool", label: "Pooling", icon: "↓" },
    { id: "arch", label: "Full Architecture", icon: "⊟" },
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
            }}>LESSON 6</span>
            <span style={{ color: C.textDim, fontSize: 13 }}>Stage 2: Deep Learning Fundamentals</span>
          </div>
          <h1 style={{
            fontSize: 28,
            fontWeight: 800,
            margin: "8px 0 6px",
            background: `linear-gradient(90deg, ${C.text}, ${C.accent})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            letterSpacing: "-0.5px",
          }}>
            Convolutional Neural Networks
          </h1>
          <p style={{ color: C.textDim, fontSize: 14, margin: 0, lineHeight: 1.6 }}>
            Encoding <em>spatial structure</em> into the network itself. Instead of learning every pixel connection, learn small reusable <code style={{
              background: C.accent + "22",
              color: C.accent,
              padding: "2px 6px",
              borderRadius: 4,
              fontSize: 13,
            }}>filters</code> that detect features anywhere in the image.
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
              The Big Idea
            </div>
            <div style={{ fontSize: 13, color: C.textDim, lineHeight: 1.6 }}>
              Your from-scratch network treated inputs as a flat vector — fine for 2D points, disastrous for images.
              CNNs exploit a simple insight: <strong style={{ color: C.text }}>nearby pixels are related</strong>.
              A 3×3 filter that detects a vertical edge works at <em>every</em> position in the image — this is <strong style={{ color: C.text }}>weight sharing</strong>, and it's what makes vision tractable.
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
                padding: "10px 12px",
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

        {tab === "why" && (
          <div style={{
            background: C.surface,
            borderRadius: 14,
            border: `1px solid ${C.border}`,
            padding: 20,
          }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>
              Why can't we just use fully connected layers?
            </h3>
            <p style={{ margin: "0 0 16px", fontSize: 13, color: C.textDim }}>
              Toggle between the two approaches to see the problem — and the solution.
            </p>
            <WhyNotFCDemo />
          </div>
        )}

        {tab === "conv" && (
          <div style={{
            background: C.surface,
            borderRadius: 14,
            border: `1px solid ${C.border}`,
            padding: 20,
          }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>
              How Convolution Works
            </h3>
            <p style={{ margin: "0 0 16px", fontSize: 13, color: C.textDim }}>
              A small filter slides across the image. At each position, multiply element-wise and sum.
              The result is a <strong style={{ color: C.accent }}>feature map</strong> — a heatmap of where that feature appears.
            </p>
            <ConvolutionDemo />
          </div>
        )}

        {tab === "pool" && (
          <div style={{
            background: C.surface,
            borderRadius: 14,
            border: `1px solid ${C.border}`,
            padding: 20,
          }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>
              Max Pooling: Downsample and Gain Invariance
            </h3>
            <p style={{ margin: "0 0 16px", fontSize: 13, color: C.textDim }}>
              After convolution, max pooling takes the strongest activation from each 2×2 block.
              This shrinks the spatial dimensions while keeping the important features.
            </p>
            <PoolingDemo />
          </div>
        )}

        {tab === "arch" && (
          <div>
            <div style={{
              background: C.surface,
              borderRadius: 14,
              border: `1px solid ${C.border}`,
              padding: 20,
              marginBottom: 16,
            }}>
              <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>
                The Feature Hierarchy
              </h3>
              <p style={{ margin: "0 0 16px", fontSize: 13, color: C.textDim }}>
                Hover over each layer. Early layers detect simple features (edges). Later layers combine them into complex features (eyes, wheels). This hierarchy is how CNNs "see."
              </p>
              <HierarchyDemo />
            </div>

            {/* PyTorch code */}
            <div style={{
              background: C.surface,
              borderRadius: 14,
              border: `1px solid ${C.border}`,
              padding: 20,
            }}>
              <h3 style={{ margin: "0 0 12px", fontSize: 15, color: C.text }}>
                CNN in PyTorch — Same Training Loop!
              </h3>
              <pre style={{
                background: C.bg,
                padding: 16,
                borderRadius: 10,
                fontSize: 11.5,
                lineHeight: 1.7,
                overflowX: "auto",
                fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                border: `1px solid ${C.border}`,
                margin: 0,
              }}>
                <span style={{ color: C.textDim }}>{"# Define a CNN — compare to CircleNet!\n"}</span>
                <span style={{ color: C.purple }}>{"class"}</span>{" "}<span style={{ color: C.green }}>{"ImageNet"}</span>{"(nn.Module):\n"}
                {"    "}<span style={{ color: C.purple }}>{"def"}</span>{" __init__(self):\n"}
                {"        super().__init__()\n"}
                {"        "}<span style={{ color: C.textDim }}>{"# Conv layers: learn filters\n"}</span>
                {"        self.conv1 = "}<span style={{ color: C.blue }}>{"nn.Conv2d(3, 16, 3, padding=1)"}</span>{"  "}<span style={{ color: C.textDim }}>{"# 3 channels → 16 filters\n"}</span>
                {"        self.conv2 = "}<span style={{ color: C.blue }}>{"nn.Conv2d(16, 32, 3, padding=1)"}</span>{" "}<span style={{ color: C.textDim }}>{"# 16 → 32 filters\n"}</span>
                {"        self.conv3 = "}<span style={{ color: C.blue }}>{"nn.Conv2d(32, 64, 3, padding=1)"}</span>{" "}<span style={{ color: C.textDim }}>{"# 32 → 64 filters\n"}</span>
                {"        self.pool = nn.MaxPool2d(2, 2)\n"}
                {"        "}<span style={{ color: C.textDim }}>{"# FC layer: classify\n"}</span>
                {"        self.fc = "}<span style={{ color: C.accent }}>{"nn.Linear(64 * 4 * 4, 10)"}</span>{"  "}<span style={{ color: C.textDim }}>{"# 10 classes\n"}</span>
                {"\n"}
                {"    "}<span style={{ color: C.purple }}>{"def"}</span>{" forward(self, x):\n"}
                {"        x = self.pool("}<span style={{ color: C.blue }}>{"F.relu(self.conv1(x))"}</span>{")  "}<span style={{ color: C.textDim }}>{"# 32→16\n"}</span>
                {"        x = self.pool("}<span style={{ color: C.blue }}>{"F.relu(self.conv2(x))"}</span>{")  "}<span style={{ color: C.textDim }}>{"# 16→8\n"}</span>
                {"        x = self.pool("}<span style={{ color: C.blue }}>{"F.relu(self.conv3(x))"}</span>{")  "}<span style={{ color: C.textDim }}>{"# 8→4\n"}</span>
                {"        x = x.view(x.size(0), -1)        "}<span style={{ color: C.textDim }}>{"# flatten\n"}</span>
                {"        x = self.fc(x)                    "}<span style={{ color: C.textDim }}>{"# classify\n"}</span>
                {"        "}<span style={{ color: C.purple }}>{"return"}</span>{" x\n\n"}
                <span style={{ color: C.textDim }}>{"# Training loop — EXACTLY THE SAME!\n"}</span>
                <span style={{ color: C.purple }}>{"for"}</span>{" X_batch, y_batch "}<span style={{ color: C.purple }}>{"in"}</span>{" loader:\n"}
                {"    predictions = model(X_batch)\n"}
                {"    loss = loss_fn(predictions, y_batch)\n"}
                {"    "}<span style={{ color: C.accent }}>{"loss.backward()"}</span>{"       "}<span style={{ color: C.textDim }}>{"# autograd handles conv gradients!\n"}</span>
                {"    optimizer.step()\n"}
                {"    optimizer.zero_grad()"}
              </pre>
              <div style={{
                marginTop: 14,
                padding: "12px 16px",
                background: C.accent + "10",
                borderRadius: 8,
                border: `1px solid ${C.accent}30`,
              }}>
                <div style={{ fontSize: 13, color: C.text, lineHeight: 1.6 }}>
                  <strong style={{ color: C.accent }}>Notice: the training loop didn't change.</strong>{" "}
                  You swapped <code>CircleNet</code> for a CNN, and everything else stays the same.
                  PyTorch's autograd computes gradients through conv layers, pooling, and ReLU — all automatically.
                  This is why learning the loop pattern matters.
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
