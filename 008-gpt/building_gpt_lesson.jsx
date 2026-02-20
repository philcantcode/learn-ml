import { useState, useEffect, useRef, useMemo } from "react";

const C = {
  bg: "#060910",
  surface: "#0c1220",
  surfaceLight: "#141e38",
  border: "#1e2d52",
  borderLight: "#2d4270",
  text: "#e4e8f4",
  textDim: "#6878a0",
  accent: "#e0502e",
  accentDim: "#982e18",
  green: "#28e898",
  greenDim: "#0c604a",
  red: "#f06060",
  blue: "#4890f8",
  purple: "#9068f0",
  cyan: "#30d8f0",
  yellow: "#f0c020",
  orange: "#f08020",
};

// ─── Causal Mask Demo ───
const TOKENS_SEQ = ["The", "cat", "sat", "on", "the", "mat"];

function CausalMaskDemo() {
  const [selectedToken, setSelectedToken] = useState(3);

  return (
    <div>
      <div style={{ fontSize: 13, color: C.textDim, marginBottom: 14, lineHeight: 1.6 }}>
        In the full transformer (Lesson 8), every token attends to <em>every</em> other. But in GPT, a token can only
        attend to tokens <strong style={{ color: C.accent }}>before it</strong> (and itself). This is <strong style={{ color: C.text }}>causal masking</strong> —
        it prevents the model from "cheating" by looking at future tokens it's supposed to predict.
      </div>

      {/* Token sequence with selection */}
      <div style={{ display: "flex", gap: 4, marginBottom: 14, justifyContent: "center" }}>
        {TOKENS_SEQ.map((t, i) => (
          <div key={i} onClick={() => setSelectedToken(i)} style={{
            padding: "8px 14px", borderRadius: 8, cursor: "pointer",
            background: i === selectedToken ? C.accent + "25" : i <= selectedToken ? C.blue + "15" : C.surface,
            border: `2px solid ${i === selectedToken ? C.accent : i <= selectedToken ? C.blue + "60" : C.border}`,
            color: i === selectedToken ? C.accent : i <= selectedToken ? C.blue : C.textDim,
            fontSize: 14, fontWeight: i === selectedToken ? 800 : 500,
            fontFamily: "'JetBrains Mono', monospace",
            transition: "all 0.15s",
            opacity: i > selectedToken ? 0.35 : 1,
          }}>
            {t}
          </div>
        ))}
      </div>

      {/* Attention mask grid */}
      <div style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-start" }}>
        <div>
          <div style={{ fontSize: 10, color: C.textDim, marginBottom: 4, textAlign: "center", fontWeight: 600, letterSpacing: 0.5, textTransform: "uppercase" }}>
            Causal Attention Mask
          </div>
          <div style={{ display: "flex", marginLeft: 54, marginBottom: 2 }}>
            {TOKENS_SEQ.map((t, i) => (
              <div key={i} style={{ width: 48, textAlign: "center", fontSize: 10, color: C.textDim, fontFamily: "'JetBrains Mono', monospace" }}>{t}</div>
            ))}
          </div>
          {TOKENS_SEQ.map((t, r) => (
            <div key={r} style={{ display: "flex", alignItems: "center" }}>
              <div style={{
                width: 50, textAlign: "right", paddingRight: 6, fontSize: 10,
                color: r === selectedToken ? C.accent : C.textDim,
                fontWeight: r === selectedToken ? 700 : 400,
                fontFamily: "'JetBrains Mono', monospace",
              }}>{t}</div>
              {TOKENS_SEQ.map((_, col) => {
                const canAttend = col <= r;
                const isSelected = r === selectedToken;
                const isVisible = isSelected && canAttend;
                return (
                  <div key={col} style={{
                    width: 46, height: 34, margin: 1, borderRadius: 4,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    background: isVisible ? C.blue + "40" : canAttend ? C.surfaceLight : C.red + "08",
                    border: isVisible ? `1.5px solid ${C.blue}` : canAttend ? `1px solid ${C.border}` : `1px solid ${C.red}20`,
                    fontSize: 14, transition: "all 0.15s",
                    color: canAttend ? (isVisible ? C.blue : C.textDim) : C.red + "60",
                  }}>
                    {canAttend ? "✓" : "✗"}
                  </div>
                );
              })}
            </div>
          ))}
        </div>

        <div style={{ flex: "1 1 250px", minWidth: 220, paddingTop: 20 }}>
          <div style={{
            padding: "14px 16px", borderRadius: 10,
            background: C.accent + "10", border: `1px solid ${C.accent}30`,
            marginBottom: 12,
          }}>
            <div style={{ fontSize: 13, color: C.text, lineHeight: 1.6 }}>
              <strong style={{ color: C.accent }}>"{TOKENS_SEQ[selectedToken]}"</strong> (position {selectedToken}) can see:{" "}
              <strong style={{ color: C.blue }}>
                {TOKENS_SEQ.slice(0, selectedToken + 1).map((t, i) => (
                  <span key={i}>{i > 0 ? ", " : ""}{t}</span>
                ))}
              </strong>
              {selectedToken < 5 && (
                <span style={{ color: C.red }}> — but NOT {TOKENS_SEQ.slice(selectedToken + 1).join(", ")}</span>
              )}
            </div>
          </div>

          <div style={{ fontSize: 12, color: C.textDim, lineHeight: 1.7 }}>
            <strong style={{ color: C.yellow }}>Why causal masking?</strong> During training, the model predicts the next token
            at every position simultaneously. If "sat" could see "on", it would trivially know
            the answer. The mask forces each token to predict the future using only the past — exactly
            like generation, where future tokens don't exist yet.
          </div>

          <div style={{
            marginTop: 12, padding: "10px 14px", borderRadius: 8,
            background: C.surfaceLight, border: `1px solid ${C.border}`,
            fontSize: 12, color: C.textDim,
          }}>
            <strong style={{ color: C.green }}>Implementation:</strong> add <code style={{ color: C.green }}>-∞</code> to masked
            positions before softmax. Since softmax(−∞) = 0, those tokens get zero attention weight.
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Next-Token Prediction Demo ───
function NextTokenDemo() {
  const [step, setStep] = useState(0);
  const sequence = [
    { tokens: ["The"], pred: "cat", probs: [["cat", 0.35], ["dog", 0.18], ["old", 0.12], ["big", 0.08], ["small", 0.06]] },
    { tokens: ["The", "cat"], pred: "sat", probs: [["sat", 0.28], ["is", 0.20], ["was", 0.14], ["ran", 0.09], ["ate", 0.07]] },
    { tokens: ["The", "cat", "sat"], pred: "on", probs: [["on", 0.42], ["down", 0.18], ["in", 0.10], ["by", 0.08], ["quietly", 0.05]] },
    { tokens: ["The", "cat", "sat", "on"], pred: "the", probs: [["the", 0.55], ["a", 0.20], ["my", 0.06], ["his", 0.04], ["her", 0.03]] },
    { tokens: ["The", "cat", "sat", "on", "the"], pred: "mat", probs: [["mat", 0.25], ["floor", 0.18], ["roof", 0.10], ["bed", 0.09], ["couch", 0.08]] },
  ];

  const s = sequence[step];
  const maxProb = Math.max(...s.probs.map(p => p[1]));

  return (
    <div>
      <div style={{ fontSize: 13, color: C.textDim, marginBottom: 14, lineHeight: 1.6 }}>
        GPT's training objective is breathtakingly simple: <strong style={{ color: C.text }}>predict the next token</strong>.
        Given all previous tokens, what comes next? This single task, repeated over trillions of tokens,
        is what teaches the model language, facts, reasoning, and code.
      </div>

      {/* Sequence so far */}
      <div style={{
        display: "flex", alignItems: "center", gap: 4, marginBottom: 16,
        padding: "12px 16px", background: C.surface, borderRadius: 10, border: `1px solid ${C.border}`,
        flexWrap: "wrap",
      }}>
        {s.tokens.map((t, i) => (
          <span key={i} style={{
            padding: "6px 10px", borderRadius: 6,
            background: C.blue + "18", border: `1px solid ${C.blue}40`,
            fontSize: 14, fontWeight: 600, color: C.blue,
            fontFamily: "'JetBrains Mono', monospace",
          }}>{t}</span>
        ))}
        <span style={{ fontSize: 20, color: C.accent, margin: "0 4px" }}>→</span>
        <span style={{
          padding: "6px 10px", borderRadius: 6,
          background: C.accent + "20", border: `2px solid ${C.accent}`,
          fontSize: 14, fontWeight: 800, color: C.accent,
          fontFamily: "'JetBrains Mono', monospace",
        }}>?</span>
      </div>

      {/* Probability distribution */}
      <div style={{
        background: C.surface, borderRadius: 10, border: `1px solid ${C.border}`,
        padding: "14px 18px", marginBottom: 12,
      }}>
        <div style={{ fontSize: 11, color: C.textDim, marginBottom: 8, fontWeight: 600 }}>
          Model's predicted probability distribution:
        </div>
        {s.probs.map(([token, prob], i) => (
          <div key={i} style={{
            display: "flex", alignItems: "center", gap: 8, marginBottom: 4,
          }}>
            <div style={{
              width: 60, textAlign: "right", fontSize: 12, fontWeight: i === 0 ? 700 : 400,
              color: i === 0 ? C.green : C.textDim,
              fontFamily: "'JetBrains Mono', monospace",
            }}>{token}</div>
            <div style={{
              flex: 1, height: 22, background: C.bg, borderRadius: 4,
              overflow: "hidden", border: `1px solid ${C.border}`,
            }}>
              <div style={{
                width: `${(prob / maxProb) * 100}%`, height: "100%",
                background: i === 0 ? `linear-gradient(90deg, ${C.green}80, ${C.green})` : `linear-gradient(90deg, ${C.blue}40, ${C.blue}60)`,
                borderRadius: 4,
                transition: "width 0.3s ease",
              }} />
            </div>
            <div style={{
              width: 44, fontSize: 11, color: i === 0 ? C.green : C.textDim,
              fontFamily: "'JetBrains Mono', monospace", fontWeight: i === 0 ? 700 : 400,
            }}>{(prob * 100).toFixed(0)}%</div>
          </div>
        ))}
        <div style={{ fontSize: 10, color: C.textDim, marginTop: 6, textAlign: "center" }}>
          The correct next token is "<strong style={{ color: C.green }}>{s.pred}</strong>" — the loss pushes the model to assign it higher probability.
        </div>
      </div>

      {/* Step controls */}
      <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
        {sequence.map((_, i) => (
          <button key={i} onClick={() => setStep(i)} style={{
            padding: "6px 14px", borderRadius: 8,
            background: i === step ? C.accent : C.surface,
            border: `1.5px solid ${i === step ? C.accent : C.border}`,
            color: i === step ? "#000" : C.textDim,
            fontSize: 12, fontWeight: i === step ? 700 : 400,
            cursor: "pointer", fontFamily: "inherit",
          }}>
            {i + 1}
          </button>
        ))}
      </div>

      <div style={{
        marginTop: 14, padding: "12px 16px", borderRadius: 10,
        background: C.yellow + "08", border: `1px solid ${C.yellow}25`,
        fontSize: 12, color: C.textDim, lineHeight: 1.7,
      }}>
        <strong style={{ color: C.yellow }}>The key insight (2018–2020):</strong> This simple next-token objective, scaled to
        billions of parameters and trillions of tokens, produces models that can write code, translate languages,
        reason about math, and carry on conversations. Nobody designed these capabilities — they <em>emerged</em> from scale.
        This is the GPT story.
      </div>
    </div>
  );
}

// ─── Temperature / Sampling Demo ───
function SamplingDemo() {
  const [temperature, setTemperature] = useState(1.0);
  const [samples, setSamples] = useState([]);
  const [topK, setTopK] = useState(0);

  const baseLogits = { "mat": 2.5, "floor": 1.8, "roof": 1.0, "bed": 0.9, "couch": 0.8, "table": 0.4, "wall": 0.2, "chair": 0.1, "tree": -0.2, "moon": -1.5 };
  const tokens = Object.keys(baseLogits);
  const logitValues = Object.values(baseLogits);

  const probs = useMemo(() => {
    const scaled = logitValues.map(l => l / Math.max(temperature, 0.01));
    const maxS = Math.max(...scaled);
    const exps = scaled.map(s => Math.exp(s - maxS));
    let sum = exps.reduce((a, b) => a + b, 0);

    if (topK > 0 && topK < tokens.length) {
      const indexed = exps.map((e, i) => ({ e, i })).sort((a, b) => b.e - a.e);
      const keep = new Set(indexed.slice(0, topK).map(x => x.i));
      const filtered = exps.map((e, i) => keep.has(i) ? e : 0);
      sum = filtered.reduce((a, b) => a + b, 0);
      return filtered.map(e => e / sum);
    }

    return exps.map(e => e / sum);
  }, [temperature, topK]);

  const generateSample = () => {
    const r = Math.random();
    let cum = 0;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i];
      if (r < cum) {
        setSamples(prev => [tokens[i], ...prev.slice(0, 11)]);
        return;
      }
    }
    setSamples(prev => [tokens[tokens.length - 1], ...prev.slice(0, 11)]);
  };

  const maxProb = Math.max(...probs);

  return (
    <div>
      <div style={{ fontSize: 13, color: C.textDim, marginBottom: 14, lineHeight: 1.6 }}>
        After "The cat sat on the", the model outputs probabilities for every possible next token.
        <strong style={{ color: C.text }}> Temperature</strong> controls how "creative" the sampling is.
        <strong style={{ color: C.text }}> Top-K</strong> limits which tokens can be chosen.
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 20, marginBottom: 16, flexWrap: "wrap", alignItems: "flex-end" }}>
        <div>
          <div style={{ fontSize: 11, color: C.textDim, marginBottom: 4 }}>
            Temperature: <strong style={{ color: C.accent, fontFamily: "'JetBrains Mono', monospace" }}>{temperature.toFixed(2)}</strong>
          </div>
          <input type="range" min="0.01" max="2.5" step="0.01" value={temperature}
            onChange={e => setTemperature(parseFloat(e.target.value))}
            style={{ width: 200, accentColor: C.accent }} />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: C.textDim, marginTop: 2 }}>
            <span>Deterministic</span><span>Creative</span><span>Chaotic</span>
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: C.textDim, marginBottom: 4 }}>
            Top-K: <strong style={{ color: C.cyan, fontFamily: "'JetBrains Mono', monospace" }}>{topK === 0 ? "off" : topK}</strong>
          </div>
          <input type="range" min="0" max="10" step="1" value={topK}
            onChange={e => setTopK(parseInt(e.target.value))}
            style={{ width: 120, accentColor: C.cyan }} />
        </div>
        <button onClick={generateSample} style={{
          padding: "10px 20px", borderRadius: 8, background: C.accent,
          border: "none", color: "#000", fontSize: 13, fontWeight: 700,
          cursor: "pointer", fontFamily: "inherit",
        }}>
          Sample Token
        </button>
        <button onClick={() => { for (let i = 0; i < 8; i++) setTimeout(generateSample, i * 80); }} style={{
          padding: "10px 16px", borderRadius: 8, background: C.surfaceLight,
          border: `1.5px solid ${C.border}`, color: C.textDim, fontSize: 12,
          cursor: "pointer", fontFamily: "inherit",
        }}>
          Sample ×8
        </button>
      </div>

      {/* Probability bars */}
      <div style={{
        background: C.surface, borderRadius: 10, border: `1px solid ${C.border}`,
        padding: "14px 18px", marginBottom: 12,
      }}>
        {tokens.map((tok, i) => {
          const isZeroed = probs[i] < 0.001;
          return (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
              <div style={{
                width: 50, textAlign: "right", fontSize: 12,
                color: isZeroed ? C.textDim + "40" : probs[i] === maxProb ? C.green : C.text,
                fontFamily: "'JetBrains Mono', monospace",
                fontWeight: probs[i] === maxProb ? 700 : 400,
                textDecoration: isZeroed ? "line-through" : "none",
              }}>{tok}</div>
              <div style={{
                flex: 1, height: 18, background: C.bg, borderRadius: 3,
                overflow: "hidden", border: `1px solid ${C.border}`,
              }}>
                <div style={{
                  width: `${(probs[i] / maxProb) * 100}%`, height: "100%",
                  background: isZeroed ? "transparent" :
                    probs[i] === maxProb ? C.green + "80" : C.blue + "50",
                  borderRadius: 3, transition: "width 0.2s ease",
                }} />
              </div>
              <div style={{
                width: 44, fontSize: 10, textAlign: "right",
                color: isZeroed ? C.textDim + "40" : C.textDim,
                fontFamily: "'JetBrains Mono', monospace",
              }}>{(probs[i] * 100).toFixed(1)}%</div>
            </div>
          );
        })}
      </div>

      {/* Sampled tokens history */}
      {samples.length > 0 && (
        <div style={{
          background: C.surface, borderRadius: 10, border: `1px solid ${C.border}`,
          padding: "10px 16px",
        }}>
          <div style={{ fontSize: 10, color: C.textDim, marginBottom: 6, fontWeight: 600 }}>
            Sampled tokens (most recent first):
          </div>
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
            {samples.map((s, i) => (
              <span key={i} style={{
                padding: "4px 10px", borderRadius: 6, fontSize: 12,
                background: i === 0 ? C.accent + "20" : C.surfaceLight,
                border: `1px solid ${i === 0 ? C.accent + "50" : C.border}`,
                color: i === 0 ? C.accent : C.textDim,
                fontFamily: "'JetBrains Mono', monospace",
                fontWeight: i === 0 ? 700 : 400,
              }}>{s}</span>
            ))}
          </div>
        </div>
      )}

      <div style={{
        marginTop: 14, display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8,
      }}>
        {[
          { t: "T → 0", desc: "Always picks highest prob. Deterministic, repetitive.", color: C.blue },
          { t: "T = 1.0", desc: "Samples according to model's learned distribution. Balanced.", color: C.green },
          { t: "T → 2+", desc: "Near-uniform sampling. Creative but often nonsensical.", color: C.red },
        ].map((item, i) => (
          <div key={i} style={{
            padding: "10px 12px", background: C.surface,
            border: `1px solid ${C.border}`, borderRadius: 8,
          }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: item.color, marginBottom: 2, fontFamily: "'JetBrains Mono', monospace" }}>{item.t}</div>
            <div style={{ fontSize: 10, color: C.textDim, lineHeight: 1.4 }}>{item.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── GPT Architecture Overview ───
function GPTArchitecture() {
  const [hoveredPart, setHoveredPart] = useState(null);

  const info = {
    tok: { color: C.yellow, text: "Token embedding: learned lookup table. Each of ~50,000 tokens maps to a dense vector (e.g., 768-dim for GPT-2). This IS the Word2Vec idea (2013) — but trained end-to-end with the whole model." },
    pos: { color: C.orange, text: "Positional embedding: another learned lookup table. Position 0 → vector, position 1 → vector, etc. Transformers have no built-in notion of order — this is how they know 'cat sat' differs from 'sat cat'." },
    block: { color: C.purple, text: "Transformer block — exactly what you built in Lesson 8. Causal multi-head attention + FFN + residual connections + layer norm. GPT-2 Small stacks 12 of these; GPT-3 stacks 96." },
    head: { color: C.accent, text: "Language model head: a single linear layer that projects from model dimension back to vocabulary size (~50,000). The output is logits — one score per possible next token. Softmax converts to probabilities." },
  };

  return (
    <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
      {/* Diagram */}
      <div style={{ flex: "0 0 280px" }}>
        <svg width={280} height={440}>
          {/* Input tokens */}
          <text x={140} y={18} textAnchor="middle" fill={C.textDim} fontSize={11}>"The cat sat on"</text>

          {/* Token embed */}
          <rect x={40} y={30} width={200} height={34} rx={8} fill={C.yellow + "12"} stroke={C.yellow} strokeWidth={1.5}
            onMouseEnter={() => setHoveredPart("tok")} onMouseLeave={() => setHoveredPart(null)} style={{ cursor: "default" }} />
          <text x={140} y={52} textAnchor="middle" fill={C.yellow} fontSize={11} fontWeight={600}>Token Embedding</text>

          {/* Plus */}
          <text x={140} y={78} textAnchor="middle" fill={C.green} fontSize={16} fontWeight={700}>+</text>

          {/* Positional embed */}
          <rect x={40} y={85} width={200} height={34} rx={8} fill={C.orange + "12"} stroke={C.orange} strokeWidth={1.5}
            onMouseEnter={() => setHoveredPart("pos")} onMouseLeave={() => setHoveredPart(null)} style={{ cursor: "default" }} />
          <text x={140} y={107} textAnchor="middle" fill={C.orange} fontSize={11} fontWeight={600}>Positional Embedding</text>

          {/* Arrow */}
          <line x1={140} y1={119} x2={140} y2={142} stroke={C.blue} strokeWidth={2} />

          {/* Transformer blocks */}
          {[0, 1, 2].map(i => (
            <g key={i}>
              <rect x={30} y={142 + i * 60} width={220} height={48} rx={10}
                fill={C.purple + "10"} stroke={i === 0 ? C.purple : C.border} strokeWidth={i === 0 ? 1.5 : 1}
                onMouseEnter={() => setHoveredPart("block")} onMouseLeave={() => setHoveredPart(null)} style={{ cursor: "default" }} />
              <text x={140} y={162 + i * 60} textAnchor="middle" fill={i === 0 ? C.purple : C.textDim} fontSize={10} fontWeight={600}>
                Transformer Block {i === 0 ? "1" : i === 1 ? "2" : "..."}
              </text>
              <text x={140} y={178 + i * 60} textAnchor="middle" fill={C.textDim} fontSize={9}>
                {i === 0 ? "Causal Attn + FFN + Skip" : i === 1 ? "Same structure" : "× N total"}
              </text>
              {i < 2 && <line x1={140} y1={190 + i * 60} x2={140} y2={202 + i * 60} stroke={C.blue} strokeWidth={2} />}
            </g>
          ))}

          {/* Arrow to head */}
          <line x1={140} y1={310} x2={140} y2={340} stroke={C.blue} strokeWidth={2} />

          {/* LM Head */}
          <rect x={30} y={340} width={220} height={40} rx={10}
            fill={C.accent + "12"} stroke={C.accent} strokeWidth={1.5}
            onMouseEnter={() => setHoveredPart("head")} onMouseLeave={() => setHoveredPart(null)} style={{ cursor: "default" }} />
          <text x={140} y={356} textAnchor="middle" fill={C.accent} fontSize={11} fontWeight={700}>Linear → Softmax</text>
          <text x={140} y={372} textAnchor="middle" fill={C.accent} fontSize={10}>→ next token probabilities</text>

          {/* Output */}
          <line x1={140} y1={380} x2={140} y2={405} stroke={C.text} strokeWidth={2} />
          <text x={140} y={422} textAnchor="middle" fill={C.text} fontSize={11} fontWeight={600}>
            "the" (71%) "a" (15%) ...
          </text>
        </svg>
      </div>

      {/* Info */}
      <div style={{ flex: "1 1 300px", minWidth: 260 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: C.text, marginBottom: 10 }}>
          The GPT Architecture
        </div>
        <div style={{ fontSize: 13, color: C.textDim, lineHeight: 1.7, marginBottom: 14 }}>
          GPT is a <strong style={{ color: C.text }}>decoder-only</strong> transformer. It drops the encoder half entirely — 
          just stack transformer blocks with <strong style={{ color: C.accent }}>causal masking</strong>, train on
          next-token prediction, and scale. That's the whole thing.
        </div>

        {hoveredPart && (
          <div style={{
            padding: "12px 16px", borderRadius: 10, marginBottom: 14,
            background: info[hoveredPart].color + "10",
            border: `1px solid ${info[hoveredPart].color}30`,
            fontSize: 12, color: C.textDim, lineHeight: 1.7,
          }}>
            {info[hoveredPart].text}
          </div>
        )}

        {/* Scaling table */}
        <div style={{ fontSize: 11, color: C.textDim, fontWeight: 600, marginBottom: 6, letterSpacing: 0.5, textTransform: "uppercase" }}>
          The scaling story:
        </div>
        <div style={{ borderRadius: 8, overflow: "hidden", border: `1px solid ${C.border}` }}>
          {[
            { model: "GPT-1 (2018)", params: "117M", layers: 12, dim: 768, data: "~5GB books", color: C.textDim },
            { model: "GPT-2 (2019)", params: "1.5B", layers: 48, dim: 1600, data: "~40GB web", color: C.blue },
            { model: "GPT-3 (2020)", params: "175B", layers: 96, dim: 12288, data: "~570GB mixed", color: C.purple },
            { model: "GPT-4 / Claude (2023+)", params: "???", layers: "Many", dim: "Large", data: "Trillions of tokens", color: C.accent },
          ].map((m, i) => (
            <div key={i} style={{
              display: "grid", gridTemplateColumns: "1fr 60px 44px 54px",
              padding: "8px 12px", gap: 8,
              background: i % 2 === 0 ? C.surface : C.surfaceLight,
              fontSize: 11, alignItems: "center",
            }}>
              <span style={{ color: m.color, fontWeight: 600 }}>{m.model}</span>
              <span style={{ color: C.textDim, fontFamily: "'JetBrains Mono', monospace", textAlign: "right" }}>{m.params}</span>
              <span style={{ color: C.textDim, fontFamily: "'JetBrains Mono', monospace", textAlign: "right" }}>{m.layers}L</span>
              <span style={{ color: C.textDim, fontFamily: "'JetBrains Mono', monospace", textAlign: "right", fontSize: 9 }}>{m.dim}d</span>
            </div>
          ))}
        </div>
        <div style={{ fontSize: 10, color: C.textDim, marginTop: 4, textAlign: "center" }}>
          Same architecture — just bigger. From 117M to 175B+ parameters.
        </div>
      </div>
    </div>
  );
}

// ─── Tokenization Demo ───
function TokenizationDemo() {
  const [input, setInput] = useState("The cat sat on the mat.");
  
  // Simple word-level "tokenization" for demo
  const simpleTokens = input.split(/(\s+|(?=[.,!?;:])|(?<=[.,!?;:]))/).filter(t => t.trim());
  
  // Simulated BPE-style tokens
  const bpeExamples = {
    "The cat sat on the mat.": ["The", " cat", " sat", " on", " the", " mat", "."],
    "unbelievable": ["un", "bel", "iev", "able"],
    "tokenization is fascinating": ["token", "ization", " is", " fasc", "inating"],
    "GPT-4 is amazing!": ["G", "PT", "-", "4", " is", " amazing", "!"],
    "日本語テスト": ["日", "本", "語", "テ", "スト"],
  };
  
  const bpeTokens = bpeExamples[input] || simpleTokens;
  const colors = [C.blue, C.green, C.purple, C.accent, C.cyan, C.yellow, C.orange, C.red];

  return (
    <div>
      <div style={{ fontSize: 13, color: C.textDim, marginBottom: 14, lineHeight: 1.6 }}>
        Text goes in as a string, but the model needs <strong style={{ color: C.text }}>numbers</strong>.
        Tokenization splits text into sub-word units, each mapped to an integer. GPT-2 uses Byte-Pair Encoding (BPE) with ~50,000 tokens.
      </div>

      {/* Preset examples */}
      <div style={{ display: "flex", gap: 6, marginBottom: 12, flexWrap: "wrap" }}>
        {Object.keys(bpeExamples).map((ex, i) => (
          <button key={i} onClick={() => setInput(ex)} style={{
            padding: "6px 12px", borderRadius: 6, fontSize: 11,
            background: input === ex ? C.accent + "20" : C.surface,
            border: `1px solid ${input === ex ? C.accent : C.border}`,
            color: input === ex ? C.accent : C.textDim,
            cursor: "pointer", fontFamily: "inherit",
          }}>
            {ex.length > 24 ? ex.slice(0, 22) + "..." : ex}
          </button>
        ))}
      </div>

      {/* Token visualization */}
      <div style={{
        background: C.surface, borderRadius: 10, border: `1px solid ${C.border}`,
        padding: "14px 18px",
      }}>
        <div style={{ fontSize: 10, color: C.textDim, marginBottom: 8, fontWeight: 600, letterSpacing: 0.5, textTransform: "uppercase" }}>
          BPE Tokens (simulated):
        </div>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 10 }}>
          {bpeTokens.map((t, i) => (
            <div key={i} style={{
              padding: "8px 12px", borderRadius: 6,
              background: colors[i % colors.length] + "15",
              border: `1.5px solid ${colors[i % colors.length]}50`,
              fontSize: 14, fontWeight: 600,
              color: colors[i % colors.length],
              fontFamily: "'JetBrains Mono', monospace",
              display: "flex", flexDirection: "column", alignItems: "center", gap: 2,
            }}>
              <span>{t.replace(/ /g, "·")}</span>
              <span style={{ fontSize: 9, opacity: 0.6 }}>ID: {(i * 1337 + 42) % 50000}</span>
            </div>
          ))}
        </div>
        <div style={{ fontSize: 11, color: C.textDim }}>
          {bpeTokens.length} tokens → {bpeTokens.length} integers → {bpeTokens.length} embedding vectors → into the transformer
        </div>
      </div>

      <div style={{
        marginTop: 12, padding: "10px 14px", borderRadius: 8,
        background: C.yellow + "08", border: `1px solid ${C.yellow}25`,
        fontSize: 11, color: C.textDim, lineHeight: 1.6,
      }}>
        <strong style={{ color: C.yellow }}>Why sub-word?</strong> Word-level: "unbelievable" is one token, "unbelievably" is another — 
        vocabulary explodes. Character-level: sequences get very long. 
        BPE finds a middle ground: common words stay whole, rare words split into reusable pieces.
      </div>
    </div>
  );
}

// ─── Three Paradigms ───
function ParadigmsDemo() {
  const [selected, setSelected] = useState("decoder");
  
  const paradigms = {
    encoder: {
      name: "Encoder-Only", year: "2018", model: "BERT", color: C.blue,
      mask: "Bidirectional — sees ALL tokens",
      training: "Masked language modeling: hide 15% of tokens, predict them",
      useCase: "Classification, NER, search, understanding tasks",
      diagram: ["[CLS]", "The", "[MASK]", "sat", "→", "cat"],
      note: "Great at understanding, can't generate text. Used for search engines, spam filters.",
      status: "Still used in production, but overshadowed by decoder-only models.",
    },
    decoder: {
      name: "Decoder-Only", year: "2018–2020", model: "GPT", color: C.accent,
      mask: "Causal — only sees PAST tokens",
      training: "Next-token prediction: predict the very next token",
      useCase: "Text generation, chat, code, reasoning — EVERYTHING",
      diagram: ["The", "cat", "sat", "on", "→", "the"],
      note: "The dominant paradigm. GPT-4, Claude, Llama, Gemini — all decoder-only.",
      status: "This is what 'LLM' means in 2026. Winner of the architecture wars.",
    },
    encdec: {
      name: "Encoder-Decoder", year: "2019–2020", model: "T5 / BART", color: C.green,
      mask: "Encoder: bidirectional. Decoder: causal",
      training: "Sequence-to-sequence: encode input, decode output",
      useCase: "Translation, summarization, structured generation",
      diagram: ["[EN] The cat", "→", "[DE] Le chat"],
      note: "The original transformer design. Good for translation. Largely replaced by decoder-only.",
      status: "Niche use. Most tasks work better with a big decoder-only model.",
    },
  };

  const p = paradigms[selected];

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: 16 }}>
        {Object.entries(paradigms).map(([key, val]) => (
          <button key={key} onClick={() => setSelected(key)} style={{
            flex: 1, padding: "10px 8px", borderRadius: 8,
            background: selected === key ? val.color + "18" : C.surface,
            border: `1.5px solid ${selected === key ? val.color : C.border}`,
            color: selected === key ? val.color : C.textDim,
            fontSize: 12, fontWeight: selected === key ? 700 : 500,
            cursor: "pointer", fontFamily: "inherit",
          }}>
            {val.name}
            <div style={{ fontSize: 9, marginTop: 2, opacity: 0.7 }}>{val.model} ({val.year})</div>
          </button>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 14 }}>
        {[
          { label: "Attention Mask", value: p.mask },
          { label: "Training Objective", value: p.training },
          { label: "Use Cases", value: p.useCase },
          { label: "Status in 2026", value: p.status },
        ].map((item, i) => (
          <div key={i} style={{
            padding: "10px 14px", borderRadius: 8,
            background: C.surface, border: `1px solid ${C.border}`,
          }}>
            <div style={{ fontSize: 10, color: C.textDim, marginBottom: 3, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 }}>{item.label}</div>
            <div style={{ fontSize: 12, color: C.text, lineHeight: 1.5 }}>{item.value}</div>
          </div>
        ))}
      </div>

      <div style={{
        padding: "12px 16px", borderRadius: 10,
        background: p.color + "10", border: `1px solid ${p.color}30`,
        fontSize: 12, color: C.textDim, lineHeight: 1.6,
      }}>
        <strong style={{ color: p.color }}>{p.note}</strong>
      </div>
    </div>
  );
}

// ─── Timeline ───
function Timeline() {
  const events = [
    { year: "2013", label: "Word2Vec", color: C.textDim, done: true },
    { year: "2015", label: "ResNet", color: C.textDim, done: true },
    { year: "2017", label: "Transformer", color: C.textDim, done: true },
    { year: "2018", label: "GPT / BERT", color: C.accent, current: true },
    { year: "2020", label: "GPT-3", color: C.accent, current: true },
    { year: "2022", label: "ChatGPT", color: C.blue, done: false },
    { year: "2024", label: "Reasoning", color: C.blue, done: false },
    { year: "2026", label: "Agents", color: C.blue, done: false },
  ];

  return (
    <div style={{ display: "flex", gap: 2 }}>
      {events.map((e, i) => (
        <div key={i} style={{
          flex: 1, padding: "6px 4px", textAlign: "center",
          borderTop: `3px solid ${e.current ? e.color : e.done ? C.green : C.border}`,
          opacity: e.done ? 0.6 : e.current ? 1 : 0.4,
        }}>
          <div style={{ fontSize: 9, color: C.textDim }}>{e.year}</div>
          <div style={{ fontSize: 10, fontWeight: 700, color: e.current ? e.color : e.done ? C.green : C.textDim }}>{e.label}</div>
          {e.current && <div style={{ fontSize: 7, color: C.accent, fontWeight: 700 }}>NOW</div>}
        </div>
      ))}
    </div>
  );
}


export default function GPTLesson() {
  const [tab, setTab] = useState("paradigms");

  const tabs = [
    { id: "paradigms", label: "Three Paradigms", icon: "◈" },
    { id: "causal", label: "Causal Masking", icon: "◣" },
    { id: "predict", label: "Next-Token", icon: "→" },
    { id: "tokens", label: "Tokenization", icon: "▦" },
    { id: "sampling", label: "Sampling", icon: "🎲" },
    { id: "arch", label: "Full GPT", icon: "⊞" },
  ];

  return (
    <div style={{
      background: C.bg, minHeight: "100vh", color: C.text,
      fontFamily: "'Instrument Sans', 'DM Sans', system-ui, sans-serif",
    }}>
      {/* Header */}
      <div style={{
        background: `linear-gradient(135deg, ${C.surface} 0%, ${C.bg} 100%)`,
        borderBottom: `1px solid ${C.border}`, padding: "32px 24px 24px",
      }}>
        <div style={{ maxWidth: 900, margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
            <span style={{
              background: C.accent, color: "#000", fontSize: 10, fontWeight: 800,
              padding: "3px 10px", borderRadius: 20, letterSpacing: 1,
            }}>LESSON 9</span>
            <span style={{ color: C.textDim, fontSize: 13 }}>Stage 4: Large Language Models</span>
          </div>
          <h1 style={{
            fontSize: 28, fontWeight: 800, margin: "8px 0 6px",
            background: `linear-gradient(90deg, ${C.text}, ${C.accent})`,
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          }}>
            Building GPT
          </h1>
          <p style={{ color: C.textDim, fontSize: 14, margin: "0 0 14px", lineHeight: 1.6 }}>
            <strong style={{ color: C.yellow }}>2018–2020</strong> — Take the transformer, keep only the decoder half, train it to predict the next token, 
            and scale. The result: GPT. This architecture, largely unchanged, powers every frontier model in <strong style={{ color: C.accent }}>2026</strong>.
          </p>
          <Timeline />
        </div>
      </div>

      {/* Key insight */}
      <div style={{ maxWidth: 900, margin: "20px auto 0", padding: "0 24px" }}>
        <div style={{
          background: C.accent + "10", border: `1px solid ${C.accent}40`,
          borderRadius: 12, padding: "16px 20px",
          display: "flex", gap: 14, alignItems: "flex-start",
        }}>
          <span style={{ fontSize: 22, marginTop: 2 }}>💡</span>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: C.accent, marginBottom: 4 }}>The Key Insight</div>
            <div style={{ fontSize: 13, color: C.textDim, lineHeight: 1.6 }}>
              The transformer is a general-purpose architecture (2017). The GPT insight (2018) was: <strong style={{ color: C.text }}>just use the decoder half, 
              train it to predict the next token on massive text, and scale</strong>. 
              No task-specific design. No handcrafted features. Just prediction at scale. 
              Language understanding, reasoning, code generation, translation — all <em>emerge</em> from this single objective.
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ maxWidth: 900, margin: "20px auto 0", padding: "0 24px" }}>
        <div style={{ display: "flex", gap: 3, background: C.surface, borderRadius: 10, padding: 4, flexWrap: "wrap" }}>
          {tabs.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              flex: "1 1 auto", padding: "9px 10px",
              background: tab === t.id ? C.surfaceLight : "transparent",
              border: tab === t.id ? `1px solid ${C.borderLight}` : "1px solid transparent",
              borderRadius: 8, color: tab === t.id ? C.text : C.textDim,
              fontSize: 12, fontWeight: tab === t.id ? 700 : 500,
              cursor: "pointer", fontFamily: "inherit", transition: "all 0.2s",
              whiteSpace: "nowrap",
            }}>{t.icon} {t.label}</button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div style={{ maxWidth: 900, margin: "16px auto 0", padding: "0 24px 60px" }}>
        {tab === "paradigms" && (
          <div style={{ background: C.surface, borderRadius: 14, border: `1px solid ${C.border}`, padding: 20 }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>Three Transformer Paradigms (2018–2020)</h3>
            <p style={{ margin: "0 0 14px", fontSize: 13, color: C.textDim }}>
              The 2017 transformer was encoder-decoder. Researchers quickly discovered you could use <em>just</em> one half — 
              and the decoder-only approach won decisively by 2023.
            </p>
            <ParadigmsDemo />
          </div>
        )}

        {tab === "causal" && (
          <div style={{ background: C.surface, borderRadius: 14, border: `1px solid ${C.border}`, padding: 20 }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>Causal (Masked) Self-Attention</h3>
            <p style={{ margin: "0 0 14px", fontSize: 13, color: C.textDim }}>
              Click any token to see what it can attend to. The triangular mask ensures each token
              can only see the past — exactly matching what happens during generation.
            </p>
            <CausalMaskDemo />
          </div>
        )}

        {tab === "predict" && (
          <div style={{ background: C.surface, borderRadius: 14, border: `1px solid ${C.border}`, padding: 20 }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>Next-Token Prediction (2018)</h3>
            <p style={{ margin: "0 0 14px", fontSize: 13, color: C.textDim }}>
              Step through each position. The model sees the tokens so far and outputs a probability 
              distribution over ALL possible next tokens. Training maximizes the probability of the correct one.
            </p>
            <NextTokenDemo />
          </div>
        )}

        {tab === "tokens" && (
          <div style={{ background: C.surface, borderRadius: 14, border: `1px solid ${C.border}`, padding: 20 }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>Tokenization: Text → Numbers</h3>
            <p style={{ margin: "0 0 14px", fontSize: 13, color: C.textDim }}>
              Before any of this works, text must become integers. Click the examples to see how 
              Byte-Pair Encoding splits different inputs.
            </p>
            <TokenizationDemo />
          </div>
        )}

        {tab === "sampling" && (
          <div style={{ background: C.surface, borderRadius: 14, border: `1px solid ${C.border}`, padding: 20 }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>Sampling & Temperature</h3>
            <p style={{ margin: "0 0 14px", fontSize: 13, color: C.textDim }}>
              The model gives us probabilities. How do we pick a token? Adjust temperature and top-K,
              then hit "Sample" repeatedly to see the effect.
            </p>
            <SamplingDemo />
          </div>
        )}

        {tab === "arch" && (
          <div style={{ background: C.surface, borderRadius: 14, border: `1px solid ${C.border}`, padding: 20 }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>The Complete GPT Architecture</h3>
            <p style={{ margin: "0 0 14px", fontSize: 13, color: C.textDim }}>
              Hover over each component. Every piece is something you've already learned across the previous lessons. 
              GPT is just your transformer block from Lesson 8, stacked and trained on next-token prediction.
            </p>
            <GPTArchitecture />

            <div style={{
              marginTop: 20, padding: "14px 18px", borderRadius: 10,
              background: C.accent + "10", border: `1px solid ${C.accent}30`,
            }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: C.accent, marginBottom: 6 }}>
                Distance to state-of-the-art (2026):
              </div>
              <div style={{ fontSize: 12, color: C.textDim, lineHeight: 1.8 }}>
                You now understand the <strong style={{ color: C.text }}>complete GPT architecture</strong>. To reach 2026's frontier, 
                what remains is:<br />
                <strong style={{ color: C.yellow }}>• 2022: RLHF</strong> — training models to be helpful assistants (next lesson)<br />
                <strong style={{ color: C.cyan }}>• 2024: RLVR</strong> — training models to reason (lesson after that)<br />
                <strong style={{ color: C.purple }}>• Engineering</strong> — MoE, flash attention, KV-cache, quantization, LoRA<br />
                The architecture you learned today is the same one running in production. The gap is now about
                <em> training recipes and engineering</em>, not architecture.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
