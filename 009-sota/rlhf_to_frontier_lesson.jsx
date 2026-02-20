import { useState, useRef, useEffect } from "react";

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

// ─── Base Model vs Assistant Demo ───
function BaseVsAssistant() {
  const [mode, setMode] = useState("base");

  const examples = [
    {
      prompt: "What is the capital of France?",
      base: "What is the capital of Germany? What is the capital of Italy? What is the capital of Spain? These are common geography questions that students often...",
      assistant: "The capital of France is Paris.",
    },
    {
      prompt: "Write a haiku about rain.",
      base: "Write a haiku about snow.\nWrite a haiku about wind.\nWrite a haiku about the sun.\nPoetry prompts for creative writing class...",
      assistant: "Drops on quiet leaves,\nRhythm on the windowpane —\nEarth drinks and is still.",
    },
    {
      prompt: "Is it safe to eat raw chicken?",
      base: "Is it safe to eat raw fish? Many cultures have traditions of eating raw meat. In some recipes, chicken tartare is prepared by...",
      assistant: "No — raw chicken is not safe to eat. It commonly carries Salmonella and Campylobacter bacteria that can cause serious food poisoning. Always cook chicken to an internal temperature of 165°F (74°C).",
    },
  ];

  const [exIdx, setExIdx] = useState(0);
  const ex = examples[exIdx];

  return (
    <div>
      <div style={{ fontSize: 13, color: C.textDim, marginBottom: 14, lineHeight: 1.6 }}>
        Your GPT from Lesson 9 was trained on next-token prediction. It learned language, facts, even reasoning patterns.
        But it has <strong style={{ color: C.red }}>no concept of being helpful</strong>. It just continues text.
        Toggle between modes to see the difference.
      </div>

      <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
        <button onClick={() => setMode("base")} style={{
          flex: 1, padding: "10px 12px", borderRadius: 8,
          background: mode === "base" ? C.red + "18" : C.surface,
          border: `1.5px solid ${mode === "base" ? C.red : C.border}`,
          color: mode === "base" ? C.red : C.textDim,
          fontSize: 12, fontWeight: mode === "base" ? 700 : 500,
          cursor: "pointer", fontFamily: "inherit",
        }}>
          Base Model (GPT-3, 2020)
          <div style={{ fontSize: 9, marginTop: 2, opacity: 0.7 }}>Just predicts next token</div>
        </button>
        <button onClick={() => setMode("assistant")} style={{
          flex: 1, padding: "10px 12px", borderRadius: 8,
          background: mode === "assistant" ? C.green + "18" : C.surface,
          border: `1.5px solid ${mode === "assistant" ? C.green : C.border}`,
          color: mode === "assistant" ? C.green : C.textDim,
          fontSize: 12, fontWeight: mode === "assistant" ? 700 : 500,
          cursor: "pointer", fontFamily: "inherit",
        }}>
          After RLHF (ChatGPT, 2022)
          <div style={{ fontSize: 9, marginTop: 2, opacity: 0.7 }}>Aligned to be helpful</div>
        </button>
      </div>

      {/* Example selector */}
      <div style={{ display: "flex", gap: 4, marginBottom: 12 }}>
        {examples.map((_, i) => (
          <button key={i} onClick={() => setExIdx(i)} style={{
            padding: "6px 14px", borderRadius: 6, fontSize: 11,
            background: exIdx === i ? C.accent + "20" : C.surface,
            border: `1px solid ${exIdx === i ? C.accent : C.border}`,
            color: exIdx === i ? C.accent : C.textDim,
            cursor: "pointer", fontFamily: "inherit",
          }}>Example {i + 1}</button>
        ))}
      </div>

      {/* Prompt */}
      <div style={{
        padding: "10px 14px", borderRadius: "10px 10px 0 0",
        background: C.surfaceLight, border: `1px solid ${C.border}`,
        borderBottom: "none",
      }}>
        <div style={{ fontSize: 10, color: C.textDim, fontWeight: 600, marginBottom: 4, textTransform: "uppercase", letterSpacing: 0.5 }}>
          User prompt:
        </div>
        <div style={{ fontSize: 13, color: C.text, fontFamily: "'JetBrains Mono', monospace" }}>{ex.prompt}</div>
      </div>

      {/* Response */}
      <div style={{
        padding: "12px 14px", borderRadius: "0 0 10px 10px",
        background: mode === "base" ? C.red + "06" : C.green + "06",
        border: `1px solid ${mode === "base" ? C.red + "30" : C.green + "30"}`,
      }}>
        <div style={{ fontSize: 10, color: mode === "base" ? C.red : C.green, fontWeight: 600, marginBottom: 6, textTransform: "uppercase", letterSpacing: 0.5 }}>
          {mode === "base" ? "Base model output:" : "RLHF-aligned output:"}
        </div>
        <div style={{
          fontSize: 13, color: C.text, lineHeight: 1.6,
          fontFamily: "'JetBrains Mono', monospace",
          whiteSpace: "pre-wrap",
        }}>
          {mode === "base" ? ex.base : ex.assistant}
        </div>
      </div>

      <div style={{
        marginTop: 12, padding: "10px 14px", borderRadius: 10,
        background: C.accent + "08", border: `1px solid ${C.accent}25`,
        fontSize: 12, color: C.textDim, lineHeight: 1.6,
      }}>
        {mode === "base" ? (
          <>
            <strong style={{ color: C.red }}>The base model doesn't answer — it continues.</strong> It was trained
            to predict what text comes next in a document. A question in training data is usually followed
            by more questions, not answers.
          </>
        ) : (
          <>
            <strong style={{ color: C.green }}>The aligned model understands intent.</strong> RLHF taught the model
            that when a human asks a question, they want a direct, helpful answer — and that safety matters
            (see the raw chicken example).
          </>
        )}
      </div>
    </div>
  );
}

// ─── Three Phase Pipeline ───
function PipelineDemo() {
  const [phase, setPhase] = useState(0);

  const phases = [
    {
      title: "Phase 1: Supervised Fine-Tuning (SFT)",
      year: "2022",
      color: C.blue,
      icon: "📝",
      desc: "Train the base model on curated (instruction, response) pairs written by human contractors. This teaches the model the FORMAT of being an assistant — that questions get answers, instructions get followed.",
      details: [
        { label: "Input", value: "~100K human-written instruction/response pairs" },
        { label: "Method", value: "Standard fine-tuning — same next-token prediction loss" },
        { label: "Result", value: "Model answers questions instead of continuing text" },
        { label: "Limitation", value: "Quality capped by human-written examples" },
      ],
      code: `# SFT is just more training with curated data
# Format: each example is an instruction + ideal response
dataset = [
    {"instruction": "What is 2+2?",
     "response": "2+2 equals 4."},
    {"instruction": "Write a poem about dogs",
     "response": "Loyal companions with..."},
    # ... ~100,000 examples
]

# Same training loop as always!
for batch in dataloader:
    logits = model(batch["instruction"] + batch["response"])
    loss = cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()`,
    },
    {
      title: "Phase 2: Reward Model Training",
      year: "2022",
      color: C.purple,
      icon: "⚖️",
      desc: "Humans compare pairs of model outputs and pick which is better. A separate neural network (the reward model) learns to predict human preferences — it assigns a scalar score to any response.",
      details: [
        { label: "Input", value: "~300K human preference comparisons (A vs B)" },
        { label: "Method", value: "Train a model to predict which response humans prefer" },
        { label: "Output", value: "A function: reward(prompt, response) → score" },
        { label: "Key insight", value: "Easier to judge quality than to produce it" },
      ],
      code: `# Human labelers compare two responses
comparison = {
    "prompt": "Explain gravity",
    "response_A": "Gravity is a force...",  # ← detailed, accurate
    "response_B": "Stuff falls down lol",   # ← unhelpful
    "human_preference": "A"                  # human picks A
}

# Reward model learns from these comparisons
# Loss: reward(A) should be > reward(B) when A is preferred
loss = -log(sigmoid(reward(A) - reward(B)))

# After training: reward_model("Explain gravity", response) → 4.2
# Works on ANY new prompt/response pair!`,
    },
    {
      title: "Phase 3: RL Optimization (PPO / DPO)",
      year: "2022",
      color: C.green,
      icon: "🎯",
      desc: "Use reinforcement learning to fine-tune the SFT model to maximize the reward model's score. The model learns to generate responses that humans would rate highly — going beyond what any single human wrote.",
      details: [
        { label: "Method", value: "PPO (Proximal Policy Optimization) or DPO (Direct Preference Optimization)" },
        { label: "Signal", value: "Reward model score guides the optimization" },
        { label: "Constraint", value: "KL penalty prevents model from drifting too far from SFT baseline" },
        { label: "Result", value: "Responses better than any in the SFT training set" },
      ],
      code: `# PPO: generate response, score it, update model
for prompt in prompts:
    response = model.generate(prompt)
    reward = reward_model(prompt, response)
    
    # RL update: increase probability of high-reward responses
    # with KL penalty to stay close to SFT model
    loss = -reward + β * KL(model || sft_model)
    loss.backward()
    optimizer.step()

# DPO (simpler alternative, 2023):
# Skip the reward model entirely!
# Directly optimize on preference pairs:
loss = -log(sigmoid(
    β * (log π(chosen) - log π_ref(chosen))
      - β * (log π(rejected) - log π_ref(rejected))
))
# π = current model, π_ref = reference (SFT) model`,
    },
  ];

  const p = phases[phase];

  return (
    <div>
      {/* Phase selector */}
      <div style={{ display: "flex", gap: 4, marginBottom: 16 }}>
        {phases.map((ph, i) => (
          <button key={i} onClick={() => setPhase(i)} style={{
            flex: 1, padding: "10px 8px", borderRadius: 8, textAlign: "center",
            background: phase === i ? ph.color + "18" : C.surface,
            border: `1.5px solid ${phase === i ? ph.color : C.border}`,
            color: phase === i ? ph.color : C.textDim,
            fontSize: 12, fontWeight: phase === i ? 700 : 500,
            cursor: "pointer", fontFamily: "inherit",
          }}>
            <div>{ph.icon}</div>
            <div style={{ marginTop: 2 }}>Phase {i + 1}</div>
            <div style={{ fontSize: 9, opacity: 0.7, marginTop: 1 }}>
              {i === 0 ? "SFT" : i === 1 ? "Reward Model" : "RL"}
            </div>
          </button>
        ))}
      </div>

      {/* Phase content */}
      <div style={{
        background: p.color + "06", borderRadius: 12,
        border: `1px solid ${p.color}25`, padding: "16px 20px", marginBottom: 12,
      }}>
        <h4 style={{ margin: "0 0 6px", fontSize: 15, color: p.color }}>{p.title}</h4>
        <p style={{ margin: "0 0 14px", fontSize: 13, color: C.textDim, lineHeight: 1.6 }}>{p.desc}</p>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 14 }}>
          {p.details.map((d, i) => (
            <div key={i} style={{
              padding: "8px 12px", borderRadius: 8,
              background: C.surface, border: `1px solid ${C.border}`,
            }}>
              <div style={{ fontSize: 9, color: p.color, fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 2 }}>{d.label}</div>
              <div style={{ fontSize: 11, color: C.text, lineHeight: 1.4 }}>{d.value}</div>
            </div>
          ))}
        </div>

        <pre style={{
          background: C.bg, padding: 14, borderRadius: 8, margin: 0,
          fontSize: 11, lineHeight: 1.6, overflowX: "auto",
          fontFamily: "'JetBrains Mono', monospace",
          color: C.green, border: `1px solid ${C.border}`,
        }}>{p.code}</pre>
      </div>

      {/* Phase flow arrow */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
        padding: "10px 0",
      }}>
        {phases.map((ph, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{
              padding: "6px 14px", borderRadius: 20,
              background: i <= phase ? ph.color + "20" : C.surface,
              border: `1.5px solid ${i <= phase ? ph.color : C.border}`,
              color: i <= phase ? ph.color : C.textDim,
              fontSize: 11, fontWeight: 700,
            }}>
              {i === 0 ? "SFT Model" : i === 1 ? "Reward Model" : "Final Model"}
            </div>
            {i < 2 && <span style={{ color: C.textDim, fontSize: 16 }}>→</span>}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Preference Comparison Interactive ───
function PreferenceDemo() {
  const [selected, setSelected] = useState(null);
  const [score, setScore] = useState(0);

  const examples = [
    {
      prompt: "How do I pick a lock?",
      a: { text: "Sure! First, you'll need a tension wrench and a pick. Insert the tension wrench into the bottom of the keyhole and apply slight pressure...", label: "Responds with instructions" },
      b: { text: "I can help with locksmithing questions. If you're locked out, I'd recommend calling a licensed locksmith. If you're interested in the hobby of lock sport, I can point you to legitimate resources and communities.", label: "Helpful but responsible" },
      better: "b",
      why: "Response B is preferred: it's helpful without enabling potential harm. The reward model learns that safety-aware responses score higher."
    },
    {
      prompt: "Explain quantum computing to a 5-year-old",
      a: { text: "Quantum computing leverages superposition and entanglement of qubits in Hilbert space to perform parallel computations that classical Turing machines cannot efficiently simulate.", label: "Technically correct but wrong audience" },
      b: { text: "Imagine you have a magical coin. A regular coin is heads OR tails. But a quantum coin can be heads AND tails at the same time! A quantum computer uses lots of these magical coins to solve really big puzzles super fast.", label: "Age-appropriate, uses analogy" },
      better: "b",
      why: "Response B is preferred: it matches the audience. The reward model learns that adapting to the user's level is valued over raw accuracy."
    },
    {
      prompt: "What's 15% of 240?",
      a: { text: "36", label: "Correct, terse" },
      b: { text: "15% of 240 = 0.15 × 240 = 36. So 15% of 240 is 36.", label: "Shows work" },
      better: "b",
      why: "Response B is preferred: showing work helps the user learn and verify. The reward model learns that transparent reasoning is valued."
    },
  ];

  const [exIdx, setExIdx] = useState(0);
  const ex = examples[exIdx];

  const handleSelect = (choice) => {
    setSelected(choice);
    if (choice === ex.better) {
      setScore(s => s + 1);
    }
  };

  const nextExample = () => {
    setExIdx((exIdx + 1) % examples.length);
    setSelected(null);
  };

  return (
    <div>
      <div style={{ fontSize: 13, color: C.textDim, marginBottom: 14, lineHeight: 1.6 }}>
        <strong style={{ color: C.text }}>Be the human labeler.</strong> Read the prompt and two responses, then pick which is better.
        This is exactly what thousands of contractors did to create the preference data for RLHF.
        Your score: <strong style={{ color: C.purple }}>{score}/{exIdx + (selected ? 1 : 0)}</strong>
      </div>

      {/* Prompt */}
      <div style={{
        padding: "10px 14px", borderRadius: 10,
        background: C.surfaceLight, border: `1px solid ${C.border}`,
        marginBottom: 12,
      }}>
        <div style={{ fontSize: 10, color: C.textDim, fontWeight: 600, marginBottom: 4, textTransform: "uppercase", letterSpacing: 0.5 }}>Prompt:</div>
        <div style={{ fontSize: 13, color: C.text, fontFamily: "'JetBrains Mono', monospace" }}>{ex.prompt}</div>
      </div>

      {/* Two responses */}
      <div style={{ display: "flex", gap: 10, marginBottom: 12 }}>
        {[["a", ex.a], ["b", ex.b]].map(([key, resp]) => {
          const isSelected = selected === key;
          const isCorrect = selected && key === ex.better;
          const isWrong = selected && key !== ex.better && selected === key;
          const borderColor = isCorrect ? C.green : isWrong ? C.red : isSelected ? C.accent : C.border;

          return (
            <div key={key} onClick={() => !selected && handleSelect(key)} style={{
              flex: 1, padding: "12px 14px", borderRadius: 10, cursor: selected ? "default" : "pointer",
              background: isCorrect ? C.green + "08" : isWrong ? C.red + "08" : C.surface,
              border: `2px solid ${borderColor}`,
              transition: "all 0.2s",
              opacity: selected && !isSelected && key !== ex.better ? 0.5 : 1,
            }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: C.textDim, marginBottom: 6 }}>
                Response {key.toUpperCase()}
                {isCorrect && <span style={{ color: C.green, marginLeft: 6 }}>✓ Preferred</span>}
                {isWrong && <span style={{ color: C.red, marginLeft: 6 }}>✗</span>}
              </div>
              <div style={{ fontSize: 12, color: C.text, lineHeight: 1.5, marginBottom: 6 }}>{resp.text}</div>
              <div style={{ fontSize: 10, color: C.textDim, fontStyle: "italic" }}>{resp.label}</div>
            </div>
          );
        })}
      </div>

      {/* Feedback */}
      {selected && (
        <div style={{
          padding: "10px 14px", borderRadius: 10,
          background: C.purple + "10", border: `1px solid ${C.purple}30`,
          fontSize: 12, color: C.textDim, lineHeight: 1.6, marginBottom: 12,
        }}>
          <strong style={{ color: C.purple }}>Reward model signal:</strong> {ex.why}
        </div>
      )}

      <button onClick={nextExample} style={{
        padding: "8px 18px", borderRadius: 8,
        background: C.accent, border: "none", color: "#000",
        fontSize: 12, fontWeight: 700, cursor: "pointer", fontFamily: "inherit",
      }}>
        Next Comparison →
      </button>
    </div>
  );
}

// ─── RLVR and Reasoning Models (2024-2026) ───
function ReasoningModels() {
  const [showThinking, setShowThinking] = useState(false);

  return (
    <div>
      <div style={{ fontSize: 13, color: C.textDim, marginBottom: 16, lineHeight: 1.7 }}>
        <strong style={{ color: C.yellow }}>2024–2026</strong> brought the biggest shift since RLHF: <strong style={{ color: C.text }}>reasoning models</strong>.
        Instead of RL from <em>human feedback</em>, train with RL from <em>verifiable rewards</em> — math problems,
        code tests, logic puzzles where correctness is objective. The model learns to "think" before answering.
      </div>

      {/* RLHF vs RLVR comparison */}
      <div style={{ display: "flex", gap: 10, marginBottom: 16 }}>
        <div style={{
          flex: 1, padding: "14px 16px", borderRadius: 10,
          background: C.purple + "08", border: `1px solid ${C.purple}25`,
        }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: C.purple, marginBottom: 8 }}>
            RLHF (2022)
          </div>
          <div style={{ fontSize: 11, color: C.textDim, lineHeight: 1.6 }}>
            Signal: <strong style={{ color: C.text }}>human preferences</strong><br />
            Reward: trained reward model (subjective)<br />
            Result: helpful, harmless assistants<br />
            Limit: reward model can be gamed, expensive human annotation
          </div>
        </div>
        <div style={{
          flex: 1, padding: "14px 16px", borderRadius: 10,
          background: C.cyan + "08", border: `1px solid ${C.cyan}25`,
        }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: C.cyan, marginBottom: 8 }}>
            RLVR (2024–2026)
          </div>
          <div style={{ fontSize: 11, color: C.textDim, lineHeight: 1.6 }}>
            Signal: <strong style={{ color: C.text }}>verifiable correctness</strong><br />
            Reward: did the code pass tests? Is the math right?<br />
            Result: models that reason through problems<br />
            Breakthrough: DeepSeek-R1 showed this works at scale
          </div>
        </div>
      </div>

      {/* Thinking demo */}
      <div style={{
        background: C.surface, borderRadius: 12, border: `1px solid ${C.border}`,
        padding: "14px 18px", marginBottom: 14,
      }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: C.text, marginBottom: 8 }}>
          Inference-Time Compute: "Thinking" before answering
        </div>
        <div style={{
          padding: "10px 14px", borderRadius: 8,
          background: C.surfaceLight, border: `1px solid ${C.border}`, marginBottom: 10,
        }}>
          <div style={{ fontSize: 10, color: C.textDim, fontWeight: 600, marginBottom: 4 }}>PROMPT:</div>
          <div style={{ fontSize: 12, color: C.text, fontFamily: "'JetBrains Mono', monospace" }}>
            If a train leaves at 3pm going 60mph, and another leaves at 4pm going 90mph from the same station in the same direction, when does the second train catch up?
          </div>
        </div>

        <button onClick={() => setShowThinking(!showThinking)} style={{
          padding: "6px 14px", borderRadius: 6, fontSize: 11,
          background: showThinking ? C.cyan + "18" : C.surface,
          border: `1px solid ${showThinking ? C.cyan : C.border}`,
          color: showThinking ? C.cyan : C.textDim,
          cursor: "pointer", fontFamily: "inherit", marginBottom: 10,
        }}>
          {showThinking ? "Hide" : "Show"} thinking tokens
        </button>

        {showThinking && (
          <div style={{
            padding: "10px 14px", borderRadius: 8, marginBottom: 10,
            background: C.cyan + "06", border: `1px dashed ${C.cyan}30`,
            fontSize: 12, color: C.cyan, fontFamily: "'JetBrains Mono', monospace",
            lineHeight: 1.8, whiteSpace: "pre-wrap",
          }}>
{`Let me work through this step by step.

Train A: leaves at 3pm, speed = 60 mph
Train B: leaves at 4pm, speed = 90 mph

When Train B starts (4pm), Train A has a 1-hour head start.
Distance of Train A at 4pm = 60 × 1 = 60 miles

After 4pm, the gap closes at (90 - 60) = 30 mph.
Time to close 60-mile gap = 60 ÷ 30 = 2 hours

So Train B catches up at 4pm + 2 hours = 6pm.`}
          </div>
        )}

        <div style={{
          padding: "10px 14px", borderRadius: 8,
          background: C.green + "06", border: `1px solid ${C.green}25`,
          fontSize: 12, color: C.text, fontFamily: "'JetBrains Mono', monospace",
        }}>
          <div style={{ fontSize: 10, color: C.green, fontWeight: 600, marginBottom: 4 }}>ANSWER:</div>
          The second train catches up at 6:00 PM.
        </div>
      </div>

      <div style={{
        padding: "12px 16px", borderRadius: 10,
        background: C.accent + "08", border: `1px solid ${C.accent}25`,
        fontSize: 12, color: C.textDim, lineHeight: 1.7,
      }}>
        <strong style={{ color: C.accent }}>The key innovation:</strong> the model generates these thinking tokens
        on its own — nobody wrote chain-of-thought templates. RL with verifiable rewards (is the answer correct?)
        taught the model that <em>reasoning step by step</em> leads to more correct answers and therefore
        higher reward. The thinking behavior <strong style={{ color: C.text }}>emerged from reinforcement learning</strong>.
      </div>
    </div>
  );
}

// ─── Full Timeline to 2026 ───
function FrontierMap() {
  const [hovered, setHovered] = useState(null);

  const items = [
    { year: "2017", label: "Transformer", desc: "Attention is all you need. The architecture.", color: C.textDim, y: "✓ Lesson 8" },
    { year: "2018–20", label: "GPT Scaling", desc: "Decoder-only + next-token + scale = emergent abilities.", color: C.textDim, y: "✓ Lesson 9" },
    { year: "2022", label: "RLHF / ChatGPT", desc: "SFT → Reward Model → PPO. Base model becomes assistant.", color: C.green, y: "✓ THIS LESSON" },
    { year: "2022", label: "Chinchilla Laws", desc: "Scaling laws: optimal compute allocation. 20× more data per parameter than GPT-3.", color: C.blue, y: "" },
    { year: "2023", label: "Open Source Wave", desc: "LLaMA, Mistral, Qwen. RLHF + open weights democratizes frontier models.", color: C.blue, y: "" },
    { year: "2023", label: "DPO", desc: "Direct Preference Optimization: skip the reward model, optimize preferences directly. Simpler, often as effective.", color: C.purple, y: "" },
    { year: "2024", label: "MoE + Long Context", desc: "Mixture of Experts (only activate some params per token). Context windows → 128K–1M+ tokens.", color: C.orange, y: "" },
    { year: "2024–25", label: "RLVR / Reasoning", desc: "DeepSeek-R1: RL with verifiable rewards. Models learn to think. o1, Claude extended thinking.", color: C.cyan, y: "✓ THIS LESSON" },
    { year: "2025–26", label: "Agents & Tools", desc: "Models that browse, code, plan, and execute multi-step tasks. Computer use, MCP, tool calling.", color: C.accent, y: "← You are here" },
  ];

  return (
    <div>
      <div style={{ fontSize: 13, color: C.textDim, marginBottom: 14, lineHeight: 1.6 }}>
        Everything from 2017 to 2026 — where each concept you've learned fits, and what the frontier looks like today.
      </div>

      {items.map((item, i) => (
        <div key={i}
          onMouseEnter={() => setHovered(i)}
          onMouseLeave={() => setHovered(null)}
          style={{
            display: "flex", gap: 12, alignItems: "flex-start",
            padding: "10px 14px", marginBottom: 2, borderRadius: 8,
            background: hovered === i ? item.color + "08" : "transparent",
            borderLeft: `3px solid ${item.color}`,
            transition: "all 0.15s", cursor: "default",
          }}
        >
          <div style={{
            width: 60, flexShrink: 0, fontSize: 11, fontWeight: 700,
            color: item.color, fontFamily: "'JetBrains Mono', monospace",
          }}>{item.year}</div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: item.color }}>{item.label}</div>
            <div style={{ fontSize: 11, color: C.textDim, lineHeight: 1.5, marginTop: 2 }}>{item.desc}</div>
          </div>
          {item.y && (
            <div style={{
              fontSize: 10, color: item.y.includes("THIS") ? C.accent : C.green,
              fontWeight: 700, whiteSpace: "nowrap", marginTop: 2,
            }}>{item.y}</div>
          )}
        </div>
      ))}

      <div style={{
        marginTop: 16, padding: "14px 18px", borderRadius: 12,
        background: `linear-gradient(135deg, ${C.accent}12 0%, ${C.purple}08 100%)`,
        border: `1px solid ${C.accent}30`,
      }}>
        <div style={{ fontSize: 14, fontWeight: 800, color: C.accent, marginBottom: 8 }}>
          You've reached the frontier.
        </div>
        <div style={{ fontSize: 13, color: C.textDim, lineHeight: 1.7 }}>
          From vectors and dot products to the complete modern LLM stack: architecture (transformer),
          pre-training (next-token prediction), alignment (RLHF), and reasoning (RLVR). The remaining
          topics — MoE, flash attention, KV-cache, quantization, LoRA — are <em>engineering optimizations</em>,
          not conceptual breakthroughs. You understand the ideas. The rest is scale and craft.
        </div>
      </div>
    </div>
  );
}

// ─── Engineering Techniques Overview ───
function EngineeringTechniques() {
  const [expanded, setExpanded] = useState(null);

  const techniques = [
    {
      name: "Flash Attention",
      year: "2022",
      color: C.yellow,
      oneLiner: "Computes attention without materializing the full N×N matrix in memory",
      detail: "Standard attention creates an N×N matrix (where N is sequence length). For N=128K that's 16 billion entries — impossible. Flash Attention computes attention in tiles using GPU SRAM, reducing memory from O(N²) to O(N). This is why modern models can handle 128K+ token contexts. It's a pure engineering optimization — same math, dramatically less memory.",
    },
    {
      name: "KV-Cache",
      year: "2020+",
      color: C.green,
      oneLiner: "Cache key/value tensors during generation to avoid recomputation",
      detail: "During generation, you produce one token at a time. Without KV-cache, generating token 1000 requires recomputing attention for all 999 previous tokens. With KV-cache, you store the K and V matrices from previous steps and only compute Q for the new token. This turns O(N²) per token into O(N). It's why LLMs can generate tokens so fast despite having billions of parameters.",
    },
    {
      name: "Mixture of Experts (MoE)",
      year: "2024",
      color: C.purple,
      oneLiner: "Only activate a fraction of the model's parameters for each token",
      detail: "A 400B parameter MoE model might only use 50B parameters for any given token. The FFN layer is replaced by multiple 'expert' FFNs, and a router network decides which 2 experts (out of maybe 16) handle each token. This means you get the knowledge capacity of a huge model with the inference cost of a much smaller one. DeepSeek-V3 and Mixtral use this approach.",
    },
    {
      name: "Quantization",
      year: "2023+",
      color: C.cyan,
      oneLiner: "Reduce number precision from 16-bit to 8-bit or 4-bit",
      detail: "A 70B parameter model in FP16 needs ~140GB of memory. In 4-bit quantization, it fits in ~35GB — runnable on consumer GPUs. The model weights are stored at lower precision with minimal quality loss. Techniques like GPTQ and AWQ find optimal quantization schemes. This is why you can run Llama 3 70B on a gaming PC.",
    },
    {
      name: "LoRA",
      year: "2023",
      color: C.orange,
      oneLiner: "Fine-tune a large model by training only tiny adapter matrices",
      detail: "Instead of updating all 70B parameters, LoRA freezes the original weights and adds small low-rank matrices (maybe 0.1% of total params). Only these adapters are trained. This lets you fine-tune a frontier model on a single GPU in hours instead of needing a cluster. QLoRA combines this with quantization — fine-tune a 4-bit quantized model with LoRA adapters.",
    },
  ];

  return (
    <div>
      <div style={{ fontSize: 13, color: C.textDim, marginBottom: 14, lineHeight: 1.6 }}>
        These are the engineering innovations that make modern LLMs <em>practical</em>.
        None change the fundamental architecture — they make it faster, cheaper, and possible to run on smaller hardware.
      </div>

      {techniques.map((t, i) => (
        <div key={i} style={{
          marginBottom: 4, borderRadius: 8, overflow: "hidden",
          border: `1px solid ${expanded === i ? t.color + "40" : C.border}`,
          transition: "all 0.2s",
        }}>
          <div onClick={() => setExpanded(expanded === i ? null : i)} style={{
            display: "flex", alignItems: "center", gap: 10,
            padding: "10px 14px", cursor: "pointer",
            background: expanded === i ? t.color + "08" : C.surface,
          }}>
            <div style={{
              width: 48, fontSize: 10, fontWeight: 700, color: t.color,
              fontFamily: "'JetBrains Mono', monospace",
            }}>{t.year}</div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: expanded === i ? t.color : C.text }}>{t.name}</div>
              <div style={{ fontSize: 11, color: C.textDim }}>{t.oneLiner}</div>
            </div>
            <div style={{ color: C.textDim, fontSize: 14 }}>{expanded === i ? "−" : "+"}</div>
          </div>
          {expanded === i && (
            <div style={{
              padding: "10px 14px 12px",
              background: C.bg, borderTop: `1px solid ${C.border}`,
              fontSize: 12, color: C.textDim, lineHeight: 1.7,
            }}>{t.detail}</div>
          )}
        </div>
      ))}
    </div>
  );
}


// ─── Timeline bar ───
function Timeline() {
  const events = [
    { year: "2017", label: "Transformer", color: C.textDim, done: true },
    { year: "2018", label: "GPT", color: C.textDim, done: true },
    { year: "2020", label: "GPT-3", color: C.textDim, done: true },
    { year: "2022", label: "RLHF", color: C.green, current: true },
    { year: "2024", label: "RLVR", color: C.cyan, current: true },
    { year: "2025", label: "Reasoning", color: C.cyan, current: true },
    { year: "2026", label: "Now", color: C.accent, current: true },
  ];

  return (
    <div style={{ display: "flex", gap: 2 }}>
      {events.map((e, i) => (
        <div key={i} style={{
          flex: 1, padding: "6px 4px", textAlign: "center",
          borderTop: `3px solid ${e.current ? e.color : C.green}`,
          opacity: e.done ? 0.5 : 1,
        }}>
          <div style={{ fontSize: 9, color: C.textDim }}>{e.year}</div>
          <div style={{ fontSize: 10, fontWeight: 700, color: e.current ? e.color : C.green }}>{e.label}</div>
          {e.year === "2026" && <div style={{ fontSize: 7, color: C.accent, fontWeight: 700 }}>HERE</div>}
        </div>
      ))}
    </div>
  );
}


export default function RLHFLesson() {
  const [tab, setTab] = useState("problem");

  const tabs = [
    { id: "problem", label: "The Problem", icon: "⚠" },
    { id: "pipeline", label: "RLHF Pipeline", icon: "⚙" },
    { id: "judge", label: "Be the Labeler", icon: "⚖" },
    { id: "reasoning", label: "Reasoning (2024)", icon: "🧠" },
    { id: "engineering", label: "Engineering", icon: "⚡" },
    { id: "frontier", label: "Full Map", icon: "🗺" },
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
            }}>LESSONS 10–11</span>
            <span style={{ color: C.textDim, fontSize: 13 }}>Stage 4–5: LLMs → Frontier</span>
          </div>
          <h1 style={{
            fontSize: 28, fontWeight: 800, margin: "8px 0 6px",
            background: `linear-gradient(90deg, ${C.text}, ${C.green})`,
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          }}>
            From Base Model to the 2026 Frontier
          </h1>
          <p style={{ color: C.textDim, fontSize: 14, margin: "0 0 14px", lineHeight: 1.6 }}>
            <strong style={{ color: C.green }}>2022</strong> — RLHF turns GPT into ChatGPT.{" "}
            <strong style={{ color: C.cyan }}>2024</strong> — RLVR creates reasoning models.{" "}
            <strong style={{ color: C.accent }}>2026</strong> — where we are today. This is the final lesson.
          </p>
          <Timeline />
        </div>
      </div>

      {/* Key insight */}
      <div style={{ maxWidth: 900, margin: "20px auto 0", padding: "0 24px" }}>
        <div style={{
          background: C.green + "10", border: `1px solid ${C.green}40`,
          borderRadius: 12, padding: "16px 20px",
          display: "flex", gap: 14, alignItems: "flex-start",
        }}>
          <span style={{ fontSize: 22, marginTop: 2 }}>💡</span>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: C.green, marginBottom: 4 }}>The Key Insight</div>
            <div style={{ fontSize: 13, color: C.textDim, lineHeight: 1.6 }}>
              Pre-training teaches the model <em>what language is</em>. RLHF teaches it <em>what humans want</em>.
              RLVR teaches it <em>how to think</em>. Same architecture throughout — the transformer block from
              Lesson 8, stacked and trained with increasingly sophisticated objectives. Everything you've
              learned builds on everything before it.
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ maxWidth: 900, margin: "20px auto 0", padding: "0 24px" }}>
        <div style={{ display: "flex", gap: 3, background: C.surface, borderRadius: 10, padding: 4, flexWrap: "wrap" }}>
          {tabs.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              flex: "1 1 auto", padding: "9px 8px",
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
        {tab === "problem" && (
          <div style={{ background: C.surface, borderRadius: 14, border: `1px solid ${C.border}`, padding: 20 }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>The Alignment Problem (2020–2022)</h3>
            <p style={{ margin: "0 0 14px", fontSize: 13, color: C.textDim }}>
              GPT-3 is brilliant at predicting text but useless as an assistant. Toggle between modes to see why.
            </p>
            <BaseVsAssistant />
          </div>
        )}

        {tab === "pipeline" && (
          <div style={{ background: C.surface, borderRadius: 14, border: `1px solid ${C.border}`, padding: 20 }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>The RLHF Training Pipeline (2022)</h3>
            <p style={{ margin: "0 0 14px", fontSize: 13, color: C.textDim }}>
              Three phases turn a base model into an assistant. Click through each phase — notice how each builds on the previous.
            </p>
            <PipelineDemo />
          </div>
        )}

        {tab === "judge" && (
          <div style={{ background: C.surface, borderRadius: 14, border: `1px solid ${C.border}`, padding: 20 }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>Be the Human Labeler</h3>
            <p style={{ margin: "0 0 14px", fontSize: 13, color: C.textDim }}>
              This is Phase 2 in action. Pick which response is better — you're creating the signal that trains the reward model.
            </p>
            <PreferenceDemo />
          </div>
        )}

        {tab === "reasoning" && (
          <div style={{ background: C.surface, borderRadius: 14, border: `1px solid ${C.border}`, padding: 20 }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>Reasoning Models & RLVR (2024–2026)</h3>
            <p style={{ margin: "0 0 14px", fontSize: 13, color: C.textDim }}>
              The latest paradigm shift. Models that think before answering — and the thinking emerged from RL, not from human templates.
            </p>
            <ReasoningModels />
          </div>
        )}

        {tab === "engineering" && (
          <div style={{ background: C.surface, borderRadius: 14, border: `1px solid ${C.border}`, padding: 20 }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>Engineering the Frontier</h3>
            <p style={{ margin: "0 0 14px", fontSize: 13, color: C.textDim }}>
              These techniques make the architecture practical at scale. Click each to expand — none change the fundamental ideas, but all are essential for production.
            </p>
            <EngineeringTechniques />
          </div>
        )}

        {tab === "frontier" && (
          <div style={{ background: C.surface, borderRadius: 14, border: `1px solid ${C.border}`, padding: 20 }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 16, color: C.text }}>The Complete Map: 2017 → 2026</h3>
            <p style={{ margin: "0 0 14px", fontSize: 13, color: C.textDim }}>
              Every concept you've learned, placed on the timeline. Hover for details.
            </p>
            <FrontierMap />
          </div>
        )}
      </div>
    </div>
  );
}
