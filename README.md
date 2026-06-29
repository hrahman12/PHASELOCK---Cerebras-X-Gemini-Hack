<p align="center">
  <img src="assets/frontpage.png" alt="PhaseLock — TorusNet × FusionCoreNet" width="900"><br>
  <img src="assets/page 2.png" alt="Two engines, one fabric" width="900"><br>
  <img src="assets/page 3.png" alt="From prompt to verified consensus" width="900"><br>
  <img src="assets/page 4.png" alt="The twenty cortices" width="900"><br>
  <img src="assets/page 5.png" alt="Mapped to the Cerebras WSE-3" width="900">
</p>

# PhaseLock — TorusNet × FusionCoreNet

**Reasoning-stability control for multi-agent reasoning, inspired by the Cerebras Wafer-Scale Engine.**

PhaseLock runs two cooperating agents inside one dashboard:

- **TorusNet** — the *reasoning* engine. It maps a society of 15,000 specialist agents onto a 3D toroidal mesh, routes sparse "wavelet" packets between them, and produces a final answer using **Gemma‑4 (`gemma-4-31b`) on Cerebras**.
- **FusionCoreNet** — the *control* engine. It never answers the user. It watches TorusNet's graph state, computes real stability math (graph Laplacian, entropy, pressure/capacity, transport, PID, Lyapunov, loop-stability), and issues repair commands that push the reasoning toward convergence.

> TorusNet thinks. FusionCoreNet controls the physics of the thinking.

---

## What it actually is (honest summary)

- The **final answer is real** — a live call to Gemma‑4 on Cerebras.
- The **control-room math is real** — entropy, β‑like pressure, q‑like loop risk, graph Laplacian λ₂, Lyapunov energy, and a PID controller are all computed from the live graph snapshot, and every repair/visual event is triggered by a metric crossing a threshold (nothing is random).
- The **15,000-agent torus and reactor plasma are a visualization** — a clustered, aggregated representation of the reasoning dynamics, not 15,000 literal model instances.

Naming is deliberately honest throughout: **β‑like**, **q‑like**, **Lyapunov‑style**, **Cerebras‑inspired fabric twin** — analogues to physics, not claims of literal hardware or plasma.

---

## The three pages

| Page | File | Role |
|---|---|---|
| **Home** | `TorusNet x FusionCoreNet.dc.html` | Landing page — overview, architecture, the WSE‑3 spec sheet |
| **TorusNet** | `radiant-hertz/web/torusnet.html` | Reasoning agent — prompt input, 3D torus, reasoning terminal, live telemetry, Cerebras execution + fabric twin |
| **FusionCore** | `radiant-hertz/web/fusioncore.html` | Control agent — reactor visualization, math engines, equation trace, repair log, convergence timeline |

A top bar moves between **Home ↔ TorusNet ↔ FusionCore**. The two agents hand off through a state snapshot stored in the browser (`localStorage`): TorusNet emits it on each run; FusionCore reads it, analyzes it, and writes repair commands back.

---

## The loop (how to use it)

1. Open **TorusNet**, enter a prompt, click **Run**. Watch the fabric select agents, route, and converge. A **consensus score** is produced and a snapshot is emitted.
2. Open **FusionCore**, click **Analyze** — the engines diagnose the snapshot and propose repairs (verifier bombardment, magnetic reroute, cool overload, open bridge, break loop, …), each traced to the equation that fired it.
3. Click **Apply Repairs → TorusNet** (or **Run Full System · Auto-Converge** to loop automatically until the Lyapunov energy stops dropping).
4. Back on **TorusNet**, run again — consensus climbs. Repeat to push it toward the target (diminishing returns near the top, by design).

---

## FusionCoreNet math engines

| Engine | Equation | Detects | Repairs |
|---|---|---|---|
| Graph Laplacian | `L = D − A` (algebraic connectivity λ₂) | bottlenecks, isolated clusters | OPEN_BRIDGE, BOOST_CONFIDENCE_PATH |
| Information Entropy | `H = −Σ pᵢ log pᵢ` | disagreement, uncertainty | VERIFIER_BOMBARDMENT, QUARANTINE_NOISE |
| Pressure–Capacity | `β‑like = pressure / capacity` | overload, congestion | COOL_OVERLOAD, MAGNETIC_REROUTE |
| Graph Transport | `ρ′ = ρ − dt·div(J)` | flux bottlenecks, starvation | MAGNETIC_REROUTE, OPEN_BRIDGE |
| PID Control | `Kp·e + Ki·∫e + Kd·Δe` | repair intensity | scales repair strength |
| Lyapunov Stability | `ΔV < 0` | converging vs diverging | FINALIZE_IF_STABLE, warnings |
| Loop Stability | `q‑like = circulation / loop_risk` | echo chambers / loops | BREAK_LOOP |

Optional, off by default: **Real Fusion Reference** (actual `plasma β = p/p_B`, Alfvén speed, safety factor — clearly separated from the AI analogues) and an **External Physics MCP** hook.

Every animation in the reactor (plasma hotspot, verifier bombardment, magnetic reroute, cooling pulse, confinement field, bridge opening, loop break, stabilization wave, warning ring) is driven by one of these computed metrics, and each control slider has its own physics‑mapped micro‑animation.

---

## Cerebras WSE‑3 grounding

Hardware figures shown in the UI are taken from Cerebras' published white papers (*The Cerebras Wafer‑Scale Architecture for Deep Learning* and *How Cerebras Solved the Wafer‑Scale Yield Challenge*):

- **900,000 active cores** (of ~970,000, ≈93% active) · **0.05 mm²** per core · **84 die regions**
- **46,225 mm²** wafer · **~4 trillion transistors** · **TSMC 5 nm**
- **214 Pb/s** fabric bandwidth · **21 PB/s** memory bandwidth
- **2‑D mesh fabric**, 5‑port routers, **single‑cycle hop**, **32‑bit packets** (16‑bit data + 16‑bit control), **24 routing colors**
- **Defect‑tolerant, self‑healing fabric** — routes around faulty cores to present one contiguous array (the same reroute behavior FusionCoreNet's repairs model)

No FLOPs are presented as measured chip telemetry; clock frequency is not shown because the papers don't state it.

---

## Running locally

It's a static front‑end — no build step.

```bash
# from radiant-hertz/web
npx serve .
# open the printed URL, then navigate Home / TorusNet / FusionCore
```

**Live Cerebras inference:** open TorusNet and paste your Cerebras API key into the **Cerebras API Key** field (Simulation Control panel). The key is stored only in your browser (`localStorage`) and is used directly for the verified answer. Without a key (or network), TorusNet falls back to a locally synthesized answer so the rest of the demo still works.

---

## Deploying

Static hosting works anywhere (Vercel, Netlify, Cloudflare Pages, GitHub Pages, Surge). For a production deploy that keeps your API key server‑side, put a small proxy in front of Cerebras (a `/api/cerebras` serverless function reading `CEREBRAS_API_KEY`) and point the client at it. A starter `deploy/` bundle with that proxy and per‑host instructions is included.

---

## Project layout

```
TorusNet x FusionCoreNet.dc.html   Home (landing)
radiant-hertz/web/
  torusnet.html / torusnet.js       Reasoning agent
  fusioncore.html / fusioncore.js   Control agent
  style.css                         Shared control-room theme
deploy/                             Static deploy bundle + Cerebras proxy
torusnet/ · fusioncorenet/          Python reference modules (mesh, routing,
                                    consensus, control) the JS logic mirrors
```

---

## Disclaimers

PhaseLock is a **mathematically‑inspired visualization and control demo**. It is **not** a nuclear‑fusion simulator, not an exact Cerebras hardware twin, and the plasma/tokamak language is metaphorical. The math engines are real computations on the reasoning graph; the fusion/plasma framing is an interpretive lens, labeled as such throughout.

---

*Built for the Cerebras × Google DeepMind Gemma hackathon.*
