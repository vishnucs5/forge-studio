# Forge Studio — Local AI Workstation

> **A professional, privacy-first local AI workstation and creative synthesis studio for NVIDIA RTX GPUs (specifically tuned for the GeForce RTX 5070 Ti 16 GB Blackwell architecture), featuring ComfyUI execution, Ollama Qwen3-Coder 30B Copilot, PyTorch LoRA Training, VRAM-aware scheduling, presets, interactive inpainting, and hardware diagnostics.**

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178C6.svg)](https://www.typescriptlang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12%2Bcu128-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%20%2F%20sm__120-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## Table of Contents
1. [Overview & System Architecture](#1-overview--system-architecture)
2. [Key Capabilities & Workstation Features](#2-key-capabilities--workstation-features)
   - [Generation Studio (txt2img & img2img)](#generation-studio-txt2img--img2img)
   - [Interactive Inpainting Mask Studio](#interactive-inpainting-mask-studio)
   - [ControlNet Conditioning](#controlnet-conditioning)
   - [LoRA Vault & Native LoRA Training](#lora-vault--native-lora-training)
   - [Workflow Presets & Diff Review](#workflow-presets--diff-review)
3. [RTX 5070 Ti VRAM Safety & Hardware Coordination](#3-rtx-5070-ti-vram-safety--hardware-coordination)
   - [Dynamic VRAM Baseline & Heuristic Estimator](#dynamic-vram-baseline--heuristic-estimator)
   - [Atomic GPU Lease Coordinator](#atomic-gpu-lease-coordinator)
   - [User VRAM Policies](#user-vram-policies)
   - [Priority Queue Scheduler & Starvation Aging](#priority-queue-scheduler--starvation-aging)
4. [Studio Copilot (Qwen3-Coder 30B via Ollama)](#4-studio-copilot-qwen3-coder-30b-via-ollama)
5. [System Diagnostics & Hardware Benchmark Suite](#5-system-diagnostics--hardware-benchmark-suite)
6. [Model Architecture Safeguards (SD 1.5 vs SDXL)](#6-model-architecture-safeguards-sd-15-vs-sdxl)
7. [Directory Structure](#7-directory-structure)
8. [Quickstart & Startup Commands](#8-quickstart--startup-commands)
9. [Keyboard Shortcuts](#9-keyboard-shortcuts)
10. [Verification & Test Suite (105 Tests)](#10-verification--test-suite-105-tests)

---

## 1. Overview & System Architecture

Forge Studio delivers a responsive, darkroom-aesthetic (`#11110F`) local generative studio built directly on top of real offline AI runtimes. It wraps ComfyUI's dynamic computational graphs, runs local PyTorch LoRA adapter training, monitors GPU memory telemetry down to the millisecond, and pairs the workflow with an advisory local Copilot powered by Qwen3-Coder 30B.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Forge Studio UI (React 18 / TypeScript / Vite)                  │
│                                                                                        │
│   ┌─────────────────────┐   ┌────────────────────────┐   ┌─────────────────────────┐   │
│   │  Generation Studio  │   │ Inpaint Mask Studio    │   │  LoRA Vault & Training  │   │
│   │  • txt2img/img2img  │   │  • Fluid Interpolation │   │  • Dual-LoRA Stacking   │   │
│   │  • Model inventory  │   │  • 15-Step Undo/Redo   │   │  • Dataset Validator    │   │
│   │  • Sampler & Steps  │   │  • Feathering (0-16px) │   │  • Native bf16 Trainer  │   │
│   │  • Contradiction Det│   │  • Invert & Zoom (2x)  │   │  • 1-Click Vault Import │   │
│   └──────────┬──────────┘   └───────────┬────────────┘   └────────────┬────────────┘   │
│              │                          │                             │                │
│              └──────────────────────────┼─────────────────────────────┘                │
│                                         ▼                                              │
│               ┌───────────────────────────────────────────────────────┐                │
│               │             Zustand Studio State Store                │                │
│               │  • Live Telemetry Stream • Queue State (P1-P3)        │                │
│               │  • Parameter Diff Previews • Preset Hub               │                │
│               └─────────────────────────┬─────────────────────────────┘                │
└─────────────────────────────────────────┼──────────────────────────────────────────────┘
                                          │ REST & WebSockets (/api/ws)
┌─────────────────────────────────────────▼──────────────────────────────────────────────┐
│                            FastAPI Backend Engine (Port 8000)                          │
│                                                                                        │
│  ┌─────────────────────────────────┐       ┌────────────────────────────────────────┐  │
│  │     GpuSafetyManager (Leases)   │       │       PriorityQueueScheduler           │  │
│  │  • Dynamic baseline (nvidia-smi)│       │  • P1/P2/P3 dispatch with aging        │  │
│  │  • 1.20x margin + 600MB graph   │       │  • 5-second SHA256 parameter hash dedupe│ │
│  │  • Single-job mutual exclusion  │       │  • Pause / Resume local queue          │  │
│  │  • 300s TTL + heartbeat renewal │       │  • Decoupled cancel-queued vs active   │  │
│  └────────────────┬────────────────┘       └───────────────────┬────────────────────┘  │
│                   │                                            │                       │
│                   └──────────────────────┬─────────────────────┘                       │
│                                          │                                             │
│  ┌─────────────────────────────┐         │         ┌────────────────────────────────┐  │
│  │     Diagnostics Service     │◄────────┼────────►│      Workflow Presets Engine   │  │
│  │  • 9 hardware/service probes│         │         │  • 7 factory built-in presets  │  │
│  │  • 30-second TTL cache      │         │         │  • Schema v1 allowlist export  │  │
│  │  • Copyable PowerShell fixes│         │         │  • Side-by-side diff review    │  │
│  └─────────────────────────────┘         │         └────────────────────────────────┘  │
│                                          │                                             │
│  ┌─────────────────────────────┐         │         ┌────────────────────────────────┐  │
│  │    Benchmark Runner         │◄────────┘         │       Error Sanitizer          │  │
│  │  • Isolated GPU lease       │                   │  • Redacts C:\Users\paths      │  │
│  │  • 5-step CUDA warm-up      │                   │  • Redacts Bearer/sk- tokens   │  │
│  │  • Latency, TPS, VRAM, CSV  │                   │  • Transient vs permanent tags │  │
│  └─────────────────────────────┘                   └────────────────────────────────┘  │
└──────────────────┬───────────────────────────────────────────────────┬─────────────────┘
                   │ HTTP / WebSocket (:8188)                          │ HTTP (:11434)
┌──────────────────▼─────────────────────────────┐ ┌──────────────────▼──────────────────┐
│             ComfyUI Diffusion Engine           │ │            Ollama LLM Engine        │
│  • PyTorch 2.12.0+cu128 (Native sm_120)        │ │  • Model: Qwen3-Coder 30B (Q4_K_M)  │
│  • Discovered Checkpoints (SD 1.5 & SDXL)      │ │  • 120s timeout & 8,192 token ceiling│
│  • ControlNet Models & Preprocessors           │ │  • 5 Structured Advisory Actions    │
│  • Asynchronous Weight Offloading (2 Streams)  │ │  • Truthful offline error handling  │
└────────────────────────────────────────────────┘ └─────────────────────────────────────┘
```

---

## 2. Key Capabilities & Workstation Features

### Generation Studio (txt2img & img2img)
- **Direct Local Inference:** Submits dynamic JSON workflows directly to ComfyUI over WebSocket with real-time step progress and preview streaming.
- **Negative Prompt Templates:** 1-click pills for *Standard Baseline*, *Photorealism* (removes CGI/plastic skin), and *Illustration/Anime*.
- **Style Contradiction Guard:** Automatically flags if your negative prompt contains `"photorealistic / real photo"` while your prompt asks for a photograph, with a 1-click **[Fix for Photorealism]** action.
- **Advanced Samplers:** Full support for `Euler`, `Euler Ancestral`, `DPM++ 2M`, `DPM++ SDE`, `DPM++ 2M SDE`, `Heun`, and `DDIM`.

### Interactive Inpainting Mask Studio
- **Fluid Continuous Strokes:** Uses connected line-and-arc interpolation (`ctx.lineTo` with round joins and caps) to guarantee smooth, seamless brush strokes regardless of mouse movement speed.
- **15-Step Undo & Redo History:** In-memory `ImageData` snapshots with toolbar buttons and standard shortcut hotkeys (<kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Y</kbd>).
- **Mask Feathering & Blurring (0–16px):** Applies a smooth Gaussian blur filter to the mask boundary, eliminating hard binary seams and blending repainted regions smoothly into the original photo.
- **1-Click Mask Invert:** Instantly inverts black and white regions to toggle between editing a subject and replacing the background.
- **Precision Zoom (Fit, 1.5×, 2.0×):** Canvas zoom container with coordinate remapping for repairing hands, eyes, faces, or micro-details.
- **Dynamic Brush Indicator:** Real-time circular cursor ring showing exact brush diameter over the canvas.
- **1-Click "Send to Inpaint":** Directly transfer any output from the Viewport Canvas or Gallery into the Inpaint Canvas with a single click.

### ControlNet Conditioning
- **Structural Guidance:** Ingests source images and applies Canny edge detection or custom preprocessors.
- **Fine Parameters:** Independent sliders for ControlNet strength ($0.0 - 2.0$) and step percentages (start % and end %).
- **Model Discovery:** Automatic detection of `.safetensors` models in `ComfyUI/models/controlnet/`.

### LoRA Vault & Native LoRA Training
- **Dual-LoRA Stacking:** Attach up to two LoRA adapters simultaneously with independent strength weights ($0.0 - 2.0$).
- **Trigger Word Insertion:** Card metadata lists trigger words with 1-click **Copy All** or individual interactive `+ [trigger]` insertion chips.
- **5-Step LoRA Training Studio:**
  1. *Goal Selection:* Character, Style, Object, Concept.
  2. *Dataset Integrity Validation:* Inspects directory, checks image formats, tests for corrupted images via PIL, detects duplicate images via MD5 hashes, and checks caption pairs.
  3. *Caption Prefixing:* Preview and apply trigger prefixes (`trigger, [caption]`) with automatic `.txt.bak` backup creation.
  4. *RTX 5070 Ti Preset:* $1024 \times 1024$ or $512 \times 512$, batch size 1, network rank/alpha 16/16, mixed precision `bf16`, gradient checkpointing.
  5. *Real PyTorch LoRA Trainer:* Uses native PyTorch LoRA layers targeting attention projections in `bf16` on `sm_120`.
  6. *1-Click Vault Sync:* Automatically saves the `.safetensors` adapter into Forge Studio's storage and links it into ComfyUI's `models/loras/`.

### Workflow Presets & Diff Review
- **7 Built-In Factory Presets (Immutable):**
  1. **Portrait:** $1024 \times 1024$, 28 steps, CFG 6.5, Euler, Portrait negative prompt.
  2. **Cinematic Landscape:** $1344 \times 768$, 30 steps, CFG 6.5, DPM++ 2M Karras.
  3. **Product Photography:** $1152 \times 896$, 32 steps, CFG 6.5, studio neutral negative.
  4. **Fast Draft:** $768 \times 768$, 16 steps, CFG 6.0, Euler (~10s).
  5. **Inpainting Repair:** $1024 \times 1024$, conservative denoise 0.55, mask padding 32.
  6. **ControlNet Composition:** $1024 \times 1024$, 28 steps, Canny preprocessor, strength 0.85.
  7. **LoRA Style Test:** $1024 \times 1024$, 26 steps, 1 LoRA at 0.75 weight.
- **Custom User Presets:** Save current workspace settings, edit, duplicate, and delete user presets.
- **Side-by-Side Diff Review Modal (`PresetReviewModal.tsx`):** Displays old vs. new values for every changed parameter, detects missing model files, shows estimated VRAM, and requires explicit user confirmation before applying.
- **Schema v1 Allowlist Export/Import:** Strictly validated JSON import/export that strips and rejects scripts, network URLs, path traversals, or raw graph ASTs.
- **Full Reproducibility:** Every completed generation stores preset metadata, settings snapshot JSON, GPU telemetry snapshot, workflow graph hash, and asset hashes. Includes **"Copy as Preset"** and **"Reproduce Exact Generation"** in the Gallery.

---

## 3. RTX 5070 Ti VRAM Safety & Hardware Coordination

Forge Studio is specifically engineered for modern 16 GB GPUs (such as the **NVIDIA GeForce RTX 5070 Ti**) to prevent Out-Of-Memory (CUDA OOM) failures and resource competition.

### Dynamic VRAM Baseline & Heuristic Estimator
- Dynamically queries physical VRAM via `GpuMonitor` and `nvidia-smi` (no hardcoded constants).
- Computes pre-flight memory footprint using:
  $$\text{VRAM}_{\text{estimated}} = \left(\text{Baseline} + \text{Latents} + \text{LoRAs} + \text{ControlNet} + 600\text{ MB}\right) \times 1.20$$
  - SD 1.5 Baseline: $2600\text{ MB} + (1.8 \times 10^{-6} \times W \times H)\text{ MB} + 200\text{ MB} \times \text{LoRAs}$
  - SDXL Baseline: $6400\text{ MB} + (3.2 \times 10^{-6} \times W \times H)\text{ MB} + 500\text{ MB} \times \text{LoRAs}$
  - ControlNet: $+1600\text{ MB}$ if active
  - Graph & runtime overhead: $+600\text{ MB}$
  - Safety margin multiplier: $1.20\times$

### Atomic GPU Lease Coordinator
- **Single Heavy Workload Execution:** Guarantees that only **one** heavy workload runs at a time across:
  - Image Generation
  - LoRA Training
  - Hardware Benchmarking
- Protected by an `asyncio.Lock()`, with a 300-second TTL and heartbeat renewal (`heartbeat(job_id)`) to automatically release orphan leases in the event of an abnormal worker exit.

### User VRAM Policies
Users can select their operating threshold directly from the header:
* **Balanced (85% limit):** Standard mode; Copilot text inference is allowed concurrently if free VRAM $\ge 4.0\text{ GB}$.
* **Maximum Quality (92% limit):** Maximizes VRAM budget for high-resolution generation; Copilot requests are temporarily blocked during heavy sampling passes.
* **Maximum Speed (78% limit):** Lower memory threshold for rapid generation without throttling.
* **Conservative (65% limit):** Strict ceiling leaving ample headroom for background 3D or video applications.

### Priority Queue Scheduler & Starvation Aging
* **Three Priority Tiers:** **High (P1)**, **Normal (P2)**, and **Low (P3)**.
* **Starvation Aging:** Waiting jobs automatically age: for every 60 seconds spent in the queue, their effective priority is decremented by 1 (e.g., P3 waiting 120s is promoted to P1), preventing low-priority starvation.
* **5-Second Deduplication Hash:** Calculates a SHA256 generation hash across prompt, negative prompt, model, resolution, steps, CFG, sampler, seed, denoise, LoRAs, and ControlNet. Prevents accidental duplicate submissions within 5 seconds.
* **Decoupled Cancellation:**
  - `Cancel All Queued` cancels queued and waiting jobs without touching running tasks or clearing completed history.
  - `Interrupt Active` sends a `/interrupt` signal directly to ComfyUI, releases the GPU lease, and updates the active generation status to `cancelled`.

---

## 4. Studio Copilot (Qwen3-Coder 30B via Ollama)

Forge Studio connects to a local instance of **Qwen3-Coder 30B** running under Ollama (`http://127.0.0.1:11434`):
* **Scheduling Guardrails:** 120-second request timeout with an 8,192 token ceiling (~32,768 characters) to prevent RAM/VRAM exhaustion.
* **Review-Only Advisory Architecture:** The Copilot **never** executes shell commands, submits generations, or modifies parameters silently. Every recommendation is presented as a parameter diff card with an explicit **[Apply Suggested Changes]** button.
* **5 Structured Actions:**
  1. `✨ Fast Generation`: Suggests 16 steps, $768 \times 768$ (or $512 \times 512$), Euler sampler, and speed-optimized prompt tokens.
  2. `✨ Quality`: Recommends 30 steps, $1024 \times 1024$, CFG 6.5, DPM++ 2M sampler, and photographic lighting modifiers.
  3. `✨ VRAM Safe Resolution`: Evaluates detected physical VRAM and base architecture to recommend optimal dimensions.
  4. `✨ Explain Slowness`: Analyzes step counts, high pixel areas, LoRA count, or VRAM pressure.
  5. `✨ Suggest Preset`: Matches user prompt intent to the closest factory built-in preset.
* **Offline Resilience:** If Ollama is offline or unreachable, the system automatically falls back to deterministic rule-based heuristics.

---

## 5. System Diagnostics & Hardware Benchmark Suite

### Settings Diagnostics (9 Probes)
Accessible from the **Settings → System Diagnostics** tab:
1. **Backend API Connectivity:** Verifies FastAPI health and response latency.
2. **ComfyUI Endpoint & WebSocket:** Verifies ComfyUI availability on port 8188.
3. **Ollama Connectivity & Model:** Checks if Ollama is running with `qwen3-coder:30b`.
4. **Checkpoint Model Availability:** Verifies discovered `.safetensors` models.
5. **ControlNet Model Presence:** Checks `models/controlnet/` directory.
6. **Live PyTorch / CUDA Execution:** Executes a live matrix multiplication ($512 \times 512$ in `bfloat16`) directly on the RTX 5070 Ti.
7. **Training Engine Readiness:** Confirms native LoRA training engine status.
8. **Output Storage Permissions:** Tests read/write/delete access in `storage/generated`.
9. **LoRA Vault Sync State:** Confirms LoRA filesystem accessibility.

> **Remediation Guidance:** Every failing diagnostic card includes a copy-to-clipboard PowerShell command showing the exact local fix. Diagnostic results are cached for 30 seconds to prevent unnecessary hardware polling.

### Hardware Benchmark Suite
Accessible from the **Settings → Hardware Benchmark** tab:
* **Explicit Execution:** Never runs automatically; requires explicit user initiation.
* **Pre-Flight GPU Isolation:** Acquires an exclusive benchmark lease from the GPU safety coordinator.
* **5-Step Unmeasured Warm-Up:** Warms up the CUDA context and VRAM caches before recording metrics.
* **Standard Test Harness:** Fixed prompt (*"studio portrait of an astronaut in intricate reflective suit..."*), fixed seed `42`, 20 steps, CFG 7.0, $512 \times 512$, Euler sampler.
* **Live Measured Metrics:**
  - Prompt-to-first-image latency (ms)
  - Total generation duration (ms)
  - Steps per second (steps/sec)
  - Peak & average VRAM usage (MB)
  - GPU utilization (%)
  - Ollama first-token latency (ms) and tokens per second (tok/sec)
  - Environment metadata (Driver, CUDA, PyTorch version, compute capability `12.0`)
* **Exporting:** 1-click export to **JSON** or **CSV** formats.

---

## 6. Model Architecture Safeguards (SD 1.5 vs SDXL)

Stable Diffusion 1.5 and Stable Diffusion XL have drastically different native training resolutions:

| Feature | Stable Diffusion 1.5 | Stable Diffusion XL |
| :--- | :--- | :--- |
| **Native Training Resolution** | **$512 \times 512$** | **$1024 \times 1024$** |
| **Receptive Field** | ~262,144 pixels | ~1,048,576 pixels ($4\times$) |
| **Effect of 1024×1024 on SD 1.5** | **Duplication defect** (multiple subjects, cloned heads/limbs) | Native, sharp single-subject rendering |
| **Recommended Aspect Ratios** | $512 \times 512$, $512 \times 768$, $768 \times 512$ | $1024 \times 1024$, $896 \times 1152$, $1152 \times 896$ |

### Automated UI Protections in Forge Studio:
1. **Dynamic Architecture Detection:** Automatically inspects the active checkpoint name and updates the UI badge to `SD 1.5 Architecture (Native 512px)` or `SDXL Architecture (Native 1024px)`.
2. **Architecture-Matched Dimension Presets:** Selecting an SD 1.5 checkpoint dynamically displays 512px-based presets; selecting an SDXL model displays 1024px-based presets.
3. **High-Resolution Duplication Warning:** If an SD 1.5 model is paired with a resolution $> 768\text{px}$, an amber warning banner appears explaining the duplication risk with a 1-click **[Fix to 512×512]** button.
4. **Style Contradiction Alert:** Alerts you when a negative prompt contains `"real photo / photorealistic"` while the positive prompt requests a photo, providing a 1-click **[Fix for Photorealism]** button.

---

## 7. Directory Structure

```text
forge-studio/
├── frontend/                     # React 18, TypeScript, Vite, Tailwind CSS, Zustand
│   ├── src/
│   │   ├── components/
│   │   │   ├── CopilotDrawer.tsx         # Qwen3-Coder 30B chat & 5 structured actions
│   │   │   ├── ErrorBanner.tsx           # Sanitized errors & PowerShell remediation
│   │   │   ├── GalleryView.tsx           # Output grid, metadata & reproduction
│   │   │   ├── GenerationSettings.tsx    # Dimensions, samplers, seeds, VRAM alerts
│   │   │   ├── Header.tsx                # Real VRAM gauge, policy selector, status
│   │   │   ├── InpaintCanvas.tsx         # Smooth brush, undo/redo, feathering, zoom
│   │   │   ├── KeyboardShortcutsModal.tsx# Hotkey reference dialog
│   │   │   ├── LeftRail.tsx              # Dark workstation navigation rail
│   │   │   ├── LorasView.tsx             # LoRA Vault, trigger words & import
│   │   │   ├── PresetReviewModal.tsx     # Side-by-side parameter diff review
│   │   │   ├── PresetsView.tsx           # 7 built-in + user preset management
│   │   │   ├── PreviewPanel.tsx          # Viewport canvas, zoom, send to inpaint
│   │   │   ├── PromptComposer.tsx        # Prompts, trigger chips, contradiction alert
│   │   │   ├── RecentGenerationsStrip.tsx# Priority badges, pause/resume, cancel
│   │   │   ├── SettingsSummary.tsx       # Live VRAM footprint & parameter chips
│   │   │   ├── SettingsView.tsx          # 9 diagnostics probes & benchmark runner
│   │   │   └── TrainingView.tsx          # 5-step LoRA training wizard
│   │   ├── store/
│   │   │   └── useStudioStore.ts         # Global Zustand workstation store
│   │   └── types/
│   │       └── index.ts                  # Shared TypeScript interface definitions
├── backend/                      # Python 3.11+, FastAPI, SQLAlchemy Async, SQLite WAL
│   ├── app/
│   │   ├── api/
│   │   │   ├── comfyui.py                # ComfyUI proxy & asset endpoints
│   │   │   ├── controlnet.py             # ControlNet model & type discovery
│   │   │   ├── copilot.py                # Structured advisory actions & chat
│   │   │   ├── diagnostics.py            # 9 diagnostics checks & benchmark runner
│   │   │   ├── generations.py            # Synthesis lifecycle, duplicate prevention
│   │   │   ├── health.py                 # Service health checks
│   │   │   ├── loras.py                  # LoRA Vault index, import, and sync
│   │   │   ├── models.py                 # Checkpoint discovery
│   │   │   ├── presets.py                # Factory presets, CRUD, diff, export/import
│   │   │   ├── queue.py                  # Pause, resume, reorder, cancel-all, interrupt
│   │   │   ├── status.py                 # Live telemetry & observed process monitor
│   │   │   ├── training.py               # Dataset validation & LoRA training
│   │   │   └── websocket.py              # Real-time WebSocket step & preview broker
│   │   ├── models/                       # SQLAlchemy models (Generation, LoRA, Preset, Training)
│   │   ├── schemas/                      # Pydantic v2 schemas with typed validation
│   │   └── services/
│   │       ├── benchmark_service.py      # Isolated hardware benchmark harness
│   │       ├── comfyui_client.py         # Graph injector, dual-LoRA chaining
│   │       ├── diagnostics_service.py    # 9 diagnostic probes with 30s cache TTL
│   │       ├── error_sanitizer.py        # Redacts paths/tokens, classifies retryability
│   │       ├── gpu_monitor.py            # Real VRAM, temp, and process sampler
│   │       ├── gpu_safety.py             # Atomic lease coordinator & heuristic estimator
│   │       ├── lora_service.py           # Vault file scanner & metadata manager
│   │       ├── lora_trainer.py           # Native PyTorch bf16 LoRA trainer
│   │       ├── ollama_client.py          # 120s timeout, token ceiling, structured JSON
│   │       ├── preset_engine.py          # 7 factory presets, allowlist export/import
│   │       ├── queue_scheduler.py        # Priority queue, starvation aging, deduplication
│   │       └── training_engine.py        # Training coordinator
│   └── tests/                            # Pytest suite (105 tests across 11 modules)
├── workflows/
│   ├── txt2img.json                      # ComfyUI Text-to-Image graph template
│   ├── img2img.json                      # ComfyUI Image-to-Image graph template
│   ├── inpaint.json                      # ComfyUI Inpainting graph template
│   └── controlnet.json                   # ComfyUI ControlNet graph template
├── start-all.ps1                         # 1-Click PowerShell startup script
├── stop-all.ps1                          # 1-Click PowerShell shutdown script
└── README.md
```

---

## 8. Quickstart & Startup Commands

### Prerequisites
- Windows 10/11 with NVIDIA GeForce RTX GPU (RTX 5070 Ti or compatible)
- Python 3.11 with CUDA 12.8 nightly PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cu128`)
- Node.js 18+ and npm
- Ollama installed with `qwen3-coder:30b`
- ComfyUI installed at `c:\vishnu\ComfyUI`

### 1-Click Automated Startup
```powershell
# From the forge-studio directory:
.\start-all.ps1
```

### Manual Component Startup

#### 1. Start Ollama Service (Port 11434)
```powershell
ollama serve
```

#### 2. Start ComfyUI Engine (Port 8188)
```powershell
cd c:\vishnu\ComfyUI
c:\vishnu\.venv\Scripts\python.exe main.py --listen 127.0.0.1 --port 8188
```

#### 3. Start Forge Studio Backend (Port 8000)
```powershell
cd c:\vishnu\forge-studio\backend
$env:PYTHONPATH="."
c:\vishnu\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

#### 4. Start Forge Studio Frontend (Port 5173)
```powershell
cd c:\vishnu\forge-studio\frontend
npm run dev
```

Open **[http://localhost:5173/](http://localhost:5173/)** in your browser.

---

## 9. Keyboard Shortcuts

| Shortcut | Scope | Description |
| :--- | :--- | :--- |
| <kbd>Ctrl</kbd> + <kbd>Enter</kbd> (or <kbd>⌘</kbd>+<kbd>Enter</kbd>) | Global | Submit active generation job |
| <kbd>Ctrl</kbd> + <kbd>D</kbd> | Global | Randomize seed |
| <kbd>Ctrl</kbd> + <kbd>K</kbd> | Global | Toggle Studio Copilot assistant drawer |
| <kbd>Ctrl</kbd> + <kbd>G</kbd> | Global | Toggle Gallery view |
| <kbd>Ctrl</kbd> + <kbd>1</kbd> | Global | Switch to Text-to-Image mode |
| <kbd>Ctrl</kbd> + <kbd>2</kbd> | Global | Switch to Image-to-Image mode |
| <kbd>Ctrl</kbd> + <kbd>Z</kbd> | Inpaint Canvas | Undo last brush stroke / invert |
| <kbd>Ctrl</kbd> + <kbd>Y</kbd> / <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd> | Inpaint Canvas | Redo undone brush stroke |
| <kbd>[</kbd> / <kbd>]</kbd> | Inpaint Canvas | Decrease / Increase brush size by 4px |
| <kbd>?</kbd> | Global | Open Keyboard Shortcuts modal |
| <kbd>Esc</kbd> | Global | Close open modals or cancel active job |

---

## 10. Verification & Test Suite (105 Tests)

Forge Studio includes a comprehensive automated test suite covering all services, APIs, schedulers, and guardrails:

```powershell
cd c:\vishnu\forge-studio\backend
$env:PYTHONPATH="."
c:\vishnu\.venv\Scripts\python.exe -m pytest -v
```

### Test Coverage Summary:
* `test_comfyui_adapter.py` (6 tests): Graph generation, dual-LoRA chaining, cancel via `/interrupt`, queue clearing.
* `test_copilot.py` (7 tests): Qwen3-Coder chat, parameter parsing, offline heuristics.
* `test_copilot_and_error_sanitizer.py` (21 tests): 120s timeout, token ceiling truncation, 5 advisory action cards, path and Bearer token redaction, transient vs permanent classification.
* `test_diagnostics_and_benchmark.py` (11 tests): 9 diagnostic probes, 30s cache TTL, live PyTorch CUDA tensor execution, benchmark mutual exclusion, metric sampling, JSON/CSV exports.
* `test_gallery.py` (2 tests): Prompt search, mode filtering, inspector metadata.
* `test_generation_form.py` (9 tests): Form bounds validation, negative prompt checks, ControlNet input validation.
* `test_gpu_safety.py` (18 tests): Dynamic baseline VRAM calculation, policy limits (Balanced, Quality, Speed, Conservative), atomic GPU lease coordinator with TTL and heartbeat.
* `test_health.py` (3 tests): Health probes, database connectivity.
* `test_lora_vault.py` (4 tests): LoRA indexing, metadata updates, trigger words, filesystem sync.
* `test_presets.py` (8 tests): 7 factory built-in presets, user preset CRUD, immutability, side-by-side diff review, schema v1 allowlist export/import, generation reproduction.
* `test_queue_scheduler.py` (11 tests): Priority dispatch (P1–P3), starvation aging, 5-second SHA256 parameter deduplication, pause/resume, decoupled cancel-all-queued vs interrupt-active, crash recovery.
* `test_training.py` (5 tests): Dataset integrity validator, caption prefix preview/apply with `.txt.bak` backup creation, native PyTorch trainer execution.

```text
============================ 105 passed in 45.71s =============================
```

### Frontend Typecheck & Production Build:
```powershell
cd c:\vishnu\forge-studio\frontend
npm run build
```
Executes `tsc` type checking and Vite production bundling with **0 errors**.

---

## License

This project is licensed under the [MIT License](LICENSE).
