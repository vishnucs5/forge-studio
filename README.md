# Forge Studio — Local AI Workstation

> **A professional, privacy-first local AI workstation and creative synthesis studio for NVIDIA RTX GPUs (universally compatible across all GeForce RTX 30-series Ampere, 40-series Ada Lovelace, and 50-series Blackwell graphics cards), featuring ComfyUI execution, Ollama Qwen3-Coder 30B Copilot, PyTorch LoRA Training, VRAM-aware scheduling, presets, interactive inpainting, and hardware diagnostics.**

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178C6.svg)](https://www.typescriptlang.org/)
[![NVIDIA RTX](https://img.shields.io/badge/NVIDIA%20RTX-30%20%7C%2040%20%7C%2050%20Series-76B900.svg)](https://www.nvidia.com/geforce/)
[![Compute Capability](https://img.shields.io/badge/Compute-sm__86%20%7C%20sm__89%20%7C%20sm__120-green.svg)](https://developer.nvidia.com/cuda-gpus)
[![PyTorch](https://img.shields.io/badge/PyTorch-cu124%20%7C%20cu128-EE4C2C.svg)](https://pytorch.org/)

---

## Table of Contents
1. [Overview & System Architecture](#1-overview--system-architecture)
2. [Key Capabilities & Workstation Features](#2-key-capabilities--workstation-features)
   - [Generation Studio (txt2img & img2img)](#generation-studio-txt2img--img2img)
   - [Interactive Inpainting Mask Studio](#interactive-inpainting-mask-studio)
   - [ControlNet Conditioning](#controlnet-conditioning)
   - [LoRA Vault & Native LoRA Training](#lora-vault--native-lora-training)
   - [Workflow Presets & Diff Review](#workflow-presets--diff-review)
3. [Universal NVIDIA RTX 30, 40 & 50 Series Compatibility & VRAM Coordination](#3-universal-nvidia-rtx-30-40--50-series-compatibility--vram-coordination)
   - [Hardware Generation & Compute Capability Matrix](#hardware-generation--compute-capability-matrix)
   - [4-Tier VRAM Allocation & Scaling Architecture](#4-tier-vram-allocation--scaling-architecture)
   - [Hardware-Adaptive LoRA Training Presets](#hardware-adaptive-lora-training-presets)
   - [Dynamic VRAM Baseline & Heuristic Estimator](#dynamic-vram-baseline--heuristic-estimator)
   - [Atomic GPU Lease Coordinator](#atomic-gpu-lease-coordinator)
   - [User VRAM Policies & Tier-Aware Copilot Headroom](#user-vram-policies--tier-aware-copilot-headroom)
   - [Priority Queue Scheduler & Starvation Aging](#priority-queue-scheduler--starvation-aging)
4. [Studio Copilot (Qwen3-Coder 30B via Ollama)](#4-studio-copilot-qwen3-coder-30b-via-ollama)
5. [System Diagnostics & Hardware Benchmark Suite](#5-system-diagnostics--hardware-benchmark-suite)
6. [Model Architecture Safeguards (SD 1.5 vs SDXL)](#6-model-architecture-safeguards-sd-15-vs-sdxl)
7. [Directory Structure](#7-directory-structure)
8. [Quickstart & Startup Commands](#8-quickstart--startup-commands)
9. [Keyboard Shortcuts](#9-keyboard-shortcuts)
10. [Verification & Test Suite (110 Tests)](#10-verification--test-suite-110-tests)

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

## 3. Universal NVIDIA RTX 30, 40 & 50 Series Compatibility & VRAM Coordination

Forge Studio features an adaptive hardware profiling engine (`GpuHardwareProfiler`) that dynamically identifies GPU microarchitecture, compute capability (`sm_86`, `sm_89`, `sm_120`), total physical memory, and supported tensor precision formats. It scales all generation, training, and Copilot workflows smoothly from 8 GB desktop cards up to 32 GB enthusiast workstations.

### Hardware Generation & Compute Capability Matrix

| Generation | Architecture | Compute Capability | Key Graphic Cards | Hardware Precision | PyTorch Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RTX 50-Series** | **Blackwell** | `sm_120` (12.0) | RTX 5070 (12GB), RTX 5070 Ti (16GB), RTX 5080 (16GB), RTX 5090 (32GB) | `fp16`, `bf16`, `fp8`, `tf32` | PyTorch Nightly (`cu128`) |
| **RTX 40-Series** | **Ada Lovelace** | `sm_89` (8.9) | RTX 4060 (8GB), RTX 4060 Ti (8/16GB), RTX 4070 (12GB), RTX 4080 (16GB), RTX 4090 (24GB) | `fp16`, `bf16`, `fp8`, `tf32` | PyTorch Stable (`cu124` / `cu121`) |
| **RTX 30-Series** | **Ampere** | `sm_86` / `sm_80` (8.6 / 8.0) | RTX 3060 (8/12GB), RTX 3070 (8GB), RTX 3080 (10/12GB), RTX 3090 (24GB) | `fp16`, `bf16`, `tf32` | PyTorch Stable (`cu124` / `cu121`) |

### 4-Tier VRAM Allocation & Scaling Architecture

The system categorizes GPU memory into 4 operational tiers to guarantee zero Out-Of-Memory (CUDA OOM) failures:

1. **Tier 1: Compact 8 GB ($\le 8,500\text{ MB}$):**
   - *Target Hardware:* RTX 3050, RTX 3060 8GB, RTX 3070, RTX 3070 Ti, RTX 4060, RTX 4060 Ti 8GB.
   - *Generation Baseline:* 512–768px for SD 1.5; 768–896px for SDXL.
   - *Copilot Coordination:* Requires $\ge 3,000\text{ MB}$ headroom; Copilot requests are safely blocked during active diffusion to protect workstation memory.
   - *Training Preset:* Rank 8, Alpha 8, Resolution 512px, Batch size 1, Gradient Checkpointing enabled.

2. **Tier 2: Mid 12 GB ($8,501 - 13,000\text{ MB}$):**
   - *Target Hardware:* RTX 3060 12GB, RTX 3080 10/12GB, RTX 4070 / Super 12GB, RTX 4070 Ti 12GB, RTX 5070 12GB.
   - *Generation Baseline:* Native SDXL 1024×1024 sampling without VRAM paging.
   - *Copilot Coordination:* Allowed concurrently if free VRAM $\ge 3,500\text{ MB}$.
   - *Training Preset:* Rank 16, Alpha 16, Resolution 1024px, Batch size 1, Gradient Checkpointing enabled.

3. **Tier 3: High 16 GB ($13,001 - 18,000\text{ MB}$):**
   - *Target Hardware:* RTX 4060 Ti 16GB, RTX 4070 Ti Super 16GB, RTX 4080 / Super 16GB, RTX 5070 Ti 16GB, RTX 5080 16GB.
   - *Generation Baseline:* 1024×1024 to 1536×1536 high-res passes, dual-LoRA stacking, ControlNet preprocessors.
   - *Copilot Coordination:* Full concurrent execution with $\ge 4,096\text{ MB}$ headroom.
   - *Training Preset:* Rank 16, Alpha 16, Mixed precision `bf16`, Batch size 1, Gradient Checkpointing enabled (~12.5 GB peak).

4. **Tier 4: Enthusiast 24 GB+ ($> 18,000\text{ MB}$):**
   - *Target Hardware:* RTX 3090, RTX 3090 Ti, RTX 4090 24GB, RTX 5090 32GB.
   - *Generation Baseline:* Ultra-high resolution 1536–2048px, concurrent models loaded in VRAM.
   - *Copilot Coordination:* Copilot inference permitted alongside rendering at all times.
   - *Training Preset:* Rank 32/64, Resolution 1024px, Batch size 2–4, full uncheckpointed backpropagation.

### Hardware-Adaptive LoRA Training Presets

The training engine auto-detects installed hardware and loads optimal hyperparameter profiles:

```json
{
  "tier1_compact_8gb": {
    "resolution": 512, "batch_size": 1, "network_dim": 8, "network_alpha": 8,
    "mixed_precision": "fp16", "gradient_checkpointing": true, "estimated_vram_gb": 6.8
  },
  "tier2_mid_12gb": {
    "resolution": 1024, "batch_size": 1, "network_dim": 16, "network_alpha": 16,
    "mixed_precision": "bf16", "gradient_checkpointing": true, "estimated_vram_gb": 10.5
  },
  "tier3_high_16gb": {
    "resolution": 1024, "batch_size": 1, "network_dim": 16, "network_alpha": 16,
    "mixed_precision": "bf16", "gradient_checkpointing": true, "estimated_vram_gb": 12.5
  },
  "tier4_enthusiast_24gb": {
    "resolution": 1024, "batch_size": 2, "network_dim": 32, "network_alpha": 32,
    "mixed_precision": "bf16", "gradient_checkpointing": false, "estimated_vram_gb": 18.0
  }
}
```

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

### User VRAM Policies & Tier-Aware Copilot Headroom
Users can select their operating threshold directly from the header:
* **Balanced (85% limit):** Standard mode; Copilot text inference is allowed concurrently when free VRAM satisfies the tier headroom threshold ($3.0\text{ GB}$ on Tier 1, $3.5\text{ GB}$ on Tier 2, $4.0\text{ GB}$ on Tiers 3 & 4).
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
* `test_gpu_compatibility.py` (5 tests): Universal Ampere (`sm_86`), Ada Lovelace (`sm_89`), and Blackwell (`sm_120`) profiling, 4-tier VRAM allocation scaling, adaptive hyperparameter presets, tier-aware Copilot headroom thresholds.
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
============================ 110 passed in 28.81s =============================
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
