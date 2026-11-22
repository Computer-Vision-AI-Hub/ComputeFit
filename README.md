# ComputeFit

**See which models your hardware can run before you buy anything.**

ComputeFit is a browser-based hardware advisor for AI workloads. Pick a device, and every model in the catalog shows whether it fits, how fast it runs and how much of the device it uses. Open a model to set your own targets (cameras, users, latency), find the bottleneck, get a free fix or upgrade options, and export a shareable report.

It covers five model families:

| Type | Examples | Typical bottleneck |
|---|---|---|
| **Computer vision** | Detection, segmentation, OBB, pose, classification, SAM, depth (YOLO26, RT-DETR, ...) | Compute |
| **Image generation** | Stable Diffusion, SDXL, SD 3.5, FLUX | Compute |
| **LLM** | Llama, Qwen, Mistral and others | Memory bandwidth |
| **VLM** (vision-language) | Qwen2.5-VL, Gemma 3, SmolVLM, Moondream | Compute, then bandwidth |
| **VLA** (robot control) | π0, OpenVLA, SmolVLA, GR00T | Reaction time |

ComputeFit is part of the [Computer Vision AI Hub](https://computer-vision-ai-hub.github.io/ComputerVisionAIHub.github.io/) family. Everything runs in the browser, and nothing you enter leaves your machine.

---

### Workflow

1. **Pick your hardware.** Choose a device in the bar above the models, or "I don't have hardware yet" to see the cheapest device for each model.
2. **Scan the cards.** Each card shows the verdict, speed and device usage. Filter to "Fits my hardware".
3. **Check fit.** Open a model to set your own targets, then see the bottleneck, a free fix, upgrade options and the deploy command.

Keyboard: press `/` to focus search, `Esc` to close dialogs.

---

## Functionalities

### Model browser
- Cards for every model, with a verdict and key numbers for the selected device.
- Search by model, task or family.
- Filter by type (vision, image generation, LLM, VLM, VLA) and by "Fits my hardware".
- Sort by catalog order, best fit first, lightest first, or name A-Z.

### Check fit (per-model analysis)
- Editable workload targets, per model type:
  - Vision: cameras, FPS, max delay, precision, runtime, input size.
  - Image generation: images/hour, max seconds per image, resolution, steps, offloading.
  - LLM: users, context length, prompt and output tokens, speed and first-word targets.
  - VLM: images per request, answer length, requests per minute, max seconds per answer.
  - VLA: control rate (Hz) and reaction-time limit.
- Pass/fail checks against each target, a memory breakdown, and the identified bottleneck.
- **Free fix:** tries cheaper settings on your current hardware and shows the first one that passes. The order is optimized runtime, then lower precision, then smaller input or fewer image tokens, then a smaller model.
- **Budget / Recommended / Performance** hardware options, with filters for budget, power limit, edge-only, and buy vs cloud rental.
- **Accuracy trade-off table** for each precision (speed, memory, typical quality change).
- **Deployment recommendation** with a ready-to-use command.
- **Rent vs buy** break-even, including electricity cost.

### Stack planner
Run several models on one device. It adds up memory (shared runtime counted once) and combined load, and spreads models across several GPUs. Presets: *Smart retail camera*, *Warehouse robot*, *Document pipeline*, *Edge assistant*.

### Bring your own data
- **Add a model:** import any LLM or VLM from Hugging Face by `org/model` id, or paste its `config.json`.
- **Add a device:** enter any GPU's memory, bandwidth, FP32/FP16 throughput, price and power. Devices are saved in your browser (`localStorage`) and can be copied as a `data.json` entry.
- **Live catalog:** reads the Computer Vision AI Hub `models.json` on each visit (with a bundled snapshot as fallback) and pulls trending LLMs and VLMs from the Hugging Face API (cached for 24 hours).
- **`data.json` overrides:** an optional file next to `index.html` that merges entries by `id`, so prices, devices and models can be updated without editing the page:
  ```json
  { "hardware": [], "vision": [], "gen": [], "llm": [], "vlm": [], "vla": [] }
  ```

### Shareable reports
Every result and every stack can become a report with targets, results, accuracy trade-off, hardware options, deployment steps and assumptions. A report can be shared as a link, printed or saved to PDF, or downloaded as a single HTML file. It carries a report ID and an integrity hash of its inputs. Opening a shared link recomputes the hash and warns if the link was altered. This detects edits and mistakes. It is not a certificate-based digital signature.

---

## Hardware catalog

127 built-in devices in eight groups:

| Group | Count | Examples |
|---|---|---|
| **Edge devices** | 9 | Raspberry Pi 5 (CPU, Hailo-8L, Hailo-8), Jetson Nano, Orin Nano Super, Orin NX, AGX Orin, Intel mini PC |
| **Phones & tablets** | 5 | iPhone 14 Pro to iPhone 18 Pro (Apple A-series) |
| **Mini PCs & dev kits** | 2 | AMD Ryzen AI Max+ 395 128GB, NVIDIA DGX Spark (GB10) |
| **Laptop GPUs** | 20 | GTX 1650 to RTX 50-series laptop parts |
| **Desktop GPUs** | 45 | NVIDIA GTX 10 to RTX 50, AMD Radeon, Intel Arc |
| **Workstation GPUs** | 11 | RTX A4000/A5000/A6000, RTX Ada, RTX PRO Blackwell, Radeon PRO W7900 |
| **Data center GPUs** | 18 | T4, V100, L4, A10G, L40S, A100, H100 and newer |
| **Cloud instances** | 17 | AWS, Google Cloud and others, priced per hour |

Each device is described by memory, bandwidth, FP16/FP8 TFLOPS, vision throughput, price, cloud price per hour and power (TDP).

---

## Metrics

All metrics are documented in-app in the **Metrics explained** section. Each `?` next to a result links to its entry.

### General
| Metric | Meaning |
|---|---|
| **Verdict** | *Meets your targets* (all checks pass with ≥15% spare), *Meets targets, no margin*, or *Does not meet targets*. |
| **Headroom (×)** | `min(capacity ÷ demand, limit ÷ delay, 0.9 × free memory ÷ memory needed)`. 1.0× is exactly at the limit, and ≥1.5× is the bar for "Recommended". |
| **Bottleneck** | The single limiting resource: memory, compute, memory bandwidth, or CPU pre/post-processing. |

### Computer vision
| Metric | Formula / meaning |
|---|---|
| **Delay per frame (ms)** | `runtime overhead + model compute + pre/post-processing` |
| **FPS** | `devices × min(batch × 1000 ÷ inference ms, CPU pre/post capacity)` |
| **Memory (GB)** | Weights + activations + runtime context |
| **GFLOPs** | Published GFLOPs × (your size ÷ reference size)² |
| **Precision** | FP32, FP16 (default), INT8 (~1.7× faster, needs calibration) |
| **Runtime** | TensorRT, OpenVINO, NCNN, HailoRT, Core ML, ONNX Runtime (~30% slower), PyTorch (~2× slower) |

### Image generation
| Metric | Formula / meaning |
|---|---|
| **Time per image (s)** | Idle time `steps × time per step + VAE decode + text encoding`, plus queueing at peak rate |
| **Images per hour** | `GPUs × 3600 ÷ idle time × images per request` |
| **Steps / steps per second** | Denoising passes (25-30 typical, 4 for distilled models) |
| **Precision** | FP16, FP8 (~1.4× faster on FP8 hardware), NF4/GGUF Q4 (~3.5× smaller) |

### LLM
| Metric | Formula / meaning |
|---|---|
| **Speed per user (tokens/s)** | `max(bytes read ÷ bandwidth, FLOPs ÷ compute) + overhead` |
| **Time to first word (ms)** | `≈ 2 × active params × prompt tokens ÷ (TFLOPS × 45%)` |
| **KV cache** | `2 × layers × KV heads × head dim × context × users × bytes` |

### VLM
| Metric | Formula / meaning |
|---|---|
| **Time per answer (s)** | Vision encoder + reading image and text tokens + answer tokens ÷ tokens/s |
| **Answers per minute** | `parallel requests × 60 ÷ time per answer` |
| **Image tokens** | `(resolution ÷ pixels per token)²` |

### VLA (robot control)
| Metric | Formula / meaning |
|---|---|
| **Reaction time (ms)** | Vision encoder + reading cameras and instruction + action generation + overhead |
| **Actions per second (Hz)** | `actions per chunk ÷ time per inference` |
| **Chunk and denoising steps** | Fewer flow/diffusion steps are faster, and halving them saves roughly 10-40% |

### Hardware and cost
| Metric | Meaning |
|---|---|
| **TFLOPS / effective TFLOPS** | Vendor peak throughput vs what a model actually gets (about 30% of peak for vision CNNs) |
| **Memory bandwidth (GB/s)** | Decides LLM generation speed |
| **Power (W)** | Typical board power, used for the power filter and electricity cost |
| **Price / cloud price** | Approximate USD street price, or USD per GPU-hour |
| **Rent vs buy** | `buy = price × GPUs × 1.25`. Pays off after `buy ÷ (rent − power)` months. |

Usable-memory assumptions: 92% of a GPU, 65% of a Jetson, 70% of Apple unified memory, 60% of system RAM.

---

## Accuracy and limitations

ComputeFit is a **roofline-style estimate**. It combines published model sizes with device specs, then calibrates against published benchmarks. **Expect real results within roughly ±20-30%.**

**Calibrated against:** YOLO26 latencies on a T4 with TensorRT (within ~0.3 ms), RT-DETR-l on T4, SAM 2.1 on A100, typical SDXL and FLUX speeds on RTX 4090 and H100, LLM speeds from memory bandwidth, and VLA latencies (π0 about 73 ms on an RTX 4090, OpenVLA about 6 Hz on an RTX 4090, GR00T N1 about 64 ms on an L40).

**Approximate:** FLOPs for YOLO26 seg/pose/OBB/cls variants, edge-device speeds, all VLM results, and all prices (last checked Nov 2026).

Always validate with a benchmark on the real device before buying.

---


### Data sources
- [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/): model sizes, FLOPs and latency
- [Computer Vision AI Hub catalog](https://computer-vision-ai-hub.github.io/ComputerVisionAIHubLibrary.github.io/): custom models and base architectures
- [Hugging Face Hub](https://huggingface.co/): model configs for LLM and VLM import
- Vendor spec sheets: memory, bandwidth, TFLOPS and power

---

## Privacy

All calculations run locally in your browser. The only network requests are to the public Hub catalog, the Hugging Face API (for trending models and imports), and an optional `data.json` next to the page. Custom devices, imported models and cached trending data are stored in your browser's `localStorage`.
