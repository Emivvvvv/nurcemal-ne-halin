---
title: Nurcemal Ne Halin
sub_title: Real-Time Emotion Detection with Rust
author: Emirhan Tala
---

Overview & Architecture
---

**What it does:**
- Captures live video from webcam
- Detects faces and identifies emotions using AI
- Displays emotion-specific images and plays audio
- Supports customizable "emotion packs"

<!-- pause -->

**Why Rust?**

<!-- pause -->

- Because Sefa abi asked for it :D

<!-- pause -->

- And it is fun since you handle everything by yourself

<!-- pause -->

**A small funfact:**

<!-- pause -->

- This presentation is also running on a terminal with a Rust crate :)


<!-- end_slide -->

The Two-Stage Hybrid Approach
---

```
Stage 1: Face Detection          Stage 2: Emotion Recognition
        ↓                                  ↓
  Haar Cascade                        ONNX Model
```

<!-- pause -->

**Why Two Stages?**
- Running deep learning on entire image is slow and wasteful
- Most of the image doesn't contain faces
- **Strategy**: Quickly find faces first, then run expensive AI only on face regions

<!-- end_slide -->

Stage 1: OpenCV & Haar Cascade
---

**OpenCV** = Most popular computer vision library (Google, Microsoft, Intel)

**Haar Cascade** = Classical ML algorithm from 2001 for real-time face detection

<!-- pause -->

**How it works:**

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->


```
┌─────┬─────┐     ┌─────────┐     ┌──┬──┬──┐
│ ■■■ │ □□□ │     │ ■■■■■■■ │     │■■│□□│■■│
│ ■■■ │ □□□ │     │ □□□□□□□ │     │■■│□□│■■│
└─────┴─────┘     └─────────┘     └──┴──┴──┘
  Eye Region        Nose Bridge      Eye-Nose-Eye
```

<!-- column: 1 -->

![image:width:70%](./haar_cascade.png)

<!-- pause -->

<!-- column_layout: [1] -->
<!-- column: 0 -->

**Cascade of Classifiers:**
```
Image Region
    ↓
[Check 1: Simple] → Reject 50% of non-faces (fast!)
    ↓
[Check 2: Moderate] → Reject 30% more
    ↓
[Check 3: Complex] → Final verification
    ↓
Face Detected! ✓
```

Most non-face regions fail early and are rejected immediately.

<!-- end_slide -->

Stage 1: OpenCV & Haar Cascade
---

**Key Parameters We Use:**

```rust
detect_multi_scale(
    &gray_image,
    scale_factor: 1.1,      // How much to shrink image each pass
    min_neighbors: 5,       // Overlapping detections needed
    min_size: 40x40,        // Smallest face to detect
)
```

- **Scale Factor (1.1)**: Shrink image by 10% each iteration to detect faces at different distances
- **Min Neighbors (5)**: Need 5 overlapping detections to confirm it's a real face (reduces false positives)
- **Min Size (40x40 pixels)**: Ignore anything smaller (speeds up processing, filters noise)

<!-- end_slide -->

Stage 2: ONNX & Deep Learning
---

**ONNX (Open Neural Network Exchange)** = Universal format for ML models

Created by Microsoft & Facebook (2017)

<!-- pause -->

**The Problem:**
```
Before ONNX:
Train in PyTorch → Deploy = Install 500+ MB framework ❌
```

**The Solution:**
```
With ONNX:
Train in PyTorch → Export to .onnx (5 MB) → Deploy anywhere ✓
```

<!-- pause -->

**Benefits:**
- 2-3x faster than PyTorch/TensorFlow
- No Python runtime needed
- Works on any platform (Windows, macOS, Linux, mobile)
- Industry standard (Microsoft Bing, Facebook, NVIDIA)

<!-- end_slide -->

HSEmotion Model & CNN Architecture
---

**Our Model:**
- **Type**: Convolutional Neural Network (CNN)
- **Input**: 260x260 RGB face image
- **Output**: 8 emotion probabilities
- **Emotions**: Angry, Disgusted, Scared, Happy, Sad, Surprised, Neutral, Contempt

<!-- pause -->

**How CNNs Recognize Emotions:**

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->

![image:width:100%](./cnn.png)

<!-- column: 1 -->

```
Input Face (260x260)
    ↓
[Conv Layers] → Learn edges, corners, textures
    ↓
[More Conv Layers] → Learn eye shapes, mouth curves, eyebrows
    ↓
[Deep Layers]
    ↓
Learn patterns:
    "Raised eyebrows + wide eyes = surprised"
    "Downturned mouth + furrowed brow = sad"
    ↓
[Final Layers] → Emotion probabilities
    ↓
Output: [Happy: 75%, Neutral: 15%, Surprised: 10%, ...]
```
<!-- column_layout: [1] -->
<!-- column: 0 -->
<!-- pause -->

**Why Deep Learning?**
Emotions are complex and subtle—requires understanding facial muscle movements that classical algorithms can't capture.

<!-- end_slide -->

The Complete Pipeline
---

```
┌──────────────────────────────────────────────────────┐
│              CAMERA FRAME (640x480)                  │
└─────────────────────┬────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────┐
│          STAGE 1: FACE DETECTION (OpenCV)            │
│                                                      │
│  • Convert to grayscale                              │
│  • Run Haar Cascade at multiple scales               │
│  • Find face bounding boxes                          │
└─────────────────────┬────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────┐
│                    PREPROCESSING                     │
│                                                      │
│  • Extract face region                               │
│  • Resize to 260x260                                 │
│  • Normalize pixels (0-255 → 0.0-1.0)                │
└─────────────────────┬────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────┐
│          STAGE 2: EMOTION RECOGNITION (ONNX)         │
│                                                      │
│  • Create tensor [1, 3, 260, 260]                    │
│  • Run CNN model (HSEmotion)                         │
│  • Apply softmax → probabilities                     │
└─────────────────────┬────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────┐
│                    FINAL RESULT                      │
│                                                      │
│        Emotion: Happy | Confidence: 75%              │
└──────────────────────────────────────────────────────┘
```

<!-- end_slide -->

Performance & Accuracy Optimizations
---

**1. Frame Skipping**
- Process every 5th frame (30 FPS → 6 FPS processing)
- Reduces CPU usage by 80%
- Still feels responsive

<!-- pause -->

**2. Confidence Threshold Tuning**
```
Raw Model Output → Softmax → Probabilities
[2.1, -0.5, 1.8, ...] → [0.45, 0.08, 0.32, ...]
                              ↓
                    Filter: Keep if > 25%
```
- Only show emotions with >25% confidence
- Prevents flickering between uncertain predictions
- Lower threshold = better detection, higher = more stable

<!-- end_slide -->

Performance & Accuracy Optimizations
---

**3. Haar Cascade Parameter Tuning**

The face detector scans the image at multiple scales and positions:

```
Scale Factor (1.1):
┌────┐  ┌──────┐  ┌──────────┐
│ 40 │→ │  44  │→ │    48    │  (Smaller steps = more thorough)
└────┘  └──────┘  │          │
                  └──────────┘
```
<!-- pause -->
**Min Neighbors (3)**: How many overlapping detections confirm a face
```

Detection 1: ┌────┐
Detection 2:  ┌────┐
Detection 3:   ┌────┐   → 5 overlaps = Real face ✓
Detection 4:    ┌────┐
Detection 5:     ┌────┐
```
<!-- pause -->
**Min Face Size (40x40)**: Smallest face to detect
- Smaller = detects distant faces but slower
- Larger = faster but misses small faces

<!-- end_slide -->

Performance & Accuracy Optimizations
---

**4. Multi-Threading Architecture**

We use **3 types of threads**:

```
┌─────────────────┐         ┌──────────────────┐         ┌──────────────┐
│   UI Thread     │         │  Worker Thread   │         │ Audio Thread │
│  (Main/egui)    │         │  (AI Processing) │         │ (Per sound)  │
│                 │         │                  │         │              │
│  • Capture 30   │         │  • Receive every │         │  • Play MP3  │
│    FPS camera   │         │    5th frame     │         │  • Auto-exit │
│  • Render UI    │         │  • Face detect   │         │              │
│  • Handle input │◄────────┤  • Emotion AI    ├────────►│              │
│  • Display      │ Channel │  • Send results  │ Channel │              │
└─────────────────┘         └──────────────────┘         └──────────────┘
```

**Thread Breakdown:**
1. **UI Thread**: Runs continuously, handles camera + rendering + user input
2. **Worker Thread**: Processes frames for AI (face detection + emotion recognition)
3. **Audio Threads**: Spawned on-demand when emotion changes, plays sound then exits

<!-- pause -->

**Why this design?**
- UI never blocks waiting for AI
- Audio plays independently without blocking anything
- Channels (mpsc + broadcast) for thread-safe communication

<!-- end_slide -->

<!-- jump_to_middle -->

Teşekkürler!
---