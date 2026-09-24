# Human Intent Prediction for Adaptive Lower-Limb Exoskeletons

<div align="center">

![Project Banner](https://img.shields.io/badge/Research-Exoskeleton%20AI-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Deep%20Learning](https://img.shields.io/badge/Deep%20Learning-CNN%2FLSTM%2FAttention-5F9EA0)
![Docs](https://img.shields.io/badge/Docs-PDF%20%26%20Monograph-green)

</div>

A research project focused on predictive human locomotor intent recognition for adaptive lower-limb exoskeleton control.

## Overview

This work investigates how to anticipate user movement intentions before the motion is fully executed, enabling more natural, responsive, and safe assistance from wearable robotic systems.

The project combines:
- biomechanics-aware signal analysis,
- deep learning for temporal intent prediction,
- multimodal locomotion modeling,
- experimental evaluation and comparative analysis.

## Why this project matters

Lower-limb exoskeletons need to move in sync with the user rather than react only after motion has already started. The goal is to predict intent early enough to support walking transitions, gait adaptation, and safe assistance in real time.

This repository covers the full research workflow, from data generation and preprocessing to model training, experiments, and monograph documentation.

## Key research directions

- Human intent prediction from wearable sensor data
- Temporal modeling of locomotor transitions
- Adaptive assistance for gait and posture control
- Domain generalization across users
- Few-shot and transfer learning strategies
- Practical deployment considerations for wearable robotics

## Repository structure

```text
.
├── Monograph_Full.md         # Full research monograph in Markdown
├── Monograph_Full.pdf        # PDF export of the documentation
├── README.md                 # Project landing page
├── code/
│   ├── data_generator.py     # Synthetic and structured data generation
│   ├── preprocessing.py     # Feature engineering and signal processing
│   ├── models.py            # Deep learning architectures
│   ├── training.py          # Training pipeline
│   ├── run_experiments.py   # Experiment runner
│   └── requirements.txt     # Project dependencies
├── sections/
│   ├── section_1_introduction.md
│   ├── section_2_literature_review.md
│   ├── section_3_methodology.md
│   ├── section_4_experiments.md
│   ├── section_5_discussion.md
│   ├── section_6_conclusion.md
│   └── section_7_references.md
└── ...
```

## Documentation

- Full PDF documentation: https://drive.google.com/file/d/1H6S0WTeTOPKSrNXUZ-H9zbas9I5VvW2L/view?usp=sharing
- Full monograph source: `Monograph_Full.md`

## Project goals

1. Improve predictive accuracy for locomotor intent recognition.
2. Reduce latency between human intention and assistive response.
3. Develop models robust to user variability and noisy measurements.
4. Support real-world deployment in adaptive exoskeleton control systems.

## Methodology snapshot

The project investigates deep architectures such as CNNs, LSTMs, ConvLSTM, and attention-based temporal models for predicting locomotor state transitions using sensor-driven signals.

## Status

This repository is intended for research documentation, reproducible experiments, and academic sharing.

## Contact

Project author: Amanat Korgan

---

<p align="center">
  <strong>Designed for research, reproducibility, and practical exoskeleton intelligence.</strong>
</p>
