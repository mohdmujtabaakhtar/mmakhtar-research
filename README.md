# Trustworthy & Clinical Speech AI: Research Code

**Mohd Mujtaba Akhtar** · [Homepage](https://mohdmujtabaakhtar.github.io/) · [Google Scholar](https://scholar.google.com/citations?user=bwHU6i8AAAAJ) · [GitHub](https://github.com/mohdmujtabaakhtar)

[![tests](https://github.com/mohdmujtabaakhtar/mmakhtar-research/actions/workflows/tests.yml/badge.svg)](https://github.com/mohdmujtabaakhtar/mmakhtar-research/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)

This repository collects code for my publications (2025–2026) on speech and audio AI. The work is built around one question: **how can representations from large pre-trained speech, audio and vision models be fused and structured so that they stay reliable for the settings that matter most?** That means clinical assessment, synthetic-speech forensics and paralinguistic understanding across languages.

All papers share one pipeline: **frozen foundation-model features → a light, task-specific downstream model**. The shared pieces (feature extraction, subject-wise cross-validation, metrics and non-Euclidean geometry) live in [`common/`](common/), and each paper adds its own model under [`papers/`](papers/).

## Research themes

- 🩺 **Clinical speech AI:** detecting neurological and respiratory conditions from voice, face and cough (ALS/stroke, Alzheimer's disease, tuberculosis), including zero-shot transfer across languages.
- 🔍 **Synthetic-speech forensics:** detecting deepfakes from neural audio codecs and diffusion TTS, and attributing them to their source generator, including open-set settings, clinical speech, singing voice and low-resource Indic languages.
- 🎭 **Paralinguistics and emotion:** speech emotion recognition across languages and from non-verbal vocalisations, via fusion of heterogeneous foundation models.
- 📐 **Geometry-aware fusion:** hyperbolic, spherical and optimal-transport methods for aligning representations from different pre-trained models.

## Papers

**Code status.** ✅ **Reference implementation**: re-implemented from the paper, tested and runnable end to end. 🗓 **Planned**: paper summary and citation only for now. The reference implementations are clean re-implementations written from each paper's method section, **not the original experimental code**. Each paper's README lists every detail the paper left open and the choice made for it.

### 🩺 Clinical speech AI

| Paper | Venue | Links | Code |
|---|---|---|---|
| **DIVINE:** Coordinating Multimodal Disentangled Representations for Oro-Facial Neurological Disorder Assessment<br><sub>**M. M. Akhtar\***, Girish\*, M. Singh</sub> | **EACL 2026** Main<br>🏆 Social Impact Award | [Paper](https://aclanthology.org/2026.eacl-long.248/) | ✅ [`papers/divine`](papers/divine) |
| **ORBIT:** Synergizing Zero-Shot Cross-Lingual Alzheimer Detection with Language-Invariant Multimodal Bi-Geometric Adversarial Learning<br><sub>Girish\*, **M. M. Akhtar\***, F. Sheth\*, M. Singh, J. Gerard, P. McClean, K. Wong-Lin</sub> | **INTERSPEECH 2026**<br>Oral | [arXiv](https://arxiv.org/abs/2606.17254) | 🗓 |
| **COBALT:** From Signals to Patterns: Non-Invasive Tuberculosis Detection from Cough Audio using Bandit Weighted Hyperbolic Prototypes<br><sub>**M. M. Akhtar\***, Girish\*, S. Wadhwa, M. Singh, N. Ma</sub> | **INTERSPEECH 2026**<br>Oral | [arXiv](https://arxiv.org/abs/2606.17337) | 🗓 |
| **HCFD:** A Benchmark for Audio Deepfake Detection in Healthcare<br><sub>**M. M. Akhtar\***, Girish\*, M. Singh</sub> | **ACL 2026** Findings | [Paper](https://aclanthology.org/2026.findings-acl.1739/) | 🗓 |

### 🔍 Synthetic-speech detection and source attribution

| Paper | Venue | Links | Code |
|---|---|---|---|
| **Indic-CodecFake meets SATYAM:** Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages<br><sub>Girish\*, **M. M. Akhtar\***, O. C. Phukan, A. B. Buduru</sub> | **ACL 2026** Findings | [Paper](https://aclanthology.org/2026.findings-acl.2159/) | 🗓 |
| **SIGNAL:** Bridging Attribution and Open-Set Detection using Graph-Augmented Instance Learning in Synthetic Speech<br><sub>**M. M. Akhtar\***, Girish\*, F. Sheth\*, M. Singh</sub> | **EACL 2026** Main | [Paper](https://aclanthology.org/2026.eacl-long.250/) | 🗓 |
| **RHYME:** Curved Worlds, Clear Boundaries: Generalizing Speech Deepfake Detection using Hyperbolic and Spherical Geometry Spaces<br><sub>F. Sheth\*, Girish\*, **M. M. Akhtar\***, M. Singh</sub> | **IJCNLP-AACL 2025** Main | [Paper](https://aclanthology.org/2025.ijcnlp-long.104/) | 🗓 |
| **MiCuNet:** Towards Attribution of Generators and Emotional Manipulation in Cross-Lingual Synthetic Speech using Geometric Learning<br><sub>Girish\*, **M. M. Akhtar\***, F. Sheth\*, M. Singh</sub> | **IJCNLP-AACL 2025** Findings | [Paper](https://aclanthology.org/2025.findings-ijcnlp.37/) | 🗓 |
| Towards Source Attribution of Singing Voice Deepfake with Multimodal Foundation Models<br><sub>O. C. Phukan\*, Girish\*, **M. M. Akhtar\***, S. R. Behera, P. Mallick, P. B. Reddy, A. B. Buduru, R. Sharma</sub> | **INTERSPEECH 2025** | [arXiv](https://arxiv.org/abs/2506.03364) | 🗓 |
| Source Tracing of Synthetic Speech Systems Through Paralinguistic Pre-Trained Representations<br><sub>Girish\*, **M. M. Akhtar\***, O. C. Phukan\*, D. Singh\*, S. R. Behera, P. B. Reddy, A. B. Buduru, R. Sharma</sub> | **EUSIPCO 2025** | [arXiv](https://arxiv.org/abs/2506.01157) | 🗓 |

### 🎭 Emotion and paralinguistics

| Paper | Venue | Links | Code |
|---|---|---|---|
| **NOVA-ARC:** Prosody as Supervision: Bridging the Non-Verbal–Verbal for Multilingual Speech Emotion Recognition<br><sub>Girish\*, **M. M. Akhtar\***, M. Singh</sub> | **ACL 2026** Main | [Paper](https://aclanthology.org/2026.acl-long.1940/) | 🗓 |
| **PARROT:** Synergizing Mamba and Attention-based SSL Pre-Trained Models via Parallel Branch Hadamard Optimal Transport for Speech Emotion Recognition<br><sub>O. C. Phukan\*, **M. M. Akhtar\***, Girish\*, S. R. Behera, J. S. K. Patibandla, A. B. Buduru, R. Sharma</sub> | **INTERSPEECH 2025** | [arXiv](https://arxiv.org/abs/2506.01138) | 🗓 |
| Strong Alone, Stronger Together: Synergizing Modality-Binding Foundation Models with Optimal Transport for Non-Verbal Emotion Recognition<br><sub>O. C. Phukan, **M. M. Akhtar\***, Girish\*, S. R. Behera, S. Kalita, A. B. Buduru, R. Sharma, S. R. M. Prasanna</sub> | **ICASSP 2025** | [Paper](https://ieeexplore.ieee.org/abstract/document/10889257) | 🗓 |
| **RENO:** Are Mamba-Based Audio Foundation Models the Best Fit for Non-Verbal Emotion Recognition?<br><sub>**M. M. Akhtar\***, O. C. Phukan\*, Girish\*, S. R. Behera, A. C. Nayak, S. K. Nayak, A. B. Buduru, R. Sharma</sub> | **EUSIPCO 2025** | [arXiv](https://arxiv.org/abs/2506.02258) | 🗓 |

### 🛡️ Multimodal content safety

| Paper | Venue | Links | Code |
|---|---|---|---|
| **SNIFR:** Boosting Fine-Grained Child Harmful Content Detection Through Audio-Visual Alignment with Cascaded Cross-Transformer<br><sub>O. C. Phukan, **M. M. Akhtar\***, Girish\*, S. R. Behera, A. O. Siddiqui, S. Jain, P. Mallick, J. S. K. Patibandla, P. B. Reddy, A. B. Buduru, R. Sharma</sub> | **INTERSPEECH 2025** | [arXiv](https://arxiv.org/abs/2506.03378) | 🗓 |

<sub>\* Equal contribution.</sub>

## Repository layout

```
mmakhtar-research/
├── common/                  shared code used by every paper
│   ├── features/            frozen feature extraction: speech (WavLM, wav2vec 2.0, HuBERT,
│   │                        Whisper, x-vector, TRILLsson) and video (VideoMAE, VideoMAE V2, ViViT)
│   ├── data.py              manifest-based datasets, subject-wise k-fold splits, synthetic data
│   ├── baselines.py         FCN / CNN probes and concatenation fusion used as baselines
│   ├── geometry.py          Poincaré-ball and hypersphere operations
│   ├── losses.py            KL, reparameterisation, sparsity
│   ├── metrics.py           accuracy, macro-F1, MAE, RMSE, EER
│   └── training.py          training loop with early stopping
├── papers/
│   └── divine/              one folder per paper: model, training script, config, README
├── tests/                   unit tests (run on every push)
├── CITATION.bib             BibTeX for all papers above
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/mohdmujtabaakhtar/mmakhtar-research.git
cd mmakhtar-research
pip install -r requirements.txt

pytest                                         # unit tests, ~10 s on CPU
python -m papers.divine.train --synthetic      # end-to-end smoke test, no downloads
```

Each paper's README explains how to extract features for its dataset, the manifest format and how to train and evaluate.

## Citation

BibTeX entries for every paper are in [`CITATION.bib`](CITATION.bib). If you use this code, please cite the relevant paper.

## Contact

Questions and collaboration requests are welcome: open an issue or email **mmakhtar.research@gmail.com**.

## License

Code is released under the [MIT License](LICENSE). Datasets used in the papers keep their own licences and access conditions. Clinical datasets in particular are available only from their original custodians.
