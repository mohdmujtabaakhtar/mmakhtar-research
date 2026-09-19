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
- 🔍 **Synthetic-speech forensics:** detecting deepfakes from neural audio codecs and diffusion TTS, and attributing them to their source generator, including open-set settings, clinical speech, singing voice, emotional manipulation and low-resource Indic and South-East Asian languages, with audio-language models as detectors.
- 🎭 **Paralinguistics and emotion:** speech emotion recognition across languages and from non-verbal vocalisations, via fusion of heterogeneous foundation models.
- 🛡️ **Multimodal content safety:** audio-visual detection of harmful content in videos watched by children.
- 📐 **Geometry-aware fusion:** hyperbolic, spherical and optimal-transport methods for aligning representations from different pre-trained models.

## Papers

**Code status:** all 16 papers have code. Every model is tested and runs end to end on generated data, and each paper's ablations are config switches.

### 🩺 Clinical speech AI

| Paper | Venue | Links | Code |
|---|---|---|---|
| **DIVINE:** Coordinating Multimodal Disentangled Representations for Oro-Facial Neurological Disorder Assessment<br><sub>**M. M. Akhtar\***, Girish\*, M. Singh</sub> | **EACL 2026** Main<br>🏆 Social Impact Award | [Paper](https://aclanthology.org/2026.eacl-long.248/) | ✅ [`papers/divine`](papers/divine) |
| **ORBIT:** Synergizing Zero-Shot Cross-Lingual Alzheimer Detection with Language-Invariant Multimodal Bi-Geometric Adversarial Learning<br><sub>Girish\*, **M. M. Akhtar\***, F. Sheth\*, M. Singh, J. Gerard, P. McClean, K. Wong-Lin</sub> | **INTERSPEECH 2026**<br>Oral | [arXiv](https://arxiv.org/abs/2606.17254) | ✅ [`papers/orbit`](papers/orbit) |
| **COBALT:** From Signals to Patterns: Non-Invasive Tuberculosis Detection from Cough Audio using Bandit Weighted Hyperbolic Prototypes<br><sub>**M. M. Akhtar\***, Girish\*, S. Wadhwa, M. Singh, N. Ma</sub> | **INTERSPEECH 2026**<br>Oral | [arXiv](https://arxiv.org/abs/2606.17337) | ✅ [`papers/cobalt`](papers/cobalt) |
| **HCFD:** A Benchmark for Audio Deepfake Detection in Healthcare (**PHOENIX-Mamba**)<br><sub>**M. M. Akhtar\***, Girish\*, M. Singh</sub> | **ACL 2026** Findings | [Paper](https://aclanthology.org/2026.findings-acl.1739/) · [Project page](https://helixometry.github.io/HCFD/) | ✅ [`papers/hcfd`](papers/hcfd) |

### 🔍 Synthetic-speech detection and source attribution

| Paper | Venue | Links | Code |
|---|---|---|---|
| **Indic-CodecFake meets SATYAM:** Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages<br><sub>Girish\*, **M. M. Akhtar\***, O. C. Phukan\*, A. B. Buduru</sub> | **ACL 2026** Findings | [Paper](https://aclanthology.org/2026.findings-acl.2159/) · [Project page](https://helixometry.github.io/IndicFake/) | ✅ [`papers/satyam`](papers/satyam) |
| **GARUDA:** Bridging the SEA Gap: An Initial Benchmark for Neural Audio Codec-Synthesized Speech Deepfakes in South-East Asian Languages<br><sub>O. C. Phukan\*, Girish\*, **M. M. Akhtar\***, A. B. Buduru</sub> | **IJCAI 2026** | [arXiv](https://arxiv.org/abs/2606.15968) · [Project page](https://helixometry.github.io/SEACodecFake/) | ✅ [`papers/garuda`](papers/garuda) |
| **SIGNAL:** Bridging Attribution and Open-Set Detection using Graph-Augmented Instance Learning in Synthetic Speech<br><sub>**M. M. Akhtar\***, Girish\*, F. Sheth\*, M. Singh</sub> | **EACL 2026** Main | [Paper](https://aclanthology.org/2026.eacl-long.250/) | ✅ [`papers/signal`](papers/signal) |
| **RHYME:** Curved Worlds, Clear Boundaries: Generalizing Speech Deepfake Detection using Hyperbolic and Spherical Geometry Spaces<br><sub>F. Sheth\*, Girish\*, **M. M. Akhtar\***, M. Singh</sub> | **IJCNLP-AACL 2025** Main | [Paper](https://aclanthology.org/2025.ijcnlp-long.104/) | ✅ [`papers/rhyme`](papers/rhyme) |
| **MiCuNet:** Towards Attribution of Generators and Emotional Manipulation in Cross-Lingual Synthetic Speech using Geometric Learning<br><sub>Girish\*, **M. M. Akhtar\***, F. Sheth, M. Singh</sub> | **IJCNLP-AACL 2025** Findings | [Paper](https://aclanthology.org/2025.findings-ijcnlp.37/) | ✅ [`papers/micunet`](papers/micunet) |
| **COFFE:** Towards Source Attribution of Singing Voice Deepfake with Multimodal Foundation Models<br><sub>O. C. Phukan\*, Girish\*, **M. M. Akhtar\***, S. R. Behera, P. Mallick, P. B. Reddy, A. B. Buduru, R. Sharma</sub> | **INTERSPEECH 2025** | [arXiv](https://arxiv.org/abs/2506.03364) | ✅ [`papers/coffe`](papers/coffe) |
| **TRIO:** Source Tracing of Synthetic Speech Systems Through Paralinguistic Pre-Trained Representations<br><sub>Girish\*, **M. M. Akhtar\***, O. C. Phukan\*, D. Singh\*, S. R. Behera, P. B. Reddy, A. B. Buduru, R. Sharma</sub> | **EUSIPCO 2025** | [arXiv](https://arxiv.org/abs/2506.01157) | ✅ [`papers/trio`](papers/trio) |

### 🎭 Emotion and paralinguistics

| Paper | Venue | Links | Code |
|---|---|---|---|
| **NOVA-ARC:** Prosody as Supervision: Bridging the Non-Verbal–Verbal for Multilingual Speech Emotion Recognition<br><sub>Girish\*, **M. M. Akhtar\***, M. Singh</sub> | **ACL 2026** Main | [Paper](https://aclanthology.org/2026.acl-long.1940/) | ✅ [`papers/nova_arc`](papers/nova_arc) |
| **PARROT:** Synergizing Mamba and Attention-based SSL Pre-Trained Models via Parallel Branch Hadamard Optimal Transport for Speech Emotion Recognition<br><sub>O. C. Phukan\*, **M. M. Akhtar\***, Girish\*, S. R. Behera, J. S. K. Patibandla, A. B. Buduru, R. Sharma</sub> | **INTERSPEECH 2025** | [arXiv](https://arxiv.org/abs/2506.01138) | ✅ [`papers/parrot`](papers/parrot) |
| **MATA:** Strong Alone, Stronger Together: Synergizing Modality-Binding Foundation Models with Optimal Transport for Non-Verbal Emotion Recognition<br><sub>O. C. Phukan, **M. M. Akhtar\***, Girish\*, S. R. Behera, S. Kalita, A. B. Buduru, R. Sharma, S. R. M. Prasanna</sub> | **ICASSP 2025** | [Paper](https://ieeexplore.ieee.org/abstract/document/10889257) | ✅ [`papers/mata`](papers/mata) |
| **RENO:** Are Mamba-Based Audio Foundation Models the Best Fit for Non-Verbal Emotion Recognition?<br><sub>**M. M. Akhtar\***, O. C. Phukan\*, Girish\*, S. R. Behera, A. C. Nayak, S. K. Nayak, A. B. Buduru, R. Sharma</sub> | **EUSIPCO 2025** | [arXiv](https://arxiv.org/abs/2506.02258) | ✅ [`papers/reno`](papers/reno) |

### 🛡️ Multimodal content safety

| Paper | Venue | Links | Code |
|---|---|---|---|
| **SNIFR:** Boosting Fine-Grained Child Harmful Content Detection Through Audio-Visual Alignment with Cascaded Cross-Transformer<br><sub>O. C. Phukan, **M. M. Akhtar\***, Girish\*, S. R. Behera, A. O. Siddiqui, S. Jain, P. Mallick, J. S. K. Patibandla, P. B. Reddy, A. B. Buduru, R. Sharma</sub> | **INTERSPEECH 2025** | [arXiv](https://arxiv.org/abs/2506.03378) | ✅ [`papers/snifr`](papers/snifr) |

<sub>\* Equal contribution.</sub>

## Repository layout

```
mmakhtar-research/
├── common/                  shared code used by every paper
│   ├── features/            frozen feature extraction
│   │   ├── speech.py        WavLM, wav2vec 2.0, HuBERT, UniSpeech-SAT, Whisper, voc2vec, mHuBERT-147,
│   │   │                    MMS, XLS-R, AST, x-vector, ECAPA, TRILLsson, PaSST (frames or pooled vectors)
│   │   ├── video.py         VideoMAE, VideoMAE V2, ViViT
│   │   ├── text.py          mBERT, XLM-R, multilingual E5, Qwen3-Embedding
│   │   └── spectral.py      MFCC, LFCC; STFT / CQT / wavelet spectrograms with mel, gammatone
│   │                        or linear filterbanks
│   ├── geometry.py          Poincaré ball, hypersphere and Euclidean manifolds; Fréchet mean;
│   │                        stereographic and north-pole maps
│   ├── ot.py                Sinkhorn optimal transport, batch feature transport, gradient reversal
│   ├── divergences.py       Bhattacharyya, Jensen-Shannon, Chernoff, Rényi; CCA objective
│   ├── alm.py               audio-prefix language-model head (Qwen2 or an offline toy LM), LoRA
│   ├── mamba.py             dependency-free Mamba (selective state-space) block
│   ├── layers.py            conv adapters, embedding CNN, attention pooling
│   ├── baselines.py         FCN / CNN probes and concatenation fusion used as baselines
│   ├── runner.py            shared k-fold / official-split experiment runner and CLI
│   ├── data.py              manifest-based datasets, subject-wise k-fold splits
│   ├── training.py          training loop, AdamW + warm-up/cosine, early stopping
│   ├── config.py            YAML configs with command-line overrides (--set key=value)
│   ├── losses.py, metrics.py
├── papers/                  one folder per paper: model, training script, configs, README
│   ├── divine/              DIVINE (EACL 2026)
│   ├── orbit/               ORBIT (INTERSPEECH 2026)
│   ├── cobalt/              COBALT (INTERSPEECH 2026)
│   ├── hcfd/                PHOENIX-Mamba (ACL 2026 Findings)
│   ├── satyam/              SATYAM (ACL 2026 Findings)
│   ├── garuda/              GARUDA (IJCAI 2026)
│   ├── signal/              SIGNAL (EACL 2026)
│   ├── rhyme/               RHYME (IJCNLP-AACL 2025)
│   ├── micunet/             MiCuNet (IJCNLP-AACL 2025 Findings)
│   ├── coffe/               COFFE, singing-voice source attribution (INTERSPEECH 2025)
│   ├── trio/                TRIO, source tracing (EUSIPCO 2025)
│   ├── nova_arc/            NOVA-ARC (ACL 2026)
│   ├── parrot/              PARROT (INTERSPEECH 2025)
│   ├── mata/                MATA, "Strong Alone, Stronger Together" (ICASSP 2025)
│   ├── reno/                RENO (EUSIPCO 2025)
│   └── snifr/               SNIFR (INTERSPEECH 2025)
├── tests/                   unit tests (run on every push)
├── assets/                  architecture figures
├── CITATION.bib             BibTeX for all papers above
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/mohdmujtabaakhtar/mmakhtar-research.git
cd mmakhtar-research
pip install -r requirements.txt

pytest                                          # unit tests, ~30 s on CPU

# end-to-end smoke tests on generated data (no downloads, ~10 s to 3 min each on CPU)
python -m papers.divine.train   --synthetic
python -m papers.orbit.train    --synthetic
python -m papers.cobalt.train   --synthetic
python -m papers.hcfd.train     --synthetic
python -m papers.satyam.train   --synthetic     # uses a tiny offline LM in place of Qwen2
python -m papers.garuda.train   --synthetic     # uses a tiny offline LM in place of Qwen2
python -m papers.signal.train   --synthetic
python -m papers.rhyme.train    --synthetic
python -m papers.micunet.train  --synthetic
python -m papers.coffe.train    --synthetic
python -m papers.trio.train     --synthetic
python -m papers.nova_arc.train --synthetic
python -m papers.parrot.train   --synthetic
python -m papers.mata.train     --synthetic
python -m papers.reno.train     --synthetic
python -m papers.snifr.train    --synthetic
```

Every paper's model has switches for the ablations in its paper, e.g.
`--set model.geometry=euclidean`, and most training scripts also run the paper's baselines with `--model`.

Each paper's README explains how to extract features for its dataset, the manifest format and how to train and evaluate.

## Citation

BibTeX entries for every paper are in [`CITATION.bib`](CITATION.bib). If you use this code, please cite the relevant paper.

## Contact

Questions and collaboration requests are welcome: open an issue or email **mmakhtar.research@gmail.com**.

## License

Code is released under the [MIT License](LICENSE). Datasets used in the papers keep their own licences and access conditions. Clinical datasets in particular are available only from their original custodians.
