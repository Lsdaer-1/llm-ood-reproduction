# Multi-modal OOD detection reproduction

This repository is my reproduction and extension of ideas from *Exploring Large Language Models for Multi-Modal Out-of-Distribution Detection* (Findings of EMNLP 2023). It starts from a CLIP maximum-concept-matching baseline and adds LLM-generated class descriptors, descriptor calibration, object cues, and retrieval-based analysis.

## Result snapshot

The strongest local experiment used Oxford-IIIT Pet as in-distribution data and a hard 500-image NINCO subset as OOD data.

| Method | AUROC ↑ | FPR95 ↓ |
| --- | ---: | ---: |
| CLIP MCM baseline | 0.9793 | 0.080 |
| Descriptor scoring | 0.9629 | 0.102 |
| Calibrated descriptors | 0.9714 | 0.092 |
| Calibrated descriptors + object cues | **0.9864** | **0.058** |

All rows use 500 ID and 500 OOD images. The JSON summaries under [`results/`](results/) preserve the parameters and measured values.

## Pipeline

1. Encode images and ID class prompts with CLIP.
2. Use maximum similarity as the MCM confidence baseline.
3. Generate multiple natural-language descriptors per class.
4. Calibrate descriptor sets with retrieval evidence.
5. Combine descriptor similarity with detected object words.
6. Report AUROC and FPR95.

## Run the baseline

```bash
pip install -r requirements.txt
python scripts/run_mcm.py \
  --id_dataset cifar10 \
  --ood_dataset cifar100 \
  --device cpu
```

The repository includes experiment scripts, dataset adapters, generated descriptor JSON, and compact result summaries. Image datasets, downloaded CLIP weights, YOLO weights, and the large retrieval database are intentionally excluded.

Some descriptor-generation scripts use the OpenAI API. Set `OPENAI_API_KEY` in the environment; do not store it in source files.

## Reproducibility notes

- The hard NINCO subset was selected during exploratory analysis, so it should not be interpreted as a standard benchmark split.
- Generated descriptors and object lists can vary with model version and sampling behavior.
- Local model paths in generated summaries have been replaced with portable model identifiers.

## Paper

The reproduced paper is [Exploring Large Language Models for Multi-Modal Out-of-Distribution Detection](https://aclanthology.org/2023.findings-emnlp.351/) by Yi Dai, Hao Lang, Kaisheng Zeng, Fei Huang, and Yongbin Li. This repository is an independent reproduction, not an official implementation.
