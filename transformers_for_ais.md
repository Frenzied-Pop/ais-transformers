# 🌊 Using Transformers for AIS Data: A Primer

This guide outlines how to use transformer-based models, especially BERT-like architectures, for AIS (Automatic Identification System) data. The focus is on unsupervised learning, particularly masked language modeling (MLM), to detect vessel behavior patterns, generate embeddings, and build maritime intelligence tools.

---

## 📌 Overview

**AIS data** is inherently spatiotemporal and sparse, making it suitable for sequence modeling. Like NLP, AIS consists of time-ordered tokens: position, speed, direction, status, etc.

Transformer-based models, originally built for text, can be adapted to:

- Reconstruct missing AIS points (denoising)
- Embed vessel behavior for clustering
- Detect anomalies or loitering patterns
- Predict next positions or behaviors

---

## 🚢 Use Cases

| Use Case                 | Transformer Role           | Output                            |
| ------------------------ | -------------------------- | --------------------------------- |
| Trajectory Prediction    | Decoder or Encoder-Decoder | Next lat/lon or movement state    |
| Behavior Classification  | BERT-style encoder         | Normal / Loitering / STS Transfer |
| Embedding for Clustering | Self-supervised encoder    | Fixed-size vector per voyage      |
| Anomaly Detection        | MLM / Autoencoder          | Score or reconstruction error     |

---

## 🧠 Masked Language Modeling (MLM) for AIS

### 🎯 Concept

- Randomly mask 15% of AIS tokens
- Train the model to predict masked tokens from context
- No labels required — fully self-supervised

### ✅ Why it Works

- Captures normal vessel behavior patterns
- Learns contextual dependencies (e.g., SOG ↔ Lat/Lon ↔ COG)
- Embeddings from this pretraining generalize well to downstream tasks

### 🔠 Example Tokens

```text
[CLS] [Tanker] [Flag:Panama] [Grid:343x221] [SOG:8-10] [COG:SE] [Gap:10min] [MASK]
```

---

## 🔍 Tokenization Strategies for AIS

| Feature                  | Tokenization                        | Notes                    |
| ------------------------ | ----------------------------------- | ------------------------ |
| Latitude/Longitude       | Spatial grid cell (e.g. 0.05°)      | Unique token per cell    |
| Speed Over Ground (SOG)  | Binned (e.g., 0-2, 2-5, 5-10 knots) | Use domain knowledge     |
| Course Over Ground (COG) | Binned (e.g., N, NE, E, ...)        | 8 or 16 bins works well  |
| Time Delta               | Quantized (e.g., <1m, 1–5m, ...)    | Handle irregular spacing |
| Port, Flag, MMSI prefix  | One-hot or embedded                 | Useful as static context |

---

## 🧱 Model Options

### 🔷 Transformer Variants

| Model                     | Strength                                           |
| ------------------------- | -------------------------------------------------- |
| BERT                      | Bidirectional context, good for embedding behavior |
| BART                      | Sequence reconstruction or prediction              |
| Longformer / BigBird      | Long-range dependencies, sparse attention          |
| Informer / TimesNet       | Optimized for irregular, long-range time series    |
| ELECTRA / DeBERTa         | Efficient alternatives to BERT                     |
| MAE (Masked Autoencoders) | Continuous-valued denoising                        |

### 🔶 Non-Transformer Models

| Model                   | Notes                      |
| ----------------------- | -------------------------- |
| GRU-D / Time-aware LSTM | Handle gaps and time decay |
| TCNs (Temporal CNNs)    | Low-compute, interpretable |

### 🟢 Lightweight / Fusion Models

| Model                 | Use                                      |
| --------------------- | ---------------------------------------- |
| TinyBERT / DistilBERT | Compact for fast inference               |
| Perceiver IO          | Multi-modal AIS + ELINT + RF             |
| TabTransformer        | Embed tabular metadata (e.g., ship info) |

---

## 🧪 Embedding + Clustering Pipeline

1. Tokenize AIS sequences into fixed-length chunks
2. Train a masked BERT-like model to reconstruct missing tokens
3. Use `[CLS]` token or averaged embeddings as voyage representation
4. Apply:
   - HDBSCAN or OPTICS for clustering
   - KNN for similarity search
   - t-SNE or UMAP for visual inspection

---

## 🧠 Why This Works for Intelligence

- Unsupervised pretraining mimics how analysts spot patterns
- No need for labeled data (which is rare in maritime intelligence)
- Attention maps provide interpretability
- Embeddings transfer to:
  - Loitering detection
  - Dark fleet tracking
  - Sanctions enforcement

---

## 🔒 JWICS Note

If CIA AIML's S3 bucket contains:

- Transformer-based pretrained encoders for behavior modeling
- YOLOv11 for SAR spoof detection or port surveillance

…request updomaining to JWICS if mission-relevant. Highlight:

- Pattern discovery for dark ship tracking
- AI-assisted triage of vessel behavior anomalies
- Pretrained models enabling GPU-efficient experimentation

---

## 📎 Next Steps

- Build a prototype: tokenized sequences + MLM + unsupervised loss
- Request GPU access or mirror access to CIA-hosted models
- Explore TimesNet or Informer for forecasting and dense time series
- Develop a voyage embedding store for retrieval and comparison

---

## 📨 Contact / Credits

This guide was prepared as part of an internal research effort on the application of large language models and time series transformers to maritime domain awareness.

*Contributor: Tim*

---

**[GitHub Repo Placeholder]**: `https://github.com/yourusername/ais-transformers`

> 🧠 "The patterns are there. We just need to mask the noise until the signal shines through."

