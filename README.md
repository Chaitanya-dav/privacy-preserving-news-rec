# Privacy-Preserving Multimodal News Recommender

A complete end-to-end **privacy-preserving federated news recommendation system** trained
on the [MIND dataset](https://msnews.github.io/) with multimodal article understanding
(text + image) and differential privacy.

## 🏆 Results at a Glance

| Setup | AUC | MRR | nDCG@5 | nDCG@10 |
|-------|-----|-----|--------|---------|
| Centralized (5 epochs) | **0.9748** | 0.9683 | 0.9700 | 0.9785 |
| Federated + DP (5 rounds) | **0.9795** | 0.9717 | 0.9762 | **0.9802** |

> Federated model **matches or exceeds** centralized on all metrics with full privacy protection.

---

## Architecture

```
Article ──► DistilBERT CLS (768D) ─┐
            ResNet-18 Pool (512D) ──┼──► Concat Proj (256D) ──► News Vector
                                    ┘
User History ──► [News Vectors] ──► Mean Pooling ──► User Vector
                                               │
Candidates ──► [News Vectors] ─────────────────┘
                                               │
                                    Dot Product Score ──► Top-K Recs
```

**Privacy Stack:** Flower FedAvg → Gradient Clipping → DP-SGD Noise → Secure Aggregation Masking

---

## Project Structure

```
privacy_preserving_multimodal_news_rec/
├── configs/                  # YAML configs (base, federated, privacy)
├── preprocessing/            # MIND parser, multimodal extractor, embedding cache
├── models/                   # NewsEncoder, UserEncoder, Recommender
├── centralized/              # Centralized trainer & training script
├── federated/                # Flower client, server, partitioner
├── privacy/                  # Differential privacy, secure aggregation
├── evaluation/               # AUC, MRR, nDCG metrics
├── backend/                  # Flask REST API
│   ├── routes/               # recommend.py, metrics.py, news.py
│   └── services/             # model_loader.py, recommender_service.py
├── frontend/                 # Interactive web dashboard
│   ├── index.html            # Single-page app (4 sections)
│   ├── css/dashboard.css     # Dark glassmorphism UI
│   └── js/dashboard.js      # Chart.js + API integration
├── tests/                    # Unit tests (preprocessing, models, federated)
├── data/                     # Raw MIND data + processed embeddings
├── results/                  # Checkpoints + final_report.md
├── generate_report.py        # Phase 20 final report generator
├── requirements.txt
└── wsgi.py                   # Production WSGI entry
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the backend API
```bash
py backend/app.py
# API at http://localhost:5000
```

### 3. Open the dashboard
Open `frontend/index.html` in a browser (or serve with any static server).

### 4. Generate final report
```bash
py generate_report.py
# Output: results/final_report.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/recommend/` | Top-K personalised recs |
| GET | `/api/recommend/demo` | Demo recommendations |
| GET | `/api/metrics/` | Full training history |
| GET | `/api/metrics/summary` | Final metrics per mode |
| GET | `/api/news/` | Paginated news catalogue |
| GET | `/api/news/categories` | Category counts |
| GET | `/api/news/<id>` | Single article metadata |

---

## 20-Phase Roadmap — All Complete

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Environment & Config | ✅ |
| 2 | MIND Preprocessing | ✅ |
| 3 | Text Encoder (DistilBERT) | ✅ |
| 4 | Image Encoder (ResNet-18) | ✅ |
| 5 | Multimodal Fusion | ✅ |
| 6 | User Encoder | ✅ |
| 7 | Recommender Model | ✅ |
| 8 | Embedding Cache | ✅ |
| 9 | Dataset & DataLoader | ✅ |
| 10 | Centralized Training | ✅ |
| 11 | Evaluation Metrics | ✅ |
| 12 | Federated Partitioner | ✅ |
| 13 | Federated Client (Flower) | ✅ |
| 14 | Federated Server / FedAvg | ✅ |
| 15 | Differential Privacy | ✅ |
| 16 | Secure Aggregation | ✅ |
| 17 | Unit Tests | ✅ |
| 18 | Backend REST API | ✅ |
| 19 | Frontend Dashboard | ✅ |
| 20 | Integration & Final Report | ✅ |