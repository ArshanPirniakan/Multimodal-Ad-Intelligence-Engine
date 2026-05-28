# Multimodal Ad Engine

A multimodal deep learning pipeline that scores real estate ad copy for predicted engagement and automatically rewrites underperforming listings using Gemini.

## How It Works

The model fuses two branches:

- **Vision branch** — EfficientNet-B0 (frozen, ImageNet weights) extracts a 1280-dim embedding from the property image
- **Text branch** — `all-MiniLM-L6-v2` (frozen) encodes the listing description into a 384-dim embedding via mean pooling
- **Fusion head** — a small MLP takes the concatenated 1664-dim vector and outputs an engagement score between 0 and 1

If the predicted score falls below the threshold (default: 0.45), the pipeline calls Gemini to rewrite the ad copy.

```
image ──► EfficientNet-B0 ──► 1280-dim ──┐
                                          ├──► FusionHead ──► score [0, 1]
text  ──► MiniLM ──────────►  384-dim ──┘
                                          │
                                          └── score < 0.45 ──► Gemini rewrite
```

## Setup

```bash
pip install torch torchvision transformers pandas numpy pillow
pip install google-generativeai  # optional, needed for Gemini rewrite
```

Set your Gemini API key if you want the rewrite feature:

```bash
export GEMINI_API_KEY=your_key_here
```

## Usage

**Run the demo (no dataset needed):**

```bash
python multimodal_ad_engine.py
```

**Train on a dataset:**

```bash
python multimodal_ad_engine.py data/raw/brasil_real_estate.csv
```

The CSV needs at minimum a description column (`description`, `title`, `ad_description`, or `text`) and optionally a price column (`price`, `price_usd`, `precio`, or `valor`) and an `image_path` column pointing to local image files.

**Run inference from Python:**

```python
from multimodal_ad_engine import run_inference

result = run_inference(
    image_path="property.jpg",
    description="Spacious 3-bed apartment with city views...",
    metadata={
        "bedrooms": 3,
        "price": 450000,
        "location": "São Paulo",
        "property_type": "apartment",
    }
)

print(result["predicted_score"])   # e.g. 0.38
print(result["flagged"])           # True if below threshold
print(result["optimized_copy"])    # Gemini rewrite if flagged
```

## Configuration

All settings live in the `Config` dataclass at the top of the file. Key ones:

| Parameter | Default | Description |
|---|---|---|
| `engagement_threshold` | `0.45` | Scores below this trigger a Gemini rewrite |
| `batch_size` | `32` | Training batch size |
| `learning_rate` | `1e-4` | AdamW learning rate |
| `num_epochs` | `10` | Training epochs |
| `dropout` | `0.3` | Dropout rate in fusion head |
| `checkpoint_path` | `model.pth` | Where the best model is saved |
| `gemini_model` | `gemini-2.5-flash` | Gemini model used for rewriting |

Device is auto-detected: CUDA > MPS > CPU.

## Engagement Score

The training label is computed from two signals:

- **Price score** — normalized price relative to the dataset min/max (50% weight)
- **Text score** — description length normalized to 500 chars (50% weight)

This is a proxy for engagement. If you have real click or conversion data, replace `compute_engagement_score` with your actual labels for better results.

## Project Structure

```
multimodal_ad_engine.py   # everything: model, training, inference, Gemini rewrite
data/
  raw/
    brasil_real_estate.csv
model.pth                 # saved after training
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- transformers
- torchvision
- pandas, numpy, Pillow
- google-generativeai (optional)
