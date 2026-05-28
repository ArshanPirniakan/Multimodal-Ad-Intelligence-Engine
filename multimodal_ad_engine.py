import os
import warnings
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as tv_models

from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    image_size: int = 224
    max_text_length: int = 128
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    dropout: float = 0.3
    engagement_threshold: float = 0.45
    text_backbone: str = "sentence-transformers/all-MiniLM-L6-v2"
    vision_backbone: str = "efficientnet_b0"
    image_embed_dim: int = 1280
    text_embed_dim: int = 384
    fusion_hidden_dim: int = 512
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    checkpoint_path: Path = Path("model.pth")
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    gemini_model: str = "gemini-2.5-flash"
    dataset_csv: Path = Path("data/raw/brasil_real_estate.csv")


cfg = Config()


def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((cfg.image_size + 32, cfg.image_size + 32)),
        transforms.RandomCrop(cfg.image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_backbone)
    return tokenizer


def tokenize_text(text: str, tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    if not isinstance(text, str) or not text.strip():
        text = "no description available"
    encoding = tokenizer(
        text,
        max_length=cfg.max_text_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
    }


def load_image(image_path: str, transform: transforms.Compose) -> torch.Tensor:
    try:
        img = Image.open(image_path).convert("RGB")
        return transform(img)
    except Exception:
        img = Image.new("RGB", (cfg.image_size, cfg.image_size), color=(128, 128, 128))
        return transform(img)


def normalize_price(price: float, price_min: float, price_max: float) -> float:
    if price_max == price_min:
        return 0.5
    return float(np.clip((price - price_min) / (price_max - price_min), 0.0, 1.0))


def compute_engagement_score(row: pd.Series, price_min: float, price_max: float) -> float:
    price_score = normalize_price(row.get("price", 0.0), price_min, price_max)
    desc_len = len(str(row.get("description", "")))
    text_score = float(np.clip(desc_len / 500.0, 0.0, 1.0))
    score = 0.5 * price_score + 0.5 * text_score
    return float(np.clip(score, 0.0, 1.0))


class RealEstateDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        image_transform: transforms.Compose,
        price_min: float,
        price_max: float,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.price_min = price_min
        self.price_max = price_max

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        description = str(row.get("description", ""))
        image_path = str(row.get("image_path", ""))
        text_tensors = tokenize_text(description, self.tokenizer)
        image_tensor = load_image(image_path, self.image_transform)
        label = compute_engagement_score(row, self.price_min, self.price_max)
        return {
            "input_ids": text_tensors["input_ids"],
            "attention_mask": text_tensors["attention_mask"],
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
        }


def build_dataloaders(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    train_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:
    price_col = df["price"].dropna()
    price_min = float(price_col.min())
    price_max = float(price_col.max())
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    train_dataset = RealEstateDataset(train_df, tokenizer, get_train_transforms(), price_min, price_max)
    val_dataset = RealEstateDataset(val_df, tokenizer, get_val_transforms(), price_min, price_max)
    pin = cfg.device == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    return train_loader, val_loader


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    count = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / count


class VisionBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool = backbone.avgpool
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


class TextBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(cfg.text_backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return mean_pool(outputs.last_hidden_state, attention_mask)


class FusionHead(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MultimodalAdEngine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_branch = VisionBranch()
        self.text_branch = TextBranch()
        fusion_in_dim = cfg.image_embed_dim + cfg.text_embed_dim
        self.fusion_head = FusionHead(fusion_in_dim)

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        visual_embed = self.vision_branch(image)
        text_embed = self.text_branch(input_ids, attention_mask)
        fused = torch.cat([visual_embed, text_embed], dim=-1)
        return self.fusion_head(fused)


def compute_r2(preds: torch.Tensor, targets: torch.Tensor) -> float:
    ss_res = torch.sum((targets - preds) ** 2).item()
    ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
    return 1.0 - ss_res / (ss_tot + 1e-9)


def train_epoch(
    model: MultimodalAdEngine,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        preds = model(images, input_ids, attention_mask)
        loss = loss_fn(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def validate_epoch(
    model: MultimodalAdEngine,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_preds: list = []
    all_targets: list = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            preds = model(images, input_ids, attention_mask)
            loss = loss_fn(preds, labels)
            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())
    all_preds_t = torch.cat(all_preds)
    all_targets_t = torch.cat(all_targets)
    r2 = compute_r2(all_preds_t, all_targets_t)
    return total_loss / max(len(loader), 1), r2


def run_training(df: pd.DataFrame) -> None:
    device = torch.device(cfg.device)
    logger.info(f"Training on device: {device}")
    tokenizer = load_tokenizer()
    train_loader, val_loader = build_dataloaders(df, tokenizer)
    model = MultimodalAdEngine().to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)
    loss_fn = nn.SmoothL1Loss()
    best_val_loss = float("inf")
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, r2 = validate_epoch(model, val_loader, loss_fn, device)
        logger.info(
            f"Epoch {epoch}/{cfg.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"R²: {r2:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.checkpoint_path)
            logger.info(f"Checkpoint saved (val_loss={best_val_loss:.4f})")
    logger.info("Training complete.")


def load_trained_model(device: torch.device) -> MultimodalAdEngine:
    model = MultimodalAdEngine().to(device)
    if cfg.checkpoint_path.exists():
        model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device, weights_only=True))
        logger.info(f"Loaded checkpoint from {cfg.checkpoint_path}")
    else:
        logger.warning("No checkpoint found. Using untrained model weights.")
    model.eval()
    return model


def predict_score(
    model: MultimodalAdEngine,
    tokenizer: AutoTokenizer,
    image_path: str,
    description: str,
    device: torch.device,
) -> float:
    transform = get_val_transforms()
    image_tensor = load_image(image_path, transform).unsqueeze(0).to(device)
    text_tensors = tokenize_text(description, tokenizer)
    input_ids = text_tensors["input_ids"].unsqueeze(0).to(device)
    attention_mask = text_tensors["attention_mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(image_tensor, input_ids, attention_mask)
    return float(score.item())


def rewrite_with_gemini(
    original_copy: str,
    metadata: Dict[str, Any],
) -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        return "[google-generativeai not installed. Run: pip install google-generativeai]"
    if not cfg.gemini_api_key:
        return "[GEMINI_API_KEY environment variable not set.]"
    genai.configure(api_key=cfg.gemini_api_key)
    gemini = genai.GenerativeModel(cfg.gemini_model)
    bedrooms = metadata.get("bedrooms", "N/A")
    price = metadata.get("price", "N/A")
    location = metadata.get("location", "N/A")
    property_type = metadata.get("property_type", "property")
    prompt = f"""You are an elite real estate copywriter specializing in high-conversion digital advertising.

The following advertisement copy has been flagged as underperforming by an AI scoring system:

--- ORIGINAL COPY ---
{original_copy}
---

Property details (must be preserved accurately):
- Type: {property_type}
- Bedrooms: {bedrooms}
- Price: {price}
- Location: {location}

Your task: Rewrite this ad copy to maximize engagement. Follow these rules strictly:
1. Open with a compelling hook that creates emotional desire or urgency.
2. Highlight unique visual and lifestyle assets of the property.
3. Preserve ALL factual details (bedrooms, price, location) without distortion.
4. End with a clear, high-converting call to action.
5. Keep it under 120 words. Do not use buzzword fluff or generic phrases like "dream home."

Return only the rewritten copy. No explanations, no headers."""
    response = gemini.generate_content(prompt)
    return response.text.strip()


def run_inference(
    image_path: str,
    description: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    device = torch.device(cfg.device)
    tokenizer = load_tokenizer()
    model = load_trained_model(device)
    score = predict_score(model, tokenizer, image_path, description, device)
    result: Dict[str, Any] = {
        "predicted_score": round(score, 4),
        "threshold": cfg.engagement_threshold,
        "flagged": score < cfg.engagement_threshold,
        "original_copy": description,
        "optimized_copy": None,
    }
    if result["flagged"] and metadata is not None:
        logger.info(f"Score {score:.4f} below threshold {cfg.engagement_threshold}. Triggering Gemini rewrite.")
        optimized = rewrite_with_gemini(description, metadata)
        result["optimized_copy"] = optimized
    else:
        logger.info(f"Score {score:.4f} is above threshold. Ad copy is performing well.")
    return result


def load_brasil_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    path = csv_path or str(cfg.dataset_csv)
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Raw shape: {df.shape}")
    description_col = None
    for candidate in ["description", "title", "ad_description", "text"]:
        if candidate in df.columns:
            description_col = candidate
            break
    if description_col is None:
        raise ValueError(f"No description column found. Available columns: {list(df.columns)}")
    if description_col != "description":
        df = df.rename(columns={description_col: "description"})
    price_col = None
    for candidate in ["price", "price_usd", "precio", "valor"]:
        if candidate in df.columns:
            price_col = candidate
            break
    if price_col and price_col != "price":
        df = df.rename(columns={price_col: "price"})
    elif not price_col:
        df["price"] = 0.0
    if "image_path" not in df.columns:
        df["image_path"] = ""
    df["description"] = df["description"].fillna("").astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df = df[df["description"].str.strip() != ""].reset_index(drop=True)
    logger.info(f"Cleaned shape: {df.shape}")
    return df


def demo_without_dataset() -> None:
    logger.info("Running demo with synthetic data (no dataset CSV provided).")
    device = torch.device(cfg.device)
    tokenizer = load_tokenizer()
    model = MultimodalAdEngine().to(device)
    model.eval()
    dummy_image = torch.zeros(1, 3, cfg.image_size, cfg.image_size).to(device)
    sample_description = (
        "Spacious 3-bedroom apartment in the heart of São Paulo. "
        "Modern finishes, open kitchen, balcony with city views. "
        "Close to major metro lines and business districts."
    )
    text_tensors = tokenize_text(sample_description, tokenizer)
    input_ids = text_tensors["input_ids"].unsqueeze(0).to(device)
    attention_mask = text_tensors["attention_mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(dummy_image, input_ids, attention_mask)
    predicted = float(score.item())
    logger.info(f"Demo predicted score: {predicted:.4f}")
    flagged = predicted < cfg.engagement_threshold
    logger.info(f"Flagged for rewrite: {flagged}")
    if flagged:
        logger.info("Would trigger Gemini rewrite with GEMINI_API_KEY set.")
    logger.info("Demo complete. Pass a real dataset CSV to run_training() for full training.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        df = load_brasil_dataset(csv_path)
        run_training(df)
    else:
        demo_without_dataset()
