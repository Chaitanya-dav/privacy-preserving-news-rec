"""MIND Dataset Downloader and High-Fidelity Synthetic Demo Generator.

Supports downloading the official Microsoft News Dataset (MINDsmall)
as well as generating a self-contained, realistic multimodal demo dataset
for rapid offline prototyping, development, and test execution.
"""

import os
import zipfile
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

MIND_URLS = {
    "small": {
        "train": "https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_train.zip",
        "dev": "https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_dev.zip",
    },
    "large": {
        "train": "https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_train.zip",
        "dev": "https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_dev.zip",
    },
}

CATEGORIES = [
    ("technology", ["ai", "gadgets", "cybersecurity", "software"]),
    ("finance", ["markets", "crypto", "banking", "realestate"]),
    ("sports", ["football", "basketball", "tennis", "formula1"]),
    ("health", ["nutrition", "medicine", "fitness", "mentalhealth"]),
    ("politics", ["elections", "policy", "diplomacy", "governance"]),
    ("science", ["astronomy", "climate", "physics", "biology"]),
    ("entertainment", ["movies", "music", "gaming", "celebrity"]),
]

CATEGORY_COLORS = {
    "technology": (33, 150, 243),
    "finance": (76, 175, 80),
    "sports": (255, 152, 0),
    "health": (233, 30, 99),
    "politics": (156, 39, 176),
    "science": (0, 188, 212),
    "entertainment": (255, 87, 34),
}

HEADLINE_TEMPLATES = {
    "technology": [
        ("Next-Gen Multimodal AI Model Outperforms Human Benchmarks", "Breakthrough architectures combine vision, text, and privacy-preserving federated intelligence."),
        ("Quantum Computing Prototype Solves Complex Optimization Problems", "Engineers demonstrate fault-tolerant quantum error correction running at near zero temperatures."),
        ("Cybersecurity Alert: Zero-Day Vulnerability Patched Across Global Clouds", "Major cloud providers deploy urgent kernel patches to protect distributed enterprise workloads."),
        ("Autonomous Drone Fleet Delivers Critical Medical Supplies to Remote Regions", "Electric autonomous delivery systems achieve record range and reliability during high-altitude trials."),
        ("Open-Source Foundation Releases State-of-the-Art Edge Reasoning Model", "Lightweight language models achieve desktop-class inference speed directly on mobile silicon."),
    ],
    "finance": [
        ("Global Markets Rally Following Central Bank Interest Rate Cut", "Equities surge across Asia and Europe as inflation metrics moderate to target quarterly levels."),
        ("Decentralized Finance Protocol Reaches Record Total Value Locked", "Institutional liquidity pools fuel renewed momentum in zero-knowledge privacy financial networks."),
        ("Green Energy Investment Index Outperforms Fossil Fuel Equities", "Renewable energy infrastructure funds report unprecedented annual dividend yields and capital growth."),
        ("Fintech Startup Unveils Cross-Border Instant Payment Infrastructure", "Low-fee settlement rail utilizes distributed cryptographic ledgers to settle payments in seconds."),
        ("Tech Sector Earnings Surpass Forecasts Amid Robust Enterprise Cloud Demand", "Leading software conglomerates report double-digit revenue expansion driven by automated services."),
    ],
    "sports": [
        ("Underdog Team Claims Championship Glory in Thrilling Overtime Finish", "A sensational buzzer-beater seals an unexpected comeback victory in the national finals."),
        ("Star Forward Breaks Decades-Old Scoring Record with Hat-Trick", "Veteran striker enters the history books after an electrifying performance in front of seventy thousand fans."),
        ("Grand Prix Weekend Sees Record-Breaking Laps in Unpredictable Weather", "Tire strategy in sudden rain proves decisive as young driver claims first career podium."),
        ("Marathon Runner Shatters World Record in Breathtaking Solo Run", "Paced to perfection, the reigning champion crosses the line under the historic two-hour barrier."),
        ("Olympic Gymnastics Final Showcases Never-Before-Seen Difficulty Vault", "Gold medalist receives standing ovation after flawless execution of complex acrobatic routine."),
    ],
    "health": [
        ("Clinical Trial Reveals Promising mRNA Vaccine Against Multiple Flu Strains", "Phase 3 clinical data demonstrates broad neutralizing antibody response across diverse viral lineages."),
        ("Study Identifies Daily Walking Habit as Key Driver of Longevity and Brain Health", "Researchers track sixty thousand participants to measure cognitive resilience and cardiovascular markers."),
        ("Innovative Nanoparticle Drug Delivery System Targets Cancer Cells Selectively", "Oncology researchers report dramatic tumor reduction with minimal systemic side effects."),
        ("Nutritional Science Study Reassesses Mediterranean Diet for Metabolic Resilience", "Longitudinal dietary cohort shows marked reduction in vascular inflammation and metabolic dysfunction."),
        ("Mental Wellness App Utilizes Biofeedback to Mitigate Work-Related Stress", "Wearable sensor integration provides real-time respiratory guidance during acute stress episodes."),
    ],
    "politics": [
        ("International Summit Concludes with Historic Clean Energy Agreement", "World leaders sign binding multilateral treaty accelerating carbon neutrality targets by a full decade."),
        ("Bipartisan Privacy Legislation Advances Through Parliamentary Committee", "Comprehensive data protection bill mandates zero-knowledge proofs and strict algorithmic accountability."),
        ("Diplomatic Envoy Brokers Ceasefire in Longstanding Regional Dispute", "Peacekeepers monitor transition corridor as humanitarian corridors reopen for displaced citizens."),
        ("Electoral Reform Commission Proposes Digital Voting Verification Standards", "Independent security audit guarantees end-to-end cryptographic verifiability for future ballots."),
        ("Municipal Council Approves Subsidized High-Speed Transit Corridor", "Urban planning initiative aims to connect underserved suburbs with zero-emission light rail transit."),
    ],
    "science": [
        ("Deep Space Observatory Captures High-Resolution Images of Exoplanet Atmosphere", "Infrared spectrographic data confirms water vapor, carbon monoxide, and methane absorption bands."),
        ("Particle Physicists Discover Novel Subatomic Resonance at High Energies", "Collision data reveals an exotic four-quark state challenging conventional quantum chromodynamics models."),
        ("Marine Biologists Discover Bioluminescent Ecosystem in Abyssal Ocean Trench", "Submersible expedition identifies dozens of previously undocumented deep-sea invertebrate species."),
        ("Fusion Reactor Prototype Sustains Stable Plasma for Over Twenty Minutes", "Superconducting magnetic coils maintain high core temperatures in landmark fusion milestone."),
        ("Paleontologists Unearth Pristine Fossilized Dinosaur Nest with Intact Embryos", "Fossil bed dating back eighty million years offers fresh insights into early avian reproductive behavior."),
    ],
    "entertainment": [
        ("Independent Sci-Fi Epic Sweeps Top Honors at Annual Film Festival", "Visionary director's practical-effects masterpiece wins best picture, cinematography, and score."),
        ("Acoustic World Tour Sells Out Global Arenas in Less Than Ten Minutes", "Multi-platinum singer-songwriter announces extended international stadium tour dates for next summer."),
        ("Next-Generation Open-World RPG Breaks Concurrent Player Records on Launch", "Rich storytelling and dynamic weather physics earn widespread acclaim from critics and gamers worldwide."),
        ("Curator Unveils Retrospective Exhibition Celebrating Surrealist Pioneers", "Rarely exhibited oil paintings and interactive multimedia installations draw record museum crowds."),
        ("Animated Feature Film Stuns Audiences with Hand-Drawn Fluid Animation Style", "Critics praise the emotional coming-of-age narrative and lush watercolor visual aesthetic."),
    ],
}


def download_file(url: str, dest_path: Path) -> Path:
    """Download a file with streaming progress logging."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s", url, dest_path)
    print(f"Downloading: {url} -> {dest_path.name}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    if downloaded % (5 * chunk_size) == 0 or downloaded == total_size:
                        print(f"  Progress: {pct:.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)")

    return dest_path


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a ZIP archive to a destination directory."""
    extract_to.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s to %s", zip_path, extract_to)
    print(f"Extracting {zip_path.name} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete for {zip_path.name}")


def download_mind(
    variant: str = "small",
    raw_dir: str = "data/raw",
    splits: Tuple[str, ...] = ("train", "dev"),
) -> Dict[str, Path]:
    """Download official MIND dataset splits and extract them.

    Args:
        variant: 'small' or 'large'
        raw_dir: Root directory for raw data
        splits: Tuple of splits to download ('train', 'dev')

    Returns:
        Dictionary mapping split name to its directory path.
    """
    if variant not in MIND_URLS:
        raise ValueError(f"Unknown variant '{variant}'. Must be 'small' or 'large'.")

    base_raw = Path(raw_dir)
    result_paths = {}

    for split in splits:
        split_dir = base_raw / split
        news_file = split_dir / "news.tsv"
        behaviors_file = split_dir / "behaviors.tsv"

        if news_file.exists() and behaviors_file.exists():
            print(f"MIND {variant} {split} dataset already exists at {split_dir}")
            result_paths[split] = split_dir
            continue

        url = MIND_URLS[variant][split]
        zip_path = base_raw / f"MIND{variant}_{split}.zip"

        if not zip_path.exists():
            download_file(url, zip_path)

        extract_zip(zip_path, split_dir)
        result_paths[split] = split_dir

    return result_paths


def generate_synthetic_image(
    news_id: str,
    category: str,
    title: str,
    output_path: Path,
    image_size: Tuple[int, int] = (224, 224),
) -> Path:
    """Generate a clean synthetic thumbnail image for multimodal experiments.

    Args:
        news_id: Article ID string
        category: Article category
        title: Article title
        output_path: Target image file path (.jpg)
        image_size: (width, height) tuple, default (224, 224)

    Returns:
        Path to generated JPEG image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_color = CATEGORY_COLORS.get(category, (100, 100, 100))

    # Create image with smooth vertical gradient based on category color
    img = Image.new("RGB", image_size, color=base_color)
    draw = ImageDraw.Draw(img)

    w, h = image_size
    # Draw geometric/abstract backdrop pattern
    for i in range(h):
        blend = i / h
        r = int(base_color[0] * (1 - blend * 0.4) + 15 * blend)
        g = int(base_color[1] * (1 - blend * 0.4) + 20 * blend)
        b = int(base_color[2] * (1 - blend * 0.4) + 30 * blend)
        draw.line([(0, i), (w, i)], fill=(r, g, b))

    # Draw card overlay
    card_margin = 12
    draw.rounded_rectangle(
        [card_margin, card_margin, w - card_margin, h - card_margin],
        radius=10,
        outline=(255, 255, 255, 180),
        width=2,
    )

    # Category badge
    draw.rectangle([card_margin + 8, card_margin + 8, card_margin + 100, card_margin + 26], fill=(0, 0, 0, 160))
    draw.text((card_margin + 12, card_margin + 10), category.upper(), fill=(255, 255, 255))

    # Center article ID & icon box
    box_w, box_h = 90, 40
    box_x = (w - box_w) // 2
    box_y = (h - box_h) // 2
    draw.rounded_rectangle([box_x, box_y, box_x + box_w, box_y + box_h], radius=6, fill=(255, 255, 255))
    draw.text((box_x + 10, box_y + 12), news_id, fill=(20, 20, 20))

    # Footer banner
    footer_y = h - card_margin - 35
    draw.rectangle([card_margin + 4, footer_y, w - card_margin - 4, h - card_margin - 4], fill=(0, 0, 0, 190))
    short_title = title[:24] + ("..." if len(title) > 24 else "")
    draw.text((card_margin + 8, footer_y + 8), short_title, fill=(240, 240, 240))

    img.save(output_path, "JPEG", quality=90)
    return output_path


def generate_demo_dataset(
    output_dir: str = "data/raw",
    num_news: int = 80,
    num_users: int = 40,
    num_train_impressions: int = 150,
    num_dev_impressions: int = 50,
    seed: int = 42,
    generate_images: bool = True,
) -> Dict[str, Path]:
    """Generate a realistic, MIND-compliant demo dataset with multimodal assets.

    Creates valid `train/news.tsv`, `train/behaviors.tsv`, `dev/news.tsv`,
    and `dev/behaviors.tsv` along with synthetic news thumbnail images.

    Args:
        output_dir: Root raw data directory
        num_news: Number of news articles to create
        num_users: Number of simulated users
        num_train_impressions: Number of train impression sessions
        num_dev_impressions: Number of dev impression sessions
        seed: Random seed for deterministic reproducibility
        generate_images: Whether to generate JPEG images for articles

    Returns:
        Dictionary of created file paths.
    """
    random.seed(seed)
    base_dir = Path(output_dir)
    train_dir = base_dir / "train"
    dev_dir = base_dir / "dev"
    images_dir = base_dir / "images"

    train_dir.mkdir(parents=True, exist_ok=True)
    dev_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating realistic MIND demo dataset (Seed: {seed})...")

    # 1. Synthesize News Articles
    news_pool = []
    category_list = list(HEADLINE_TEMPLATES.keys())

    for idx in range(1, num_news + 1):
        news_id = f"N{idx:05d}"
        cat = category_list[(idx - 1) % len(category_list)]
        subcat = random.choice(dict(CATEGORIES)[cat])
        templates = HEADLINE_TEMPLATES[cat]
        base_title, base_abstract = templates[(idx - 1) % len(templates)]
        title = f"{base_title} (Edition #{idx})"
        abstract = f"{base_abstract} Comprehensive updates report ongoing progress across {cat} domains."
        url = f"https://assets.msn.com/news/{cat}/{news_id}"
        title_entities = "[]"
        abstract_entities = "[]"

        news_pool.append({
            "news_id": news_id,
            "category": cat,
            "subcategory": subcat,
            "title": title,
            "abstract": abstract,
            "url": url,
            "title_entities": title_entities,
            "abstract_entities": abstract_entities,
        })

        if generate_images:
            img_path = images_dir / f"{news_id}.jpg"
            if not img_path.exists():
                generate_synthetic_image(news_id, cat, title, img_path)

    print(f"  Synthesized {len(news_pool)} multimodal news articles.")

    # Write news.tsv to both train and dev (in MIND, corpus is shared/split)
    def write_news_tsv(dest_path: Path, articles: List[Dict[str, str]]):
        with open(dest_path, "w", encoding="utf-8") as f:
            for item in articles:
                line = "\t".join([
                    item["news_id"],
                    item["category"],
                    item["subcategory"],
                    item["title"],
                    item["abstract"],
                    item["url"],
                    item["title_entities"],
                    item["abstract_entities"],
                ])
                f.write(line + "\n")

    train_news_tsv = train_dir / "news.tsv"
    dev_news_tsv = dev_dir / "news.tsv"
    write_news_tsv(train_news_tsv, news_pool)
    write_news_tsv(dev_news_tsv, news_pool)

    # 2. Synthesize Users & Interaction Behaviors
    all_news_ids = [n["news_id"] for n in news_pool]
    user_ids = [f"U{u:05d}" for u in range(1, num_users + 1)]

    # Assign persistent user interest profiles (favorite categories)
    user_preferred_cats = {}
    for u in user_ids:
        fav_cats = random.sample(category_list, k=random.randint(1, 3))
        user_preferred_cats[u] = fav_cats

    def generate_behaviors(
        dest_path: Path,
        num_impressions: int,
        start_id: int = 1,
    ) -> int:
        """Write behaviors.tsv with realistic user histories and candidate impressions."""
        with open(dest_path, "w", encoding="utf-8") as f:
            for imp_num in range(1, num_impressions + 1):
                imp_id = start_id + imp_num - 1
                user_id = random.choice(user_ids)
                fav_cats = user_preferred_cats[user_id]

                # User history: 5 to 20 past clicked articles biased towards preferred categories
                history_candidates = [
                    n["news_id"] for n in news_pool if n["category"] in fav_cats
                ]
                if len(history_candidates) < 5:
                    history_candidates = all_news_ids

                history_len = random.randint(5, min(25, len(history_candidates)))
                user_history = random.sample(history_candidates, k=history_len)
                history_str = " ".join(user_history)

                # Impressions: 1 to 2 positive clicks, 4 to 8 non-clicked negatives
                # Prefer candidates from favorite categories for positive clicks
                pos_pool = [nid for nid in history_candidates if nid not in user_history]
                if not pos_pool:
                    pos_pool = all_news_ids
                neg_pool = [nid for nid in all_news_ids if nid not in user_history and nid not in pos_pool]
                if not neg_pool:
                    neg_pool = all_news_ids

                num_pos = random.randint(1, 2)
                num_neg = random.randint(4, 7)

                pos_clicks = random.sample(pos_pool, k=min(num_pos, len(pos_pool)))
                neg_clicks = random.sample(neg_pool, k=min(num_neg, len(neg_pool)))

                impression_items = [f"{nid}-1" for nid in pos_clicks] + [f"{nid}-0" for nid in neg_clicks]
                random.shuffle(impression_items)
                impressions_str = " ".join(impression_items)

                timestamp = f"11/15/2026 {random.randint(8, 22):02d}:{random.randint(0, 59):02d}:00 AM"

                line = f"{imp_id}\t{user_id}\t{timestamp}\t{history_str}\t{impressions_str}\n"
                f.write(line)

        return start_id + num_impressions

    train_behaviors_tsv = train_dir / "behaviors.tsv"
    dev_behaviors_tsv = dev_dir / "behaviors.tsv"

    next_id = generate_behaviors(train_behaviors_tsv, num_train_impressions, start_id=1)
    generate_behaviors(dev_behaviors_tsv, num_dev_impressions, start_id=next_id)

    print(f"  Generated {num_train_impressions} train impressions, {num_dev_impressions} dev impressions.")
    print("MIND demo dataset generation complete!")

    return {
        "train_news": train_news_tsv,
        "train_behaviors": train_behaviors_tsv,
        "dev_news": dev_news_tsv,
        "dev_behaviors": dev_behaviors_tsv,
        "images_dir": images_dir,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_demo_dataset()
