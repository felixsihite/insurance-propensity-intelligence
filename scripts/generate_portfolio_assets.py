"""Generate website-ready portfolio cover images from dashboard screenshots."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_DIR = PROJECT_ROOT / "outputs" / "dashboard_screenshots"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "portfolio_assets"
ASSET_DIR = PROJECT_ROOT / "streamlit_app" / "assets" / "portfolio_assets"
Font = ImageFont.FreeTypeFont | ImageFont.ImageFont


def font(size: int, bold: bool = False) -> Font:
    filename = "segoeuib.ttf" if bold else "segoeui.ttf"
    candidates = [
        Path("C:/Windows/Fonts") / filename,
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def rounded_paste(canvas: Image.Image, image: Image.Image, box: tuple[int, int], radius: int = 28) -> None:
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, image.size[0], image.size[1]), radius=radius, fill=255)
    canvas.paste(image, box, mask)


def draw_metric(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, value: str, colors: dict[str, str]) -> None:
    x, y = xy
    draw.rounded_rectangle((x, y, x + 260, y + 112), radius=16, fill=colors["card"], outline=colors["border"], width=2)
    draw.text((x + 22, y + 20), label, fill=colors["muted"], font=font(22))
    draw.text((x + 22, y + 54), value, fill=colors["text"], font=font(36, bold=True))


def make_cover(theme_name: str) -> Path:
    dark = theme_name == "dark"
    colors = {
        "background": "#0B1F33" if dark else "#D6E4F0",
        "panel": "#102A44" if dark else "#F7FAFC",
        "card": "#132F4C" if dark else "#F8FBFE",
        "border": "#315B7F" if dark else "#B8C8D8",
        "text": "#F8FAFC" if dark else "#172033",
        "muted": "#CBD5E1" if dark else "#52616B",
        "accent": "#38BDF8" if dark else "#1F7A8C",
        "accent_2": "#2DD4BF" if dark else "#2563EB",
    }

    source = SCREENSHOT_DIR / f"executive_summary_{theme_name}.png"
    if not source.exists():
        raise FileNotFoundError(f"Dashboard screenshot is missing: {source}")

    canvas = Image.new("RGB", (1600, 900), colors["background"])
    draw = ImageDraw.Draw(canvas)

    draw.rounded_rectangle((52, 52, 1548, 848), radius=34, fill=colors["panel"], outline=colors["border"], width=2)
    draw.rectangle((52, 52, 1548, 134), fill="#07111F" if dark else "#0B1F33")
    draw.text((90, 82), "Insurance Customer Propensity Prediction", fill="#F8FAFC", font=font(30, bold=True))
    draw.text((1290, 86), "Data Science Portfolio", fill="#38BDF8", font=font(24, bold=True))

    draw.text((94, 192), "Propensity Intelligence", fill=colors["text"], font=font(56, bold=True))
    subtitle = "Customer ranking, model lift, segmentation, and campaign targeting."
    draw.text((98, 268), subtitle, fill=colors["muted"], font=font(25))

    draw_metric(draw, (98, 362), "ROC-AUC", "0.858", colors)
    draw_metric(draw, (390, 362), "PR-AUC", "0.368", colors)
    draw_metric(draw, (98, 506), "Top Decile Lift", "3.22x", colors)
    draw_metric(draw, (390, 506), "Top 30% Capture", "79.4%", colors)

    draw.rounded_rectangle((98, 678, 690, 756), radius=22, fill=colors["accent"])
    draw.text((132, 700), "Scoring dashboard + reproducible ML pipeline", fill="#06111F" if dark else "#FFFFFF", font=font(23, bold=True))

    screenshot = Image.open(source).convert("RGB")
    screenshot = screenshot.resize((790, 444), Image.Resampling.LANCZOS)
    rounded_paste(canvas, screenshot, (730, 230), radius=26)
    draw.rounded_rectangle((730, 230, 1520, 674), radius=26, outline=colors["accent_2"], width=3)

    output = OUTPUT_DIR / f"insurance_propensity_cover_{theme_name}.png"
    canvas.save(output, quality=95)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for theme_name in ["light", "dark"]:
        output = make_cover(theme_name)
        shutil.copy2(output, ASSET_DIR / output.name)
        generated.append(str(output.relative_to(PROJECT_ROOT)))
        print(output)
    manifest = {
        "portfolio_covers": generated,
        "dashboard_screenshots": [
            str(path.relative_to(PROJECT_ROOT))
            for path in sorted((PROJECT_ROOT / "outputs" / "dashboard_screenshots").glob("*.png"))
        ],
        "analysis_charts": [
            str(path.relative_to(PROJECT_ROOT))
            for path in sorted((PROJECT_ROOT / "outputs" / "charts").glob("*.png"))
        ],
    }
    (OUTPUT_DIR / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
