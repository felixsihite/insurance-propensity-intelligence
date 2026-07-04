import pandas as pd
from PIL import Image

from insurance_propensity.config import PROJECT_ROOT


def test_submission_and_scored_outputs_exist():
    submission = pd.read_csv(PROJECT_ROOT / "outputs" / "submission" / "submission.csv")
    scored = pd.read_csv(PROJECT_ROOT / "outputs" / "predictions" / "test_customer_scores.csv")

    assert submission.shape == (127_037, 2)
    assert list(submission.columns) == ["id", "Response"]
    assert scored.shape[0] == 127_037
    assert scored["propensity_score"].between(0, 1).all()
    assert scored["propensity_decile"].between(1, 10).all()


def test_portfolio_visual_assets_are_available():
    screenshot_dir = PROJECT_ROOT / "outputs" / "dashboard_screenshots"
    cover_dir = PROJECT_ROOT / "outputs" / "portfolio_assets"

    screenshots = sorted(screenshot_dir.glob("*.png"))
    covers = sorted(cover_dir.glob("insurance_propensity_cover_*.png"))

    assert len(screenshots) == 14
    assert len(covers) == 2

    for path in screenshots + covers:
        image = Image.open(path)
        assert image.width >= 1200
        assert image.height >= 720
        assert path.stat().st_size > 20_000


def test_professional_documentation_pack_exists():
    docs = [
        "CASE_STUDY.md",
        "MODEL_CARD.md",
        "DATA_DICTIONARY.md",
        "DEPLOYMENT_GUIDE.md",
        "PORTFOLIO_WEBSITE_COPY.md",
    ]
    for name in docs:
        path = PROJECT_ROOT / "docs" / name
        assert path.exists()
        assert path.stat().st_size > 500