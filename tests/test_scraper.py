import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import json
from datetime import datetime, timezone

from scripts.scrape_catalog import SHLCatalogScraper


@pytest.fixture
def scraper():
    return SHLCatalogScraper()


# 1. test_fetch_page_success
@pytest.mark.asyncio
async def test_fetch_page_success(scraper):
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html>Sample Content</html>"
    mock_client.get.return_value = mock_response

    result = await scraper.fetch_page(mock_client, "https://www.shl.com/test")
    assert result == "<html>Sample Content</html>"
    # Only assert the URL was called — don't assert exact kwargs
    # because fetch_page uses follow_redirects=True internally
    call_args = mock_client.get.call_args
    assert call_args[0][0] == "https://www.shl.com/test"


# 2. test_fetch_page_retries_on_500
@pytest.mark.asyncio
async def test_fetch_page_retries_on_500(scraper):
    mock_client = AsyncMock()
    mock_response_500 = MagicMock()
    mock_response_500.status_code = 500
    mock_response_500.raise_for_status.side_effect = Exception("Internal Server Error")
    mock_client.get.return_value = mock_response_500

    with patch("tenacity.nap.time.sleep", return_value=None):
        with pytest.raises(Exception):
            await scraper.fetch_page(mock_client, "https://www.shl.com/fail")

    assert mock_client.get.call_count == 3


# 3. test_parse_catalog_page_extracts_individual_tests
@pytest.mark.asyncio
async def test_parse_catalog_page_extracts_individual_tests(scraper):
    mock_html = """
    <div>
        <h2>Individual Test Solutions</h2>
        <a href="/solutions/products/verify-numerical/">Verify - Numerical Reasoning</a>
        <a href="/solutions/products/opq32r/">OPQ32r</a>
    </div>
    """
    results = await scraper.parse_catalog_page(mock_html)
    assert len(results) >= 1
    urls = [r["url"] for r in results]
    assert any("verify-numerical" in u or "opq32r" in u for u in urls)


# 4. test_parse_catalog_page_skips_prepackaged
@pytest.mark.asyncio
async def test_parse_catalog_page_skips_prepackaged(scraper):
    mock_html = """
    <div>
        <h2>Individual Test Solutions</h2>
        <a href="/solutions/products/opq32r/">OPQ32r</a>
    </div>
    <div>
        <h2>Pre-packaged Job Solutions</h2>
        <a href="/solutions/products/graduate-pack/">Graduate Pack</a>
    </div>
    """
    results = await scraper.parse_catalog_page(mock_html)
    urls = [r["url"] for r in results]
    assert "https://www.shl.com/solutions/products/graduate-pack/" not in urls


# 5. test_parse_assessment_page_extracts_description
@pytest.mark.asyncio
async def test_parse_assessment_page_extracts_description(scraper):
    mock_html = """
    <html>
        <head>
            <meta name="description" content="This is a secure personality evaluation tool.">
        </head>
        <body><main>Main content backup description text.</main></body>
    </html>
    """
    res = await scraper.parse_assessment_page(
        mock_html, "https://www.shl.com/opq", "OPQ32r"
    )
    assert "personality evaluation tool" in res["description"]


# 6. test_parse_assessment_page_detects_remote_testing
@pytest.mark.asyncio
async def test_parse_assessment_page_detects_remote_testing(scraper):
    mock_html = (
        "<html><body>This assessment supports remote proctoring "
        "and online testing.</body></html>"
    )
    res = await scraper.parse_assessment_page(
        mock_html, "https://www.shl.com/test", "Test"
    )
    assert res["remote_testing"] is True


# 7. test_parse_assessment_page_detects_adaptive
@pytest.mark.asyncio
async def test_parse_assessment_page_detects_adaptive(scraper):
    mock_html = (
        "<html><body>Features our computer adaptive testing "
        "framework.</body></html>"
    )
    res = await scraper.parse_assessment_page(
        mock_html, "https://www.shl.com/test", "Test"
    )
    assert res["adaptive"] is True


# 8. test_parse_assessment_page_extracts_duration
@pytest.mark.asyncio
async def test_parse_assessment_page_extracts_duration(scraper):
    mock_html = (
        "<html><body>The test has a time configuration of "
        "45 minutes for completion.</body></html>"
    )
    res = await scraper.parse_assessment_page(
        mock_html, "https://www.shl.com/test", "Test"
    )
    assert res["duration_minutes"] == 45


# 9. test_infer_test_type_personality
@pytest.mark.asyncio
async def test_infer_test_type_personality(scraper):
    mock_html = (
        "<html><body>Description of an OPQ occupational "
        "questionnaire.</body></html>"
    )
    res = await scraper.parse_assessment_page(
        mock_html, "https://www.shl.com/opq", "OPQ Personality Test"
    )
    assert res["test_type"] == "P"


# 10. test_infer_test_type_ability
@pytest.mark.asyncio
async def test_infer_test_type_ability(scraper):
    mock_html = (
        "<html><body>Measures numerical and inductive reasoning "
        "ability.</body></html>"
    )
    res = await scraper.parse_assessment_page(
        mock_html, "https://www.shl.com/num", "Numerical Reasoning"
    )
    assert res["test_type"] == "A"


# 11. test_infer_test_type_knowledge
@pytest.mark.asyncio
async def test_infer_test_type_knowledge(scraper):
    mock_html = (
        "<html><body>A technical evaluation test covering "
        "modern java and python rules.</body></html>"
    )
    res = await scraper.parse_assessment_page(
        mock_html, "https://www.shl.com/code", "Coding Knowledge"
    )
    assert res["test_type"] == "K"


# 12. test_infer_test_type_default
@pytest.mark.asyncio
async def test_infer_test_type_default(scraper):
    mock_html = (
        "<html><body>Random context data with zero standard "
        "categorizations.</body></html>"
    )
    res = await scraper.parse_assessment_page(
        mock_html, "https://www.shl.com/rand", "Mystery Product"
    )
    assert res["test_type"] == "A"


# 13. test_save_creates_valid_json
def test_save_creates_valid_json(scraper, tmp_path):
    scraper.assessments = [
        {
            "name": "Verify - Numerical Reasoning",
            "url": "https://www.shl.com/verify-num",
            "test_type": "A",
            "description": "Numerical ability assessment",
            "remote_testing": True,
            "adaptive": True,
            "duration_minutes": 30,
            "languages": ["English"],
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    file_path = tmp_path / "catalog.json"
    scraper.save(file_path)

    assert file_path.exists()
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert "assessments" in data
    assert len(data["assessments"]) == 1


# 14. test_save_metadata_fields
def test_save_metadata_fields(scraper, tmp_path):
    scraper.assessments = []
    file_path = tmp_path / "catalog_meta.json"
    scraper.save(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "scraped_at" in data
    assert "total" in data
    assert "assessments" in data
    assert data["total"] == 0