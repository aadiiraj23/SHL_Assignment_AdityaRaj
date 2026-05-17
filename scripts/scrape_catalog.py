import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import httpx
import structlog
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential


class SHLCatalogScraper:
    BASE_URL = "https://www.shl.com"
    CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

    def __init__(self) -> None:
        self.logger = structlog.get_logger(__name__)
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        self.timeout = httpx.Timeout(20.0)
        self.assessments: list[dict] = []
        self.visited_urls: set[str] = set()
        self._next_page_url: str | None = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def fetch_page(self, client: httpx.AsyncClient, url: str) -> str:
        self.logger.info("fetching_catalog", url=url)
        response = await client.get(url, headers=self.headers, follow_redirects=True)
        if response.status_code >= 400:
            response.raise_for_status()
        return response.text

    async def parse_catalog_page(self, html: str) -> list[dict]:
        soup = BeautifulSoup(html, "lxml")
        self._next_page_url = self._extract_next_page_url(soup)

        section_header = soup.find(
            string=re.compile(r"\bIndividual Test Solutions\b", re.IGNORECASE)
        )
        if not section_header:
            self.logger.warning("section_not_found", hint="Individual Test Solutions header missing")
            return []

        section_container = self._find_section_container(section_header)
        if not section_container:
            self.logger.warning("section_container_not_found")
            return []

        assessments: list[dict] = []
        for anchor in section_container.find_all("a", href=True):
            href = anchor.get("href", "").strip()
            name = anchor.get_text(strip=True)
            if not href or not name:
                continue
            if "/solutions/products/" not in href:
                continue
            full_url = urljoin(self.BASE_URL, href)
            assessments.append({"name": name, "url": full_url})

        self.logger.info("catalog_page_parsed", found=len(assessments))
        return self._dedupe_assessments(assessments)

    async def parse_assessment_page(self, html: str, url: str, name: str) -> dict:
        soup = BeautifulSoup(html, "lxml")
        description = self._extract_description(soup)
        text_blob = soup.get_text(" ", strip=True)
        lowered = text_blob.lower()

        duration = self._extract_duration_minutes(text_blob)
        languages = self._extract_languages(soup)
        test_type = self._infer_test_type(name, description)

        return {
            "name": name,
            "url": url,
            "test_type": test_type,
            "description": description,
            "remote_testing": "remote" in lowered,
            "adaptive": "adaptive" in lowered,
            "duration_minutes": duration,
            "languages": languages,
            "scraped_at": self._now_iso(),
        }

    async def scrape_all(self) -> list[dict]:
        start_time = time.monotonic()
        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers=self.headers,
            follow_redirects=True
        ) as client:
            page_url: str | None = self.CATALOG_URL
            all_links: list[dict] = []

            while page_url:
                html = await self.fetch_page(client, page_url)
                page_links = await self.parse_catalog_page(html)
                all_links.extend(page_links)
                page_url = self._next_page_url

            self.logger.info("catalog_links_collected", total=len(all_links))

            for index, item in enumerate(all_links, start=1):
                url = item["url"]
                if url in self.visited_urls:
                    continue
                self.visited_urls.add(url)

                try:
                    html = await self.fetch_page(client, url)
                    assessment = await self.parse_assessment_page(
                        html, url=url, name=item["name"]
                    )
                except Exception as exc:
                    self.logger.warning("assessment_failed", url=url, error=str(exc))
                    assessment = {
                        "name": item["name"],
                        "url": url,
                        "test_type": self._infer_test_type(item["name"], ""),
                        "description": "",
                        "remote_testing": False,
                        "adaptive": False,
                        "duration_minutes": None,
                        "languages": [],
                        "scraped_at": self._now_iso(),
                    }

                self.assessments.append(assessment)
                self.logger.info(
                    "scraped_progress",
                    current=index,
                    total=len(all_links),
                    name=item["name"],
                )
                await asyncio.sleep(0.5)

        self.logger.info(
            "scrape_complete",
            total=len(self.assessments),
            elapsed_seconds=round(time.monotonic() - start_time, 2),
        )
        return self.assessments

    def save(self, output_path: Path) -> None:
        payload = {
            "scraped_at": self._now_iso(),
            "total": len(self.assessments),
            "assessments": self.assessments,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.logger.info(
            "saved_catalog",
            path=str(output_path),
            total=len(self.assessments)
        )

    def _find_section_container(self, header_node) -> BeautifulSoup | None:
        current = header_node
        for _ in range(5):
            if not current:
                return None
            parent = current.parent
            if parent and parent.name in {"section", "div", "article"}:
                return parent
            current = parent
        return None

    def _extract_next_page_url(self, soup: BeautifulSoup) -> str | None:
        link = soup.find("a", attrs={"rel": "next"})
        if link and link.get("href"):
            return urljoin(self.BASE_URL, link["href"])
        pager = soup.find("a", string=re.compile(r"next", re.IGNORECASE))
        if pager and pager.get("href"):
            return urljoin(self.BASE_URL, pager["href"])
        return None

    def _extract_description(self, soup: BeautifulSoup) -> str:
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            return meta["content"].strip()
        main = soup.find("main") or soup.find("div", class_=re.compile("content", re.I))
        if main:
            return main.get_text(" ", strip=True)
        return ""

    def _extract_duration_minutes(self, text: str) -> int | None:
        match = re.search(r"(\d{1,3})\s*(minutes|minute|min)\b", text, re.I)
        if match:
            return int(match.group(1))
        return None

    def _extract_languages(self, soup: BeautifulSoup) -> list[str]:
        language_block = soup.find(string=re.compile(r"Languages?", re.I))
        if language_block:
            parent = language_block.parent
            if parent:
                items = parent.find_all("li")
                if items:
                    return [
                        item.get_text(strip=True)
                        for item in items
                        if item.get_text(strip=True)
                    ]
                text = parent.get_text(" ", strip=True)
                match = re.search(r"Languages?:\s*(.+)", text, re.I)
                if match:
                    raw = match.group(1)
                    return [lang.strip() for lang in raw.split(",") if lang.strip()]
        return []

    def _infer_test_type(self, name: str, description: str) -> str:
        text = f"{name} {description}".lower()
        if any(word in text for word in ["personality", "opq", "mq"]):
            return "P"
        if any(word in text for word in ["ability", "verbal", "numerical", "inductive", "deductive"]):
            return "A"
        if any(word in text for word in ["java", "python", "coding", "knowledge", "verify"]):
            return "K"
        if "situational" in text:
            return "S"
        if any(word in text for word in ["biodata", "motivation"]):
            return "B"
        if "competency" in text:
            return "C"
        return "A"

    def _dedupe_assessments(self, assessments: list[dict]) -> list[dict]:
        seen: set[str] = set()
        unique: list[dict] = []
        for item in assessments:
            url = item.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            unique.append(item)
        return unique

    def _now_iso(self) -> str:
        return (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )


if __name__ == "__main__":
    async def main() -> None:
        scraper = SHLCatalogScraper()
        assessments = await scraper.scrape_all()
        output = Path("data/catalog.json")
        output.parent.mkdir(exist_ok=True)
        scraper.save(output)
        print(f"Done. Saved {len(assessments)} assessments to {output}")

    asyncio.run(main())