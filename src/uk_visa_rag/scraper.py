"""Scrape UK Immigration Rules from the GOV.UK Content API."""

import logging
from pathlib import Path
from typing import List, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.gov.uk"
API_BASE = "https://www.gov.uk/api/content"
IMMIGRATION_RULES_PATH = "/guidance/immigration-rules"


def _clean_html(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(separator="\n").strip()


def _api_path_to_web_url(api_path: str) -> str:
    return BASE_URL + api_path.replace("/api/content", "")


class ImmigrationRulesScraper:
    """Fetch all sections of the UK Immigration Rules from GOV.UK."""

    def __init__(self, timeout: int = 10):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "UK Immigration RAG/1.0"})
        self.timeout = timeout

    def _fetch_json(self, path: str) -> dict:
        url = API_BASE + path if not path.startswith("http") else path
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def fetch_sections(self) -> pd.DataFrame:
        """Fetch all 105 immigration rule sections and return as a DataFrame.

        Columns: title, path, web_url, text
        """
        root = self._fetch_json(IMMIGRATION_RULES_PATH)
        sections = root.get("links", {}).get("sections", [])
        logger.info("Found %d sections", len(sections))

        records: List[Dict] = []
        for sec in sections:
            title = sec.get("title", "")
            api_path = sec.get("api_path", "")
            try:
                data = self._fetch_json(api_path)
                body_html = data.get("details", {}).get("body", "")
                text = _clean_html(body_html) if body_html else ""
                records.append({
                    "title": title,
                    "path": api_path,
                    "web_url": _api_path_to_web_url(api_path),
                    "text": text,
                })
                logger.info("Fetched: %s", title)
            except Exception as e:
                logger.warning("Failed: %s (%s)", title, e)

        return pd.DataFrame(records)

    def fetch_and_save(self, data_dir: str | Path = "data") -> pd.DataFrame:
        """Fetch sections and persist to CSV + JSON."""
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        df = self.fetch_sections()
        df.to_csv(data_dir / "immigration_rules.csv", index=False)
        df.to_json(data_dir / "immigration_rules.json", orient="records", indent=2)
        logger.info("Saved %d sections to %s", len(df), data_dir)
        return df
