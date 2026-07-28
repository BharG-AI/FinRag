"""Download 10-K filings from SEC EDGAR and save cleaned text.

Usage:
    python -m finrag.ingest AAPL MSFT
    python -m finrag.ingest AAPL --form 10-Q --limit 2

EDGAR is free but has rules: identify yourself in the User-Agent and stay
under 10 requests/second. Set FINRAG_SEC_USER_AGENT to your name/email.
"""

import argparse
import re
import time

import requests
from bs4 import BeautifulSoup

from .config import get_settings

TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"


def _get(url: str, user_agent: str) -> requests.Response:
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    time.sleep(0.15)  # stay politely under EDGAR's rate limit
    return resp


def resolve_cik(ticker: str, user_agent: str) -> int:
    data = _get(TICKER_URL, user_agent).json()
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return int(row["cik_str"])
    raise ValueError(f"Ticker not found on EDGAR: {ticker}")

def latest_filings(cik: int, form: str, limit: int, user_agent: str) -> list[dict]:
    subs = _get(SUBMISSIONS_URL.format(cik=cik), user_agent).json()
    recent = subs["filings"]["recent"]
    out = []
    for form_type, accession, doc, date in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"], recent["filingDate"]
    ):
        if form_type == form:
            out.append(
                {"accession": accession.replace("-", ""), "doc": doc, "date": date}
            )
        if len(out) >= limit:
            break
    return out


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SEC filings as text")
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--form", default="10-K")
    parser.add_argument("--limit", type=int, default=1, help="filings per ticker")
    args = parser.parse_args()

    settings = get_settings()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    for ticker in args.tickers:
        cik = resolve_cik(ticker, settings.sec_user_agent)
        for filing in latest_filings(cik, args.form, args.limit, settings.sec_user_agent):
            url = ARCHIVE_URL.format(cik=cik, accession=filing["accession"], doc=filing["doc"])
            print(f"{ticker}: fetching {filing['date']} {args.form}")
            text = html_to_text(_get(url, settings.sec_user_agent).text)
            year = filing["date"][:4]
            out = settings.raw_dir / f"{ticker.upper()}_{args.form}_{year}.txt"
            out.write_text(text)
            print(f"  -> {out} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
