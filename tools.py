import os
import json
import re
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

DEFAULT_SAVE_PATH = os.getenv("SAVE_PATH", "research_output.txt")
MAX_WIKI_CHARS = int(os.getenv("WIKI_MAX_CHARS", "100"))
MAX_FETCH_CHARS = int(os.getenv("FETCH_MAX_CHARS", "2000"))


def save_to_txt(data: str, filename: str = DEFAULT_SAVE_PATH):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Try to parse structured JSON if provided
    parsed = None
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except Exception:
            parsed = None
    elif isinstance(data, dict):
        parsed = data

    md_report = []
    if isinstance(parsed, dict) and {
        "topic",
        "summary",
        "sources",
        "tools_used",
    }.issubset(parsed.keys()):
        # Structured ResearchResponse -> render nicely in Markdown
        topic = parsed.get("topic")
        summary = parsed.get("summary")
        sources = parsed.get("sources") or []
        tools_used = parsed.get("tools_used") or []

        md_report.append(f"# Research Report: {topic}")
        md_report.append("")
        md_report.append(f"- Timestamp: {timestamp}")
        if tools_used:
            md_report.append(f"- Tools used: {', '.join(tools_used)}")
        md_report.append("")
        md_report.append("## Summary")
        md_report.append(summary or "")
        md_report.append("")
        md_report.append("## Sources")
        if isinstance(sources, list) and sources:
            for src in sources:
                md_report.append(f"- {src}")
        else:
            md_report.append("- (none)")
        md_report.append("")
        md_report.append("---")
        md_report.append("")
    else:
        # Fallback: wrap raw text in a Markdown section
        md_report.append("# Research Report")
        md_report.append("")
        md_report.append(f"- Timestamp: {timestamp}")
        md_report.append("")
        md_report.append("## Content")
        md_report.append(str(data))
        md_report.append("")
        md_report.append("---")
        md_report.append("")

    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n".join(md_report))

    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=MAX_WIKI_CHARS)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)


def fetch_page_content(url: str) -> str:
    headers = {"User-Agent": "AI_Agents/1.0 (+https://github.com/)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_FETCH_CHARS]


fetch_tool = Tool(
    name="fetch_page",
    func=fetch_page_content,
    description="Fetches and sanitizes web page text for analysis. Input should be a URL.",
)