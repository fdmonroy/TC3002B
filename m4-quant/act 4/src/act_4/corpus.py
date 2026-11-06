"""Corpus fetching and cleaning for the text similarity activity.

Data layer for the whole activity: the cleaned film dialogue and the
Project Gutenberg edition of "Alice's Adventures in Wonderland" are stored
as flowing text files under ``data/raw`` and loaded through
:func:`load_corpus`.
"""

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")

# Mapping of corpus name to the file that holds the cleaned final text.
CORPUS_FILES: dict[str, str] = {
    "shrek": "shrek_original.txt",
    "shrek_2": "shrek_2.txt",
    "toy_story": "toy_story.txt",
    "alicia": "alicia.txt",
}

# Canonical transcript sources per film.
# NOTE: the brief's original Shrek URL (Transcript_(Shrek)) does not exist on
# the wiki ("Shrek (film) (transcript)" is the canonical page there), and
# transcripts.fandom.com retired its Toy Story transcript, so that film uses
# the canonical IMSDb screenplay instead. Fandom wiki pages are fetched
# through the MediaWiki action API of the same wiki because direct GETs of
# /wiki/ pages are blocked by bot protection.
SCRIPT_SOURCES: dict[str, str] = {
    "shrek": "https://shrek.fandom.com/wiki/Shrek_(film)_(transcript)",
    "shrek_2": "https://shrek.fandom.com/wiki/Shrek_2/Transcript",
    "toy_story": "https://imsdb.com/scripts/Toy-Story.html",
}

ALICE_URL: str = "https://www.gutenberg.org/files/11/11-0.txt"

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_TOC_CLASS = re.compile(r"toc(level|section)")
_INTRO_LINE = "This is the transcript for"
_PUNCT_SPACE = re.compile(r"\s+([,.;:!?…])")
_ILLUSTRATION = re.compile(r"\[Illustration\]\s*")
_SPEAKER = re.compile(
    r"^\[?[A-Za-z0-9][A-Za-z0-9 #'’,/&…()?.!\-]{0,28}[A-Za-z0-9.\]]?:[\s.]*"
)
_CUE_LINE = re.compile(r"^\[?[A-Z][A-Z0-9 .,'&()/#!?–—\"-]*[)A-Z0-9.!?'\\]?\s*$")
_SLUG_LINE = re.compile(
    r"^\s*(INT|EXT|FADE|CUT TO|DISSOLVE|THE END|CONTINUED|SFX|BEGIN|END|"
    r"ORIGINAL|FINAL)"
)
_START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
_END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"


def _fetch_url(url: str) -> str:
    """Download the body of a URL and decode it as text."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    charset = response.headers.get_content_charset() or _sniff_charset(raw)
    return raw.decode(charset, errors="replace")


def _sniff_charset(raw: bytes) -> str:
    """Guess the encoding of an HTML body from its meta tag, default utf-8."""
    head = raw[:2048].decode("ascii", errors="ignore")
    match = re.search(r'charset=["\']?([a-zA-Z0-9_-]+)', head)
    return match.group(1) if match else "utf-8"


def _fandom_api_url(wiki_url: str) -> str:
    """Turn a Fandom wiki page URL into its MediaWiki action API parse URL."""
    parts = urllib.parse.urlsplit(wiki_url)
    page = urllib.parse.unquote(parts.path.removeprefix("/wiki/"))
    query = urllib.parse.urlencode(
        {"action": "parse", "page": page, "prop": "text",
         "redirects": "1", "format": "json"}
    )
    return f"{parts.scheme}://{parts.netloc}/api.php?{query}"


def _fetch_wiki_page_html(wiki_url: str) -> str:
    """Fetch the rendered HTML of a wiki page through its action API."""
    payload = json.loads(_fetch_url(_fandom_api_url(wiki_url)))
    if "error" in payload:
        raise ValueError(f"Wiki API error for {wiki_url}: {payload['error'].get('info')}")
    return payload["parse"]["text"]["*"]


def _wiki_line_text(paragraph) -> str:
    """Extract one transcript entry from a <p> or <li> element as plain text."""
    text = paragraph.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    text = _PUNCT_SPACE.sub(r"\1", text)
    return text.strip()


def _remove_wiki_chrome(soup: BeautifulSoup) -> None:
    """Strip navigation, tables, media and icons from a parsed wiki page."""
    for tag in soup.find_all(
        ["script", "style", "table", "figure", "figcaption", "svg", "img", "input"]
    ):
        tag.decompose()
    for tag in soup.find_all(id=re.compile(r"^toc", re.IGNORECASE)):
        tag.decompose()
    for item in soup.find_all("li", class_=_TOC_CLASS):
        parent = item.find_parent("ul")
        if parent is not None:
            parent.decompose()


def _clean_wiki_transcript(soup: BeautifulSoup) -> str:
    """Join only the spoken lines of a wiki transcript page.

    Each dialogue entry starts with a speaker tag (``Name:``); the tag is
    dropped and the spoken text is kept, so the result is continuous
    dialogue without scene descriptions, headings or wiki chrome.
    """
    _remove_wiki_chrome(soup)
    lines: list[str] = []
    for entry in soup.find_all(["p", "li"]):
        text = _wiki_line_text(entry)
        if not text or text.startswith(_INTRO_LINE):
            continue
        speaker = _SPEAKER.match(text)
        if speaker is None:
            continue
        lines.append(text[speaker.end():].strip())
    return " ".join(lines)


def _is_cue_line(line: str) -> bool:
    """Tell an ALL CAPS character cue from a slug or transition line."""
    if not line or not _CUE_LINE.match(line) or _SLUG_LINE.match(line):
        return False
    return line.upper() == line and any(c.isalpha() for c in line)


def _clean_screenplay(pre) -> str:
    """Reduce a hosted screenplay to its flowing dialogue text.

    Walks the DOM of the hosting ``<pre>`` block: bold ALL CAPS lines are
    character cues, slug and transition lines are skipped, and the text
    block that follows a cue up to the next blank line is that character's
    dialogue.
    """
    children = list(pre.children)
    start = next(
        (i for i, c in enumerate(children)
         if getattr(c, "name", None) == "b" and "FADE IN" in c.get_text()),
        0,
    )
    end = next(
        (i for i, c in enumerate(children)
         if getattr(c, "name", None) == "b" and "THE END" in c.get_text()),
        len(children),
    )
    blocks: list[str] = []
    expect_dialogue = False
    for child in children[start:end]:
        if getattr(child, "name", None) == "b":
            cue = re.sub(r"\s+", " ", child.get_text(" ", strip=True)).strip()
            expect_dialogue = _is_cue_line(cue)
            continue
        if not expect_dialogue:
            continue
        spoken = re.split(r"\r?\n\s*\r?\n", str(child))[0]
        spoken = re.sub(r"\s+", " ", spoken.replace("\r", " ")).strip()
        if spoken:
            blocks.append(spoken)
        expect_dialogue = False
    return " ".join(blocks)


def clean_transcript(raw_html: str) -> str:
    """Clean a raw transcript page into flowing dialogue text.

    Handles two layouts, both detected from the HTML itself: wiki
    transcript pages (dialogue as tagged rows or list lines with character
    names, from which nav, tables, image captions, icons and speaker tags
    are stripped) and hosted screenplays wrapped in a ``<pre>`` block.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    pre = soup.find("pre")
    if pre is not None and "FADE IN" in pre.get_text():
        return _clean_screenplay(pre)
    return _clean_wiki_transcript(soup)


def strip_gutenberg_boilerplate(raw_text: str) -> str:
    """Remove the Project Gutenberg license header and footer of an ebook."""
    start = raw_text.find(_START_MARKER)
    if start >= 0:
        remainder = raw_text[start + len(_START_MARKER):]
        newline = remainder.find("\n")
        raw_text = remainder[newline + 1:] if newline >= 0 else remainder
    end = raw_text.find(_END_MARKER)
    if end >= 0:
        raw_text = raw_text[:end]
    raw_text = _ILLUSTRATION.sub(" ", raw_text)
    return re.sub(r"\s+", " ", raw_text).strip()


def fetch_corpus(name: str, data_dir: Path = RAW_DIR) -> Path:
    """Fetch, clean and persist a corpus, returning the cleaned file path.

    Idempotent: skips the network and returns the stored file when the
    cleaned file for ``name`` already exists and is non-empty.
    """
    if name not in CORPUS_FILES:
        raise ValueError(f"Unknown corpus name: {name!r}. Known: {sorted(CORPUS_FILES)}")
    cleaned_path = data_dir / CORPUS_FILES[name]
    if cleaned_path.exists() and cleaned_path.stat().st_size > 0:
        return cleaned_path

    if name == "alicia":
        text = strip_gutenberg_boilerplate(_fetch_url(ALICE_URL))
    elif name in SCRIPT_SOURCES:
        source = SCRIPT_SOURCES[name]
        raw = _fetch_wiki_page_html(source) if "/wiki/" in source else _fetch_url(source)
        text = clean_transcript(raw)
    else:
        raise ValueError(f"No source configured for corpus: {name!r}")

    data_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path.write_text(text, encoding="utf-8")
    return cleaned_path


def load_corpus(name: str, data_dir: Path = RAW_DIR) -> str:
    """Return the cleaned text of a corpus, building it on first use."""
    return fetch_corpus(name, data_dir).read_text(encoding="utf-8")
