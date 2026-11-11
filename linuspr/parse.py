"""Pure parsers for the lkml.iu.edu hypermail archive (LKML mirror).

No network in this module — everything takes HTML strings, so all corpus logic
is hermetically testable against saved fixtures in tests/fixtures/.

Archive shape (verified 2026-11-10 against /hypermail/linux/kernel/2406.0/):

  index.html   thread-ordered list, one entry per message:
                 <li><strong><a name="00001" href="00001.html">SUBJECT</a> </strong>
                     <em>Author </em><small>(Sat Jun 01 2024 - 00:36:02 EST)</small>
               `name` is the stable message id within the month. <li>/<ul> are
               left unclosed in places, so thread depth is tracked via <ul> only.

  NNNNN.html   <!--X-Message-Id: ...-->   real Message-ID (headers mangled:
               <!--X-Subject: ...-->      emails are xxxx'd out by the mirror,
               <!--X-Date: ...-->         but these comments survive)
               <strong>From: </strong>AUTHOR
               <!--X-Body-of-Message--> BODY <!--X-Body-of-Message-End-->
                 body quoted text renders as `&gt;<i> ... </i>` lines
               "Previous message: </strong> <a href="NNNNN.html"> Author: "SUBJ""
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

_ENTRY_RE = re.compile(
    r'<a name="(?P<mid>\d{5}(?:\.\d+)?)" href="(?P<file>\d{5}(?:\.\d+)?)\.html">'
    r"(?P<subject>.*?)</a>"
    r"\s*</strong>\s*<em>(?P<author>.*?)</em>\s*<small>\((?P<date>.*?)\)</small>",
    re.S,
)

_INDEX_TOKEN_RE = re.compile(
    r"<(/?)ul\b[^>]*>|<a name=\"(?P<mid>\d{5}(?:\.\d+)?)\"", re.I
)

_FROM_RE = re.compile(r"From:\s*</strong>(?P<author>.*?)(?:<|\n)", re.S)
_DATE_RE = re.compile(r"Date:\s*</strong>(?P<v>.*?)(?:<|\n)", re.S)
_TITLE_RE = re.compile(r"<title>(?P<t>.*?)</title>", re.S | re.I)
_LORE_RE = re.compile(r"https?://lore\.kernel\.org/(\w+)/([^/\"\s<]+)")

_BODY_RE = re.compile(
    r"<!--X-Body-of-Message-->(?P<body>.*?)<!--X-Body-of-Message-End-->", re.S
)
_MSGID_RE = re.compile(r"<!--X-Message-Id:\s*(?P<v>.*?)\s*-->", re.S)
_XSUBJ_RE = re.compile(r"<!--X-Subject:\s*(?P<v>.*?)\s*-->", re.S)
_XDATE_RE = re.compile(r"<!--X-Date:\s*(?P<v>.*?)\s*-->", re.S)
_PREV_RE = re.compile(
    r'Previous message: </strong>\s*<a href="(?P<file>\d{5}(?:\.\d+)?)\.html">'
    r'\s*(?P<author>[^:"]{0,160}?):\s*(?:&quot;|")(?P<subject>.*?)(?:&quot;|"|</a>)',
    re.S,
)

_TAG_RE = re.compile(r"<[^>]*>")
_SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)


@dataclass
class IndexEntry:
    """One row of a month index page."""

    mid: str
    subject: str
    author: str
    date: str
    depth: int = 0
    parent: str | None = None

    @property
    def filename(self) -> str:
        return f"{self.mid}.html"


PATCH_BODY_RE = re.compile(r"\n-{3,}\s*\n")
DIFF_START_RE = re.compile(r"^(diff --git |--- a/)", re.M)


@dataclass
class Message:
    mid: str
    month: str
    subject: str
    author: str
    date: str
    body: str
    msgid: str = ""
    prev_mid: str | None = None
    prev_author: str = ""
    quoted: str = ""
    own: str = ""
    lore_links: list[tuple[str, str]] = field(default_factory=list)

    @property
    def is_reply(self) -> bool:
        return self.subject.lower().startswith("re:")

    @property
    def is_patch_thread(self) -> bool:
        return "[patch" in self.subject.lower()


def _clean(text: str) -> str:
    text = _SCRIPT_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    return html.unescape(text).strip()


def parse_index(index_html: str) -> list[IndexEntry]:
    """Parse a month index page into entries in hypermail's thread order."""
    entries: list[IndexEntry] = []
    depth = 0
    spine: list[str] = []
    seen: set[str] = set()
    for m in _INDEX_TOKEN_RE.finditer(index_html):
        if m.group("mid"):
            em = _ENTRY_RE.match(index_html, m.start())
            if not em or em["mid"] in seen:
                continue
            seen.add(em["mid"])
            entries.append(
                IndexEntry(
                    mid=em["mid"],
                    subject=_clean(em["subject"]),
                    author=_clean(em["author"]),
                    date=_clean(em["date"]),
                    depth=max(depth - 1, 0),
                    parent=spine[-1] if spine else None,
                )
            )
            spine.append(em["mid"])
        elif m.group(1):  # </ul>
            depth = max(depth - 1, 0)
            if spine:
                spine.pop()
        else:  # <ul>
            depth += 1
    return entries


def parse_message(msg_html: str, *, mid: str, month: str) -> Message:
    """Parse one hypermail message page into subject/author/date/body parts."""
    body_m = _BODY_RE.search(msg_html)
    body = _normalise_body(_clean(body_m.group("body")) if body_m else "")

    subj_m = _XSUBJ_RE.search(msg_html)
    if subj_m:
        subj = _clean(subj_m.group("v"))
    else:
        t = _TITLE_RE.search(msg_html)
        subj = re.sub(
            r"^Linux-Kernel Archive:\s*", "", _clean(t.group("t")) if t else ""
        )

    frm = _FROM_RE.search(msg_html)
    dt = _XDATE_RE.search(msg_html) or _DATE_RE.search(msg_html)
    msgid = _MSGID_RE.search(msg_html)
    prev = _PREV_RE.search(msg_html)

    own_lines: list[str] = []
    quoted_lines: list[str] = []
    for line in body.splitlines():
        (quoted_lines if line.lstrip().startswith(">") else own_lines).append(line)

    return Message(
        mid=mid,
        month=month,
        subject=subj,
        author=_clean(frm.group("author")) if frm else "",
        date=_clean(dt.group("v")) if dt else "",
        body=body,
        msgid=html.unescape(msgid.group("v")) if msgid else "",
        prev_mid=prev.group("file") if prev else None,
        prev_author=_clean(prev.group("author")) if prev else "",
        quoted="\n".join(quoted_lines),
        own="\n".join(own_lines).strip(),
        lore_links=[(a, b) for a, b in _LORE_RE.findall(msg_html)],
    )


def patch_text(msg: Message) -> str:
    """The posted patch itself (prose + diff), cut from the message body.

    hypermail renders the whole posting, so a patch submission carries
    `---\ndiff --git a/...` verbatim. Returns "" for messages that are not
    patch postings (replies, review chatter).
    """
    body = msg.body
    m = PATCH_BODY_RE.search(body)
    if not m:
        m = DIFF_START_RE.search(body)
        if not m:
            return ""
        start = m.start()
    else:
        start = m.end()
        diff = DIFF_START_RE.search(body, start)
        if diff:
            start = diff.start()
    chunk = body[start:].strip()
    # a posting is only worth taking if it actually looks like a patch
    if not DIFF_START_RE.search(chunk) and "@@" not in chunk:
        return ""
    return chunk


def _normalise_body(body: str) -> str:
    lines = [ln.rstrip() for ln in body.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)
