"""Create a small, clean JSONL corpus from extracted Enron email files.

Only standard-library modules are used so this first processing step is easy to
run and inspect.  It expects the common Enron ``maildir``-style layout, where
each email is an RFC 822 message stored in an individual file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterator


WHITESPACE = re.compile(r"[ \t]+")
BLANK_LINES = re.compile(r"\n{3,}")


def iter_email_files(input_dir: Path, include_path: str | None) -> Iterator[Path]:
    """Yield files in deterministic order, optionally restricted by path text."""
    for path in sorted(candidate for candidate in input_dir.rglob("*") if candidate.is_file()):
        relative_path = path.relative_to(input_dir).as_posix()
        if include_path is None or include_path.lower() in relative_path.lower():
            yield path


def clean_text(value: str) -> str:
    """Normalize whitespace without discarding email content or quoted evidence."""
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = "\n".join(WHITESPACE.sub(" ", line).strip() for line in value.split("\n"))
    return BLANK_LINES.sub("\n\n", value).strip()


def plain_text_body(message) -> str:
    """Extract text/plain content, avoiding attachments and HTML where possible."""
    parts = message.walk() if message.is_multipart() else [message]
    text_parts: list[str] = []
    for part in parts:
        if part.get_content_disposition() == "attachment":
            continue
        if part.get_content_type() != "text/plain":
            continue
        try:
            text_parts.append(part.get_content())
        except (LookupError, UnicodeError):
            continue
    return clean_text("\n\n".join(text_parts))


def normalized_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).isoformat()
    except (TypeError, ValueError, IndexError, OverflowError):
        return clean_text(value)


def parse_email(path: Path, input_dir: Path) -> dict[str, object] | None:
    """Parse one email file into the canonical Phase 4 document shape."""
    try:
        raw_bytes = path.read_bytes()
        message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    except (OSError, UnicodeError):
        return None

    body = plain_text_body(message)
    subject = clean_text(str(message.get("subject", "")))
    if not body and not subject:
        return None

    relative_path = path.relative_to(input_dir).as_posix()
    document_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
    metadata = {
        "subject": subject or None,
        "from": clean_text(str(message.get("from", ""))) or None,
        "to": clean_text(str(message.get("to", ""))) or None,
        "cc": clean_text(str(message.get("cc", ""))) or None,
        "date": normalized_date(message.get("date")),
        "message_id": clean_text(str(message.get("message-id", ""))) or None,
    }
    text = "\n".join(piece for piece in (f"Subject: {subject}" if subject else "", body) if piece)
    return {
        "document_id": document_id,
        "source_path": relative_path,
        "metadata": metadata,
        "body": body,
        "text": text,
        "verification": {
            "status": "unverified",
            "reason": "Email content is corpus evidence, not independently verified fact.",
        },
    }


def build_corpus(input_dir: Path, output_file: Path, max_documents: int, include_path: str | None) -> int:
    """Write parsed emails as JSONL and return the number of retained documents."""
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    retained = 0
    with output_file.open("w", encoding="utf-8", newline="\n") as destination:
        for path in iter_email_files(input_dir, include_path):
            document = parse_email(path, input_dir)
            if document is None:
                continue
            destination.write(json.dumps(document, ensure_ascii=False) + "\n")
            retained += 1
            if retained >= max_documents:
                break
    return retained


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess a bounded Enron email subset into JSONL.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Extracted Enron mail directory.")
    parser.add_argument("--output-file", type=Path, required=True, help="Destination JSONL file.")
    parser.add_argument("--max-documents", type=int, default=200, help="Maximum valid emails to retain (default: 200).")
    parser.add_argument("--include-path", help="Only include source files whose relative path contains this text.")
    args = parser.parse_args()
    if args.max_documents < 1:
        parser.error("--max-documents must be at least 1")
    count = build_corpus(args.input_dir, args.output_file, args.max_documents, args.include_path)
    print(f"Wrote {count} documents to {args.output_file}")


if __name__ == "__main__":
    main()
