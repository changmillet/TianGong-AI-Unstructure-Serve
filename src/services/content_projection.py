"""Project normalized MinerU content into the existing business chunk contract.

Image selection and vision-result validation belong to the calling pipeline.
HTTP serialization and optional null fields remain outside this projection.
"""

from collections.abc import Callable, Sequence

from src.utils.text_output import clean_text


def _coerce_text_parts(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = clean_text(value)
        return [cleaned] if cleaned.strip() else []
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            parts.extend(_coerce_text_parts(item))
        return parts
    cleaned = clean_text(str(value))
    return [cleaned] if cleaned.strip() else []


def _image_captions(item: dict, *, include_generated: bool = True) -> list[str]:
    captions = _coerce_text_parts(item.get("img_caption") or item.get("image_caption"))
    if not include_generated:
        for field in ("image_caption", "chart_caption"):
            if field in item:
                return _coerce_text_parts(item[field])
        if captions and captions[-1] == item.get("content"):
            captions = captions[:-1]
    return captions


def _image_footnotes(item: dict) -> list[str]:
    return _coerce_text_parts(item.get("img_footnote") or item.get("image_footnote"))


def image_text(item: dict, *, include_generated: bool = True) -> str:
    captions = _image_captions(item, include_generated=include_generated)
    footnotes = _image_footnotes(item)
    return clean_text("\n".join([*captions, *footnotes]))


def table_text(item: dict) -> str:
    return clean_text(
        "\n".join(
            filter(
                None,
                [
                    "\n".join(item.get("table_caption", [])),
                    item.get("table_body", ""),
                    "\n".join(item.get("table_footnote", [])),
                ],
            )
        )
    )


def list_text(item: dict) -> str:
    items = item.get("list_items") or []
    return clean_text("\n".join(items) if items else item.get("text", ""))


def project_legacy_content(
    content_list: Sequence[dict],
    *,
    chunk_type: bool,
    image_text_resolver: Callable[[dict], str | None] | None = None,
    strict_type: bool = False,
) -> list[dict]:
    """Keep source order/pages while preserving plain and image-mode behavior.

    Without a resolver, plain parsing uses only legacy image notes and drops
    text that becomes empty after cleaning. Image pipelines provide their own
    resolver and retain their existing empty text/table chunks. ``strict_type``
    preserves the synchronous image builder's required type field.
    """
    plain = image_text_resolver is None
    results = []
    for item in content_list:
        # Image builders historically validate a page even for a skipped block;
        # plain parsing only converts the page when a nonempty chunk is emitted.
        page_number = None if plain else int(item.get("page_idx", 0)) + 1
        kind = item["type"] if strict_type else item.get("type")
        text = None
        if kind in ("text", "equation"):
            candidate = item.get("text", "")
            nonempty = candidate and candidate.strip() if plain else candidate.strip()
            if nonempty:
                text = clean_text(candidate)
        elif kind in ("header", "footer"):
            if not chunk_type:
                continue
            candidate = item.get("text", "")
            if plain:
                if candidate and candidate.strip():
                    text = clean_text(candidate)
            else:
                cleaned = clean_text(candidate)
                if cleaned.strip():
                    text = cleaned
        elif kind == "list":
            if plain and not (
                any(value.strip() for value in item.get("list_items", []))
                or item.get("text", "").strip()
            ):
                continue
            cleaned = list_text(item)
            if plain or cleaned.strip():
                text = cleaned
        elif kind == "table" and (
            item.get("table_caption") or item.get("table_body") or item.get("table_footnote")
        ):
            text = table_text(item)
        elif kind == "image":
            if plain:
                if item.get("img_caption") or item.get("img_footnote"):
                    text = clean_text(
                        "\n".join(
                            [*(item.get("img_caption") or []), *(item.get("img_footnote") or [])]
                        )
                    )
            else:
                text = image_text_resolver(item)
        if text is None or (plain and not text):
            continue
        chunk = {
            "text": text,
            "page_number": int(item.get("page_idx", 0)) + 1 if plain else page_number,
        }
        if chunk_type:
            if kind == "text" and item.get("text_level") is not None:
                chunk["type"] = "title"
            elif kind in ("header", "footer"):
                chunk["type"] = kind
            elif kind == "image" and not plain:
                chunk["type"] = "image"
        results.append(chunk)
    return results
