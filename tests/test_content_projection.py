"""Freeze business output differences before consolidating content projection."""

from copy import deepcopy

import pytest

from src.utils.text_output import build_plain_text


@pytest.fixture
def project(app, monkeypatch):
    from src.services import gpu_scheduler, mineru_sci_service, mineru_service_full
    from src.services import mineru_with_images_service as images
    from src.services import two_stage_pipeline

    def run(mode, contents, *, chunk_type, return_txt=True):
        if mode in {"plain", "sci"}:
            monkeypatch.setattr(
                mineru_service_full, "parse_doc", lambda *_a, **_k: (contents, "unused", None)
            )
            monkeypatch.setattr(mineru_sci_service, "parse_doc", mineru_service_full.parse_doc)
            result = gpu_scheduler._actual_parse(
                "unused.pdf",
                "sci" if mode == "sci" else "default",
                {"chunk_type": chunk_type, "return_txt": return_txt},
            )
            return result["result"], result["txt"]
        if mode == "images":
            replacements = {
                id(item): "  Printed image\nVision label  "
                for item in contents
                if item.get("img_path")
            }
            result = images._build_result_items(contents, replacements, chunk_type=chunk_type)
            return result, build_plain_text(result) if return_txt else None
        result, txt = two_stage_pipeline._merge_content(
            contents,
            [{"seq": 3, "vision_text": "Vision label"}],
            chunk_type=chunk_type,
            return_txt=return_txt,
        )
        return [item.model_dump(exclude_none=True) for item in result], txt

    return run


@pytest.mark.parametrize("mode", ["plain", "sci", "images", "two-stage"])
@pytest.mark.parametrize("chunk_type", [False, True])
@pytest.mark.parametrize("return_txt", [False, True])
def test_reading_order_source_pages_types_and_txt(project, mode, chunk_type, return_txt):
    contents = [
        {"type": "header", "text": "Header", "page_idx": 10},
        {"type": "text", "text": "Title\ud800", "text_level": 1, "page_idx": 10},
        {"type": "text", "text": "  Body  ", "page_idx": 10},
        {"type": "list", "list_items": ["A", "B"], "page_idx": 10},
        {"type": "equation", "text": "x = 1", "page_idx": 11},
        {
            "type": "table",
            "table_caption": ["Measurements"],
            "table_body": "<table>42</table>",
            "table_footnote": ["Units"],
            "page_idx": 11,
        },
        {
            "type": "image",
            "img_path": "figure.png",
            "img_caption": ["Printed image", "SDK diagram"],
            "image_caption": ["Printed image"],
            "content": "SDK diagram",
            "img_footnote": ["Note"],
            "__image_seq": 3,
            "page_idx": 11,
        },
        {"type": "image", "img_caption": ["Caption only"], "page_idx": 12},
        {"type": "page_number", "text": "13", "page_idx": 12},
        {"type": "unknown", "text": "Ignore", "page_idx": 12},
        {"type": "footer", "text": "Footer", "page_idx": 12},
    ]
    original = deepcopy(contents)
    expected = [
        {"text": "Title", "page_number": 11},
        {"text": "  Body  ", "page_number": 11},
        {"text": "A\nB", "page_number": 11},
        {"text": "x = 1", "page_number": 12},
        {"text": "Measurements\n<table>42</table>\nUnits", "page_number": 12},
        {
            "text": {
                "plain": "Printed image\nSDK diagram\nNote",
                "sci": "Printed image\nSDK diagram\nNote",
                "images": "Printed image\nVision label",
                "two-stage": "Printed image\nNote\nVision label",
            }[mode],
            "page_number": 12,
        },
        {"text": "Caption only", "page_number": 13},
    ]
    if chunk_type:
        expected[0]["type"] = "title"
        if mode in {"images", "two-stage"}:
            expected[-2]["type"] = expected[-1]["type"] = "image"
        expected.insert(0, {"text": "Header", "page_number": 11, "type": "header"})
        expected.append({"text": "Footer", "page_number": 13, "type": "footer"})
    result, txt = project(mode, contents, chunk_type=chunk_type, return_txt=return_txt)
    assert result == expected
    expected_txt = "Title\n\n" if chunk_type else "Title\n"
    expected_txt += "Body\nA\nB\nx = 1\nMeasurements\n<table>42</table>\nUnits\n"
    expected_txt += expected[-3 if chunk_type else -2]["text"] + "\nCaption only"
    if chunk_type:
        expected_txt = "Header\n" + expected_txt + "\nFooter"
    assert txt == (expected_txt if return_txt else None)
    assert contents == original


@pytest.mark.parametrize("mode", ["plain", "sci", "images", "two-stage"])
def test_cleaned_empty_text_keeps_existing_mode_semantics(project, mode):
    contents = [
        {"type": "text", "text": "\ud800", "page_idx": 0},
        {"type": "table", "table_caption": ["\ud800"], "page_idx": 0},
        {"type": "list", "list_items": ["\ud800"], "page_idx": 0},
        {"type": "header", "text": "\ud800", "page_idx": 0},
    ]
    result, txt = project(mode, contents, chunk_type=True)
    expected = [] if mode in {"plain", "sci"} else [{"text": "", "page_number": 1}] * 2
    assert result == expected
    assert txt == ""


@pytest.mark.parametrize("mode", ["plain", "sci"])
def test_sci_current_path_preserves_reference_section(project, mode):
    contents = [
        {"type": "text", "text": "References", "text_level": 1, "page_idx": 1},
        {"type": "text", "text": "Citation", "page_idx": 1},
    ]
    result, _ = project(mode, contents, chunk_type=True)
    assert result == [
        {"text": "References", "page_number": 2, "type": "title"},
        {"text": "Citation", "page_number": 2},
    ]


def test_two_stage_filtered_figure_retains_sdk_content(project):
    contents = [
        {
            "type": "image",
            "img_path": "filtered.png",
            "img_caption": ["Printed", "SDK content"],
            "content": "SDK content",
            "page_idx": 7,
        }
    ]
    result, txt = project("two-stage", contents, chunk_type=True)
    assert result == [{"text": "Printed\nSDK content", "page_number": 8, "type": "image"}]
    assert txt == "Printed\nSDK content"


def test_two_stage_missing_assigned_vision_result_still_fails(project):
    contents = [{"type": "image", "img_path": "figure.png", "__image_seq": 9}]
    with pytest.raises(RuntimeError, match="Missing or empty vision result for seq=9"):
        project("two-stage", contents, chunk_type=True)


@pytest.mark.parametrize("mode", ["plain", "sci", "images", "two-stage"])
def test_modern_caption_only_fallback_remains_specific_to_sync_images(project, mode):
    contents = [{"type": "image", "image_caption": "Printed caption", "page_idx": 2}]
    result, _ = project(mode, contents, chunk_type=True)
    expected = (
        [{"text": "Printed caption", "page_number": 3, "type": "image"}] if mode == "images" else []
    )
    assert result == expected


@pytest.mark.parametrize("mode", ["plain", "sci", "images", "two-stage"])
def test_skipped_block_page_validation_preserves_existing_mode_boundary(project, mode):
    contents = [{"type": "unknown", "page_idx": "invalid"}]
    if mode in {"plain", "sci"}:
        assert project(mode, contents, chunk_type=True) == ([], "")
    else:
        with pytest.raises(ValueError):
            project(mode, contents, chunk_type=True)


@pytest.mark.parametrize("mode", ["plain", "sci", "images", "two-stage"])
def test_only_text_blocks_become_titles(project, mode):
    contents = [
        {"type": "equation", "text": "x", "text_level": 1},
        {"type": "list", "list_items": ["A"], "text_level": 1},
        {"type": "table", "table_body": "42", "text_level": 1},
    ]
    result, txt = project(mode, contents, chunk_type=True)
    assert result == [{"text": value, "page_number": 1} for value in ("x", "A", "42")]
    assert txt == "x\nA\n42"
