import base64
import csv
import io
import json
import re
from html.parser import HTMLParser

import pytest
from PIL import Image

import crnn.annotate as annotate


def _csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path, count=3):
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for index in range(count):
        Image.new("RGB", (40 + index, 20), (index, 20, 40)).save(images / f"{index}.png")
        rows.append({"id": f"test/{index}.png", "image_path": f"images/{index}.png", "label": "word"})
    manifest = tmp_path / "test.csv"
    _csv(manifest, ("id", "image_path", "label"), rows)
    annotations = tmp_path / "test_annotations.csv"
    _csv(annotations, annotate.ANNOTATION_FIELDS,
         [{"id": row["id"], "blur_manual": "", "distortion": "", "notes": ""} for row in rows])
    return manifest, annotations, [row["id"] for row in rows]


def test_import_subset_by_id_preserves_blanks_and_records_source(tmp_path):
    manifest, destination, ids = _fixture(tmp_path)
    existing = annotate.read_annotation_rows(destination)
    existing[ids[0]]["distortion"] = "none"
    _csv(destination, annotate.ANNOTATION_FIELDS, list(existing.values()))
    incoming = tmp_path / "download.csv"
    _csv(incoming, annotate.ANNOTATION_FIELDS,
         [{"id": ids[2], "blur_manual": " SEVERE ", "distortion": "curved", "notes": 'Chữ mờ, "cong"\nDòng 2'},
          {"id": ids[0], "blur_manual": "clear", "distortion": "", "notes": ""}])
    event = annotate.merge_annotations(manifest, destination, incoming, annotator="reviewer_1")
    saved = annotate.read_annotation_rows(destination)
    assert list(saved) == ids
    assert saved[ids[0]]["distortion"] == "none"
    assert saved[ids[2]]["blur_manual"] == "severe"
    assert saved[ids[2]]["notes"] == 'Chữ mờ, "cong"\nDòng 2'
    assert saved[ids[1]]["blur_manual"] == ""
    assert event["num_changed_samples"] == 2
    metadata = json.loads(annotate.annotation_provenance_path(destination).read_text(encoding="utf-8"))
    assert metadata["annotations_sha256"] == annotate.sha256_file(destination)
    assert metadata["field_provenance"][ids[0]]["blur_manual"]["source"] == "human_review"
    assert "distortion" not in metadata["field_provenance"][ids[0]]


def test_conflicts_need_explicit_overwrite_and_blanks_never_clear(tmp_path):
    manifest, destination, ids = _fixture(tmp_path)
    existing = annotate.read_annotation_rows(destination)
    existing[ids[0]]["blur_manual"] = "clear"
    existing[ids[0]]["notes"] = "keep note"
    _csv(destination, annotate.ANNOTATION_FIELDS, list(existing.values()))
    incoming = tmp_path / "download.csv"
    _csv(incoming, annotate.ANNOTATION_FIELDS, [{"id": ids[0], "blur_manual": "severe", "distortion": "", "notes": ""}])
    original = destination.read_bytes()
    with pytest.raises(ValueError, match="Conflicting"):
        annotate.merge_annotations(manifest, destination, incoming, annotator="reviewer_2")
    assert destination.read_bytes() == original
    annotate.merge_annotations(manifest, destination, incoming, annotator="reviewer_2", source="ai_review", overwrite=True)
    saved = annotate.read_annotation_rows(destination)
    assert saved[ids[0]]["blur_manual"] == "severe"
    assert saved[ids[0]]["notes"] == "keep note"
    metadata = json.loads(annotate.annotation_provenance_path(destination).read_text(encoding="utf-8"))
    assert metadata["field_provenance"][ids[0]]["blur_manual"]["source"] == "ai_review"


@pytest.mark.parametrize("bad_row", [
    {"id": "unknown", "blur_manual": "clear", "distortion": "", "notes": ""},
    {"id": "test/0.png", "blur_manual": "blurry", "distortion": "", "notes": ""},
    {"id": "test/0.png", "blur_manual": "", "distortion": "wide", "notes": ""},
])
def test_invalid_last_row_leaves_saved_annotations_unchanged(tmp_path, bad_row):
    manifest, destination, ids = _fixture(tmp_path)
    incoming = tmp_path / "download.csv"
    _csv(incoming, annotate.ANNOTATION_FIELDS,
         [{"id": ids[1], "blur_manual": "clear", "distortion": "none", "notes": ""}, bad_row])
    original = destination.read_bytes()
    with pytest.raises(ValueError):
        annotate.merge_annotations(manifest, destination, incoming, annotator="reviewer")
    assert destination.read_bytes() == original
    assert not annotate.annotation_provenance_path(destination).exists()


@pytest.mark.parametrize("content", [
    "id,blur_manual,distortion,notes\na,clear,none,\na,severe,curved,\n",
    "id,blur_manual,distortion,notes,notes\na,clear,none,,\n",
    "id,blur_manual,distortion,notes\na,clear,none,ok,extra\n",
    "id,blur_manual,distortion,notes\na,clear,none\n",
])
def test_csv_structure_and_duplicate_ids_are_rejected(tmp_path, content):
    path = tmp_path / "malformed.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        annotate.read_annotation_rows(path)


def test_atomic_csv_replacement_failure_retains_original(tmp_path, monkeypatch):
    manifest, destination, ids = _fixture(tmp_path)
    incoming = tmp_path / "download.csv"
    _csv(incoming, annotate.ANNOTATION_FIELDS, [{"id": ids[0], "blur_manual": "clear", "distortion": "none", "notes": ""}])
    original = destination.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(annotate.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        annotate.merge_annotations(manifest, destination, incoming, annotator="reviewer")
    assert destination.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_export_is_offline_paginated_full_resolution_and_escaped(tmp_path):
    manifest, annotations, ids = _fixture(tmp_path, 3)
    saved = annotate.read_annotation_rows(annotations)
    malicious = '</script><script>alert("bad")</script>'
    saved[ids[0]].update({"blur_manual": "clear", "notes": malicious})
    _csv(annotations, annotate.ANNOTATION_FIELDS, list(saved.values()))
    original = annotations.read_bytes()
    output = tmp_path / "review"
    metadata = annotate.export_review_bundle(manifest, annotations, output, page_size=2)
    assert metadata["selected_ids"] == ids
    assert metadata["annotator"] == "human_review_pending"
    assert annotations.read_bytes() == original
    assert sorted(path.name for path in (output / "chunks").iterdir()) == ["page_0001.js", "page_0002.js"]
    html = (output / "index.html").read_text(encoding="utf-8")
    assert malicious not in html
    assert "\\u003c/script\\u003e" in html
    assert "localStorage" in html and "textContent" in html
    assert "https://" not in html and "fetch(" not in html
    boot = json.loads(re.search(r"const boot=(.+);", html).group(1))
    assert boot["initial"][ids[0]]["blur_manual"] == "clear"
    assert boot["initial"][ids[0]]["notes"] == malicious
    javascript = (output / "chunks" / "page_0001.js").read_text(encoding="utf-8")
    payload = json.loads(javascript.removeprefix("window.receivePage(0,").removesuffix(");\n"))
    with Image.open(io.BytesIO(base64.b64decode(payload[0]["image"].split(",", 1)[1]))) as image:
        assert image.size == (40, 20)
    with pytest.raises(FileExistsError):
        annotate.export_review_bundle(manifest, annotations, output)


def test_rendered_dropdowns_have_every_selectable_annotation_value():
    class SelectOptions(HTMLParser):
        def __init__(self):
            super().__init__()
            self.select = None
            self.option_open = False
            self.option_value = None
            self.option_text = ""
            self.values = {}

        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            if tag == "select":
                self.select = attributes["id"]
                self.values[self.select] = []
            elif tag == "option":
                assert self.select is not None and not self.option_open
                self.option_open = True
                self.option_value = attributes.get("value")
                self.option_text = ""

        def handle_data(self, data):
            if self.option_open:
                self.option_text += data

        def handle_endtag(self, tag):
            if tag == "option":
                assert self.option_open, "Option closing tag without a selectable option"
                self.values[self.select].append(self.option_value if self.option_value is not None else self.option_text.strip())
                self.option_open = False
            elif tag == "select":
                assert not self.option_open
                self.select = None

    parser = SelectOptions()
    parser.feed(annotate._HTML_TEMPLATE.replace("__BOOT_JSON__", "{}"))
    assert parser.values["blur"] == ["", *annotate.BLUR_TAGS]
    assert parser.values["distortion"] == ["", *annotate.DISTORTION_TAGS]


def test_sample_selection_is_reproducible_and_independent_of_errors(tmp_path):
    manifest, annotations, ids = _fixture(tmp_path, 10)
    first = annotate.export_review_bundle(manifest, annotations, tmp_path / "first", sample_size=4, seed=42, page_size=2)
    second = annotate.export_review_bundle(manifest, annotations, tmp_path / "second", sample_size=4, seed=42, page_size=2)
    assert first["selected_ids"] == second["selected_ids"]
    assert len(first["selected_ids"]) == 4
    assert first["selection_rule"] == "uniform_sample_without_replacement"
    assert first["selected_ids"] == [identifier for identifier in ids if identifier in first["selected_ids"]]
    with pytest.raises(ValueError):
        annotate.export_review_bundle(manifest, annotations, tmp_path / "bad", sample_size=0)


def test_stale_provenance_is_rejected_before_writing(tmp_path):
    manifest, destination, ids = _fixture(tmp_path)
    incoming = tmp_path / "download.csv"
    _csv(incoming, annotate.ANNOTATION_FIELDS, [{"id": ids[0], "blur_manual": "clear", "distortion": "none", "notes": ""}])
    annotate.annotation_provenance_path(destination).write_text(json.dumps({"annotations_sha256": "stale"}), encoding="utf-8")
    original = destination.read_bytes()
    with pytest.raises(ValueError, match="hash differs"):
        annotate.merge_annotations(manifest, destination, incoming, annotator="reviewer")
    assert destination.read_bytes() == original


def test_import_preserves_sampling_metadata_and_rejects_a_different_manifest(tmp_path):
    manifest, destination, ids = _fixture(tmp_path)
    incoming = tmp_path / "download.csv"
    _csv(incoming, annotate.ANNOTATION_FIELDS, [{"id": ids[0], "blur_manual": "clear", "distortion": "none", "notes": ""}])
    review = tmp_path / "review_metadata.json"
    information = {"manifest_sha256": annotate.sha256_file(manifest), "source": "human_review",
                   "annotator": "reviewer", "selected_ids": [ids[0]],
                   "selection_rule": "uniform_sample_without_replacement", "seed": 42,
                   "manifest_sample_count": 3, "selected_sample_count": 1}
    review.write_text(json.dumps(information), encoding="utf-8")
    event = annotate.merge_annotations(manifest, destination, incoming, annotator="reviewer", review_metadata=review)
    assert event["selection"]["selected_ids"] == [ids[0]]
    assert event["review_metadata_sha256"] == annotate.sha256_file(review)
    original = destination.read_bytes()
    review.write_text(json.dumps({**information, "manifest_sha256": "different"}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        annotate.merge_annotations(manifest, destination, incoming, annotator="reviewer", review_metadata=review)
    assert destination.read_bytes() == original
