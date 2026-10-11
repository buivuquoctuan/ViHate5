"""Export an offline image review bundle and safely merge downloaded labels."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

ANNOTATION_FIELDS = ("id", "blur_manual", "distortion", "notes")
BLUR_TAGS = ("clear", "moderate", "severe")
DISTORTION_TAGS = ("none", "perspective", "curved", "stylized", "mixed")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_review_manifest(path: Path) -> list[dict[str, str]]:
    """Load images without importing the training framework or predictions."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if len(set(headers)) != len(headers) or not {"id", "image_path"} <= set(headers):
            raise ValueError("Review manifest requires unique id,image_path columns")
        rows = []
        seen = set()
        for number, row in enumerate(reader, 2):
            identifier = row.get("id")
            if None in row or not identifier or identifier in seen or not row.get("image_path"):
                raise ValueError(f"{path}:{number}: malformed row or duplicate/empty ID")
            seen.add(identifier)
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            image_path = image_path.resolve()
            if not image_path.is_file():
                raise ValueError(f"{path}:{number}: image not found: {image_path}")
            rows.append({"id": identifier, "image_path": str(image_path)})
    if not rows:
        raise ValueError("Review manifest must not be empty")
    return rows


def read_annotation_rows(path: Path, allowed_ids: set[str] | None = None) -> dict[str, dict[str, str]]:
    """Validate all cells before any import mutation; blanks remain unassigned."""
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if len(set(headers)) != len(headers) or set(headers) != set(ANNOTATION_FIELDS):
            raise ValueError(f"{path}: expected unique columns {','.join(ANNOTATION_FIELDS)}")
        for number, raw in enumerate(reader, 2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"{path}:{number}: malformed CSV cell count")
            identifier = raw["id"]
            if not identifier or identifier in result or (allowed_ids is not None and identifier not in allowed_ids):
                raise ValueError(f"{path}:{number}: unknown or duplicate annotation id {identifier!r}")
            blur = raw["blur_manual"].strip().lower()
            distortion = raw["distortion"].strip().lower()
            if blur and blur not in BLUR_TAGS:
                raise ValueError(f"{path}:{number}: invalid blur_manual {blur!r}")
            if distortion and distortion not in DISTORTION_TAGS:
                raise ValueError(f"{path}:{number}: invalid distortion {distortion!r}")
            result[identifier] = {"id": identifier, "blur_manual": blur, "distortion": distortion, "notes": raw["notes"]}
    return result


def annotation_provenance_path(path: Path) -> Path:
    return Path(path).with_suffix(".provenance.json")


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=ANNOTATION_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def merge_annotations(
    manifest: Path, annotations: Path, incoming_csv: Path, *, annotator: str,
    source: str = "human_review", overwrite: bool = False, review_metadata: Path | None = None,
) -> dict[str, Any]:
    """Merge nonblank incoming cells by ID; conflicting saved cells require opt-in.

    Blank incoming cells mean leave unchanged, including under --overwrite.
    CSV and provenance files are each replaced atomically after full validation.
    Provenance records the resulting CSV hash so an interrupted two-file update
    can be detected; it never invents provenance for preexisting assignments.
    """
    manifest, annotations, incoming_csv = map(Path, (manifest, annotations, incoming_csv))
    if not annotator.strip() or source not in ("human_review", "ai_review"):
        raise ValueError("Specify an annotator and source human_review or ai_review")
    if annotations.resolve() == incoming_csv.resolve():
        raise ValueError("Incoming CSV and destination annotations must be different files")
    records = read_review_manifest(manifest)
    identifiers = [row["id"] for row in records]
    allowed = set(identifiers)
    current = read_annotation_rows(annotations, allowed) if annotations.exists() else {}
    incoming = read_annotation_rows(incoming_csv, allowed)
    if not incoming:
        raise ValueError("Incoming annotation CSV is empty")
    merged = {identifier: dict(current.get(identifier, {"id": identifier, "blur_manual": "", "distortion": "", "notes": ""})) for identifier in identifiers}
    changes: dict[str, list[str]] = {}
    for identifier, row in incoming.items():
        for field in ANNOTATION_FIELDS[1:]:
            old, new = merged[identifier][field], row[field]
            if not new or new == old:
                continue
            if old and not overwrite:
                raise ValueError(f"Conflicting nonempty {field} for {identifier!r}; use --overwrite to replace it")
            merged[identifier][field] = new
            changes.setdefault(identifier, []).append(field)
    review_information = None
    if review_metadata is not None:
        review_metadata = Path(review_metadata)
        review_information = json.loads(review_metadata.read_text(encoding="utf-8"))
        if not isinstance(review_information, dict) or review_information.get("manifest_sha256") != sha256_file(manifest):
            raise ValueError("Review metadata does not match the manifest")
        claimed_source = review_information.get("source")
        if claimed_source not in (source, "human_review_pending" if source == "human_review" else source):
            raise ValueError("Review metadata source differs from --source")
        selected_ids = review_information.get("selected_ids", [])
        if not isinstance(selected_ids, list) or any(not isinstance(identifier, str) for identifier in selected_ids) or len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= allowed:
            raise ValueError("Review metadata contains unknown or duplicate selected IDs")
        if not set(changes) <= set(selected_ids):
            raise ValueError("Incoming changes contain IDs outside the selected review sample")
    payload = _csv_bytes([merged[identifier] for identifier in identifiers])
    provenance_path = annotation_provenance_path(annotations)
    previous_hash = sha256_file(annotations) if annotations.exists() else None
    metadata: dict[str, Any] = {"schema_version": 1, "events": [], "field_provenance": {}}
    if provenance_path.exists():
        metadata = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Annotation provenance must be a JSON object")
        if metadata.get("annotations_sha256") not in (None, previous_hash):
            raise ValueError("Annotation provenance hash differs from saved CSV; resolve the mismatch before importing")
        if not isinstance(metadata.get("events", []), list) or not isinstance(metadata.get("field_provenance", {}), dict):
            raise ValueError("Malformed annotation provenance")
    timestamp = datetime.now(timezone.utc).isoformat()
    event = {
        "timestamp_utc": timestamp, "source": source, "annotator": annotator.strip(),
        "manifest_sha256": sha256_file(manifest), "incoming_csv_sha256": sha256_file(incoming_csv),
        "previous_annotations_sha256": previous_hash,
        "annotations_sha256": hashlib.sha256(payload).hexdigest(),
        "changed_ids": list(changes), "changed_fields": changes,
        "num_changed_samples": len(changes), "overwrite": overwrite,
    }
    if review_information is not None:
        event["review_metadata_sha256"] = sha256_file(review_metadata)
        event["claimed_annotator"] = review_information.get("annotator")
        event["selection"] = {key: review_information.get(key) for key in
                              ("selection_rule", "seed", "manifest_sample_count", "selected_sample_count", "selected_ids")}
    metadata.setdefault("events", []).append(event)
    field_provenance = metadata.setdefault("field_provenance", {})
    for identifier, fields in changes.items():
        for field in fields:
            field_provenance.setdefault(identifier, {})[field] = {
                "source": source, "annotator": annotator.strip(), "timestamp_utc": timestamp,
            }
    metadata.update({"schema_version": 1, "annotations_sha256": event["annotations_sha256"],
                     "manifest_sha256": event["manifest_sha256"]})
    metadata_bytes = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write(annotations, payload)
    _atomic_write(provenance_path, metadata_bytes)
    return event


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _embedded_image(path: Path) -> tuple[str, int, int]:
    # Encode a trusted raster after EXIF orientation, at original resolution.
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        return uri, image.width, image.height


def export_review_bundle(
    manifest: Path, annotations: Path, output: Path, *, page_size: int = 100,
    seed: int = 42, sample_size: int | None = None,
) -> dict[str, Any]:
    manifest, annotations, output = map(Path, (manifest, annotations, output))
    if page_size < 1 or page_size > 500:
        raise ValueError("page_size must be between 1 and 500")
    records = read_review_manifest(manifest)
    if sample_size is not None and not 1 <= sample_size <= len(records):
        raise ValueError("sample_size must be between 1 and the manifest sample count")
    if output.exists():
        raise FileExistsError(f"Review output already exists: {output}; choose a new directory")
    identifiers = [row["id"] for row in records]
    current = read_annotation_rows(annotations, set(identifiers)) if annotations.exists() else {}
    initial = {identifier: current.get(identifier, {"id": identifier, "blur_manual": "", "distortion": "", "notes": ""}) for identifier in identifiers}
    selected = records
    if sample_size is not None:
        indices = sorted(random.Random(seed).sample(range(len(records)), sample_size))
        selected = [records[index] for index in indices]
    selected_ids = [row["id"] for row in selected]
    manifest_hash = sha256_file(manifest)
    metadata = {
        "schema_version": 1, "manifest_sha256": manifest_hash,
        "annotations_sha256": sha256_file(annotations) if annotations.exists() else None,
        "selection_rule": "uniform_sample_without_replacement" if sample_size is not None else "all_manifest_rows",
        "seed": seed, "manifest_sample_count": len(records), "selected_sample_count": len(selected),
        "selected_ids": selected_ids, "page_size": page_size,
        "annotator": "human_review_pending", "source": "unassigned",
        "image_policy": "embedded PNG; original resolution; EXIF orientation applied; no predictions shown",
        "existing_assignment_policy": "preserve and lock nonempty saved cells",
    }
    boot = {"allIds": identifiers, "selectedIds": selected_ids, "initial": initial,
            "pageSize": page_size, "metadata": metadata,
            "storageKey": f"crnn-review-v1:{manifest_hash}:{metadata['annotations_sha256'] or 'blank'}"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".annotation-export-", dir=output.parent) as temporary:
        staged = Path(temporary) / "bundle"
        chunks = staged / "chunks"
        chunks.mkdir(parents=True)
        for start in range(0, len(selected), page_size):
            payload = []
            for row in selected[start:start + page_size]:
                uri, width, height = _embedded_image(Path(row["image_path"]))
                payload.append({"id": row["id"], "image": uri, "width": width, "height": height})
            number = start // page_size
            (chunks / f"page_{number + 1:04d}.js").write_text(f"window.receivePage({number},{_safe_json(payload)});\n", encoding="utf-8")
        (staged / "index.html").write_text(_HTML_TEMPLATE.replace("__BOOT_JSON__", _safe_json(boot)), encoding="utf-8")
        (staged / "selection_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staged / "README.txt").write_text(
            "Open index.html directly in a browser. Keep the chunks directory beside it.\n"
            "Images retain original resolution; toggle original-size view for visual review.\n"
            "Selection uses manifest IDs only, without predictions or error-based filtering.\n"
            "Browser progress is best effort. Download CSV and review metadata before closing.\n"
            "Existing saved labels are locked. Blank categories mean unassigned, not none/clear.\n"
            "Import CSV using python -m crnn.annotate import --manifest ... --annotations ... --csv ... --annotator NAME\n"
            "Incoming blanks do not clear saved fields. Conflicts require explicit --overwrite.\n", encoding="utf-8")
        os.replace(staged, output)
    return metadata


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CRNN — gán nhãn ảnh ngoại tuyến</title>
<style>
body{font:16px system-ui,sans-serif;background:#f3f5f7;color:#14283c;margin:0}main{max-width:1100px;margin:auto;padding:22px}
h1{font-size:25px}button,select,input,textarea{font:inherit;padding:9px;border:1px solid #8b9bad;border-radius:5px}button{background:white;cursor:pointer}button:disabled{opacity:.45;cursor:default}
.toolbar,.fields{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:15px 0}.fields label{display:grid;gap:6px;flex:1;min-width:200px}
#frame{overflow:auto;min-height:150px;padding:24px;background:white;border:1px solid #bac6d3;display:flex;align-items:center;justify-content:center}#image{max-width:100%;max-height:55vh;object-fit:contain;image-rendering:auto}#image.original{max-width:none;max-height:none}
textarea{box-sizing:border-box;width:100%;min-height:80px}.small{font-size:14px;color:#40566a}.warning{padding:10px;background:#fff0ca;border:1px solid #d8ad44}#status{min-height:24px}summary{cursor:pointer}kbd{background:#e4eaf0;border:1px solid #bdc9d6;padding:0 4px}
</style></head><body><main>
<h1>Gán nhãn độ mờ và biến dạng chữ</h1>
<p>Quan sát ảnh gốc. Để trống khi chưa đánh giá hoặc chưa chắc; <b>none</b> và <b>clear</b> là nhãn đã đánh giá. Công cụ không hiển thị dự đoán của mô hình.</p>
<details><summary>Quy ước và phím tắt</summary><p><b>Độ mờ:</b> clear: cạnh/nét chữ rõ; moderate: cạnh chữ mềm nhưng phần lớn chi tiết còn phân biệt; severe: nhiều chi tiết nét chữ mất do mờ. Không dùng nền nhiều chi tiết, độ dài hay độ tương phản thấp làm nhãn mờ tự động.</p>
<p><b>Biến dạng:</b> none: không thấy biến dạng đáng kể; perspective: phối cảnh nghiêng/hội tụ; curved: đường chữ cong; stylized: hình dạng nét/font cách điệu; mixed: kết hợp nhiều loại. Ghi chú khi ranh giới khó quyết định.</p>
<p>←/→: ảnh trước/sau; 1/2/3: clear/moderate/severe; N/P/C/S/M: none/perspective/curved/stylized/mixed. Phím tắt không áp dụng khi đang nhập ghi chú hoặc chọn ô.</p></details>
<div class="toolbar"><button id="previous">← Trước</button><button id="next">Sau →</button><span id="position"></span><label><input type="checkbox" id="unlabeled"> Chỉ ảnh thiếu ít nhất một nhãn</label><label><input type="checkbox" id="original"> Xem kích thước gốc</label></div>
<p id="sampleId"></p><div id="frame"><img id="image" alt="Ảnh từ cần đánh giá"></div>
<div class="fields"><label>Độ mờ<select id="blur"><option value="">Chưa gán nhãn</option><option value="clear">clear</option><option value="moderate">moderate</option><option value="severe">severe</option></select></label>
<label>Biến dạng<select id="distortion"><option value="">Chưa gán nhãn</option><option value="none">none</option><option value="perspective">perspective</option><option value="curved">curved</option><option value="stylized">stylized</option><option value="mixed">mixed</option></select></label></div>
<label for="notes">Ghi chú</label><textarea id="notes" placeholder="Chi tiết khó phân biệt, dấu hiệu mờ hoặc biến dạng..."></textarea>
<p class="small">Ô đã có dữ liệu trong CSV nguồn được khóa để giữ nhãn hiện có. Thay đổi nhãn đã lưu cần CSV sửa riêng và nhập với --overwrite.</p>
<div class="toolbar"><label>Người gán nhãn <input id="annotator" placeholder="Tên hoặc mã người đánh giá"></label><button id="download">Tải CSV tiến độ</button><button id="downloadMetadata">Tải metadata đánh giá</button></div>
<p id="progress"></p><p id="status" role="status"></p><p id="storageWarning" class="warning" hidden>Trình duyệt không lưu được tiến độ lâu dài. Hãy tải CSV trước khi đóng trang.</p>
<p class="small">Tiến độ thuộc bộ dữ liệu và bản CSV nguồn này, lưu trong trình duyệt hiện tại. Tải CSV định kỳ; giữ cả index.html và thư mục chunks khi chuyển máy.</p>
</main><script>
'use strict';
const boot=__BOOT_JSON__;
const byId=id=>document.getElementById(id), validBlur=['','clear','moderate','severe'],validDist=['','none','perspective','curved','stylized','mixed'];
let state=JSON.parse(JSON.stringify(boot.initial)),current=0,renderToken=0,timer=null;
try{const saved=JSON.parse(localStorage.getItem(boot.storageKey)||'{}');for(const id of boot.selectedIds){const item=saved[id];if(!item||typeof item!=='object')continue;for(const field of ['blur_manual','distortion','notes']){if(state[id][field])continue;const value=item[field];if(typeof value!=='string')continue;if(field==='blur_manual'&&!validBlur.includes(value))continue;if(field==='distortion'&&!validDist.includes(value))continue;state[id][field]=value;}}}catch(error){byId('storageWarning').hidden=false;}
const cache=new Map(),pending=new Map();
window.receivePage=(number,records)=>{cache.set(number,records);if(pending.has(number)){pending.get(number).resolve(records);pending.delete(number);}for(const key of cache.keys()){if(Math.abs(key-number)>1)cache.delete(key);}};
function loadPage(number){if(cache.has(number))return Promise.resolve(cache.get(number));if(pending.has(number))return pending.get(number).promise;let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});pending.set(number,{promise,resolve,reject});const script=document.createElement('script');script.src='chunks/page_'+String(number+1).padStart(4,'0')+'.js';script.onload=()=>script.remove();script.onerror=()=>{pending.delete(number);script.remove();reject(new Error('Không đọc được trang ảnh. Giữ thư mục chunks cạnh index.html.'));};document.head.appendChild(script);return promise;}
function incomplete(id){return !state[id].blur_manual||!state[id].distortion;}
function nextIndex(direction){for(let i=current+direction;i>=0&&i<boot.selectedIds.length;i+=direction){if(!byId('unlabeled').checked||incomplete(boot.selectedIds[i]))return i;}return -1;}
function progress(){const reviewedBlur=boot.selectedIds.filter(id=>state[id].blur_manual).length,reviewedDist=boot.selectedIds.filter(id=>state[id].distortion).length,both=boot.selectedIds.filter(id=>!incomplete(id)).length;byId('progress').textContent='Mẫu đã chọn: '+boot.selectedIds.length+'; có nhãn mờ: '+reviewedBlur+'; có nhãn biến dạng: '+reviewedDist+'; đủ hai nhãn: '+both+'.';byId('previous').disabled=nextIndex(-1)<0;byId('next').disabled=nextIndex(1)<0;}
function save(){try{const selected=Object.create(null);for(const id of boot.selectedIds)selected[id]=state[id];localStorage.setItem(boot.storageKey,JSON.stringify(selected));byId('status').textContent='Đã lưu tiến độ trong trình duyệt.';}catch(error){byId('storageWarning').hidden=false;byId('status').textContent='Đã giữ tiến độ trong trang; hãy tải CSV.';}progress();}
function update(field,element){const id=boot.selectedIds[current];if(boot.initial[id][field])return;state[id][field]=element.value;clearTimeout(timer);timer=setTimeout(save,250);progress();}
async function render(){const token=++renderToken;if(byId('unlabeled').checked&&!boot.selectedIds.some(incomplete)){byId('sampleId').textContent='';byId('position').textContent='Không còn ảnh thiếu nhãn';byId('image').removeAttribute('src');for(const id of ['blur','distortion','notes'])byId(id).disabled=true;byId('status').textContent='Tất cả ảnh đã đủ hai nhãn. Bỏ bộ lọc để xem lại.';progress();return;}const id=boot.selectedIds[current],item=state[id];byId('sampleId').textContent='ID: '+id;byId('position').textContent=(current+1)+' / '+boot.selectedIds.length+' (trang '+(Math.floor(current/boot.pageSize)+1)+')';for(const [field,elementId] of [['blur_manual','blur'],['distortion','distortion'],['notes','notes']]){byId(elementId).value=item[field];byId(elementId).disabled=Boolean(boot.initial[id][field]);}byId('image').removeAttribute('src');byId('status').textContent='Đang đọc ảnh...';progress();try{const records=await loadPage(Math.floor(current/boot.pageSize));if(token!==renderToken)return;const record=records[current%boot.pageSize];if(record.id!==id)throw new Error('ID ảnh không khớp metadata.');byId('image').src=record.image;byId('image').alt='Ảnh gốc '+id+'; '+record.width+' × '+record.height+' pixels';byId('status').textContent='Ảnh gốc '+record.width+' × '+record.height+' pixels.';}catch(error){byId('status').textContent=error.message;}}
function navigate(direction){save();const index=nextIndex(direction);if(index>=0){current=index;render();}}
byId('previous').onclick=()=>navigate(-1);byId('next').onclick=()=>navigate(1);
byId('blur').onchange=()=>update('blur_manual',byId('blur'));byId('distortion').onchange=()=>update('distortion',byId('distortion'));byId('notes').oninput=()=>update('notes',byId('notes'));
byId('original').onchange=()=>byId('image').classList.toggle('original',byId('original').checked);
byId('unlabeled').onchange=()=>{if(byId('unlabeled').checked&&!incomplete(boot.selectedIds[current])){const index=boot.selectedIds.findIndex(incomplete);if(index>=0)current=index;else byId('status').textContent='Tất cả ảnh đã đủ hai nhãn.';}render();};
function download(content,type,name){save();const url=URL.createObjectURL(new Blob([content],{type}));const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function quote(value){return '"'+String(value).replaceAll('"','""')+'"';}
byId('download').onclick=()=>{const rows=[['id','blur_manual','distortion','notes'],...boot.allIds.map(id=>[id,state[id].blur_manual,state[id].distortion,state[id].notes])];download('\uFEFF'+rows.map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n','text/csv;charset=utf-8','test_annotations_review.csv');};
byId('downloadMetadata').onclick=()=>{const changedIds=boot.selectedIds.filter(id=>['blur_manual','distortion','notes'].some(field=>state[id][field]!==boot.initial[id][field]));const data={...boot.metadata,annotator:byId('annotator').value.trim()||'human_review_unspecified',source:changedIds.length?'human_review':'human_review_pending',review_timestamp_utc:new Date().toISOString(),changed_ids:changedIds,blur_assigned_ids:boot.selectedIds.filter(id=>state[id].blur_manual),distortion_assigned_ids:boot.selectedIds.filter(id=>state[id].distortion)};download(JSON.stringify(data,null,2)+'\n','application/json','annotation_review_metadata.json');};
document.addEventListener('keydown',event=>{if(['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName)||event.ctrlKey||event.metaKey||event.altKey)return;if(event.key==='ArrowLeft'){event.preventDefault();navigate(-1);}else if(event.key==='ArrowRight'){event.preventDefault();navigate(1);}else{const blur={'1':'clear','2':'moderate','3':'severe'}[event.key],dist={n:'none',p:'perspective',c:'curved',s:'stylized',m:'mixed'}[event.key.toLowerCase()];if(blur&&!byId('blur').disabled){byId('blur').value=blur;update('blur_manual',byId('blur'));}if(dist&&!byId('distortion').disabled){byId('distortion').value=dist;update('distortion',byId('distortion'));}}});
window.addEventListener('beforeunload',save);render();
</script></body></html>'''


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Create an offline visual review bundle")
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--annotations", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--page-size", type=int, default=100)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--sample-size", type=int)
    merge = commands.add_parser("import", help="Merge downloaded labels by exact ID")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--annotations", type=Path, required=True)
    merge.add_argument("--csv", type=Path, required=True)
    merge.add_argument("--annotator", required=True)
    merge.add_argument("--source", choices=("human_review", "ai_review"), default="human_review")
    merge.add_argument("--review-metadata", type=Path, help="Downloaded review metadata including the sampled cohort")
    merge.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            result = export_review_bundle(args.manifest, args.annotations, args.output,
                                          page_size=args.page_size, seed=args.seed, sample_size=args.sample_size)
            print(f"Review bundle: {args.output / 'index.html'} ({result['selected_sample_count']} images)")
        else:
            result = merge_annotations(args.manifest, args.annotations, args.csv, annotator=args.annotator,
                                       source=args.source, overwrite=args.overwrite, review_metadata=args.review_metadata)
            print(f"Imported {result['num_changed_samples']} changed samples. Annotations: {args.annotations}")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
