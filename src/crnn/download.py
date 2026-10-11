"""Download the official IIIT5K archive, verify checksum, and safely extract it."""

import argparse
import hashlib
import json
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URLS = [
    "https://cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz",
    "https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz",
]
# Independently published in the docTR IIIT5K dataset implementation.
SHA256 = "7872c9efbec457eb23f3368855e7738f72ce10927f52a382deb4966ca0ffa38e"


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive_path, output):
    output = output.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (output / member.name).resolve()
            if not target.is_relative_to(output) or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive member: {member.name}")
        archive.extractall(output, members=members, filter="data")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--archive", type=Path, help="Use an already downloaded official archive.")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    archive_path = args.archive or args.output / "IIIT5K-Word_V3.0.tar.gz"
    source = "local archive" if args.archive else None
    if not archive_path.exists():
        if args.archive:
            raise FileNotFoundError(archive_path)
        part = archive_path.with_suffix(archive_path.suffix + ".part")
        errors = []
        for url in URLS:
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "CRNN-IIIT5K-research/0.1"})
                with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as handle:
                    if "html" in response.headers.get("Content-Type", "").lower():
                        raise ValueError("Dataset endpoint returned HTML rather than an archive.")
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                if digest_file(part) != SHA256:
                    raise ValueError("Official archive SHA256 mismatch; extraction refused.")
                part.replace(archive_path)
                source = url
                break
            except (OSError, ValueError) as exc:
                errors.append(f"{url}: {exc}")
        else:
            raise RuntimeError("Cannot download verified dataset:\n" + "\n".join(errors))
    digest = digest_file(archive_path)
    if digest != SHA256:
        raise ValueError(f"Archive checksum mismatch: {digest}; expected {SHA256}.")
    safe_extract(archive_path, args.output)
    metadata_path = args.output / "source.json"
    if not source and metadata_path.exists():
        source = json.loads(metadata_path.read_text(encoding="utf-8")).get("source")
    metadata = {"source": source or "verified cached official archive", "archive_sha256": digest,
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
                "checksum_reference": "https://mindee.github.io/doctr/_modules/doctr/datasets/iiit5k.html"}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Verified dataset extracted to {args.output.resolve()}")


if __name__ == "__main__":
    main()
