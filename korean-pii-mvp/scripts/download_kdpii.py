"""Download the original v1 splits; verify Zenodo-published MD5 checksums."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

FILES = {
    "train.json": "fa633403055e4e5918b92b2d11726434",
    "valid.json": "6f89923f873c840ca773d255d058150c",
    "test.json": "08eebd98593c1bad23fbf29053b3f9d2",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/raw/kdpii-v1")
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"record": "https://zenodo.org/records/10968609", "files": {}}
    for name, expected in FILES.items():
        path = root / name
        if not path.exists():
            temporary = path.with_suffix(".json.part")
            request = urllib.request.Request(f"https://zenodo.org/records/10968609/files/{name}?download=1", headers={"User-Agent": "korean-pii-mvp/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if hashlib.md5(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {name}; temporary file retained for inspection")
            temporary.replace(path)
        content = path.read_bytes()
        if hashlib.md5(content).hexdigest() != expected:
            raise ValueError(f"Existing file checksum mismatch: {name}; refusing overwrite")
        manifest["files"][name] = {"bytes": len(content), "md5": expected, "sha256": hashlib.sha256(content).hexdigest()}
        print(name, "verified", len(content))
    (root / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
