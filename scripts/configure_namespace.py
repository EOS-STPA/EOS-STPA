#!/usr/bin/env python3
"""Configure a GitHub Pages namespace and regenerate validation outputs."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

PREFIX_RE = re.compile(r"@prefix\s+eos:\s+<([^>]+)>\s+\.")


def read_current_namespace(ontology_path: Path) -> str:
    text = ontology_path.read_text(encoding="utf-8")
    match = PREFIX_RE.search(text)
    if not match:
        raise SystemExit("Could not locate the eos: namespace in the ontology file.")
    return match.group(1)


def replace_text(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(old, new), encoding="utf-8")


def write_docs_page(path: Path, namespace: str, repository_url: str) -> None:
    document_iri = namespace[:-1] if namespace.endswith("#") else namespace
    ttl_url = "eos_stpa_avx.ttl"
    page = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>EOS-STPA ontology namespace</title>
</head>
<body>
  <main>
    <h1>EOS-STPA ontology namespace</h1>
    <p><code>{namespace}</code></p>
    <p><a href=\"{ttl_url}\">Download the Turtle/OWL ontology</a></p>
    <p><a href=\"{repository_url}\">Open the GitHub repository</a></p>
    <p>This namespace page documents the executable delayed red-light ontology validation module.</p>
  </main>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="GitHub account name")
    parser.add_argument("--repository", required=True, help="GitHub repository name")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ontology_path = root / "ontology" / "eos_stpa_avx.ttl"
    current_namespace = read_current_namespace(ontology_path)
    current_document_iri = current_namespace[:-1] if current_namespace.endswith("#") else current_namespace

    new_document_iri = f"https://{args.username}.github.io/{args.repository}/ontology"
    new_namespace = new_document_iri + "#"
    repository_url = f"https://github.com/{args.username}/{args.repository}"

    files_to_update = [
        root / "ontology" / "eos_stpa_avx.ttl",
        root / "queries" / "delayed_red_light_validation.rq",
        root / "scripts" / "run_validation.py",
        root / "README.md",
    ]

    for path in files_to_update:
        replace_text(path, current_namespace, new_namespace)
        replace_text(path, current_document_iri, new_document_iri)

    # Regenerate validation outputs using the new namespace.
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_validation.py"),
            "--ontology", str(root / "ontology" / "eos_stpa_avx.ttl"),
            "--query", str(root / "queries" / "delayed_red_light_validation.rq"),
            "--outdir", str(root / "results"),
        ],
        check=True,
        cwd=root,
    )

    docs_ontology = root / "docs" / "ontology"
    docs_ontology.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ontology_path, docs_ontology / "eos_stpa_avx.ttl")
    write_docs_page(docs_ontology / "index.html", new_namespace, repository_url)

    print(f"Configured namespace: {new_namespace}")
    print(f"Repository URL: {repository_url}")
    print("Validation outputs regenerated successfully.")


if __name__ == "__main__":
    main()
