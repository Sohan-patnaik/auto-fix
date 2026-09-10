"""
AutoFix MVP — Milestone 1 (no execution/testing loop yet):
  clone -> chunk -> embed (NVIDIA NIM) -> localize -> generate diff -> print

Usage:
    python main.py --repo https://github.com/org/repo.git --issue "Users cannot login after password reset."
"""
import argparse
import os
from dotenv import load_dotenv

from autofix import indexer, fixer

load_dotenv()


def run(repo_url: str, issue_text: str, workdir: str = "./_workdir"):
    print(f"[1/4] Cloning {repo_url} ...")
    repo_path = indexer.clone_repo(repo_url, workdir)

    print("[2/4] Indexing repo (chunking + embedding via NVIDIA NIM) ...")
    collection, n_chunks = indexer.build_index(repo_path, collection_name="autofix_repo")
    print(f"      indexed {n_chunks} chunks")

    print("[3/4] Localizing relevant code for the issue ...")
    hits = indexer.query_index(collection, issue_text, k=8)
    for h in hits:
        print(f"      - {h['filepath']} (lines {h['start_line']}-{h['end_line']})")

    print("[4/4] Generating patch ...")
    patch = fixer.generate_patch(issue_text, hits)

    print("\n" + "=" * 60)
    print("PROPOSED PATCH")
    print("=" * 60)
    print(patch)

    if patch != "NO_FIX_FOUND":
        print("\n" + "=" * 60)
        print("DRAFT PR DESCRIPTION")
        print("=" * 60)
        print(fixer.draft_pr_description(issue_text, patch))

    return patch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoFix MVP")
    parser.add_argument("--repo", required=True, help="Git repo URL to clone")
    parser.add_argument("--issue", required=True, help="Issue text to fix")
    args = parser.parse_args()
    run(args.repo, args.issue)
