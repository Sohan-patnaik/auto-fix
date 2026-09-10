# AutoFix MVP

Clones a repo, indexes it into ChromaDB using NVIDIA NIM embeddings, retrieves
code relevant to a GitHub issue, and asks a NIM-hosted LLM to generate a
unified diff patch + PR description. No execution/test loop yet (Milestone 2).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in NVIDIA_API_KEY from https://build.nvidia.com
```

## Run

```bash
python main.py --repo https://github.com/org/repo.git --issue "Users cannot login after password reset."
```

## What it does
1. `indexer.clone_repo` — shallow clones the target repo.
2. `indexer.build_index` — chunks source files (~80 lines/chunk), embeds each
   chunk with the NIM embedding model (`input_type=passage`), stores in a
   local persistent Chroma collection.
3. `indexer.query_index` — embeds the issue text (`input_type=query`) and
   retrieves the top-k most relevant chunks.
4. `fixer.generate_patch` — sends the issue + retrieved code to the NIM LLM,
   which returns a unified diff.
5. `fixer.draft_pr_description` — asks the LLM to summarize the fix as a PR
   description.

## Not in this MVP (next milestones)
- Applying the patch + running tests in Docker
- Retry loop on test failure (LangGraph)
- Opening the actual PR via GitHub API
- FastAPI wrapper + Supabase run logging

## Files
```
autofix-mvp/
├── autofix/
│   ├── nim_client.py   # OpenAI-compatible client pointed at NVIDIA NIM
│   ├── indexer.py      # clone + chunk + embed + retrieve
│   └── fixer.py        # patch + PR description generation
├── main.py             # CLI entrypoint
├── requirements.txt
└── .env.example
```
