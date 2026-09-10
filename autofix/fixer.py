"""Turns an issue + retrieved code context into a proposed unified diff patch."""
from . import nim_client

SYSTEM_PROMPT = """You are a senior software engineer fixing a reported bug.
You will be given a GitHub issue and relevant code chunks retrieved from the repo.

Respond with ONLY a unified diff patch (git diff format, starting with `--- a/<path>` / `+++ b/<path>`)
that fixes the issue. Keep the change minimal and scoped to the actual bug.
Do not include explanations, markdown fences, or any text outside the diff.
If you cannot determine a fix from the given context, respond with exactly: NO_FIX_FOUND
"""


def format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"### {c['filepath']} (lines {c['start_line']}-{c['end_line']})\n```\n{c['text']}\n```"
        )
    return "\n\n".join(parts)


def generate_patch(issue_text: str, chunks: list[dict]) -> str:
    context = format_context(chunks)
    user_prompt = f"ISSUE:\n{issue_text}\n\nRELEVANT CODE:\n{context}\n\nProduce the diff now."
    return nim_client.chat(SYSTEM_PROMPT, user_prompt, temperature=0.1).strip()


def draft_pr_description(issue_text: str, patch: str) -> str:
    system = "You write concise, professional GitHub PR descriptions."
    user = (
        f"Issue:\n{issue_text}\n\nPatch:\n{patch}\n\n"
        "Write a PR description with sections: ## Problem, ## Root Cause, ## Fix, ## Testing. "
        "Keep it under 200 words."
    )
    return nim_client.chat(system, user, temperature=0.3).strip()
