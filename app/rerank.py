"""Second stage: ask an LLM to re-rank the vector search candidates and
explain each match in one sentence.

Embeddings are good at "these texts are about similar things" but bad at
things like seniority ("needs 5+ years" vs a student) or what the person said
they want. The LLM sees only ~20 shortlisted jobs, so it stays cheap.
"""
import json
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from app.config import settings

# (system_prompt, user_prompt) -> raw text response
LLMCall = Callable[[str, str], str]

MAX_RESUME_CHARS = 6000
MAX_JOB_CHARS = 800

SYSTEM_PROMPT = """You match candidates to jobs.
You get a resume, optionally what the candidate is looking for, and a list of jobs.
Score how good each job is for this candidate from 0 to 100. Consider:
- skills and tech stack overlap
- seniority: a student or fresher is a poor fit for roles needing years of experience
- what the candidate says they want

Return only JSON in this exact shape:
{"matches": [{"job_id": <int>, "score": <int 0-100>, "reason": "<one sentence>"}]}

Rules:
- only use job_id values from the list you were given
- the reason must mention something specific from both the resume and the job
- return the best matches first"""


class Match(BaseModel):
    job_id: int
    score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1)


class RerankResponse(BaseModel):
    matches: list[Match]


class RerankError(RuntimeError):
    pass


def build_user_prompt(resume_text: str, wants: str | None, jobs: list[dict]) -> str:
    lines = ["RESUME:", resume_text[:MAX_RESUME_CHARS], ""]
    if wants:
        lines += ["CANDIDATE IS LOOKING FOR:", wants, ""]
    lines.append("JOBS:")
    for job in jobs:
        desc = " ".join(job["description"].split())[:MAX_JOB_CHARS]
        lines.append(
            f"- job_id: {job['id']} | {job['title']} at {job['company']}"
            f" ({job.get('location') or 'location not given'})\n  {desc}"
        )
    return "\n".join(lines)


def parse_response(raw: str, allowed_ids: set[int]) -> list[Match]:
    raw = raw.strip()
    # some models wrap JSON in ```json fences even when asked not to
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()

    try:
        parsed = RerankResponse.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RerankError(f"bad LLM output: {exc}") from exc

    # drop ids the model made up, and duplicates
    seen, matches = set(), []
    for m in parsed.matches:
        if m.job_id in allowed_ids and m.job_id not in seen:
            seen.add(m.job_id)
            matches.append(m)
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches


def rerank(
    llm: LLMCall,
    resume_text: str,
    jobs: list[dict],
    wants: str | None = None,
    top_k: int = 5,
    retries: int = 1,
) -> list[dict]:
    """Returns the top_k jobs with 'score' and 'reason' added."""
    if not jobs:
        return []

    by_id = {job["id"]: job for job in jobs}
    prompt = build_user_prompt(resume_text, wants, jobs)

    last_error = None
    for _ in range(retries + 1):
        try:
            matches = parse_response(llm(SYSTEM_PROMPT, prompt), set(by_id))
            break
        except RerankError as exc:
            last_error = exc
    else:
        raise last_error

    if not matches:
        raise RerankError("LLM returned no usable matches")

    return [
        {**by_id[m.job_id], "score": m.score, "reason": m.reason}
        for m in matches[:top_k]
    ]


def openai_compatible_llm() -> LLMCall | None:
    """Returns an LLM function, or None if no API key is configured."""
    if not (settings.llm_api_key and settings.llm_model):
        return None

    from openai import OpenAI

    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or None,
        timeout=settings.llm_timeout,
    )

    def call(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return resp.choices[0].message.content or ""

    return call
