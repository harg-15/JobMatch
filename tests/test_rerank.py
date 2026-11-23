import json

import pytest

from app.rerank import RerankError, build_user_prompt, parse_response, rerank

JOBS = [
    {"id": 1, "title": "Backend Intern", "company": "A", "location": "Gurugram",
     "description": "Java Spring Boot"},
    {"id": 2, "title": "ML Intern", "company": "B", "location": None,
     "description": "Python PyTorch"},
    {"id": 3, "title": "Sales Intern", "company": "C", "location": "Delhi",
     "description": "cold calling"},
]


def fake_llm(payload):
    calls = []

    def llm(system, user):
        calls.append(user)
        return payload if isinstance(payload, str) else json.dumps(payload)

    llm.calls = calls
    return llm


def test_rerank_sorts_by_score_and_keeps_reasons():
    llm = fake_llm({"matches": [
        {"job_id": 1, "score": 60, "reason": "Spring Boot matches the loan project"},
        {"job_id": 2, "score": 90, "reason": "PyTorch matches the BERT project"},
    ]})
    out = rerank(llm, "resume text", JOBS, top_k=5)
    assert [j["id"] for j in out] == [2, 1]
    assert out[0]["reason"] == "PyTorch matches the BERT project"
    assert out[0]["company"] == "B"  # original job fields are kept


def test_made_up_and_duplicate_ids_are_dropped():
    raw = json.dumps({"matches": [
        {"job_id": 99, "score": 100, "reason": "x"},
        {"job_id": 1, "score": 80, "reason": "y"},
        {"job_id": 1, "score": 70, "reason": "z"},
    ]})
    matches = parse_response(raw, allowed_ids={1, 2, 3})
    assert [(m.job_id, m.reason) for m in matches] == [(1, "y")]


def test_json_in_code_fence_is_accepted():
    raw = '```json\n{"matches": [{"job_id": 3, "score": 10, "reason": "no overlap"}]}\n```'
    assert parse_response(raw, {3})[0].job_id == 3


def test_bad_output_is_retried_then_raises():
    llm = fake_llm("sorry, I can't do that")
    with pytest.raises(RerankError):
        rerank(llm, "resume", JOBS, retries=1)
    assert len(llm.calls) == 2


def test_score_out_of_range_is_rejected():
    with pytest.raises(RerankError):
        parse_response('{"matches": [{"job_id": 1, "score": 150, "reason": "x"}]}', {1})


def test_prompt_contains_wants_and_job_ids():
    prompt = build_user_prompt("my resume", "remote backend role", JOBS)
    assert "remote backend role" in prompt
    assert "job_id: 1" in prompt and "job_id: 3" in prompt
    assert "location not given" in prompt
