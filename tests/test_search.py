from app.search import Filters, keyword_search, vector_search


def test_vector_search_finds_relevant_job(loaded, embedder):
    q = embedder.embed_query("android kotlin jetpack compose play store app")
    results = vector_search(loaded, q, limit=3)
    assert results[0]["external_id"] == "sj-009"
    # sorted by similarity, best first
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


def test_exclude_filter_removes_jobs(loaded, embedder):
    q = embedder.embed_query("sales business development internship delhi")
    results = vector_search(loaded, q, limit=30, filters=Filters(exclude_terms=["unpaid"]))
    assert results
    for r in results:
        assert "unpaid" not in (r["title"] + r["description"]).lower()


def test_exclude_terms_are_not_treated_as_like_patterns(loaded, embedder):
    # '%' should be matched literally, not as "anything"
    q = embedder.embed_query("backend")
    results = vector_search(loaded, q, limit=30, filters=Filters(exclude_terms=["%"]))
    assert len(results) == 30


def test_location_filter(loaded, embedder):
    q = embedder.embed_query("backend intern")
    results = vector_search(loaded, q, limit=30, filters=Filters(location="gurugram"))
    assert results
    assert all("Gurugram" in r["location"] for r in results)


def test_keyword_search_with_long_text(loaded):
    resume = "I know React, TypeScript and CSS. Built a chat app with Node.js and MongoDB."
    results = keyword_search(loaded, resume, limit=5)
    ids = [r["external_id"] for r in results]
    assert "sj-007" in ids or "sj-013" in ids


def test_keyword_search_with_no_real_words(loaded):
    assert keyword_search(loaded, "a an the of", limit=5) == []
