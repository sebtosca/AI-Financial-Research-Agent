from app.eval.dataset import EVAL_DATASET_VERSION, GOLDEN_EVAL_CASES

_KNOWN_TOOLS = {
    "get_stock_price",
    "get_stock_history",
    "search_financial_news",
    "analyze_sentiment",
}


def test_dataset_version_is_set():
    assert EVAL_DATASET_VERSION == "v1"


def test_dataset_is_non_empty():
    assert len(GOLDEN_EVAL_CASES) >= 8


def test_case_ids_are_unique():
    ids = [case.id for case in GOLDEN_EVAL_CASES]
    assert len(ids) == len(set(ids))


def test_expected_tools_are_known_general_tools():
    for case in GOLDEN_EVAL_CASES:
        assert set(case.expected_tools) <= _KNOWN_TOOLS, case.id


def test_private_rag_cases_have_reference_answers():
    rag_cases = [case for case in GOLDEN_EVAL_CASES if case.category == "private_rag"]
    assert rag_cases
    for case in rag_cases:
        assert case.reference_answer
        assert case.expected_rag_engaged is True


def test_rag_override_case_forces_with_rag_requested_false():
    override_cases = [case for case in GOLDEN_EVAL_CASES if case.category == "rag_override"]
    assert override_cases
    for case in override_cases:
        assert case.with_rag_requested is False
        assert case.expected_rag_engaged is False
