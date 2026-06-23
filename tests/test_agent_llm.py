from __future__ import annotations

import pytest

from agent_llm import StubLLM, classify_intent, narrate


def test_stub_list_pops_in_order():
    llm = StubLLM(["first", "second"])
    assert llm.complete("sys", "a") == "first"
    assert llm.complete("sys", "b") == "second"
    assert llm.calls == [("sys", "a"), ("sys", "b")]


def test_stub_callable_handler():
    llm = StubLLM(lambda system, prompt: f"{system}|{prompt}")
    assert llm.complete("S", "P") == "S|P"


def test_classify_intent_matches_allowed_label():
    llm = StubLLM(["  COMPARE  "])
    out = classify_intent(llm, "compare the top 3", ["refine", "compare", "done"])
    assert out == "compare"


def test_classify_intent_falls_back_to_default_on_unknown():
    llm = StubLLM(["banana"])
    out = classify_intent(llm, "???", ["refine", "compare"], default="done")
    assert out == "done"


def test_narrate_passes_through():
    llm = StubLLM(["a sentence"])
    assert narrate(llm, "sys", "prompt") == "a sentence"
