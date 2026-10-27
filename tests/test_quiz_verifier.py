"""Quiz verifier, with a fake client (no API key needed)."""

from types import SimpleNamespace

import pytest

from study_rag.llm import RefusalError
from study_rag.quiz import QuizQuestion
from study_rag.quiz.verifier import Review, verify_question


def question() -> QuizQuestion:
    return QuizQuestion(
        question="What does dropout do?",
        options=["Prevents co-adaptation", "Adds layers", "Raises the learning rate", "Nothing"],
        correct_index=0,
        explanation="Dropout stops units from co-adapting.",
        excerpt=1,
    )


class FakeVerifierMessages:
    def __init__(self, review, stop_reason="end_turn"):
        self.review, self.stop_reason, self.kwargs = review, stop_reason, {}

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            parsed_output=self.review, stop_reason=self.stop_reason, stop_details=None,
            content=[], usage=None,
        )


def test_verify_uses_structured_output_and_sends_question_and_excerpt():
    review = Review(ok=False, reasons=["B is also correct"])
    msgs = FakeVerifierMessages(review)
    out = verify_question(
        SimpleNamespace(messages=msgs), "m", question(), "Dropout randomly zeroes units."
    )
    assert out == review
    assert msgs.kwargs["output_format"] is Review
    sent = str(msgs.kwargs["messages"])
    assert "What does dropout do?" in sent and "Dropout randomly zeroes units." in sent
    assert "Prevents co-adaptation" in sent  # the options are what gets judged
    assert msgs.kwargs.get("system")


def test_verify_raises_on_refusal():
    client = SimpleNamespace(messages=FakeVerifierMessages(Review(ok=True, reasons=[]), "refusal"))
    with pytest.raises(RefusalError):
        verify_question(client, "m", question(), "text")
