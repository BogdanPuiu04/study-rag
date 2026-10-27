from __future__ import annotations

from pydantic import BaseModel


class QuizQuestion(BaseModel):
    """One multiple-choice question. Also the schema Claude must fill (structured output),
    so every field is plain: no validators the model's output could trip over."""

    question: str
    options: list[str]  # exactly four, in display order
    correct_index: int  # 0-based index into `options`
    explanation: str  # why the correct option is right, in terms of the excerpt
    excerpt: int  # 1-based number of the excerpt the question is built from


class Quiz(BaseModel):
    questions: list[QuizQuestion]
