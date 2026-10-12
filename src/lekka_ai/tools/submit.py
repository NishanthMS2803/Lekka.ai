"""Built-in tool for submitting structured agent answers and concluding tasks."""

from __future__ import annotations

from pydantic import BaseModel, Field

from lekka_ai.schemas.submit import SubmitAnswer
from lekka_ai.tools.base import Tool, default_registry


class SubmitAnswerOutput(BaseModel):
    """Result of submitting an answer through submit_answer."""

    accepted: bool = Field(..., description="Whether submission was accepted")
    message: str = Field(..., description="Status explanation or validation feedback")
    status: str | None = Field(default=None, description="Task resolution status")


class SubmitAnswerTool(Tool[SubmitAnswer, SubmitAnswerOutput]):
    """Built-in tool for submitting final answer and grounded claims."""

    name = "submit_answer"
    description = (
        "Submit the final structured answer, grounded claims, confidence, "
        "and assumptions to conclude the task."
    )
    input_model = SubmitAnswer
    output_model = SubmitAnswerOutput

    async def run(self, input_data: SubmitAnswer) -> SubmitAnswerOutput:
        """Process validated answer submission."""
        return SubmitAnswerOutput(
            accepted=True,
            message="Answer submission accepted and validated.",
            status=str(input_data.status),
        )


default_submit_answer_tool = SubmitAnswerTool()
default_registry.register(default_submit_answer_tool, overwrite=True)
