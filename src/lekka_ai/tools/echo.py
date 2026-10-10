"""Example EchoTool implementation for testing and demonstrations."""

from pydantic import BaseModel, Field

from lekka_ai.tools.base import Tool


class EchoInput(BaseModel):
    """Input payload for EchoTool."""

    message: str = Field(..., description="Message string to echo back")


class EchoOutput(BaseModel):
    """Output payload for EchoTool."""

    echoed: str = Field(..., description="The echoed message string")


class EchoTool(Tool[EchoInput, EchoOutput]):
    """Tool that returns the input message unchanged."""

    name: str = "echo"
    description: str = "Echo back the provided message string."
    input_model: type[EchoInput] = EchoInput
    output_model: type[EchoOutput] = EchoOutput

    async def run(self, input_data: EchoInput) -> EchoOutput:
        """Execute echo tool and return EchoOutput."""
        return EchoOutput(echoed=input_data.message)
