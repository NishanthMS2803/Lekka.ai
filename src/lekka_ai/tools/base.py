"""Base classes and registry for the Lekka AI tool framework."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from lekka_ai.llm.base import ToolSpec


class Tool[InT: BaseModel, OutT: BaseModel](ABC):
    """Abstract base class for all tools with schema validation."""

    name: str
    description: str
    input_model: type[InT]
    output_model: type[OutT]

    @abstractmethod
    async def run(self, input_data: InT) -> OutT:
        """Execute the tool logic using validated input model."""
        ...

    async def execute(self, raw_input: dict[str, Any] | InT) -> OutT:
        """Validate input dictionary against input_model and execute."""
        if isinstance(raw_input, self.input_model):
            validated_input = raw_input
        else:
            validated_input = self.input_model.model_validate(raw_input)
        return await self.run(validated_input)

    async def __call__(self, raw_input: dict[str, Any] | InT) -> OutT:
        """Invoke tool execution directly via call syntax."""
        return await self.execute(raw_input)

    def to_spec(self) -> ToolSpec:
        """Export tool declaration and JSON schema as a ToolSpec."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            json_schema=self.input_model.model_json_schema(),
        )


class ToolRegistry:
    """Registry for managing and looking up available agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any, Any]] = {}

    def register(self, tool: Tool[Any, Any], overwrite: bool = False) -> None:
        """Register a tool instance by its unique name."""
        if tool.name in self._tools and not overwrite:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any, Any]:
        """Retrieve a registered tool by name, raising KeyError if absent."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry.")
        return self._tools[name]

    def list(self) -> list[Tool[Any, Any]]:
        """Return a list of all registered tools."""
        return list(self._tools.values())

    def __contains__(self, name: str) -> bool:
        """Check if tool name is registered."""
        return name in self._tools


default_registry: ToolRegistry = ToolRegistry()
