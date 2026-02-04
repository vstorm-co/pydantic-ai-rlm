"""Pydantic models for structured RLM outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GroundedResponse(BaseModel):
    """A response with citation markers mapping to exact quotes from source documents.

    Example:
        ```python
        response = GroundedResponse(
            info="Revenue grew [1] driven by expansion [2]", grounding={"1": "increased by 45%", "2": "new markets in Asia"}
        )
        ```
    """

    info: str = Field(description="Response text with citation markers like [1]")
    grounding: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from citation markers to exact quotes from the source",
    )
