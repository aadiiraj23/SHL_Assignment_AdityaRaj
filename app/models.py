from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Assessment(BaseModel):
    name: str
    url: str
    test_type: str = Field(..., min_length=1, max_length=1)
    description: str = ""
    remote_testing: bool = False
    adaptive: bool = False
    duration_minutes: Optional[int] = None
    languages: list[str] = Field(default_factory=list)

    @field_validator("test_type")
    @classmethod
    def validate_test_type(cls, value: str) -> str:
        allowed = {"A", "P", "K", "B", "S", "C"}
        if value not in allowed:
            raise ValueError("test_type must be one of A, P, K, B, S, C")
        return value


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=16)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": "I am hiring a mid-level Java developer.",
                    },
                    {
                        "role": "assistant",
                        "content": "What skills and seniority are required?",
                    },
                    {
                        "role": "user",
                        "content": "About 4 years, problem solving and communication.",
                    },
                ]
            }
        }
    )

    @model_validator(mode="after")
    def validate_starts_with_user(self) -> "ChatRequest":
        if self.messages and self.messages[0].role != "user":
            raise ValueError("messages must start with a user role")
        return self


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str = ""
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=10)
    end_of_conversation: bool = False

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reply": "Here are a few assessments that match your role.",
                "recommendations": [
                    {
                        "name": "Verify - Numerical Reasoning",
                        "url": "https://www.shl.com/solutions/products/verify-numerical-reasoning/",
                        "test_type": "A",
                    },
                    {
                        "name": "OPQ32r",
                        "url": "https://www.shl.com/solutions/products/opq32r/",
                        "test_type": "P",
                    },
                ],
                "end_of_conversation": False,
            }
        }
    )

    @field_validator("recommendations")
    @classmethod
    def validate_recommendations_length(
        cls, value: list[Recommendation]
    ) -> list[Recommendation]:
        if len(value) > 10:
            raise ValueError("recommendations cannot exceed 10 items")
        return value
