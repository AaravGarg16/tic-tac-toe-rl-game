"""Request and response models for the API."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Mark = Literal["X", "O"]
Difficulty = Literal["easy", "hard"]
Status = Literal["in_progress", "X", "O", "draw"]


class MoveRequest(BaseModel):
    board: List[str] = Field(
        ...,
        min_length=9,
        max_length=9,
        description="Nine cells, each '', 'X' or 'O', in reading order.",
    )
    ai_mark: Mark = Field(..., description="The mark the agent plays.")
    difficulty: Difficulty = Field("hard", description="'easy' = Q-learning, 'hard' = DQN.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"board": ["X", "", "", "", "", "", "", "", ""], "ai_mark": "O", "difficulty": "hard"}
            ]
        }
    }


class MoveResponse(BaseModel):
    move: int = Field(..., description="Cell index the agent played, 0-8.")
    board: List[str] = Field(..., description="The board after the agent's move.")
    status: Status
    winning_line: Optional[List[int]] = Field(
        None, description="The three cells that won, if the game just ended."
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "loading"]
    agents_loaded: bool
