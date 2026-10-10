from typing import Any

from pydantic import BaseModel


class TransitionMatrixResponse(BaseModel):
    job_id: str
    response_transition_matrix: dict[str, Any]
