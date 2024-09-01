from typing import Optional
from pydantic import BaseModel

class QuizzifyArgs(BaseModel):
    topic: str
    n_questions: int
    file_url: str
    file_type: str
    lang: Optional[str] = "en"
    question_type: str