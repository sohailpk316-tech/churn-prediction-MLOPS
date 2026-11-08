from pydantic import BaseModel, Field
from typing import Literal
class StudentProfile(BaseModel):
    gender: Literal['Male','Female']
    age: int = Field(ge=15, le=60)
    course: str = Field(min_length=1, max_length=100)
    year_of_study: str
    cgpa: str
    marital_status: Literal['Yes','No']
    anxiety: Literal['Yes','No']
    panic_attack: Literal['Yes','No']
    sought_treatment: Literal['Yes','No']
