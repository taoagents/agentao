from abc import ABC
from typing import List

from pydantic import BaseModel

from agentao.helpers.classes import GeneratedProblemStatement, IssueSolution


class MinerSubmission(BaseModel):
    repo: str
    problem: GeneratedProblemStatement
    solution: IssueSolution
    miner_hotkey: str


class GraderInterface(ABC):
    def grade(self, submissions: List[MinerSubmission]) -> List[float]:
        raise NotImplementedError("AbstractGrader.grade() must be overridden")

