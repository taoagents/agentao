from dataclasses import asdict
from agentao.helpers.clients import LogContext
from agentao.validator.graders.abstract_grader import GraderInterface, MinerSubmission
from logging import Logger
from typing import List, Final
import openai
import os
from pydantic import BaseModel
import bert_score

RTC_PROMPT: Final[str] = """
You are an expert code reviewer and descriptor. You will be provided with a
git diff patch, and your job is to generate a problem statement that
describes what the patch is trying to solve.

The code repo that this patch applies to is {repo}.

The following is the git diff patch:
{patch}
"""

class RtcResponseFormat(BaseModel):
    inverse_prompt: str

class RtcGrader(GraderInterface):
    """
    Grader that evaluates a miner's patch based on semantic similarity to the problem statement.
    """
    def __init__(self, logger: Logger):
        self.logger = logger

    def inverse_prompt(self, patch: str, repo: str) -> str:
        OPENAI_CLIENT: Final[openai.Client] = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        inv_prompt = OPENAI_CLIENT.beta.chat.completions.parse(
            model='gpt-4o-mini',
            messages=[
                {"role": "system", "content": RTC_PROMPT.format(patch=patch, repo=repo)},
            ],
            response_format=RtcResponseFormat,
        ).choices[0].message.parsed.inverse_prompt

        return inv_prompt
    
    def grade(self, submissions: List[MinerSubmission], forward_pass_id: str) -> List[float]:
        problem_statements = [s.problem.problem_statement for s in submissions]
        inv_prompts = [self.inverse_prompt(s.solution.patch, s.repo) for s in submissions]

        _, _, F1 = bert_score.score(problem_statements, inv_prompts, lang='en')

        hotkey_grade = [(s.miner_hotkey, f) for s, f in zip(submissions, F1.tolist())]
        for hk, score in hotkey_grade:
            self.logger.info(f"miner {hk} got RTC score of {score}", extra=asdict(LogContext(
                    log_type="lifecycle",
                    event_type="rtc_score",
                    additional_properties={
                        "grade": score, 
                        "miner_hotkey": hk,
                        "forward_pass_id": forward_pass_id,
                    }
                )))

        return F1.tolist()
