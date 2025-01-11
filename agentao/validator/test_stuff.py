from pydantic import BaseModel, Field, ValidationError
from dataclasses import dataclass
from typing import List, Callable, Union
from jinja2 import Template
from pathlib import Path
import requests

corcel_key = "..."

from agentao.helpers.classes import FilePair


class GeneratedProblem(BaseModel):
    problem_statement: str
    dynamic_checklist: List[str]


if __name__ == "__main__":
    url = "https://api.corcel.io/v1/text/cortext/chat"

    print(GeneratedProblem.model_json_schema())

    payload = {
        "model": "claude-3-5-sonnet-20240620",
        "stream": False,
        "top_p": 1,
        "temperature": 0.01,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": """Generate a random math problem, and a checklist the solution. The output should be a json formatted like
                {
                    "problem_statement": "",
                    "dynamic_checklist": []
                }
                """
            },
        ]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": corcel_key
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.text)
    print(response.json())
