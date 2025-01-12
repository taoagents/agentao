import logging
from dataclasses import dataclass
from pathlib import Path

import ipdb

from agentao.miner.swe_agent_adapter import generate_code_patch

logger = logging.getLogger(__name__)


@dataclass
class UnsolvedIssue:
    desc: str
    local_code_path: Path


@dataclass
class IssueSolution:
    patch: str


sample_issue = """
To create a more versatile color theming capability in this software system, develop a feature that allows users to toggle between different color palettes dynamically and visualize a preview of these palettes. The implementation should include:

1. A function to list all available palettes from `seaborn/palettes.py`.
2. An interface (could be a command-line or a simple GUI) to allow users to select a palette.
3. Visualization of the chosen palette using an HTML display or image output, showing each color in the palette as a rectangular block, similar to the `_repr_html_` method of `_ColorPalette`.
4. This feature should also integrate with the existing theme settings in `seaborn/rcmod.py` so that selecting a palette from your interface updates the current theme settings, affecting the visual appearance of any plots generated thereafter.

Ensure the solution adheres to these steps while maintaining testability and efficient use of resources.
"""

from logging import Logger
logger = Logger("test_miner")

response = generate_code_patch(
    "claude-sonnet-3.5",
    UnsolvedIssue(
        desc=sample_issue,
        local_code_path=Path("../gh-issue-pull/analysis/seaborn")
    ),
    logger=logger,
)
ipdb.set_trace()
