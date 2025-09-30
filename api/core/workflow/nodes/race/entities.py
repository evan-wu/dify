from enum import Enum
from typing import Optional

from core.workflow.nodes.base import BaseNodeData


class RaceStrategy(str, Enum):
    """
    Strategy for racing parallel branches.
    """
    FIRST_COMPLETE = "first_complete"  # Take result from fastest branch
    FASTEST_VALID = "fastest_valid"  # Take first valid (non-error) result
    TIMEOUT_BEST = "timeout_best"  # Wait for timeout, then take best available
    QUALITY_RACE = "quality_race"  # Race with quality scoring


class WinCondition(str, Enum):
    """
    Conditions that determine a 'winning' result.
    """
    ANY_RESULT = "any_result"  # Any non-empty result wins
    NO_ERROR = "no_error"  # Any result without error status
    CUSTOM_VALIDATION = "custom_validation"  # Use custom validation logic


class RaceNodeData(BaseNodeData):
    """
    Race Node Data for competitive parallel execution.
    """

    type: str = "race"
    race_strategy: RaceStrategy = RaceStrategy.FIRST_COMPLETE
    win_condition: WinCondition = WinCondition.ANY_RESULT
    timeout_seconds: Optional[float] = 30.0
    max_winners: int = 1  # How many results to collect before declaring winner(s)

    # Variables to race for
    variables: list[list[str]]

    # Optional custom scoring/validation
    validation_expression: Optional[str] = None  # Custom expression to validate results
    scoring_expression: Optional[str] = None  # Custom expression to score results

    # Failure handling
    fail_on_timeout: bool = False  # Whether to fail if timeout occurs
    fail_on_all_errors: bool = True  # Whether to fail if all branches error
