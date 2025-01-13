import logging
import os
from datetime import datetime
from logging import Logger
import posthog
import pytz
from dotenv import load_dotenv
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import requests
from agentao.helpers.constants import BASE_DASHBOARD_URL
load_dotenv()

lifecycle_events = {
    "question_generated": ["question_id", "question_text"],
    "miner_submitted": ["question_id", "miner_hotkey", "patch", "response_time"],
    "solution_selected": ["question_id", "grade", "miner_hotkey"]
}

@dataclass
class LogSessionContext:
    actor_id: str
    actor_type: str
    session_id: str # This is generated by the client when it starts, and is used to identify the session in PostHog
    is_mainnet: bool
    log_version: int

    def to_dict(self):
        return asdict(self)

def record_generated_question(
    question_text: str,
    question_id: str,
    submitting_hotkey: str,
    is_mainnet: bool,
    base_url: str = BASE_DASHBOARD_URL
) -> requests.Response:
    endpoint = f"{base_url}/api/trpc/question.recordGeneratedQuestion"
    
    payload = {
        "json": {
            "question_text": question_text,
            "question_id": question_id,
            "submitting_hotkey": submitting_hotkey,
            "is_mainnet": is_mainnet
        }
    }
    
    headers = {
        "Content-Type": "application/json",
    }
    
    requests.post(
        url=endpoint, 
        headers=headers,
        json=payload
    )

def record_solution_selected(
    question_id: str,
    miner_hotkey: str,
    submitting_hotkey: str,
    is_mainnet: bool,
    grade: int,
    base_url: str = BASE_DASHBOARD_URL
) -> requests.Response:
    endpoint = f"{base_url}/api/trpc/question.recordSolutionSelected"
   
    payload = {
        "json": {
            "question_id": question_id,
            "miner_hotkey": miner_hotkey,
            "submitting_hotkey": submitting_hotkey,
            "is_mainnet": is_mainnet,
            "grade": grade
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "agentao-client/1.0"
    }
    
    requests.post(endpoint, json=payload, headers=headers)

def record_miner_submission(
    question_id: str,
    submitting_hotkey: str,
    miner_hotkey: str,
    is_mainnet: bool,
    patch: str,
    response_time: datetime,
    base_url: str = BASE_DASHBOARD_URL
) -> requests.Response:
    try:
        endpoint = f"{base_url}/api/trpc/question.recordMinerSubmitted"

        payload = {
            "json": {
                "question_id": question_id,
                "miner_hotkey": miner_hotkey,
                "submitting_hotkey": submitting_hotkey,
                "is_mainnet": is_mainnet,
                "patch": patch,
            }
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "agentao-client/1.0"
        }

        requests.post(endpoint, json=payload, headers=headers)
    except Exception as e:
        print(f"Failed to record miner submission: {e}")

def validate_lifecycle_event(event_type: str, properties: Dict[str, Any]) -> bool:
    required_properties = lifecycle_events.get(event_type)
    
    return required_properties is not None and all(prop in properties for prop in required_properties)

@dataclass
class LogContext:
    log_type: str # internal or lifecycle
    event_type: str # can be anything. this is the event id recorded. if this is a lifecycle log it must be one of question_generated, miner_submitted, or solution_selected
    flush_posthog: bool = False
    additional_properties: Optional[Dict[Any, Any]] = None

    def __post_init__(self):
        if self.log_type not in ["lifecycle", "internal"]:
            raise ValueError(f"Invalid log type: {self.log_type}. Must be one of internal or lifecycle")
        
        if self.log_type == "lifecycle" and (self.event_type not in lifecycle_events.keys() or not validate_lifecycle_event(self.event_type, self.additional_properties)):
            raise ValueError(f"Properties do not match the expected format for the event type {self.event_type}")

        return True
    
    def to_dict(self):
        return asdict(self)

class ESTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        est = pytz.timezone("America/New_York")
        ct = datetime.fromtimestamp(record.created, est)
        return ct.strftime("%Y-%m-%d %H:%M:%S")

    def format(self, record):
        # Pad the level name to 5 characters
        record.levelname = f"{record.levelname:<5}"
        return super().format(record)

formatter = ESTFormatter('%(asctime)s - %(filename)s:%(lineno)d [%(levelname)s] %(message)s')

# Get all built-in LogRecord attributes by creating a dummy record and getting its __dict__ keys
LOG_RECORD_BUILTIN_ATTRS = list(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

class AgentaoHandler(logging.Handler):
    def __init__(self, context: LogSessionContext):
        super().__init__()
        self.setFormatter(formatter)
        self._context = context

        self._posthog_enabled = False

        if os.environ.get("POSTHOG_KEY") and os.environ.get("POSTHOG_HOST"):
            try:
                posthog.api_key = os.environ["POSTHOG_KEY"]
                posthog.host = os.environ["POSTHOG_HOST"]
                self._posthog_enabled = True
            except Exception as e:
                print(f"Failed to initialize PostHog handler: {e}")
                self._posthog_enabled = False
    
    def emit(self, record):
        if self._posthog_enabled == False:
            return 
        
        try:
            properties = {}

            for key, val in record.__dict__.items():
                if key not in LOG_RECORD_BUILTIN_ATTRS:
                    properties[key] = val

            # Unwrap additional_properties if they exist
            if 'additional_properties' in properties:
                additional_props = properties.pop('additional_properties')
                if additional_props:
                    properties.update(additional_props)

            formatted_properties = {
                "description": record.message,
                **self._context.to_dict(),
                **properties
            }

            # If its a simple message log, just send it directly with only actor context
            if len(properties.keys()) == 0:
                if self._posthog_enabled:
                    posthog.capture(
                        distinct_id=self._context.actor_id,
                        event=record.message,
                        properties=formatted_properties
                    )
            
            log_type = properties.get("log_type")
            event_type = properties.get("event_type")
            flush_posthog_value = properties.get("flush_posthog") or False

            if log_type == "lifecycle":
                if not event_type or event_type not in lifecycle_events.keys():
                    raise ValueError(f"Invalid event type: {properties.get('event_type') or ''}")
                
                if not self._context.actor_type == "validator":
                    raise PermissionError("Only validators can post lifecycle events.")
                
                if self._posthog_enabled:
                    posthog.capture(
                        distinct_id=self._context.actor_id,
                        event=event_type,
                        properties=formatted_properties
                    )
                
                if event_type == "question_generated":
                    record_generated_question(
                        question_text=formatted_properties.get("question_text"),
                        question_id=formatted_properties.get("question_id"),
                        submitting_hotkey=self._context.actor_id,
                        is_mainnet=self._context.is_mainnet
                    )
                
                elif event_type == "miner_submitted":
                    record_miner_submission(
                        question_id=formatted_properties.get("question_id"),
                        submitting_hotkey=self._context.actor_id,
                        miner_hotkey=formatted_properties.get("miner_hotkey"),
                        is_mainnet=self._context.is_mainnet,
                        patch=formatted_properties.get("patch"),
                        response_time=formatted_properties.get("response_time")
                    )
                
                elif event_type == "solution_selected":
                    record_solution_selected(
                        question_id=formatted_properties.get("question_id"),
                        miner_hotkey=formatted_properties.get("miner_hotkey"),
                        submitting_hotkey=self._context.actor_id,
                        is_mainnet=self._context.is_mainnet,
                        grade=formatted_properties.get("grade")
                    )

            else:
                if self._posthog_enabled:
                    posthog.capture(
                        distinct_id=self._context.actor_id,
                        event=event_type or record.message,
                        properties=formatted_properties
                    )

            if flush_posthog_value == True:
                posthog.flush()
        except Exception:
            self.handleError(record)

def setup_logger(logger_name: str, log_session_context: LogSessionContext) -> Logger:
    logger = logging.getLogger(logger_name)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(AgentaoHandler(context=log_session_context))

    return logger

if __name__ == "__main__":
    import uuid 

    test_uuid = str(uuid.uuid4())
    log_session_context = LogSessionContext(
        actor_id="1289",
        actor_type="validator",
        session_id="283823990239",
        is_mainnet=False,
        log_version=7,
    )

    example_internal_log = LogContext(
        log_type="internal",
        event_type="this is some other event type",
        flush_posthog=True,
        additional_properties={
            "patch", "this is an example patch",
        }
    )

    example_lifecycle_log = LogContext(
        log_type="lifecycle",
        event_type="question_generated",
        additional_properties={"question_text": "new question generated!", "question_id": test_uuid}
    )

    my_logger: Logger = setup_logger(logger_name="validator_n", log_session_context=log_session_context)
    my_logger.info("internal test 1", extra=asdict(example_internal_log))
    my_logger.info("lifecycle test 1", extra=asdict(example_lifecycle_log))


    from pprint import pprint
    pprint(asdict(example_internal_log), indent=2)
    print('+++++++')
    pprint(asdict(example_lifecycle_log), indent=2)
    # query_all_logs()