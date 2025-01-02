import logging
import os
from datetime import datetime
from logging import Logger

import posthog
import pytz
from dotenv import load_dotenv

load_dotenv()


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


class PostHogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.setFormatter(formatter)

    def emit(self, record):
        try:
            event_id = getattr(record, 'event_id', None) or record.getMessage()
            description = self.format(record)  # Use formatter to format the message
            properties = getattr(record, 'properties', {})
            distinct_id = getattr(record, 'distinct_id', 'anonymous')
            event_properties = {
                'description': description,
                **properties
            }
            posthog.capture(
                distinct_id=distinct_id,
                event=event_id,
                properties=event_properties
            )
        except Exception:
            self.handleError(record)


def setup_logger() -> Logger:
    # Clear any existing handlers to avoid conflicts
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)

    # Create formatter and console handler
    formatter = ESTFormatter('%(asctime)s - %(filename)s:%(lineno)d [%(levelname)s] %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add PostHog handler if environment variables are set
    posthog_enabled = False
    if os.environ.get("POSTHOG_KEY") and os.environ.get("POSTHOG_HOST"):
        try:
            posthog.api_key = os.environ["POSTHOG_KEY"]
            posthog.host = os.environ["POSTHOG_HOST"]
            logger.addHandler(PostHogHandler())
            posthog_enabled = True
        except Exception as e:
            logger.warning(f"Failed to initialize PostHog handler: {e}")

    # Attach the posthog_enabled flag to the logger
    logger.posthog_enabled = posthog_enabled
    return logger


# Initialize the shared logger
LOGGER = setup_logger()
