from multiprocessing import Process
from mmlu import EvaluateMMLU
from base import initialize_session
from chat.api import app
import uvicorn
import time
import atexit
import logging
import warnings
from sqlalchemy.exc import SAWarning
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Disable logging for httpx
logging.getLogger("httpx").disabled = True

# Suppress all SAWarnings
warnings.filterwarnings("ignore", category=SAWarning)

# Disable logging globally
logging.disable(logging.CRITICAL)


def run_api():
    """Run the API using uvicorn."""
    uvicorn.run(app, host="localhost", port=8000, log_level="critical")


def main():
    """Main function to evaluate MMLU."""
    from examples import (
        COTAgentSystem,
        DebateAgentSystem,
        DynamicRolesAgentSystem,
        QDAgentSystem,
        ReflexionAgentSystem,
        SelfConsistencyAgentSystem,
        StepBackAgentSystem,
    )

    systems = [
        COTAgentSystem,
        DebateAgentSystem,
        DynamicRolesAgentSystem,
        QDAgentSystem,
        ReflexionAgentSystem,
        SelfConsistencyAgentSystem,
        StepBackAgentSystem,
    ]

    # Evaluate accuracy
    e = EvaluateMMLU()

    # accuracy = e.evaluate(COTAgentSystem, limit=10)

    accuracy = e.evaluate_multiple(systems, limit=10)


if __name__ == "__main__":

    # Check if the OpenAI key is set in the .env file
    if not (openai_key := os.getenv("OPENAI_API_KEY")):
        raise ValueError("Please set the OPENAI_API_KEY in the .env file.")

    # Start the API in a separate process
    api_process = Process(target=run_api)
    api_process.start()

    # Ensure the API process is terminated when the script exits
    def cleanup():
        print("Shutting down API process...")
        api_process.terminate()
        api_process.join()

    atexit.register(cleanup)

    # Allow API to initialize before running main
    time.sleep(2)

    # Run the main function
    main()
