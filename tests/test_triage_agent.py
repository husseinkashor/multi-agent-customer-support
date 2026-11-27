
import sys
from pathlib import Path

# Make the project root available for imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.triage_agent import TriageAgent


def main():
    agent = TriageAgent()

    message = (
        "My order has been delayed for 10 days. "
        "I want a refund."
    )

    result = agent.classify(message)

    print("\nCustomer message:")
    print(message)

    print("\nTriage Agent result:")
    print(result)


if __name__ == "__main__":
    main()
