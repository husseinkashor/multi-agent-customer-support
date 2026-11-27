
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_client import LLMClient


def main():
    llm = LLMClient()

    answer = llm.generate(
        prompt="A customer's order is 10 days late. Write a helpful response.",
        system_message=(
            "You are the Response Agent in a multi-agent "
            "customer support system."
        ),
    )

    print("\nResponse Agent output:")
    print(answer)


if __name__ == "__main__":
    main()
