"""
Needle in Haystack example using pydantic-ai-rlm.

This example demonstrates finding a specific pattern (magic number) hidden
in a massive text context. The RLM approach uses code execution to efficiently
search through the data without needing to process it all in the LLM context.
"""

import random

from dotenv import load_dotenv

from pydantic_ai_rlm import configure_logging, run_rlm_analysis_sync


def generate_massive_context(num_lines: int = 1_000_000, answer: str = "1298418") -> str:
    """Generate a massive text context with a hidden magic number."""
    print(f"Generating massive context with {num_lines:,} lines...")

    random_words = [
        "blah",
        "random",
        "text",
        "data",
        "content",
        "information",
        "sample",
    ]

    lines = []
    for _ in range(num_lines):
        num_words = random.randint(3, 8)
        line_words = [random.choice(random_words) for _ in range(num_words)]
        lines.append(" ".join(line_words))

    # Insert the magic number at a random position (somewhere in the middle)
    magic_position = random.randint(400_000, 600_000)
    lines[magic_position] = f"The num that is important is equal to: {answer}"

    print(f"Magic number inserted at position {magic_position:,}")

    return "\n".join(lines)


def main():
    """Run the needle-in-haystack example."""
    load_dotenv()

    configure_logging(enabled=True)

    print("=" * 60)
    print("Needle in Haystack Example (pattern matching)")
    print("=" * 60)

    answer = str(random.randint(1_000_000, 9_999_999))
    context = generate_massive_context(num_lines=1_000_000, answer=answer)

    print(f"\nContext size: {len(context):,} characters")
    print("Running RLM analysis...\n")

    query = "I'm looking for a magic number. What is it? Analyse context!"

    result = run_rlm_analysis_sync(
        context=context,
        query=query,
        model="openai:gpt-5",
        sub_model="openai:gpt-5-mini",
        grounded=True,
    )

    print(f"\nResult: {result}")
    print(f"Expected: {answer}")
    print(f"Success: {result == answer}")

    return result == answer


if __name__ == "__main__":
    main()
