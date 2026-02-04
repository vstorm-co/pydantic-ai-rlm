"""
Semantic Search example using pydantic-ai-rlm.

This example demonstrates finding information that requires semantic understanding
rather than simple pattern matching. The RLM approach uses sub-LLM queries to
understand and analyze document content to find the answer.
"""

import random

from dotenv import load_dotenv

from pydantic_ai_rlm import configure_logging, run_rlm_analysis_sync


def generate_semantic_context(num_documents: int = 500) -> tuple[str, str]:
    """
    Generate a context with many document summaries where finding the answer
    requires semantic understanding (LLM query), not just pattern matching.

    Returns the context and the expected answer (company name).
    """
    print(f"Generating semantic context with {num_documents:,} documents...")

    # Template company activities (none mention bankruptcy directly)
    activities = [
        "reported strong quarterly earnings with revenue up {pct}%",
        "announced a new product line targeting enterprise customers",
        "expanded operations into {region} markets",
        "hired {num} new engineers for their R&D department",
        "launched a sustainability initiative to reduce carbon footprint",
        "partnered with {partner} for cloud infrastructure",
        "received {amount}M in Series {round} funding",
        "opened a new headquarters in {city}",
        "released version {ver} of their flagship software",
        "acquired a small startup specializing in {tech}",
    ]

    regions = ["European", "Asian", "Latin American", "African", "Middle Eastern"]
    partners = ["AWS", "Azure", "Google Cloud", "Oracle", "IBM"]
    cities = ["Austin", "Seattle", "Denver", "Miami", "Boston", "Chicago"]
    techs = ["AI", "blockchain", "IoT", "cybersecurity", "data analytics"]
    rounds = ["A", "B", "C", "D"]

    # Generate company names
    prefixes = [
        "Tech",
        "Data",
        "Cloud",
        "Cyber",
        "Net",
        "Info",
        "Digi",
        "Smart",
        "Quantum",
        "Meta",
    ]
    suffixes = [
        "Corp",
        "Systems",
        "Labs",
        "Solutions",
        "Works",
        "Dynamics",
        "Logic",
        "Soft",
        "Ware",
        "Hub",
    ]
    companies = [f"{random.choice(prefixes)}{random.choice(suffixes)}" for _ in range(num_documents)]

    # Pick a random company to be the bankrupt one
    bankrupt_idx = random.randint(100, num_documents - 100)
    bankrupt_company = companies[bankrupt_idx]

    documents = []
    for i, company in enumerate(companies):
        if i == bankrupt_idx:
            # This company went bankrupt - but express it in varied, indirect ways
            bankruptcy_phrases = [
                f"Document {i}: {company} has ceased all operations after failing to secure emergency funding. "
                f"The board voted unanimously to wind down the business, citing insurmountable debt obligations "
                f"and inability to meet payroll. Creditors are expected to receive pennies on the dollar.",
                f"Document {i}: Following months of financial turmoil, {company} filed for Chapter 11 protection "
                f"in Delaware bankruptcy court. The once-promising startup had burned through $50M in venture capital "
                f"before running out of runway. Assets will be liquidated to pay outstanding debts.",
                f"Document {i}: {company}'s spectacular collapse shocked the industry. After defaulting on multiple "
                f"loan covenants and facing a cash crunch, the company shuttered its doors last week. "
                f"Employees were given 24 hours notice before being let go without severance.",
            ]
            documents.append(random.choice(bankruptcy_phrases))
        else:
            # Normal company activity
            activity = random.choice(activities)
            activity = activity.format(
                pct=random.randint(5, 45),
                region=random.choice(regions),
                num=random.randint(20, 200),
                partner=random.choice(partners),
                amount=random.randint(10, 500),
                round=random.choice(rounds),
                city=random.choice(cities),
                ver=f"{random.randint(1, 5)}.{random.randint(0, 9)}",
                tech=random.choice(techs),
            )
            documents.append(f"Document {i}: {company} {activity}.")

    # Shuffle to make it harder
    random.shuffle(documents)

    context = "\n\n".join(documents)
    print(f"Bankrupt company: {bankrupt_company} (was at original index {bankrupt_idx})")

    return context, bankrupt_company


def main():
    """Run the semantic understanding example."""
    load_dotenv()
    configure_logging(enabled=True)

    print("=" * 60)
    print("Semantic Search Example (requires LLM query)")
    print("=" * 60)

    context, bankrupt_company = generate_semantic_context(num_documents=500)

    print(f"\nContext size: {len(context):,} characters")
    print("Running RLM analysis...\n")

    # This query requires understanding what "went bankrupt" means semantically
    # The context uses phrases like "filed for Chapter 11", "ceased operations",
    # "spectacular collapse" - not the word "bankrupt" directly
    query = (
        "Which company in these documents went bankrupt or became insolvent? "
        "You need to find the company that failed financially - look for descriptions "
        "of business failure, shutdown, or insolvency. Return only the company name."
    )

    result = run_rlm_analysis_sync(
        context=context,
        query=query,
        model="openai:gpt-5",
        sub_model="openai:gpt-5-mini",
        grounded=True,
    )

    print(f"\nResult: {result}")
    print(f"Expected: {bankrupt_company}")
    success = bankrupt_company in str(result)
    print(f"Success: {success}")

    return success


if __name__ == "__main__":
    main()
