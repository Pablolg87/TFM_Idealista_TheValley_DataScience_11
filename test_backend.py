"""Backend integration check for the Smart Advisor MVP.

This script verifies that the backend modules can work together without
Streamlit or generative AI. It exercises the agent router against advisor.py
using a real row from the configured dataset.
"""

from pprint import pprint
from typing import Any

import advisor
from agent import SmartAdvisorAgent


def _print_section(title: str) -> None:
    """Print a readable console section title."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _get_sample_property() -> dict[str, Any]:
    """Read one valid property from the dataset for integration checks."""
    dataset = advisor.load_dataset()
    required_columns = [
        advisor.NEIGHBORHOOD_COLUMN,
        advisor.PRICE_COLUMN,
    ]
    missing_columns = [
        column for column in required_columns if column not in dataset.columns
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing}")

    sample = dataset.dropna(subset=required_columns).iloc[0]

    return {
        "neighborhood": str(sample[advisor.NEIGHBORHOOD_COLUMN]),
        "property_price": float(sample[advisor.PRICE_COLUMN]),
    }


def _print_response(title: str, response: dict[str, Any]) -> None:
    """Print a complete backend response."""
    _print_section(title)
    pprint(response, sort_dicts=False)


def main() -> None:
    """Run backend integration checks through SmartAdvisorAgent."""
    try:
        agent = SmartAdvisorAgent()
        sample_property = _get_sample_property()

        stats_response = agent.process_request(
            {
                "intent": "neighborhood_stats",
                "neighborhood": sample_property["neighborhood"],
            }
        )
        _print_response("Neighborhood statistics response", stats_response)

        comparison_response = agent.process_request(
            {
                "intent": "property_comparison",
                "neighborhood": sample_property["neighborhood"],
                "property_price": sample_property["property_price"],
            }
        )
        _print_response("Property comparison response", comparison_response)

    except Exception as exc:
        _print_section("Backend integration check failed")
        print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
