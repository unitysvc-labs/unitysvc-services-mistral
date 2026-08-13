#!/usr/bin/env python3
"""
Template-based update_services.py for Mistral AI.

Yields model dictionaries that are rendered using Jinja2 templates.

Usage: python scripts/update_services.py
"""

import os
import sys
from pathlib import Path
from typing import Iterator

import httpx

from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup
from unitysvc_sellers.params_render import write_params_from_iterator

# Provider Configuration
PROVIDER_NAME = "mistral"
PROVIDER_DISPLAY_NAME = "Mistral AI"
API_BASE_URL = "https://api.mistral.ai/v1"
ENV_API_KEY_NAME = "MISTRAL_API_KEY"

SCRIPT_DIR = Path(__file__).parent

# Model families advertised by /models that are NOT served by
# /v1/chat/completions — calling them there returns
# {"message": "Invalid model: <id>", "code": "1500"} with HTTP 400, so every
# rendered code example and connectivity test fails and the service is rejected.
#
# They need their own endpoints (/v1/ocr, the audio/realtime APIs, moderation),
# for which the platform has neither a service_type nor connectivity/code-example
# presets — tracked in unitysvc/unitysvc#1781, the same gap that made us skip
# Groq's non-chat modalities. Skip them here until that lands; the embedding and
# rerank families are handled by _determine_service_type and stay.
SKIP_SUBSTRINGS = (
    "ocr",         # /v1/ocr — document understanding
    "moderation",  # moderation API
    "tts",         # speech synthesis
    "transcribe",  # speech-to-text
    "realtime",    # streaming audio sessions
    # Voxtral *mini* is audio-understanding only and 400s on chat; voxtral-small
    # DOES serve chat completions (verified against the live API), so match the
    # family prefix rather than "voxtral".
    "voxtral-mini",
)


class ModelSource:
    """Fetches models and yields template dictionaries."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.data_fetcher = ModelDataFetcher()
        self.litellm_data = None

    def iter_models(self) -> Iterator[dict]:
        """Yield model dictionaries for template rendering."""
        # Fetch LiteLLM data once
        self.litellm_data = self.data_fetcher.fetch_litellm_model_data()

        print(f"Fetching models from {PROVIDER_DISPLAY_NAME} API...")
        try:
            r = httpx.get(
                f"{API_BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
            r.raise_for_status()
            models = r.json().get("data", [])
            print(f"Found {len(models)} models\n")
        except Exception as e:
            print(f"Error listing models: {e}")
            return

        skipped = []
        for i, model_info in enumerate(models, 1):
            model_id = model_info.get("id", "")
            print(f"[{i}/{len(models)}] {model_id}")

            reason = next(
                (kw for kw in SKIP_SUBSTRINGS if kw in model_id.lower()), None
            )
            if reason:
                skipped.append(model_id)
                print(f"  SKIP (not a chat-completions model: {reason}) — see unitysvc#1781")
                continue

            # Build template variables
            template_vars = self._build_template_vars(model_id, model_info)
            if template_vars:
                yield template_vars
                print("  OK")

        if skipped:
            print(f"\nSkipped {len(skipped)} non-chat model(s): {', '.join(skipped)}")

    def _build_template_vars(self, model_id: str, model_info: dict) -> dict:
        """Build template variables for a model."""
        service_type = self._determine_service_type(model_id)
        display_name = model_id.replace("-", " ").replace("_", " ").title()

        # Build details from LiteLLM data and model info
        details = {}
        model_data = ModelDataLookup.lookup_model_details(
            model_id, self.litellm_data or {})

        if model_data:
            for field in [
                    "max_tokens", "max_input_tokens", "max_output_tokens",
                    "mode"
            ]:
                if field in model_data:
                    details[field] = model_data[field]
            if "litellm_provider" in model_data:
                details["litellm_provider"] = model_data["litellm_provider"]

        if "owned_by" in model_info:
            details["owned_by"] = model_info["owned_by"]
        if "object" in model_info:
            details["object"] = model_info["object"]

        # Canonical (snake_case) metadata required by the platform validator
        # for LLM offerings.  Both keys must be present; null asserts
        # "unknown".  Closed-source models leave parameter_count null per
        # the canonical helper.  metadata_sources records provenance so
        # reviewers can triage stale-value reports.
        canonical = ModelDataLookup.get_canonical_metadata(
            model_id,
            fetcher=self.data_fetcher,
        )
        details["context_length"] = canonical["context_length"]
        details["parameter_count"] = canonical["parameter_count"]
        if canonical["sources"]:
            details["metadata_sources"] = canonical["sources"]

        # Extract upstream pricing for description, but set prices to 0 for BYOK
        pricing = None
        if model_data:
            if "input_cost_per_token" in model_data and "output_cost_per_token" in model_data:
                input_price = round(float(
                    model_data["input_cost_per_token"]) * 1_000_000, 4)
                output_price = round(float(
                    model_data["output_cost_per_token"]) * 1_000_000, 4)
                price_desc = (
                    f"Service provider charges "
                    f"${self._format_price(input_price)} / "
                    f"${self._format_price(output_price)} "
                    f"per 1M input/output tokens"
                )
                pricing = {
                    "type": "one_million_tokens",
                    "input": "0",
                    "output": "0",
                    "description": price_desc,
                }
                # Include cached_input if available
                if "cache_read_input_token_cost" in model_data:
                    cached_price = round(float(
                        model_data["cache_read_input_token_cost"]) * 1_000_000, 4)
                    pricing["cached_input"] = "0"
                    price_desc = (
                        f"Service provider charges "
                        f"${self._format_price(input_price)} / "
                        f"${self._format_price(output_price)} / "
                        f"${self._format_price(cached_price)} "
                        f"per 1M input/output/cached tokens"
                    )
                    pricing["description"] = price_desc

        return {
            # Folder path under specs/ == listing.name == "<provider>/<model_id>"
            # (flat layout, #1263). populate_from_iterator preserves the slash.
            "name": f"{PROVIDER_NAME}/{model_id}",
            # Offering name is the bare upstream model_id
            "offering_name": model_id,
            # Offering fields
            "display_name": display_name,
            "description": f"{display_name} language model",
            "service_type": service_type,
            "status": "ready",
            "details": details,
            "payout_price": pricing,
            # Listing fields
            "list_price": pricing,
            # Provider config (for templates)
            "provider_name": PROVIDER_NAME,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "api_base_url": API_BASE_URL,
            "env_api_key_name": ENV_API_KEY_NAME,
        }

    def _determine_service_type(self, model_id: str) -> str:
        model_lower = model_id.lower()
        if any(kw in model_lower for kw in ["embed", "embedding"]):
            return "embedding"
        if any(kw in model_lower for kw in ["rerank"]):
            return "rerank"
        if any(kw in model_lower for kw in ["vision"]):
            return "vision_language_model"
        return "llm"

    def _format_price(self, price: float) -> str:
        """Format price without trailing .0 for whole numbers."""
        if price == int(price):
            return str(int(price))
        return str(price)


def main():
    api_key = os.environ.get(ENV_API_KEY_NAME)
    if not api_key:
        print(f"Error: {ENV_API_KEY_NAME} not set")
        sys.exit(1)

    source = ModelSource(api_key)
    write_params_from_iterator(
        iterator=source.iter_models(),
        output_dir=SCRIPT_DIR.parent / "specs",
    )


if __name__ == "__main__":
    main()
