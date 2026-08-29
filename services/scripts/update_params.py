#!/usr/bin/env python3
"""
Template-based update_services.py for Mistral AI.

Yields model dictionaries that are rendered using Jinja2 templates.

Usage: python scripts/update_services.py
"""

import json
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
SPECS_DIR = SCRIPT_DIR.parent / "specs"


def committed_parameters(service_name: str) -> dict:
    """The parameters already committed for ``service_name`` ({} if it is new).

    unitysvc-sellers >= 0.3.1 keeps a committed value when the iterator yields
    ``None`` for it: from inside the writer, a lookup that failed and a lookup
    that legitimately found nothing are the same event. That is right for
    enrichment, but it means a price we FAILED to derive gets re-shipped as
    though it were this run's answer. Reading the previous value here is what
    separates the two cases — see the price guard in ``_build_template_vars``.
    """
    path = SPECS_DIR / f"{service_name}.json"
    if not path.is_file():
        return {}
    try:
        return (json.loads(path.read_text()) or {}).get("parameters") or {}
    except (OSError, ValueError):
        return {}

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
        if not self.litellm_data:
            print(
                "Error: LiteLLM model data came back empty. Every price lookup "
                "would fail, and unitysvc-sellers >= 0.3.1 would preserve the "
                "committed prices instead — re-shipping stale rate cards as "
                "though they were current."
            )
            sys.exit(1)

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
            # Not `return`. An empty iterator is indistinguishable from "the
            # upstream retired its whole catalog": with deprecate_missing the
            # writer would mark every committed service deprecated, and exiting
            # 0 would make a failed fetch look like a clean no-change run.
            sys.exit(1)

        if not models:
            print(
                "Error: upstream enumerated zero models — refusing to treat an "
                "empty enumeration as a retired catalog."
            )
            sys.exit(1)

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
        service_name = f"{PROVIDER_NAME}/{model_id}"
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

        # Extract upstream pricing for description, but set prices to 0 for BYOK.
        #
        # `pricing_note` is the bare rate card — no "Service provider charges"
        # prefix, because the copy that consumes it already names the biller.
        # It is a param in its own right so the templates can place it: the
        # listing cell puts it behind the `|` of the price-description grammar
        # (unitysvc/unitysvc#1886) and the offering description states it in
        # prose. Do NOT fold it back into `pricing["description"]` — that dict
        # feeds `payout_price` too, which is seller-facing and stays as it is.
        pricing = None
        pricing_note = None
        if model_data:
            if "input_cost_per_token" in model_data and "output_cost_per_token" in model_data:
                input_price = round(float(
                    model_data["input_cost_per_token"]) * 1_000_000, 4)
                output_price = round(float(
                    model_data["output_cost_per_token"]) * 1_000_000, 4)
                pricing_note = (
                    f"${self._format_price(input_price)} / "
                    f"${self._format_price(output_price)} "
                    f"per 1M input/output tokens"
                )
                pricing = {
                    "type": "one_million_tokens",
                    "input": "0",
                    "output": "0",
                    "description": f"Service provider charges {pricing_note}",
                }
                # Include cached_input if available
                if "cache_read_input_token_cost" in model_data:
                    cached_price = round(float(
                        model_data["cache_read_input_token_cost"]) * 1_000_000, 4)
                    pricing["cached_input"] = "0"
                    pricing_note = (
                        f"${self._format_price(input_price)} / "
                        f"${self._format_price(output_price)} / "
                        f"${self._format_price(cached_price)} "
                        f"per 1M input/output/cached tokens"
                    )
                    pricing["description"] = f"Service provider charges {pricing_note}"

        # `list_price` is nullable and the schema does not require it, so a
        # failed lookup is rejected by nothing downstream. And since
        # unitysvc-sellers 0.3.1 preserves committed values against a yielded
        # None, that failure now SHIPS THE PREVIOUS PRICE as though it
        # were this run's answer. A model that has never appeared in the LiteLLM
        # data has no committed value and nothing to silently ship; a model that
        # had one and can no longer derive it is the regression, and it is fatal.
        if pricing is None and committed_parameters(service_name).get("list_price") is not None:
            print(
                f"  FATAL: {model_id} has a committed list_price but no "
                "input_cost_per_token/output_cost_per_token in this run's "
                "LiteLLM data. Refusing to re-ship the previous price."
            )
            sys.exit(1)

        return {
            # The service's name IS its path under specs/ (flat layout, #1263).
            # unitysvc-sellers >= 0.3.1 requires this key verbatim: `name_field`
            # is gone and there is no fallback for a dict that omits it.
            "service_name": service_name,
            # Offering name is the bare upstream model_id
            "offering_name": model_id,
            # Offering fields
            "display_name": display_name,
            "description": f"{display_name} language model",
            "service_type": service_type,
            # Does this model accept a tools-bearing request? Mistral answers
            # per model on the card itself, so this is copied from the upstream
            # rather than inferred from the id or from a catalog-wide
            # assumption. It drives the `feature:func-call` tag in
            # templates/offering.json.j2 — the only tag the platform's closed
            # vocabulary lets a seller declare (unitysvc docs/tags.yml).
            #
            # Always a real bool, never None: unitysvc-sellers >= 0.3.1 keeps
            # the committed value when the iterator yields None, so a null here
            # would re-ship an old `true` after Mistral stopped advertising the
            # capability. A card without `capabilities` (the schema says there
            # is none) reads as False, which drops the tag — the safe direction:
            # a missing tag costs a catalog facet, a wrong one ships a
            # tool-calling example that 400s in the customer's hands.
            "supports_function_calling": bool(
                (model_info.get("capabilities") or {}).get("function_calling")
            ),
            "status": "ready",
            "details": details,
            "payout_price": pricing,
            # Bare upstream rate card, placed by the templates (listing price
            # cell's hover note + the offering description).
            "pricing_note": pricing_note,
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
        output_dir=SPECS_DIR,
    )


if __name__ == "__main__":
    main()
