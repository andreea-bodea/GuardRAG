"""
Shared constants for text types, column mappings, epsilon values, and display names.
"""

# === Text type column names (in the text tables, e.g. tab_text2) ===

TEXT_TYPE_WITH_PII = "text_with_pii"

PRESIDIO_TEXT_TYPES = [
    "text_pii_deleted",
    "text_pii_labeled",
    "text_pii_synthetic",
]

DIFFRACTOR_TEXT_TYPES = [
    "text_pii_dp_diffractor1",
    "text_pii_dp_diffractor2",
    "text_pii_dp_diffractor3",
]

DP_PROMPT_TEXT_TYPES = [
    "text_pii_dp_dp_prompt1",
    "text_pii_dp_dp_prompt2",
    "text_pii_dp_dp_prompt3",
]

DPMLM_TEXT_TYPES = [
    "text_pii_dp_dpmlm1",
    "text_pii_dp_dpmlm2",
    "text_pii_dp_dpmlm3",
]

# 7 text types used for TAB (original + Presidio + Diffractor)
TAB_TEXT_TYPES = (
    [TEXT_TYPE_WITH_PII]
    + PRESIDIO_TEXT_TYPES
    + DIFFRACTOR_TEXT_TYPES
)

# All 13 text types in insertion order (matches insert_responses() columns)
ORDERED_TEXT_TYPES = (
    [TEXT_TYPE_WITH_PII]
    + PRESIDIO_TEXT_TYPES
    + DIFFRACTOR_TEXT_TYPES
    + DP_PROMPT_TEXT_TYPES
    + DPMLM_TEXT_TYPES
)

# === Text-type to response-column mapping ===

TEXT_TYPE_TO_RESPONSE_COLUMN = {
    "text_pii_deleted": "response_pii_deleted",
    "text_pii_labeled": "response_pii_labeled",
    "text_pii_synthetic": "response_pii_synthetic",
    "text_pii_dp_diffractor1": "response_pii_dp_diffractor1",
    "text_pii_dp_diffractor2": "response_pii_dp_diffractor2",
    "text_pii_dp_diffractor3": "response_pii_dp_diffractor3",
    "text_pii_dp_dp_prompt1": "response_pii_dp_dp_prompt1",
    "text_pii_dp_dp_prompt2": "response_pii_dp_dp_prompt2",
    "text_pii_dp_dp_prompt3": "response_pii_dp_dp_prompt3",
    "text_pii_dp_dpmlm1": "response_pii_dp_dpmlm1",
    "text_pii_dp_dpmlm2": "response_pii_dp_dpmlm2",
    "text_pii_dp_dpmlm3": "response_pii_dp_dpmlm3",
}

# === Response column lists (for evaluation) ===

ANONYMIZATION_TYPES_RESPONSES = [
    "response_pii_deleted", "response_pii_labeled", "response_pii_synthetic",
    "response_pii_dp_diffractor1", "response_pii_dp_diffractor2", "response_pii_dp_diffractor3",
    "response_pii_dp_dp_prompt1", "response_pii_dp_dp_prompt2", "response_pii_dp_dp_prompt3",
    "response_pii_dp_dpmlm1", "response_pii_dp_dpmlm2", "response_pii_dp_dpmlm3",
]

ANONYMIZATION_TYPES_POSTPROCESSED = [
    "response_postprocessed_pii_deleted", "response_postprocessed_pii_labeled", "response_postprocessed_pii_synthetic",
    "response_postprocessed_pii_diffractor1", "response_postprocessed_pii_diffractor2", "response_postprocessed_pii_diffractor3",
    "response_postprocessed_pii_dp_prompt1", "response_postprocessed_pii_dp_prompt2", "response_postprocessed_pii_dp_prompt3",
    "response_postprocessed_pii_dpmlm1", "response_postprocessed_pii_dpmlm2", "response_postprocessed_pii_dpmlm3",
]

# Display names for aggregation tables (same order as ANONYMIZATION_TYPES_*)
METHOD_DISPLAY_NAMES = [
    "PII Deletion",
    "PII Labeling",
    "PII Synthetic data",
    "1-DIFFRACTOR (ε=1)",
    "1-DIFFRACTOR (ε=2)",
    "1-DIFFRACTOR (ε=3)",
    "DP-PROMPT (ε=150)",
    "DP-PROMPT (ε=200)",
    "DP-PROMPT (ε=250)",
    "DP-MLM (ε=50)",
    "DP-MLM (ε=75)",
    "DP-MLM (ε=100)",
]

# === Epsilon values ===

DIFFRACTOR_EPSILONS = [1, 2, 3]
DP_PROMPT_EPSILONS = [150, 200, 250]
DPMLM_EPSILONS = [50, 75, 100]