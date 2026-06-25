"""Guardrails for the Shef web application.

Centralised input validation, output sanitisation, prompt-injection
detection, PII scrubbing, and retrieval-result filtering.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from fastapi import HTTPException

# ── Size-limit constants ────────────────────────────────────────────────────

MAX_USER_MESSAGE_CHARS = 2000
MAX_HISTORY_MESSAGE_CHARS = 2000
MAX_EXTRACTED_CONTEXT_CHARS = 2000
MAX_AI_REPLY_CHARS = 6000
MAX_SEARCH_CONTEXT_CHARS = 5000
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_AUDIO_BYTES = 10 * 1024 * 1024

# ── Unsafe-input phrases ────────────────────────────────────────────────────
# Original set plus expanded jailbreak / prompt-injection patterns.

UNSAFE_INPUT_PHRASES: tuple[str, ...] = (
    # Original phrases
    "ignore previous instructions",
    "ignore all previous instructions",
    "reveal system prompt",
    "reveal your system prompt",
    "show system prompt",
    "show your system prompt",
    "developer message",
    "system message",
    "api key",
    "secret key",
    "password",
    # Expanded jailbreak patterns
    "bypass",
    "pretend you are",
    "act as",
    "disregard",
    "override",
    "jailbreak",
    "do anything now",
    "ignore all rules",
    "forget your instructions",
    "you are now",
    "new persona",
    "roleplay as",
    "dan",
)

# ── Recipe-context terms (for RAG filtering) ────────────────────────────────

RECIPE_CONTEXT_TERMS: tuple[str, ...] = (
    "recipe",
    "cook",
    "cooking",
    "ingredient",
    "meal",
    "food",
    "dish",
    "kitchen",
    "substitution",
    "filipino",
    "pinoy",
    "ulam",
)

FOOD_RELEVANCE_TERMS: tuple[str, ...] = (
    *RECIPE_CONTEXT_TERMS,
    "craving",
    "hungry",
    "eat",
    "breakfast",
    "lunch",
    "dinner",
    "snack",
    "dessert",
    "sisig",
    "adobo",
    "sinigang",
    "pancit",
    "tinola",
    "kare-kare",
    "afritada",
    "menudo",
    "lechon",
    "lumpia",
    "torta",
    "paksiw",
    "caldereta",
    "nilaga",
    "giniling",
    "laing",
    "bicol express",
    "halo-halo",
    "chicken",
    "pork",
    "beef",
    "fish",
    "shrimp",
    "egg",
    "eggs",
    "rice",
    "garlic",
    "onion",
    "tomato",
    "kamatis",
    "calamansi",
    "soy sauce",
    "vinegar",
    "pepper",
    "salt",
    "ginger",
    "luya",
    "coconut milk",
    "vegetable",
    "vegetables",
    "eggplant",
    "talong",
    "itlog",
    "sibuyas",
    "noodle",
    "noodles",
    "tofu",
    "mushroom",
    "cheese",
    "flour",
    "sugar",
    "milk",
    "apple",
    "banana",
    "carrot",
    "potato",
    "cabbage",
    "lettuce",
    "cucumber",
    "squash",
    "beans",
    "corn",
    "mango",
    "pineapple",
    "papaya",
    "bread",
    "butter",
    "oil",
)

NON_INGREDIENT_EXTRACTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bno\s+(?:edible\s+|kitchen\s+|visible\s+)*"
        r"(?:ingredients?|food|items?)\s+(?:are\s+|were\s+)?"
        r"(?:visible|detected|found|present)\b"
    ),
    re.compile(
        r"\b(?:i\s+)?(?:do\s+not|don't|cannot|can't)\s+see\s+"
        r"(?:any\s+)?(?:edible\s+|kitchen\s+|visible\s+)*"
        r"(?:ingredients?|food|items?)\b"
    ),
    re.compile(
        r"\b(?:image|photo|picture)\s+(?:does\s+not|doesn't)\s+"
        r"(?:show|contain|include)\s+(?:any\s+)?"
        r"(?:ingredients?|food|edible\s+items?)\b"
    ),
)

NON_FOOD_MEDIA_TERMS: tuple[str, ...] = (
    "person",
    "people",
    "face",
    "selfie",
    "profile photo",
    "profile picture",
    "portrait",
    "clothing",
)

# ── Output-secret patterns & markers ────────────────────────────────────────

SECRET_OUTPUT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # NVIDIA / OpenAI style keys
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bnvapi-[A-Za-z0-9_-]{8,}\b"),
    # JWT-like tokens
    re.compile(r"\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b"),
    # Generic "key = value" leaks
    re.compile(r"(?i)\b(api[_ -]?key|secret[_ -]?key|password|bearer token)\s*[:=]"),
    # AWS access key IDs
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # GCP API keys
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    # Azure-style keys (32-char hex with optional hyphens)
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    # Generic long hex secrets (≥32 hex chars)
    re.compile(r"\b[0-9a-fA-F]{32,}\b"),
)

SECRET_OUTPUT_MARKERS: tuple[str, ...] = (
    "sk-",
    "nvapi-",
    "api key",
    "secret key",
    "password",
    "bearer ",
    "akia",
    "aiza",
)

# ── PII patterns ────────────────────────────────────────────────────────────

PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Email addresses
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    # US phone numbers  (xxx) xxx-xxxx  or  xxx-xxx-xxxx  or  +1xxxxxxxxxx
    re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"),
    # PH phone numbers  09xx-xxx-xxxx  or  +639xxxxxxxxx
    re.compile(r"(?<!\d)(?:\+?63|0)9\d{2}[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"),
    # Credit card numbers – Visa (4xxx), MC (5xxx/2xxx), Amex (3xxx)
    re.compile(r"(?<!\d)[3-6]\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}(?!\d)"),
    # Amex 15-digit variant
    re.compile(r"(?<!\d)3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}(?!\d)"),
)

# ── Internal-network patterns ───────────────────────────────────────────────

INTERNAL_NETWORK_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 192.168.x.x
    re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b"),
    # 10.x.x.x
    re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    # 172.16.0.0 – 172.31.255.255
    re.compile(r"\b172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"),
    # localhost / loopback
    re.compile(r"\blocalhost\b", re.IGNORECASE),
    re.compile(r"\b127\.0\.0\.1\b"),
    re.compile(r"(?<![:\w])::1(?![:\w])"),
)

# ── Base64 injection pattern ────────────────────────────────────────────────

_BASE64_INJECTION_RE = re.compile(
    r"(?:"
    r"data:[a-z]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]{40,}"
    r"|"
    r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/=])"
    r")"
)

# ── Unicode-homoglyph normalisation map ─────────────────────────────────────
# Maps visually similar non-Latin characters to their ASCII look-alikes.

_HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic
    "\u0430": "a",  # а
    "\u0435": "e",  # е
    "\u043e": "o",  # о
    "\u0440": "p",  # р
    "\u0441": "c",  # с
    "\u0443": "y",  # у
    "\u0456": "i",  # і
    "\u0445": "x",  # х
    "\u043d": "h",  # н → h (visual)
    "\u0422": "T",  # Т
    "\u0410": "A",  # А
    "\u0412": "B",  # В
    "\u0415": "E",  # Е
    "\u041a": "K",  # К
    "\u041c": "M",  # М
    "\u041d": "H",  # Н
    "\u041e": "O",  # О
    "\u0420": "P",  # Р
    "\u0421": "C",  # С
    "\u0425": "X",  # Х
    # Greek
    "\u03b1": "a",  # α
    "\u03bf": "o",  # ο
    "\u03c1": "p",  # ρ
    "\u03b5": "e",  # ε
    "\u03b9": "i",  # ι
    # Full-width Latin
    "\uff41": "a",
    "\uff42": "b",
    "\uff43": "c",
    "\uff44": "d",
    "\uff45": "e",
    "\uff49": "i",
    "\uff4f": "o",
    "\uff50": "p",
    "\uff53": "s",
    "\uff59": "y",
}

_HOMOGLYPH_TABLE = str.maketrans(_HOMOGLYPH_MAP)

# ── L33tspeak normalisation map ─────────────────────────────────────────────

_LEET_MAP: dict[str, str] = {
    "1": "i",
    "!": "i",
    "3": "e",
    "0": "o",
    "@": "a",
    "$": "s",
    "4": "a",
    "7": "t",
    "5": "s",
    "|": "i",
    "(": "c",
}

_LEET_TABLE = str.maketrans(_LEET_MAP)


# ── Normalisation helpers ───────────────────────────────────────────────────


def _normalise_text(text: str) -> str:
    """Normalise text for guardrail matching.

    Applies NFKC unicode normalisation, homoglyph replacement, and
    l33tspeak substitution so that obfuscated inputs are caught by the
    same phrase list.
    """
    normalised = unicodedata.normalize("NFKC", text)
    normalised = normalised.translate(_HOMOGLYPH_TABLE)
    normalised = normalised.translate(_LEET_TABLE)
    return normalised.lower()


# ── Phonetic-obfuscation normalisation ──────────────────────────────────────
# Maps phonetic letter spellings to their single-character equivalents so
# inputs like "A p eye keys" → "api keys" are caught.

_PHONETIC_MAP: dict[str, str] = {
    "eye": "i",
    "aye": "i",
    "ay": "a",
    "ee": "e",
    "ess": "s",
    "arr": "r",
    "ar": "r",
    "are": "r",
    "aitch": "h",
    "ach": "h",
    "jay": "j",
    "kay": "k",
    "cue": "q",
    "que": "q",
    "pee": "p",
    "tee": "t",
    "dee": "d",
    "bee": "b",
    "cee": "c",
    "see": "c",
    "sea": "c",
    "gee": "g",
    "vee": "v",
    "wye": "y",
    "why": "y",
    "you": "u",
    "oh": "o",
    "em": "m",
    "en": "n",
    "el": "l",
    "ex": "x",
    "zee": "z",
    "zed": "z",
    "eff": "f",
    "ef": "f",
    "double-u": "w",
    "double u": "w",
    "dubya": "w",
}

# Sorted longest-first so "double-u" matches before "double"
_PHONETIC_SORTED = sorted(_PHONETIC_MAP.keys(), key=len, reverse=True)
_PHONETIC_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _PHONETIC_SORTED) + r")\b",
    re.IGNORECASE,
)

# Pattern to detect spaced-out single letters like "a p i" → "api"
_SPACED_LETTERS_RE = re.compile(
    r"(?<![a-zA-Z])([a-zA-Z])\s+(?=[a-zA-Z](?:\s+[a-zA-Z](?:\s|$)|\s*$|[^a-zA-Z]))"
)


def _normalise_phonetic(text: str) -> str:
    """Collapse phonetic letter spellings and spaced-out letters.

    Examples:
        "A p eye keys" → "api keys"
        "a p i  k e y" → "api key"
        "give me the pee ay ess ess" → "give me the pass"
    """
    lowered = text.lower()
    # Replace phonetic words first
    result = _PHONETIC_RE.sub(lambda m: _PHONETIC_MAP[m.group(1).lower()], lowered)
    # Collapse sequences of single spaced-out letters (e.g. "a p i" → "api")
    result = re.sub(r"(?<![a-zA-Z])([a-zA-Z])(?:\s+([a-zA-Z]))+(?![a-zA-Z])",
                    lambda m: m.group(0).replace(" ", ""), result)
    # Normalise whitespace
    result = " ".join(result.split())
    return result


def _has_base64_injection(text: str) -> bool:
    """Return True if the text contains a suspicious base64 block."""
    return bool(_BASE64_INJECTION_RE.search(text))


# ── Core guardrail functions ────────────────────────────────────────────────


def contains_unsafe_instruction(text: str) -> bool:
    """Check raw *and* normalised text against the blocked-phrase list."""
    lowered = text.lower()
    normalised = _normalise_text(text)
    phonetic = _normalise_phonetic(text)
    return (
        any(phrase in lowered for phrase in UNSAFE_INPUT_PHRASES)
        or any(phrase in normalised for phrase in UNSAFE_INPUT_PHRASES)
        or any(phrase in phonetic for phrase in UNSAFE_INPUT_PHRASES)
    )


def check_input(
    text: str,
    *,
    field_name: str,
    max_chars: int = MAX_USER_MESSAGE_CHARS,
) -> str:
    """Validate user-facing input text.

    Raises ``HTTPException`` (400) when the input is too long, contains
    an unsafe instruction, or embeds a suspicious base64 payload.
    """
    clean_text = text.strip()

    if len(clean_text) > max_chars:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is too long.",
        )

    if clean_text and contains_unsafe_instruction(clean_text):
        raise HTTPException(
            status_code=400,
            detail="Unsafe request. Ask Shef for recipes, ingredients, substitutions, or cooking help.",
        )

    if clean_text and _has_base64_injection(clean_text):
        raise HTTPException(
            status_code=400,
            detail="Unsafe request. Ask Shef for recipes, ingredients, substitutions, or cooking help.",
        )

    return clean_text


def check_history_text(text: str) -> str | None:
    """Sanitise a single history message. Returns ``None`` to drop it."""
    clean_text = text.strip()[:MAX_HISTORY_MESSAGE_CHARS]
    if not clean_text or contains_unsafe_instruction(clean_text):
        return None
    return clean_text


def is_non_ingredient_extraction(text: str) -> bool:
    """Return True when an attachment extractor says it found no food items."""
    normalised = _normalise_text(text)
    return any(pattern.search(normalised) for pattern in NON_INGREDIENT_EXTRACTION_PATTERNS)


def _contains_food_relevance(text: str) -> bool:
    haystack = _normalise_text(text)
    return any(term in haystack for term in FOOD_RELEVANCE_TERMS)


def _looks_like_extracted_ingredient_list(text: str) -> bool:
    haystack = _normalise_text(text)
    if any(term in haystack for term in NON_FOOD_MEDIA_TERMS):
        return False
    return "," in haystack and len(haystack.split()) <= 40


def has_recipe_relevant_input(
    message: str,
    image_ingredients: str | None = None,
    audio_transcript: str | None = None,
) -> bool:
    """Return True when the current turn has cooking intent or ingredients.

    History is intentionally excluded so an unrelated upload cannot inherit a
    previous recipe topic and generate another recipe.
    """
    clean_message = message.strip()
    clean_image_ingredients = (image_ingredients or "").strip()
    clean_audio_transcript = (audio_transcript or "").strip()

    if clean_message and _contains_food_relevance(clean_message):
        return True

    if clean_image_ingredients and not is_non_ingredient_extraction(clean_image_ingredients):
        has_food_terms = _contains_food_relevance(clean_image_ingredients)
        looks_like_ingredient_list = _looks_like_extracted_ingredient_list(clean_image_ingredients)
        return has_food_terms or looks_like_ingredient_list

    if clean_audio_transcript and _contains_food_relevance(clean_audio_transcript):
        return True

    return False


# ── Output guardrails ───────────────────────────────────────────────────────


def _contains_pii(text: str) -> bool:
    """Return True if the text contains recognisable PII."""
    return any(pattern.search(text) for pattern in PII_PATTERNS)


def _contains_internal_network(text: str) -> bool:
    """Return True if the text references internal network addresses."""
    return any(pattern.search(text) for pattern in INTERNAL_NETWORK_PATTERNS)


def check_output(text: str) -> str:
    """Inspect model output and redact if secrets, PII, or internal
    addresses are detected."""
    clean_text = text.strip()[:MAX_AI_REPLY_CHARS]
    lowered = clean_text.lower()

    # Secret / key leak detection
    if any(marker in lowered for marker in SECRET_OUTPUT_MARKERS) or any(
        pattern.search(clean_text) for pattern in SECRET_OUTPUT_PATTERNS
    ):
        return "I cannot provide that information."

    # PII detection
    if _contains_pii(clean_text):
        return "I cannot provide that information."

    # Internal network reference detection
    if _contains_internal_network(clean_text):
        return "I cannot provide that information."

    return clean_text


# ── RAG / retrieval guardrails ──────────────────────────────────────────────


def is_recipe_search_result(result: dict[str, Any]) -> bool:
    """Return True if a search result is relevant to cooking/recipes and
    does not contain unsafe instructions."""
    title = str(result.get("title") or "")
    content = str(result.get("content") or "")
    haystack = f"{title} {content}".lower()
    if contains_unsafe_instruction(haystack):
        return False
    return any(term in haystack for term in RECIPE_CONTEXT_TERMS)


def clean_search_text(value: Any, *, max_chars: int) -> str:
    """Collapse whitespace and truncate a search-result field."""
    return " ".join(str(value or "").split())[:max_chars].strip()
