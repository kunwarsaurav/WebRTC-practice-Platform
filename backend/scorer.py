"""
Unified IELTS Speaking Scorer Engine
Provides comprehensive scoring across all IELTS speaking criteria.
"""

import argparse
import json
import math
import os
import re
import sys
import threading
_transcribe_lock = threading.Lock()

import time
from dotenv import load_dotenv
load_dotenv()
from collections import Counter, defaultdict
from typing import Any, Tuple, List, Dict

# Fix for Windows: HuggingFace Hub tries to create symlinks for models,
# which crashes with WinError 1314 unless you run as Administrator.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import librosa
import numpy as np
import parselmouth  # type: ignore
import spacy
import language_tool_python
from parselmouth.praat import call  # type: ignore
def cosine_similarity(a, b):
    """Minimal cosine similarity — replaces sklearn for single vector pairs."""
    a, b = np.array(a), np.array(b)
    norms = np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1)
    return np.dot(a, b.T) / np.clip(norms, 1e-10, None)
from wordfreq import zipf_frequency

try:
    from google.cloud import speech
    from google.cloud import storage
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError as exc:
    import logging
    logging.getLogger(__name__).warning("Google Cloud Speech unavailable: %s", exc)
    speech = None
    storage = None
    GOOGLE_CLOUD_AVAILABLE = False

import uuid


# GLOBAL RESOURCES
_nlp = None
_storage_client = None
_speech_client = None

def _get_storage_client():
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client

def _get_speech_client():
    global _speech_client
    if _speech_client is None:
        _speech_client = speech.SpeechClient()
    return _speech_client

def _get_nlp():
    """Lazy load spaCy."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def init_models():
    """Pre-load all heavy AI models into RAM/VRAM to enable 5s warm starts."""
    
    print("[INIT] Booting up all AI models (Cold Start)...", file=sys.stderr)
    # 1. Load spaCy
    _get_nlp()
    
    # 2. Load Grammar tool
    _get_lt("en-GB")
        
    # 3. Load SentenceTransformer
    print("[INIT] Pre-loading SentenceTransformer model…", file=sys.stderr)
    try:
        from sentence_transformers import SentenceTransformer
        global _st_model
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        pass
        
    print("[INIT] All models successfully loaded into RAM/VRAM!", file=sys.stderr)


# FLUENCY.PY


# Constant word-sets

FILLER_WORDS = {
    "uh", "um", "er", "erm", "hmm", "hm", "ah", "oh",
    "like", "you know", "i mean", "sort of", "kind of",
    "basically", "literally", "actually", "right",
}

# Treat multi-word fillers as collapsed phrases during detection
FILLER_PHRASES = {"you know", "i mean", "sort of", "kind of"}

DISCOURSE_MARKERS = {
    "however", "therefore", "moreover", "in addition", "furthermore",
    "additionally", "consequently", "nevertheless", "nonetheless",
    "on the other hand", "on the contrary", "in contrast",
    "for example", "for instance", "such as", "in particular",
    "firstly", "secondly", "finally", "in conclusion", "to sum up",
    "in other words", "that is to say", "as a result", "because of this",
    "well", "actually", "basically", "naturally", "clearly",
    "obviously", "certainly", "indeed", "of course",
}

SELF_CORRECTION_SIGNALS = {
    "i mean", "sorry", "no wait", "actually", "let me", "i should say",
    "what i mean is", "or rather", "to be more precise",
}

# Minimum gap between words (seconds) to count as a pause
PAUSE_THRESHOLD_SEC = 0.25
LONG_PAUSE_THRESHOLD_SEC = 2.0


# Public API

def analyze_fluency(
    transcript: str,
    segments: list[dict],
    word_timestamps: list[dict],
    audio_duration: float,
) -> dict[str, Any]:
    """
    Extract all Fluency & Coherence features.

    Args:
        transcript:      Full transcript string from Whisper.
        segments:        Whisper segment list (each has 'start', 'end', 'text').
        word_timestamps: WhisperX word list (each has 'word', 'start', 'end').
        audio_duration:  Total audio file length in seconds.

    Returns:
        Dictionary of named feature values.
    """
    text_lower = transcript.lower()

    # --- Basic metrics ---
    total_speaking_time = _speaking_time(word_timestamps)
    word_count = len(word_timestamps)
    wpm = _words_per_minute(word_count, total_speaking_time)

    # --- Pause analysis ---
    pauses = _extract_pauses(word_timestamps)
    pause_count = len(pauses)
    avg_pause_dur = (sum(pauses) / len(pauses)) if pauses else 0.0
    long_pause_count = sum(1 for p in pauses if p >= LONG_PAUSE_THRESHOLD_SEC)

    # --- Disfluency detection ---
    filler_count, filler_positions = _count_fillers(word_timestamps)
    repetition_count = _count_repetitions(word_timestamps)
    self_correction_count = _count_self_corrections(text_lower)

    # --- Coherence markers ---
    dm_count, dm_variety, dm_used = _count_discourse_markers(text_lower)

    # --- Speech continuity ---
    actual_speaking_time = sum(w.get("end", 0) - w.get("start", 0) for w in word_timestamps if "start" in w and "end" in w)
    continuity_ratio = (actual_speaking_time / total_speaking_time) if total_speaking_time > 0 else 0.0

    return {
        "total_speaking_time_sec": round(total_speaking_time, 2),
        "word_count": word_count,
        "words_per_minute": round(wpm, 1),
        "pause_count": pause_count,
        "avg_pause_duration_sec": round(avg_pause_dur, 3),
        "long_pause_count": long_pause_count,
        "filler_count": filler_count,
        "repetition_count": repetition_count,
        "self_correction_count": self_correction_count,
        "discourse_marker_count": dm_count,
        "discourse_marker_variety": dm_variety,
        "discourse_markers_used": dm_used,
        "speech_continuity_ratio": round(continuity_ratio, 3),
    }


# Internal helpers

def _speaking_time(word_timestamps: list[dict]) -> float:
    """Total duration from first word start to last word end."""
    if not word_timestamps:
        return 0.0
    starts = [w["start"] for w in word_timestamps if "start" in w]
    ends = [w["end"] for w in word_timestamps if "end" in w]
    if not starts or not ends:
        return 0.0
    return max(ends) - min(starts)


def _words_per_minute(word_count: int, speaking_time_sec: float) -> float:
    """Compute WPM from word count and speaking duration."""
    if speaking_time_sec <= 0:
        return 0.0
    return word_count / (speaking_time_sec / 60.0)


def _extract_pauses(word_timestamps: list[dict]) -> list[float]:
    """
    Identify inter-word gaps above the pause threshold.
    Returns a list of pause durations in seconds.
    """
    pauses = []
    for i in range(1, len(word_timestamps)):
        prev_end = word_timestamps[i - 1].get("end", None)
        curr_start = word_timestamps[i].get("start", None)
        if prev_end is not None and curr_start is not None:
            gap = curr_start - prev_end
            if gap >= PAUSE_THRESHOLD_SEC:
                pauses.append(gap)
    return pauses


def _count_fillers(word_timestamps: list[dict]) -> tuple[int, list[int]]:
    """
    Count filler words/phrases by scanning the word list.
    Returns (count, list_of_word_indices).
    """
    count = 0
    positions = []
    words = [w.get("word", "").lower().strip(",.!?") for w in word_timestamps]

    i = 0
    while i < len(words):
        # Check 2-word filler phrase first
        if i + 1 < len(words):
            bigram = words[i] + " " + words[i + 1]
            if bigram in FILLER_PHRASES:
                count += 1
                positions.append(i)
                i += 2
                continue
        # Single-word filler
        if words[i] in FILLER_WORDS:
            count += 1
            positions.append(i)
        i += 1

    return count, positions


def _count_repetitions(word_timestamps: list[dict]) -> int:
    """
    Count consecutive or near-consecutive word repetitions.
    Uses a sliding window (bigram) approach: detects cases where
    the same content word appears twice within a 3-word window.
    """
    words = [
        w.get("word", "").lower().strip(",.!?")
        for w in word_timestamps
        if w.get("word", "").strip()
    ]
    # Filter out stop words and fillers for repetition detection
    STOP = {"the", "a", "an", "is", "it", "i", "in", "on", "at", "to",
             "of", "and", "or", "but", "so", "for", "with", "was", "are"}
    content_words = [w for w in words if w not in STOP and len(w) > 2]

    count = 0
    for i in range(1, len(content_words)):
        # Immediate repetition
        if content_words[i] == content_words[i - 1]:
            count += 1
        # Near repetition (within 2 words)
        elif i >= 2 and content_words[i] == content_words[i - 2]:
            count += 1

    return count


def _count_self_corrections(text_lower: str) -> int:
    """
    Count self-correction signals using known phrases and
    repeated-word correction patterns ("I ... I mean ...").
    """
    count = 0
    for phrase in SELF_CORRECTION_SIGNALS:
        count += text_lower.count(phrase)
    return count


def _count_discourse_markers(text_lower: str) -> tuple[int, int, list[str]]:
    """
    Count total discourse marker occurrences and unique variety.
    Returns (total_count, unique_count, list_of_markers_used).
    """
    used = []
    total = 0
    for marker in DISCOURSE_MARKERS:
        # Use word-boundary-aware matching for single words
        if " " in marker:
            occ = text_lower.count(marker)
        else:
            occ = len(re.findall(r"\b" + re.escape(marker) + r"\b", text_lower))
        if occ > 0:
            used.append(marker)
            total += occ
    return total, len(used), sorted(used)


# LEXICAL.PY



# Load spaCy model — called once at module level





# Constants

# Parts of speech considered "content words" for lexical density
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

# Words appearing below this Zipf frequency score are "advanced/less common"
# Zipf scale: 6=very common, 3=rare. Words ~3.5-4.0 are "less common" in English
ADVANCED_VOCAB_ZIPF_THRESHOLD = 4.0

# MATTR (Moving Average Type-Token Ratio) sliding window size
MATTR_WINDOW = 50

# Paraphrase marker phrases — signals the speaker is reformulating
PARAPHRASE_PHRASES = {
    "in other words", "that is to say", "put differently",
    "what i mean is", "to put it another way", "in short",
    "to clarify", "that means", "which means",
}

# Strong collocations drawn from IELTS high-frequency corpus and Oxford Collocations Dictionary.
# Covers verb+noun, adjective+noun, and noun+noun patterns relevant to common IELTS topics.
STRONG_COLLOCATIONS = {
    # Verb + Noun (actions)
    ("make", "decision"), ("make", "progress"), ("make", "difference"),
    ("make", "contribution"), ("make", "effort"), ("make", "mistake"),
    ("take", "advantage"), ("take", "responsibility"), ("take", "action"),
    ("take", "part"), ("take", "place"), ("take", "approach"),
    ("give", "opportunity"), ("give", "priority"), ("give", "impression"),
    ("pay", "attention"), ("pay", "price"), ("pay", "role"),
    ("raise", "awareness"), ("raise", "concern"), ("raise", "question"),
    ("carry", "out"), ("carry", "responsibility"),
    ("come", "across"), ("come", "conclusion"),
    ("deal", "with"), ("deal", "issue"),
    ("look", "forward"), ("look", "issue"),
    ("play", "role"), ("play", "part"),
    ("face", "challenge"), ("face", "consequence"),
    ("meet", "demand"), ("meet", "need"),
    ("reach", "goal"), ("reach", "conclusion"), ("reach", "agreement"),
    ("pose", "threat"), ("pose", "challenge"),
    ("build", "relationship"), ("build", "community"),
    ("develop", "skill"), ("develop", "understanding"),
    ("provide", "opportunity"), ("provide", "support"), ("provide", "access"),
    ("gain", "experience"), ("gain", "knowledge"), ("gain", "access"),
    ("achieve", "goal"), ("achieve", "balance"), ("achieve", "success"),
    ("address", "issue"), ("address", "problem"), ("address", "concern"),
    ("have", "impact"), ("have", "effect"), ("have", "influence"),

    # Adjective + Noun
    ("significant", "impact"), ("significant", "role"), ("significant", "increase"),
    ("major", "challenge"), ("major", "issue"), ("major", "factor"),
    ("crucial", "role"), ("crucial", "factor"),
    ("key", "factor"), ("key", "role"), ("key", "issue"),
    ("vital", "role"), ("vital", "importance"),
    ("widespread", "use"), ("widespread", "concern"),
    ("growing", "concern"), ("growing", "number"), ("growing", "demand"),
    ("increasing", "number"), ("increasing", "pressure"),
    ("strong", "influence"), ("strong", "evidence"),
    ("critical", "thinking"), ("critical", "issue"),
    ("global", "problem"), ("global", "issue"), ("global", "community"),
    ("positive", "impact"), ("negative", "impact"),
    ("long", "term"), ("short", "term"),
    ("high", "quality"), ("high", "standard"),

    # Noun + Noun (compound topics common in IELTS)
    ("social", "media"), ("social", "issue"), ("social", "problem"),
    ("social", "responsibility"), ("social", "inequality"),
    ("climate", "change"), ("climate", "crisis"),
    ("public", "transport"), ("public", "health"), ("public", "sector"),
    ("economic", "growth"), ("economic", "development"), ("economic", "impact"),
    ("human", "rights"), ("human", "nature"), ("human", "development"),
    ("mental", "health"), ("mental", "wellbeing"),
    ("living", "standard"), ("standard", "living"),
    ("quality", "life"), ("way", "life"),
    ("government", "policy"), ("education", "system"), ("health", "care"),
    ("job", "opportunity"), ("work", "life"), ("life", "expectancy"),
    ("natural", "environment"), ("natural", "resource"),
    ("population", "growth"), ("urban", "development"),
    ("technological", "advancement"), ("scientific", "research"),
    ("financial", "support"), ("financial", "crisis"),
}


# Public API

def analyze_lexical(transcript: str) -> dict[str, Any]:
    """
    Extract all Lexical Resource features from the transcript.

    Args:
        transcript: Full transcript string.

    Returns:
        Dictionary of named feature values.
    """
    nlp = _get_nlp()
    doc = nlp(transcript)

    text_lower = transcript.lower()

    # Raw tokens (exclude punctuation and whitespace)
    all_tokens = [
        t for t in doc
        if not t.is_punct and not t.is_space
    ]
    total_words = len(all_tokens)

    if total_words == 0:
        return _empty_features()

    # Lemmatized words for diversity metrics
    lemmas = [t.lemma_.lower() for t in all_tokens if t.lemma_.isalpha()]
    unique_lemmas = set(lemmas)

    # Content word tokens only
    content_tokens = [t for t in all_tokens if t.pos_ in CONTENT_POS]
    content_lemmas = [t.lemma_.lower() for t in content_tokens if t.lemma_.isalpha()]

    # --- Diversity metrics ---
    ttr = len(unique_lemmas) / len(lemmas) if lemmas else 0.0
    mattr = _moving_average_ttr(lemmas, MATTR_WINDOW)

    # --- Lexical density ---
    lexical_density = len(content_tokens) / total_words if total_words else 0.0

    # --- Advanced vocabulary ---
    adv_count, adv_words = _advanced_vocabulary(lemmas)
    adv_ratio = adv_count / len(lemmas) if lemmas else 0.0

    # --- Repetition of common content words ---
    content_freq = Counter(content_lemmas)
    high_repeat_words = {
        w: c for w, c in content_freq.items()
        if c >= 3 and len(w) > 3
    }
    repetition_frequency = len(high_repeat_words)

    # --- Paraphrase detection ---
    paraphrase_found = any(phrase in text_lower for phrase in PARAPHRASE_PHRASES)
    paraphrase_count = sum(text_lower.count(phrase) for phrase in PARAPHRASE_PHRASES)

    # --- Word length (sophistication proxy) ---
    avg_word_length = (
        sum(len(t.text) for t in all_tokens) / total_words if total_words else 0.0
    )

    # --- Collocations ---
    collocation_count = _detect_collocations(content_lemmas)

    return {
        "total_words": total_words,
        "unique_words": len(unique_lemmas),
        "type_token_ratio": round(ttr, 4),
        "moving_avg_ttr": round(mattr, 4),
        "lexical_density": round(lexical_density, 4),
        "advanced_vocab_count": adv_count,
        "advanced_vocab_ratio": round(adv_ratio, 4),
        "advanced_vocab_examples": adv_words[:10],      # sample for transparency
        "high_repetition_words": list(high_repeat_words.keys())[:10],
        "repetition_frequency": repetition_frequency,
        "paraphrase_indicator": paraphrase_found,
        "paraphrase_count": paraphrase_count,
        "avg_word_length": round(avg_word_length, 2),
        "collocations_detected": collocation_count,
    }


# Internal helpers

def _moving_average_ttr(lemmas: list[str], window: int) -> float:
    """
    Compute Moving Average Type-Token Ratio (MATTR).
    Slides a window across the lemma list and averages per-window TTR.
    Normalises for text length — longer texts don't artificially lower TTR.
    """
    if len(lemmas) < window:
        # Fall back to raw TTR for short texts
        return len(set(lemmas)) / len(lemmas) if lemmas else 0.0

    ttrs = []
    for i in range(len(lemmas) - window + 1):
        window_slice = lemmas[i: i + window]
        ttrs.append(len(set(window_slice)) / window)

    return sum(ttrs) / len(ttrs)


def _advanced_vocabulary(lemmas: list[str]) -> tuple[int, list[str]]:
    """
    Count words whose Zipf frequency falls below the threshold.
    Zipf frequency < threshold → word is less common / advanced.

    Uses the 'wordfreq' library for English frequency data.
    """
    advanced = []
    seen = set()
    for lemma in lemmas:
        if lemma in seen or not lemma.isalpha() or len(lemma) < 4:
            continue
        seen.add(lemma)
        freq = zipf_frequency(lemma, "en")
        if 0 < freq < ADVANCED_VOCAB_ZIPF_THRESHOLD:
            advanced.append(lemma)

    return len(advanced), advanced


def _detect_collocations(content_lemmas: list[str]) -> int:
    """
    Count occurrences of known strong collocations in the transcript.
    Uses a pre-defined set of educationally relevant collocation pairs.
    """
    bigrams = set(zip(content_lemmas, content_lemmas[1:]))
    matches = bigrams & STRONG_COLLOCATIONS
    return len(matches)


def _empty_features() -> dict[str, Any]:
    """Return zeroed feature dict when transcript is empty."""
    return {
        "total_words": 0,
        "unique_words": 0,
        "type_token_ratio": 0.0,
        "moving_avg_ttr": 0.0,
        "lexical_density": 0.0,
        "advanced_vocab_count": 0,
        "advanced_vocab_ratio": 0.0,
        "advanced_vocab_examples": [],
        "high_repetition_words": [],
        "repetition_frequency": 0,
        "paraphrase_indicator": False,
        "paraphrase_count": 0,
        "avg_word_length": 0.0,
        "collocations_detected": 0,
    }


# GRAMMAR.PY



# Module-level singletons (loaded once)

_lt_tool = None





def _get_lt(lang: str = "en-GB"):
    global _lt_tool
    if _lt_tool is None:
        lt_url = os.environ.get("LANGUAGETOOL_URL")
        if lt_url:
            _lt_tool = language_tool_python.LanguageTool(lang, remote_server=lt_url)
        else:
            # language_tool_python auto-downloads the LanguageTool JAR on first run
            _lt_tool = language_tool_python.LanguageTool(lang)
    return _lt_tool


# LanguageTool rule-category → error type mapping
# These category IDs correspond to LanguageTool's internal rule taxonomy

TENSE_RULE_IDS = {
    "ENGLISH_WORD_REPEAT_RULE", "PAST_TENSE_WITH_WOULD",
    "VERB_TENSE", "PERFECT_TENSE",
}
AGREEMENT_RULE_IDS = {
    "AGREEMENT_SENT_START", "PRP_VB", "DOES_X",
    "HE_VERB_AGR", "SV_AGREEMENT",
}
ARTICLE_RULE_IDS = {
    "EN_A_VS_AN", "THE_SUPERLATIVE", "MISSING_ARTICLE",
    "ARTICLE_MISSING", "ARTICLE_REDUNDANT",
}
PREPOSITION_RULE_IDS = {
    "AT_THE_WEEKEND", "PREPOSITION_AFTER", "ON_THE_WAY",
    "IN_TIME_PERIOD", "PREP_REDUNDANT",
}


# Public API

def analyze_grammar(transcript: str, lang: str = "en-GB") -> dict[str, Any]:
    """
    Extract all Grammatical Range & Accuracy features.

    Args:
        transcript: Full transcript string.
        lang:       LanguageTool language code (default 'en-GB').

    Returns:
        Dictionary of named feature values.
    """
    nlp = _get_nlp()
    lt = _get_lt(lang)

    doc = nlp(transcript)
    sentences = list(doc.sents)

    if not sentences:
        return _empty_features()

    word_count = sum(
        1 for t in doc if not t.is_punct and not t.is_space
    )

    # --- Accuracy: LanguageTool error analysis ---
    lt_matches = lt.check(transcript)
    error_breakdown = _classify_errors(lt_matches)
    total_errors = len(lt_matches)
    errors_per_100 = (total_errors / word_count * 100) if word_count else 0.0

    # --- Range: spaCy syntactic analysis ---
    sentence_stats = _analyze_sentences(sentences)

    return {
        # Accuracy metrics
        "total_grammar_errors": total_errors,
        "errors_per_100_words": round(errors_per_100, 2),
        "tense_errors": error_breakdown["tense"],
        "agreement_errors": error_breakdown["agreement"],
        "article_errors": error_breakdown["article"],
        "preposition_errors": error_breakdown["preposition"],
        "other_errors": error_breakdown["other"],
        "error_examples": error_breakdown["examples"][:5],   # first 5 for context
        # Range metrics
        "sentence_count": sentence_stats["count"],
        "avg_sentence_length": sentence_stats["avg_length"],
        "subordinate_clause_count": sentence_stats["subordinate_clauses"],
        "subordinate_clause_freq": sentence_stats["subordinate_freq"],
        "complex_sentence_ratio": sentence_stats["complex_ratio"],
        "compound_sentence_ratio": sentence_stats["compound_ratio"],
        "simple_sentence_ratio": sentence_stats["simple_ratio"],
        "sentence_variety_score": sentence_stats["variety_score"],
    }


# Internal helpers

def _classify_errors(matches: list) -> dict:
    """
    Classify LanguageTool matches into IELTS-relevant error categories.
    Returns a dict with counts per category and example messages.
    """
    counts = defaultdict(int)
    examples = []

    for match in matches:
        rule_id = match.rule_id
        category = match.category if hasattr(match, "category") else ""
        message = match.message

        # Classify by rule ID prefix or category keyword
        if _matches_any(rule_id, TENSE_RULE_IDS) or "tense" in message.lower():
            counts["tense"] += 1
        elif _matches_any(rule_id, AGREEMENT_RULE_IDS) or "agreement" in message.lower():
            counts["agreement"] += 1
        elif _matches_any(rule_id, ARTICLE_RULE_IDS) or "article" in message.lower():
            counts["article"] += 1
        elif _matches_any(rule_id, PREPOSITION_RULE_IDS) or "preposition" in message.lower():
            counts["preposition"] += 1
        else:
            counts["other"] += 1

        if len(examples) < 5:
            examples.append({
                "rule": rule_id,
                "message": message,
                "context": match.context if hasattr(match, "context") else "",
            })

    return {
        "tense": counts["tense"],
        "agreement": counts["agreement"],
        "article": counts["article"],
        "preposition": counts["preposition"],
        "other": counts["other"],
        "examples": examples,
    }


def _matches_any(value: str, rule_set: set) -> bool:
    """Check if the value matches any rule ID in the set (prefix or exact)."""
    value_upper = value.upper()
    return any(
        value_upper == r or value_upper.startswith(r)
        for r in rule_set
    )


def _analyze_sentences(sentences) -> dict:
    """
    Analyse syntactic structure of each spaCy sentence.

    Sentence types:
      Simple   — no subordinate or coordinating clauses
      Compound — joined by coordinating conjunction (cc)
      Complex  — contains at least one subordinate clause (advcl, relcl, etc.)

    Returns aggregated statistics.
    """
    # Dependency labels that indicate subordinate clauses
    SUBORDINATE_DEPS = {"advcl", "relcl", "csubj", "ccomp", "xcomp", "acl"}
    # Coordinating conjunction dependency label
    COORD_DEP = "cc"

    counts = {"simple": 0, "compound": 0, "complex": 0}
    sentence_lengths = []
    total_subordinate_clauses = 0
    count = 0

    for sent in sentences:
        tokens = [t for t in sent if not t.is_punct and not t.is_space]
        if not tokens:
            continue

        count += 1
        sentence_lengths.append(len(tokens))

        deps = {t.dep_ for t in sent}
        has_subordinate = bool(deps & SUBORDINATE_DEPS)
        has_coordination = COORD_DEP in deps

        subordinate_in_sent = sum(1 for t in sent if t.dep_ in SUBORDINATE_DEPS)
        total_subordinate_clauses += subordinate_in_sent

        if has_subordinate:
            counts["complex"] += 1
        elif has_coordination:
            counts["compound"] += 1
        else:
            counts["simple"] += 1

    if count == 0:
        return {
            "count": 0, "avg_length": 0.0,
            "subordinate_clauses": 0, "subordinate_freq": 0.0,
            "complex_ratio": 0.0, "compound_ratio": 0.0,
            "simple_ratio": 0.0, "variety_score": 0.0,
        }

    avg_length = sum(sentence_lengths) / count
    subordinate_freq = total_subordinate_clauses / count

    complex_ratio = counts["complex"] / count
    compound_ratio = counts["compound"] / count
    simple_ratio = counts["simple"] / count

    # Sentence variety: reward having all three types present
    types_present = sum(1 for v in counts.values() if v > 0)
    variety_score = round(types_present / 3.0, 3)   # 0.33, 0.67, or 1.0

    return {
        "count": count,
        "avg_length": round(avg_length, 2),
        "subordinate_clauses": total_subordinate_clauses,
        "subordinate_freq": round(subordinate_freq, 3),
        "complex_ratio": round(complex_ratio, 3),
        "compound_ratio": round(compound_ratio, 3),
        "simple_ratio": round(simple_ratio, 3),
        "variety_score": variety_score,
    }


def _empty_features() -> dict[str, Any]:
    """Return zeroed features when transcript is empty."""
    return {
        "total_grammar_errors": 0,
        "errors_per_100_words": 0.0,
        "tense_errors": 0,
        "agreement_errors": 0,
        "article_errors": 0,
        "preposition_errors": 0,
        "other_errors": 0,
        "error_examples": [],
        "sentence_count": 0,
        "avg_sentence_length": 0.0,
        "subordinate_clause_count": 0,
        "subordinate_clause_freq": 0.0,
        "complex_sentence_ratio": 0.0,
        "compound_sentence_ratio": 0.0,
        "simple_sentence_ratio": 0.0,
        "sentence_variety_score": 0.0,
    }


# PRONUNCIATION.PY




# Acoustic analysis constants

# Praat pitch analysis range — broad range suitable for all speakers
PITCH_FLOOR_HZ = 75.0
PITCH_CEILING_HZ = 500.0

# A "chunk" is a group of words separated by a pause ≥ this threshold
CHUNK_PAUSE_THRESHOLD_SEC = 0.4

# Words whose duration exceeds this are considered "long" (stress proxy)
LONG_WORD_DURATION_SEC = 0.6


# Public API

def analyze_pronunciation(
    audio_path: str,
    word_timestamps: list[dict],
    segments: list[dict],
    audio_array=None,
    sample_rate=None,
) -> dict[str, Any]:
    """
    Extract all Pronunciation & Prosodic features.

    Args:
        audio_path:      Path to the audio file.
        word_timestamps: WhisperX word-level timestamps (word, start, end).
        segments:        Whisper segment list.
        audio_array:     Pre-loaded audio numpy array (optional). If supplied,
                         avoids a redundant disk read.
        sample_rate:     Sample rate of audio_array (required if audio_array given).

    Returns:
        Dictionary of named feature values.
    """
    prosodic = _extract_prosodic_features(audio_path, audio_array, sample_rate)
    rhythmic = _extract_rhythmic_features(word_timestamps)

    return {**prosodic, **rhythmic}


# Prosodic features via parselmouth (Praat)

def _extract_prosodic_features(audio_path: str, audio_array=None, sample_rate=None) -> dict[str, Any]:
    """
    Use parselmouth to extract pitch and intensity features.
    These are the primary measurable phonological features available
    without requiring a pronunciation dictionary or phoneme aligner.

    If audio_array + sample_rate are provided, parselmouth is constructed
    from the pre-loaded array instead of re-reading the file from disk.
    """
    try:
        if audio_array is not None and sample_rate is not None:
            sound = parselmouth.Sound(audio_array, sampling_frequency=sample_rate)
        else:
            sound = parselmouth.Sound(audio_path)
    except Exception as e:
        return _empty_prosodic(f"Audio load error: {e}")

    # --- Pitch analysis ---
    pitch_obj = sound.to_pitch(
        time_step=0.01,
        pitch_floor=PITCH_FLOOR_HZ,
        pitch_ceiling=PITCH_CEILING_HZ,
    )
    pitch_values = pitch_obj.selected_array["frequency"]
    # Filter out unvoiced frames (value == 0)
    voiced_pitch = pitch_values[pitch_values > 0]

    if len(voiced_pitch) < 5:
        mean_pitch = 0.0
        pitch_range = 0.0
        pitch_std = 0.0
    else:
        mean_pitch = float(np.mean(voiced_pitch))
        pitch_range = float(np.max(voiced_pitch) - np.min(voiced_pitch))
        pitch_std = float(np.std(voiced_pitch))

    # --- Intensity analysis ---
    intensity_obj = sound.to_intensity(minimum_pitch=75.0, time_step=0.01)
    intensity_values = intensity_obj.values.T.flatten()
    intensity_values = intensity_values[intensity_values > 0]

    if len(intensity_values) < 5:
        mean_intensity = 0.0
        intensity_variation = 0.0
    else:
        mean_intensity = float(np.mean(intensity_values))
        intensity_variation = float(np.std(intensity_values))

    # --- Speaking rate via energy peaks (syllable proxy) ---
    speaking_rate = _estimate_speaking_rate(audio_path, audio_array, sample_rate)

    # --- Rhythm regularity ---
    # Coefficient of Variation (CV) of voiced segment durations
    # Low CV → regular rhythm; high CV → variable/irregular rhythm
    voiced_segment_durations = _voiced_segment_durations(pitch_obj)
    if len(voiced_segment_durations) > 2:
        cv = float(np.std(voiced_segment_durations) / np.mean(voiced_segment_durations))
        rhythm_regularity = round(cv, 4)
    else:
        rhythm_regularity = 0.0

    return {
        "mean_pitch_hz": round(mean_pitch, 2),
        "pitch_range_hz": round(pitch_range, 2),
        "pitch_std_dev_hz": round(pitch_std, 2),
        "mean_intensity_db": round(mean_intensity, 2),
        "intensity_variation_db": round(intensity_variation, 2),
        "speaking_rate_syl_per_sec": round(speaking_rate, 2),
        "rhythm_regularity_cv": rhythm_regularity,
    }


def _estimate_speaking_rate(audio_path: str, audio_array=None, sample_rate=None) -> float:
    """
    Estimate syllables per second using librosa onset detection.
    Each onset roughly corresponds to a syllable nucleus in speech.
    This is an approximation; actual syllable segmentation requires
    a dedicated phoneme aligner.

    If audio_array + sample_rate are provided, skips the disk read.
    """
    try:
        if audio_array is not None and sample_rate is not None:
            y, sr = audio_array, sample_rate
        else:
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration <= 0:
            return 0.0

        # Detect onsets — each onset is a proxy for a syllable boundary
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, units="time",
            pre_max=1, post_max=1, pre_avg=3, post_avg=3,
            delta=0.07, wait=0.06,
        )
        syllable_count = len(onset_frames)
        return syllable_count / duration
    except Exception:
        return 0.0


def _voiced_segment_durations(pitch_obj) -> list[float]:
    """
    Extract durations of continuous voiced segments from a Praat Pitch object.
    Used to measure rhythm regularity.
    """
    frame_times = pitch_obj.ts()
    pitch_values = pitch_obj.selected_array["frequency"]

    durations = []
    in_voiced = False
    seg_start = 0.0

    for t, p in zip(frame_times, pitch_values):
        if p > 0 and not in_voiced:
            # Start of voiced segment
            in_voiced = True
            seg_start = t
        elif p == 0 and in_voiced:
            # End of voiced segment
            in_voiced = False
            durations.append(t - seg_start)

    # Close any open segment at the end
    if in_voiced and len(frame_times) > 0:
        durations.append(frame_times[-1] - seg_start)

    return [d for d in durations if d > 0.05]   # filter noise < 50ms


# Rhythmic features via WhisperX word timestamps

def _extract_rhythmic_features(word_timestamps: list[dict]) -> dict[str, Any]:
    """
    Derive rhythmic and connected speech features from word-level timestamps.
    These complement the acoustic Praat features.
    """
    if not word_timestamps:
        return _empty_rhythmic()

    # Word durations
    word_durations = []
    for w in word_timestamps:
        s = w.get("start")
        e = w.get("end")
        if s is not None and e is not None and e > s:
            word_durations.append(e - s)

    if not word_durations:
        return _empty_rhythmic()

    duration_variance = float(np.var(word_durations))
    long_words = sum(1 for d in word_durations if d >= LONG_WORD_DURATION_SEC)
    long_word_ratio = long_words / len(word_durations)

    # --- Chunk detection: group words separated by pauses ---
    chunks = _detect_chunks(word_timestamps)
    chunk_count = len(chunks)
    avg_words_per_chunk = (
        np.mean([len(c) for c in chunks]) if chunks else 0.0
    )

    return {
        "word_duration_variance": round(duration_variance, 4),
        "long_word_ratio": round(long_word_ratio, 4),
        "chunk_boundary_count": chunk_count,
        "avg_words_per_chunk": round(float(avg_words_per_chunk), 2),
    }


def _detect_chunks(word_timestamps: list[dict]) -> list[list[dict]]:
    """
    Group words into 'chunks' (breath groups) by detecting pauses
    above the CHUNK_PAUSE_THRESHOLD_SEC between consecutive words.
    """
    if not word_timestamps:
        return []

    chunks = []
    current_chunk = [word_timestamps[0]]

    for i in range(1, len(word_timestamps)):
        prev_end = word_timestamps[i - 1].get("end", 0)
        curr_start = word_timestamps[i].get("start", 0)
        gap = curr_start - prev_end

        if gap >= CHUNK_PAUSE_THRESHOLD_SEC:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [word_timestamps[i]]
        else:
            current_chunk.append(word_timestamps[i])

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# Empty feature fallbacks

def _empty_prosodic(error: str = "") -> dict[str, Any]:
    return {
        "mean_pitch_hz": 0.0,
        "pitch_range_hz": 0.0,
        "pitch_std_dev_hz": 0.0,
        "mean_intensity_db": 0.0,
        "intensity_variation_db": 0.0,
        "speaking_rate_syl_per_sec": 0.0,
        "rhythm_regularity_cv": 0.0,
        "_error": error,
    }


def _empty_rhythmic() -> dict[str, Any]:
    return {
        "word_duration_variance": 0.0,
        "long_word_ratio": 0.0,
        "chunk_boundary_count": 0,
        "avg_words_per_chunk": 0.0,
    }


# RELEVANCE.PY


# Lazy loaded singleton
_st_model = None

def _get_st_model():
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            _st_model = None
    return _st_model

# Constants

# SentenceTransformer semantic similarity
# Completely off-topic responses (e.g., Color vs Apple, Name vs Reliable) will score ~0.0 to 0.27.
# Tangential responses (e.g., City benefits vs Countryside weekends) score ~0.27 to 0.45.
# Direct answers score 0.50+
OFF_TOPIC_THRESHOLD = 0.35
TANGENTIAL_THRESHOLD = 0.50

# Public API

def analyze_relevance(transcript: str, question: str) -> dict[str, Any]:
    """
    Measure semantic relevance between the transcript and the question using SentenceTransformer.
    """
    if not transcript.strip() or not question.strip():
        return _empty_features()

    nlp = _get_nlp()
    st_model = _get_st_model()
    
    if not st_model:
        # Fallback if sentence-transformers is missing
        return _empty_features()

    # Calculate overall similarity
    try:
        q_emb = st_model.encode(question)
        t_emb = st_model.encode(transcript)
        overall_sim = cosine_similarity([q_emb], [t_emb])[0][0]
    except Exception:
        overall_sim = 0.0
    
    max_sent_sim = overall_sim
    
    doc = nlp(transcript)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 3]
    if sentences:
        try:
            s_embs = st_model.encode(sentences)
            sims = cosine_similarity([q_emb], s_embs)[0]
            max_sent_sim = max(sims)
        except Exception:
            pass

    is_off_topic = float(max_sent_sim) < OFF_TOPIC_THRESHOLD
    is_tangential = OFF_TOPIC_THRESHOLD <= float(max_sent_sim) < TANGENTIAL_THRESHOLD

    return {
        "overall_similarity": round(float(overall_sim), 3),
        "max_sentence_similarity": round(float(max_sent_sim), 3),
        "is_off_topic": is_off_topic,
        "is_tangential": is_tangential,
        "question_provided": True
    }


def _empty_features() -> dict[str, Any]:
    """Return zeroed features when no transcript or question is available."""
    return {
        "overall_similarity": 0.0,
        "max_sentence_similarity": 0.0,
        "is_off_topic": False,  
        "question_provided": False
    }



# SCORER.PY



# Public API

def compute_bands(
    fluency_features: dict,
    lexical_features: dict,
    grammar_features: dict,
    pronunciation_features: dict,
    relevance_features: dict,
    exam_part: int = 1,
) -> dict[str, float]:
    """
    Compute IELTS band scores for all four criteria and the overall band.

    Returns:
        {
          "fluency": float,
          "lexical": float,
          "grammar": float,
          "pronunciation": float,
          "overall": float,
        }
    """
    fluency_band = _score_fluency(fluency_features)
    lexical_band = _score_lexical(lexical_features)
    grammar_band = _score_grammar(grammar_features)
    pronunciation_band = _score_pronunciation(pronunciation_features)

    # --- Topic Relevance Penalties ---
    is_off_topic = relevance_features.get("is_off_topic", False)
    is_tangential = relevance_features.get("is_tangential", False)
    
    if is_off_topic:
        # Apple vs Cow: completely irrelevant
        fluency_band = 1.0
        lexical_band = 1.0
        grammar_band = 1.0
        pronunciation_band = 1.0
        overall = 1.0
    else:
        if is_tangential:
            # Partially memorised or tangential. The candidate is talking about something related 
            # (e.g. hard-working) but didn't answer the prompt (e.g. alone vs group).
            # We severely penalize fluency & coherence and lexical resource.
            fluency_band = min(float(fluency_band), 5.0)
            lexical_band = min(float(lexical_band), 5.0)
            
        # --- Length Penalties by Exam Part ---
        # To demonstrate high bands, the candidate must speak appropriately for the exam part.
        word_count = fluency_features.get("word_count", 0)
        duration = fluency_features.get("total_speaking_time_sec", 0)
        
        if exam_part == 1:
            # Ideal is 15-20 seconds (approx 40-50 words). 
            # 5 to 10 seconds is too short (Band 5-6).
            if duration < 10 or word_count < 20:
                fluency_band = min(float(fluency_band), 6.0)
                lexical_band = min(float(lexical_band), 6.0)
                grammar_band = min(float(grammar_band), 6.0)
            elif duration < 14 or word_count < 30:
                fluency_band = min(float(fluency_band), 7.0)
                lexical_band = min(float(lexical_band), 7.0)
            # >= 15 seconds is ideal, no penalty.
        elif exam_part == 2:
            # Ideal is 1-2 minutes (60-120 seconds).
            if duration < 30 or word_count < 60:
                fluency_band = min(float(fluency_band), 5.0)
                lexical_band = min(float(lexical_band), 5.0)
                grammar_band = min(float(grammar_band), 5.0)
            elif duration < 45 or word_count < 90:
                fluency_band = min(float(fluency_band), 6.0)
                lexical_band = min(float(lexical_band), 6.0)
                grammar_band = min(float(grammar_band), 6.0)
            elif duration < 60 or word_count < 120:
                fluency_band = min(float(fluency_band), 7.0)
                lexical_band = min(float(lexical_band), 7.0)
        elif exam_part == 3:
            # Ideal is longer than Part 1, ~30-45 seconds (approx 60-90 words).
            if duration < 15 or word_count < 30:
                fluency_band = min(float(fluency_band), 5.5)
                lexical_band = min(float(lexical_band), 5.5)
                grammar_band = min(float(grammar_band), 5.5)
            elif duration < 25 or word_count < 50:
                fluency_band = min(float(fluency_band), 6.5)
                lexical_band = min(float(lexical_band), 6.5)
            
        # --- Scripted / Memorised Speech Penalty ---
        # Detect if the candidate sounds like they are reading a textbook
        pitch_range = pronunciation_features.get("pitch_range_hz", 0)
        pitch_std = pronunciation_features.get("pitch_std_dev_hz", 0)
        fillers = fluency_features.get("filler_count", 0)
        word_count = fluency_features.get("word_count", 0)
        density = lexical_features.get("lexical_density", 0)
        
        words_100 = max(word_count, 1) / 100.0
        filler_rate = fillers / words_100

        if word_count > 30:
            is_flat = pitch_range < 60 or pitch_std < 15
            is_too_fluent = filler_rate < 1.0
            is_written_style = density >= 0.48

            if is_flat and is_too_fluent and is_written_style:
                fluency_band = min(float(fluency_band), 5.0)
                lexical_band = min(float(lexical_band), 5.0)
                relevance_features["is_scripted"] = True

        overall = _compute_overall(
            fluency_band, lexical_band, grammar_band, pronunciation_band
        )

    return {
        "fluency": float(fluency_band),
        "lexical": lexical_band,
        "grammar": grammar_band,
        "pronunciation": pronunciation_band,
        "overall": overall,
    }


# Fluency & Coherence Band Scoring
#
# Evidence evaluated against IELTS descriptors:
#
#  Band 9: Fluent, only very occasional repetition/self-correction.
#          Hesitation only to prepare content. Fully coherent. Cohesion fully
#          appropriate.
#  Band 8: Fluent with occasional repetition. Occasional hesitation for
#          language. Coherent and relevant topic development.
#  Band 7: Long turns without noticeable effort. Some hesitation/repetition/
#          self-correction. Coherence maintained. Flexible discourse markers.
#  Band 6: Long turns. Hesitation/repetition may occasionally affect coherence.
#          Range of discourse markers, not always appropriate.
#  Band 5: Relies on repetition and self-correction. Hesitations from searching
#          for vocabulary. Overuses discourse markers.
#  Band 4: Noticeable pauses. Frequent repetition/self-correction. Limited coherence.
#  Band 3: Frequent long pauses. Limited connection of ideas.
#  Band 2: Lengthy pauses before nearly every word. Very limited communication.
#  Band 1: Speech essentially incoherent.

def _score_fluency(f: dict) -> float:
    """Evaluate fluency features against IELTS Fluency & Coherence descriptors."""

    wpm = f.get("words_per_minute", 0)
    long_pauses = f.get("long_pause_count", 99)
    pause_count = f.get("pause_count", 99)
    fillers = f.get("filler_count", 99)
    reps = f.get("repetition_count", 99)
    corrections = f.get("self_correction_count", 0)
    dm_variety = f.get("discourse_marker_variety", 0)
    dm_count = f.get("discourse_marker_count", 0)
    continuity = f.get("speech_continuity_ratio", 0)
    word_count = f.get("word_count", 0)

    # Normalise fillers and repetitions relative to speech length
    # (avoids penalising longer responses unfairly)
    words_100 = max(word_count, 1) / 100.0
    filler_rate = fillers / words_100
    rep_rate = reps / words_100
    corr_rate = corrections / words_100

    # --- Descriptor alignment evidence ---
    # Each descriptor band is characterised by a set of key observations.
    # We score how many key observations align with the extracted features.

    # Band 9: near-perfect fluency
    b9 = _evidence_score([
        wpm >= 130,                  # fast, natural delivery
        long_pauses == 0,            # no long pauses
        filler_rate < 1.0,           # fewer than 1 filler per 100 words
        rep_rate < 1.0,              # very few repetitions
        corr_rate < 1.0,             # very few self-corrections
        dm_variety >= 6,             # rich repertoire of discourse markers
        continuity >= 0.90,          # near-continuous speech
    ])

    # Band 8: highly fluent, very minor lapses
    b8 = _evidence_score([
        wpm >= 120,
        long_pauses <= 1,
        filler_rate < 2.0,
        rep_rate < 2.0,
        dm_variety >= 5,
        continuity >= 0.85,
    ])

    # Band 7: good fluency with some hesitation
    b7 = _evidence_score([
        wpm >= 100,
        long_pauses <= 2,
        filler_rate < 4.0,
        rep_rate < 4.0,
        dm_variety >= 3,
        continuity >= 0.78,
    ])

    # Band 6: extended turns with occasional coherence breaks
    b6 = _evidence_score([
        wpm >= 85,
        long_pauses <= 4,
        filler_rate < 7.0,
        rep_rate < 6.0,
        dm_variety >= 2,
        continuity >= 0.68,
    ])

    # Band 5: relies on repetition, searches for words
    b5 = _evidence_score([
        wpm >= 70,
        filler_rate >= 5.0 or rep_rate >= 5.0,
        long_pauses <= 6,
        continuity >= 0.55,
    ])

    # Band 4: noticeable pauses, frequent repetition
    b4 = _evidence_score([
        wpm >= 50,
        long_pauses <= 10,
        continuity >= 0.40,
    ])

    # Band 3: frequent long pauses, limited connection
    b3 = _evidence_score([
        wpm >= 30,
        long_pauses > 6,
        continuity < 0.55,
    ])

    # Band 2: lengthy pauses before nearly every word
    b2 = _evidence_score([
        wpm < 30,
        continuity < 0.35,
    ])

    # Band 1: essentially incoherent — handled by overall word count
    b1 = word_count > 0 and word_count < 15

    if word_count == 0:
        return 0.0
    if b1:
        return 1.0

    band = _select_band(b2, b3, b4, b5, b6, b7, b8, b9)

    # Interpolation: if the evidence is borderline, apply ±0.5
    band = _interpolate(band, {
        9: b9,
        8: b8,
        7: b7,
        6: b6,
        5: b5,
        4: b4,
        3: b3,
        2: b2,
    })

    # Hard gatekeepers
    avg_pause_dur = f.get("avg_pause_duration_sec", 0)
    if wpm < 90 or avg_pause_dur > 1.5:
        band = min(band, 5.5)

    return round(band * 2) / 2   # snap to nearest 0.5


# Lexical Resource Band Scoring
#
#  Band 9: Total flexibility and precision. Sustained idiomatic language.
#  Band 8: Wide vocabulary. Effective idiomatic use. Effective paraphrasing.
#  Band 7: Flexible use, some less common items. Effective paraphrasing.
#  Band 6: Sufficient for extended discussion. Some inappropriate choices.
#          Generally successful paraphrasing.
#  Band 5: Sufficient for familiar topics. Limited flexibility.
#          Inconsistent paraphrasing.
#  Band 4: Basic vocabulary, frequent inappropriate choices. Rarely paraphrases.
#  Band 3: Simple vocabulary, inadequate for unfamiliar topics.
#  Band 2: Very limited vocabulary, mostly isolated words.
#  Band 1: Almost no usable vocabulary.

def _score_lexical(f: dict) -> float:
    """Evaluate lexical features against IELTS Lexical Resource descriptors."""

    ttr = f.get("type_token_ratio", 0)
    mattr = f.get("moving_avg_ttr", 0)
    density = f.get("lexical_density", 0)
    adv_ratio = f.get("advanced_vocab_ratio", 0)
    adv_count = f.get("advanced_vocab_count", 0)
    rep_freq = f.get("repetition_frequency", 99)
    paraphrase = f.get("paraphrase_indicator", False)
    para_count = f.get("paraphrase_count", 0)
    avg_wlen = f.get("avg_word_length", 0)
    collocations = f.get("collocations_detected", 0)
    total_words = f.get("total_words", 0)

    # Relative repetition burden
    rep_burden = rep_freq / max(total_words / 100, 1)

    b9 = _evidence_score([
        mattr >= 0.78,
        adv_ratio >= 0.12,
        density >= 0.56,
        para_count >= 2,
        collocations >= 4,
        rep_burden < 1.0,
        avg_wlen >= 5.2,
    ])

    b8 = _evidence_score([
        mattr >= 0.72,
        adv_ratio >= 0.09,
        density >= 0.52,
        paraphrase,
        collocations >= 3,
        rep_burden < 1.5,
        avg_wlen >= 4.8,
    ])

    b7 = _evidence_score([
        mattr >= 0.65,
        adv_ratio >= 0.06,
        density >= 0.48,
        paraphrase,
        collocations >= 2,
        rep_burden < 2.5,
        avg_wlen >= 4.5,
    ])

    b6 = _evidence_score([
        mattr >= 0.58,
        adv_ratio >= 0.04,
        density >= 0.44,
        rep_burden < 4.0,
        avg_wlen >= 4.2,
    ])

    b5 = _evidence_score([
        mattr >= 0.50,
        adv_ratio >= 0.02,
        density >= 0.40,
        rep_burden < 6.0,
    ])

    b4 = _evidence_score([
        mattr >= 0.40,
        density >= 0.35,
        rep_burden >= 5.0,
    ])

    b3 = _evidence_score([
        mattr >= 0.30,
        density < 0.40,
        adv_ratio < 0.02,
    ])

    b2 = _evidence_score([
        mattr < 0.30,
        total_words < 30,
    ])

    if total_words == 0:
        return 0.0
    if total_words < 10:
        return 1.0

    band = _select_band(b2, b3, b4, b5, b6, b7, b8, b9)
    band = _interpolate(band, {9: b9, 8: b8, 7: b7, 6: b6, 5: b5, 4: b4, 3: b3, 2: b2})
    return round(band * 2) / 2


# Grammatical Range & Accuracy Band Scoring
#
#  Band 9: Structures consistently precise and accurate.
#  Band 8: Wide range. Majority error-free. Only occasional errors.
#  Band 7: Range used flexibly. Frequent error-free sentences.
#          Complex and simple used effectively.
#  Band 6: Mix of simple and complex. Errors rarely impede communication.
#  Band 5: Basic structures generally controlled. Complex attempted but prone.
#  Band 4: Mostly basic. Frequent errors. Limited subordinate clauses.
#  Band 3: Numerous errors. Limited control of structure.
#  Band 2: No basic sentence forms.
#  Band 1: Only memorized language.

def _score_grammar(f: dict) -> float:
    """Evaluate grammar features against IELTS Grammatical Range & Accuracy descriptors."""

    err_per_100 = f.get("errors_per_100_words", 99)
    complex_ratio = f.get("complex_sentence_ratio", 0)
    compound_ratio = f.get("compound_sentence_ratio", 0)
    simple_ratio = f.get("simple_sentence_ratio", 1)
    variety = f.get("sentence_variety_score", 0)
    avg_sent_len = f.get("avg_sentence_length", 0)
    subord_freq = f.get("subordinate_clause_freq", 0)
    sentence_count = f.get("sentence_count", 0)

    b9 = _evidence_score([
        err_per_100 < 1.0,
        complex_ratio >= 0.45,
        variety >= 0.9,
        subord_freq >= 1.0,
        avg_sent_len >= 14,
    ])

    b8 = _evidence_score([
        err_per_100 < 2.5,
        complex_ratio >= 0.38,
        variety >= 0.67,
        subord_freq >= 0.7,
        avg_sent_len >= 12,
    ])

    b7 = _evidence_score([
        err_per_100 < 4.0,
        complex_ratio >= 0.30,
        variety >= 0.67,
        subord_freq >= 0.5,
        avg_sent_len >= 11,
    ])

    b6 = _evidence_score([
        err_per_100 < 6.0,
        complex_ratio >= 0.20,
        variety >= 0.33,
        avg_sent_len >= 9,
    ])

    b5 = _evidence_score([
        err_per_100 < 9.0,
        complex_ratio >= 0.10,
        avg_sent_len >= 7,
    ])

    b4 = _evidence_score([
        err_per_100 < 14.0,
        complex_ratio < 0.15,
        avg_sent_len >= 5,
    ])

    b3 = _evidence_score([
        err_per_100 >= 12.0,
        complex_ratio < 0.10,
    ])

    b2 = _evidence_score([
        err_per_100 >= 20.0,
        sentence_count < 3,
    ])

    if sentence_count == 0:
        return 0.0
    if sentence_count < 2 and err_per_100 > 20:
        return 1.0

    band = _select_band(b2, b3, b4, b5, b6, b7, b8, b9)
    band = _interpolate(band, {9: b9, 8: b8, 7: b7, 6: b6, 5: b5, 4: b4, 3: b3, 2: b2})
    return round(band * 2) / 2


# Pronunciation Band Scoring
#
#  Band 9: Full range of phonological features. Flexible stress and intonation.
#          Effortlessly understood.
#  Band 8: Wide range. Appropriate rhythm and stress. Easily understood.
#  Band 7: Meets Band 6 + demonstrates some Band 8 features.
#  Band 6: Uses range of phonological features. Variable control.
#          Generally understandable.
#  Band 5: Band 4 + some Band 6 features.
#  Band 4: Limited phonological range. Frequent rhythm/stress issues.
#          Mispronunciations cause clarity problems.
#  Band 3: Some Band 4 features but inconsistent.
#  Band 2: Few acceptable phonological features. Frequently unintelligible.
#  Band 1: Only occasional recognisable words.
#
# NOTE: Without a phoneme-level recogniser we cannot directly measure
# mispronunciation. We evaluate the observable acoustic correlates of
# phonological range: pitch variation, rhythm, intonation, and chunking.

def _score_pronunciation(f: dict) -> float:
    """Evaluate pronunciation features against IELTS Pronunciation descriptors."""

    pitch_range = f.get("pitch_range_hz", 0)
    pitch_std = f.get("pitch_std_dev_hz", 0)
    intensity_var = f.get("intensity_variation_db", 0)
    speaking_rate = f.get("speaking_rate_syl_per_sec", 0)
    rhythm_cv = f.get("rhythm_regularity_cv", 99)   # lower is more regular
    avg_chunk = f.get("avg_words_per_chunk", 0)
    long_word_ratio = f.get("long_word_ratio", 0)
    word_dur_var = f.get("word_duration_variance", 0)

    # Rhythm CV: lower = regular (good); very high = erratic (poor)
    # Invert for scoring: regularity_score = 1 - min(cv, 1)
    rhythm_score = max(0.0, 1.0 - min(rhythm_cv, 1.0))

    b9 = _evidence_score([
        pitch_range >= 120,           # wide pitch range = expressive intonation
        pitch_std >= 35,              # high std dev = varied intonation
        intensity_var >= 6.0,         # stress variation
        speaking_rate >= 4.0,         # confident delivery
        rhythm_score >= 0.80,         # regular rhythm
        avg_chunk >= 7,               # long breath groups (connected speech)
    ])

    b8 = _evidence_score([
        pitch_range >= 95,
        pitch_std >= 28,
        intensity_var >= 5.0,
        speaking_rate >= 3.5,
        rhythm_score >= 0.72,
        avg_chunk >= 6,
    ])

    b7 = _evidence_score([
        pitch_range >= 75,
        pitch_std >= 22,
        intensity_var >= 4.0,
        speaking_rate >= 3.0,
        rhythm_score >= 0.62,
        avg_chunk >= 5,
    ])

    b6 = _evidence_score([
        pitch_range >= 55,
        pitch_std >= 16,
        intensity_var >= 3.0,
        speaking_rate >= 2.5,
        rhythm_score >= 0.50,
        avg_chunk >= 4,
    ])

    b5 = _evidence_score([
        pitch_range >= 40,
        pitch_std >= 10,
        speaking_rate >= 2.0,
        rhythm_score >= 0.38,
    ])

    b4 = _evidence_score([
        pitch_range >= 25,
        speaking_rate >= 1.5,
        rhythm_score >= 0.25,
    ])

    b3 = _evidence_score([
        pitch_range < 30,
        rhythm_score < 0.35,
    ])

    b2 = _evidence_score([
        pitch_range < 15,
        speaking_rate < 1.5,
        rhythm_score < 0.20,
    ])

    if speaking_rate == 0 and pitch_range == 0:
        return 0.0

    band = _select_band(b2, b3, b4, b5, b6, b7, b8, b9)
    band = _interpolate(band, {9: b9, 8: b8, 7: b7, 6: b6, 5: b5, 4: b4, 3: b3, 2: b2})
    
    if rhythm_score < 0.40:
        band = min(band, 6.0)
        
    return round(band * 2) / 2


# Overall Band Calculation

def _compute_overall(f: float, l: float, g: float, p: float) -> float:
    """
    Compute overall IELTS Speaking band as the mean of the four criteria,
    then round according to official IELTS rounding rules:

    Official rule (from IELTS test taker info):
      .25 → round DOWN to nearest whole / half band
      .75 → round UP to nearest whole / half band

    This is equivalent to: multiply by 2, then standard round, divide by 2.
    """
    mean = (f + l + g + p) / 4.0
    # IELTS rounding: 
    # score ending in .25 rounds up to .5
    # score ending in .75 rounds up to next whole band
    return math.floor(mean * 2 + 0.5) / 2.0


# Descriptor-alignment helpers

def _evidence_score(conditions: list[bool]) -> float:
    """
    Compute the fraction of descriptor conditions that are satisfied.
    Returns a value between 0.0 and 1.0.
    """
    if not conditions:
        return 0.0
    return sum(1 for c in conditions if c) / len(conditions)


def _select_band(b2, b3, b4, b5, b6, b7, b8, b9) -> float:
    """
    Select the highest IELTS band where evidence score passes the threshold (≥ 0.5).
    Works down from band 9 to band 2. Returns 2.0 if nothing qualifies above band 2.
    """
    scores = [(9, b9), (8, b8), (7, b7), (6, b6), (5, b5), (4, b4), (3, b3), (2, b2)]
    for band, score in scores:
        if score >= 0.70:
            return float(band)
    return 2.0


def _interpolate(band: float, scores: dict[int, float]) -> float:
    """
    Apply ±0.5 adjustment when evidence straddles two band levels.

    Logic:
      - If the current band score is strong (≥ 0.75) but the band below
        also has moderate evidence — stay at current band.
      - If the current band score is marginal (0.50–0.65) and the band
        above has some evidence (≥ 0.35) — nudge up by 0.5.
      - If the current band score is borderline (0.50–0.60) and the band
        above is weak — nudge down by 0.5.
    """
    current_score = scores.get(int(band), 0.5)
    above_score = scores.get(int(band) + 1, 0.0)
    below_score = scores.get(int(band) - 1, 0.0)

    # Strong evidence for current band with some evidence for band above → +0.5
    if current_score >= 0.65 and above_score >= 0.40:
        band += 0.5
    # Weak current evidence, strong evidence for band below → -0.5
    elif current_score <= 0.55 and below_score >= 0.55:
        band -= 0.5

    return max(1.0, min(9.0, band))


# FEEDBACK.PY



# Public API

def generate_feedback(
    bands: dict[str, float],
    fluency_features: dict,
    lexical_features: dict,
    grammar_features: dict,
    pronunciation_features: dict,
    relevance_features: dict,
    exam_part: int = 1,
) -> dict[str, str]:
    """
    Generate examiner-style feedback for all four IELTS criteria.

    Args:
        bands:                  Band scores {fluency, lexical, grammar, pronunciation}.
        fluency_features:       Features from fluency.analyze().
        lexical_features:       Features from lexical.analyze().
        grammar_features:       Features from grammar.analyze().
        pronunciation_features: Features from pronunciation.analyze().

    Returns:
        Dictionary {criterion: feedback_string}.
    """
    is_off_topic = relevance_features.get("is_off_topic", False)
    is_tangential = relevance_features.get("is_tangential", False)
    is_scripted = relevance_features.get("is_scripted", False)

    return {
        "fluency": _fluency_feedback(bands["fluency"], fluency_features, is_off_topic, is_tangential, is_scripted, exam_part),
        "lexical": _lexical_feedback(bands["lexical"], lexical_features, is_off_topic, is_tangential, is_scripted),
        "grammar": _grammar_feedback(bands["grammar"], grammar_features, is_tangential),
        "pronunciation": _pronunciation_feedback(bands["pronunciation"], pronunciation_features),
    }


# Fluency & Coherence Feedback

def _fluency_feedback(band: float, f: dict, is_off_topic: bool, is_tangential: bool, is_scripted: bool = False, exam_part: int = 1) -> str:
    """Generate fluency & coherence examiner commentary."""
    raw_band = _score_fluency(f)
    wpm = f.get("words_per_minute", 0)
    long_pauses = f.get("long_pause_count", 0)
    fillers = f.get("filler_count", 0)
    reps = f.get("repetition_count", 0)
    dm_variety = f.get("discourse_marker_variety", 0)
    dm_used = f.get("discourse_markers_used", [])
    word_count = f.get("word_count", 0)

    if raw_band >= 8.5:
        base = "Excellent job! You speak very naturally and smoothly, connecting your ideas perfectly without needing to stop and think."
    elif raw_band >= 7.5:
        base = "Great flow! You sound confident and your ideas connect well. When you did pause, it felt like a natural break rather than searching for words."
    elif raw_band >= 6.5:
        base = "Good effort! You kept the conversation going well. You hesitated and repeated yourself a few times, but overall it was easy to follow."
    elif raw_band >= 5.5:
        base = "You kept speaking, which is good, but you paused and repeated yourself quite a bit. Try to link your ideas more smoothly."
    elif raw_band >= 4.5:
        base = "You seem to be pausing often to search for words or fix mistakes. To get a higher score, focus on keeping your sentences flowing, even if you make a small mistake."
    elif raw_band >= 3.5:
        base = "Your speech has a lot of long pauses and repetitions, making it hard to follow your main point. Try practicing speaking out loud without stopping."
    elif raw_band >= 2.5:
        base = "You are pausing too much, which makes it difficult to understand your ideas. Try focusing on speaking continuously about simple topics first."
    else:
        base = "It was very difficult to follow your speech because of the long stops. Don't worry—start by practicing short, simple sentences without pausing."

    details = []

    duration = f.get("total_speaking_time_sec", 0)
    
    if exam_part == 1:
        if duration < 10 or word_count < 20:
            details.append("Your answer was way too short! For Part 1, aim to speak for 15-20 seconds (about 2-4 sentences).")
        elif duration < 14 or word_count < 30:
            details.append("Your answer was slightly short. Try to add one more detail to reach 15-20 seconds.")
    elif exam_part == 2:
        if duration < 30 or word_count < 60:
            details.append("Your answer was much too short. In Part 2, you need to speak continuously for 1 to 2 minutes.")
        elif duration < 60 or word_count < 120:
            details.append("Try to push yourself to speak for the full 1 to 2 minutes in Part 2.")
    elif exam_part == 3:
        if duration < 15 or word_count < 30:
            details.append("This was too short for Part 3. You need to explain your ideas deeply for 30-45 seconds.")
        elif duration < 25 or word_count < 50:
            details.append("Try to expand your reasoning more to hit that 30-45 second mark for Part 3.")

    if wpm > 0:
        if wpm >= 130:
            details.append("Your speaking speed is fast and natural.")
        elif wpm >= 100:
            details.append("Your speaking pace is great for IELTS.")
        elif wpm >= 75:
            details.append("You are speaking a bit slowly. Try picking up the pace slightly to sound more natural.")
        else:
            details.append("Your speaking speed is very slow. Practicing faster delivery will help your score.")

    if long_pauses > 0:
        details.append(f"I noticed {long_pauses} long pause{'s' if long_pauses != 1 else ''} (over 2 seconds). To score higher, try using filler phrases like 'Well, let me think...' instead of staying silent.")

    if fillers > 5:
        details.append(f"You used filler words ('um', 'ah', 'like') {fillers} times. Try to reduce these by taking a deep breath instead!")

    if dm_variety >= 4:
        examples = ", ".join(f"'{m}'" for m in dm_used[:3])
        details.append(f"Awesome! You used great linking words like {examples} to connect your ideas.")
    elif dm_variety >= 2:
        details.append("You used a few linking words, but try to use more variety (like 'however', 'moreover', 'for instance') to connect your thoughts better.")
    elif dm_variety == 0:
        details.append("You didn't use any linking words! To boost your score, start using words like 'however', 'therefore', and 'for example' to connect your sentences.")

    if is_off_topic:
        details.append("Note: It seems your response to at least one question was off-topic. Make sure you listen carefully and answer the exact question asked!")
    elif is_tangential:
        details.append("Note: Your response was partially off-topic or tangential to the question asked. Your score was penalized as a result.")

    if is_scripted:
        details.append("Note: Your response sounded like it was read from a textbook or memorised. IELTS examiners heavily penalise scripted speech. Try to sound more spontaneous and natural!")

    return _join_sentences(base, details)


# Lexical Resource Feedback

def _lexical_feedback(band: float, f: dict, is_off_topic: bool, is_tangential: bool, is_scripted: bool = False) -> str:
    """Generate Lexical Resource examiner commentary."""
    raw_band = _score_lexical(f)
    mattr = f.get("moving_avg_ttr", 0)
    adv_ratio = f.get("advanced_vocab_ratio", 0)
    adv_examples = f.get("advanced_vocab_examples", [])
    rep_freq = f.get("repetition_frequency", 0)
    paraphrase = f.get("paraphrase_indicator", False)
    para_count = f.get("paraphrase_count", 0)
    density = f.get("lexical_density", 0)
    collocations = f.get("collocations_detected", 0)

    if raw_band >= 8.5:
        base = "Fantastic vocabulary! You used a wide range of precise and advanced words perfectly. You sounded very natural."
    elif raw_band >= 7.5:
        base = "Great vocabulary! You used some impressive words and expressions naturally to explain your ideas."
    elif raw_band >= 6.5:
        base = "Good job. You have enough vocabulary to discuss things clearly, but sometimes you used the wrong word or expression. Trying to learn exact word pairings (collocations) will help you improve."
    elif raw_band >= 5.5:
        base = "Your vocabulary is okay for simple topics, but you struggle when talking about complex ideas. You tend to repeat the same words."
    elif raw_band >= 4.5:
        base = "You used very basic vocabulary and repeated the same simple words often. You need to learn more synonyms to express yourself better."
    elif raw_band >= 3.5:
        base = "Your vocabulary is very limited, which makes it hard for you to discuss unfamiliar topics. Try learning topic-specific vocabulary lists."
    else:
        base = "You only used very basic, isolated words. You need to focus on learning more English vocabulary to build full sentences."

    details = []

    if adv_ratio >= 0.08:
        ex_str = f" (like {', '.join(adv_examples[:3])})" if adv_examples else ""
        details.append(f"You used some excellent, high-level vocabulary{ex_str}! Keep it up.")
    elif adv_ratio >= 0.04:
        details.append("You used a few advanced words. To reach a Band 7+, try to incorporate even more sophisticated or less common words.")
    else:
        details.append("You mostly relied on basic, everyday words. Try learning 'less common' vocabulary to impress the examiner.")

    if mattr >= 0.65:
        details.append("You did a great job avoiding repetition by using lots of different words.")
    elif mattr >= 0.50:
        details.append("You repeated some words quite a bit. Try to use synonyms (different words with the same meaning).")
    else:
        details.append("You repeated the same words over and over. Expanding your vocabulary will fix this.")

    if paraphrase:
        details.append("Great job paraphrasing! Rephrasing ideas instead of getting stuck is exactly what examiners look for.")
    else:
        details.append("Try to use phrases like 'in other words' or 'what I mean is' to explain yourself if you get stuck.")

    if collocations >= 3:
        details.append("You naturally grouped words together (collocations) very well.")

    if is_off_topic:
        details.append("Note: Because your response drifted off-topic, your vocabulary score was penalized.")
    elif is_tangential:
        details.append("Note: Because your response was somewhat off-topic or tangential, your vocabulary score was penalized.")

    if is_scripted:
        details.append("Note: Because your response sounded memorised or read, your vocabulary score was capped. Examiners expect natural, spoken language rather than perfectly written text.")

    return _join_sentences(base, details)


# Grammatical Range & Accuracy Feedback

def _grammar_feedback(band: float, f: dict, is_tangential: bool = False) -> str:
    """Generate Grammatical Range & Accuracy examiner commentary."""
    raw_band = _score_grammar(f)
    err_per_100 = f.get("errors_per_100_words", 0)
    complex_ratio = f.get("complex_sentence_ratio", 0)
    tense_err = f.get("tense_errors", 0)
    agreement_err = f.get("agreement_errors", 0)
    article_err = f.get("article_errors", 0)
    prep_err = f.get("preposition_errors", 0)

    if raw_band >= 8.5:
        base = "Flawless grammar! You used a wide mix of complex sentences with almost zero mistakes."
    elif raw_band >= 7.5:
        base = "Excellent grammar. You confidently used complex sentences and your mistakes were very rare and minor."
    elif raw_band >= 6.5:
        base = "Good grammar! You mixed simple and complex sentences well. You made some mistakes, but they didn't make you hard to understand."
    elif raw_band >= 5.5:
        base = "You do well with basic sentences, but you make quite a few errors when you try to use complex, longer sentences. Focus on practicing sentence structures."
    elif raw_band >= 4.5:
        base = "You mostly use very simple, short sentences. Your grammar mistakes are frequent enough that it sometimes causes confusion."
    elif raw_band >= 3.5:
        base = "You made a lot of grammar errors that made it hard to understand your points. Try to master basic, simple sentences before moving on."
    else:
        base = "Your grammar is very limited. Focused study on the basic rules of English grammar is highly recommended."

    details = []

    if err_per_100 < 2 and raw_band >= 5.0:
        details.append("You made almost no mistakes. Fantastic accuracy!")
    elif err_per_100 < 5 and raw_band >= 5.0:
        details.append("Your accuracy is decent, but you still make noticeable errors. Proofread your speech mentally to catch them.")
    elif raw_band < 5.0:
        details.append("You make a lot of grammatical errors. Try to slow down and think about your grammar while speaking.")

    dominant_errors = []
    if tense_err > 0: dominant_errors.append("verb tenses (past/present/future)")
    if agreement_err > 0: dominant_errors.append("matching singular/plural subjects and verbs")
    if article_err > 0: dominant_errors.append("using 'a', 'an', or 'the' correctly")
    if prep_err > 0: dominant_errors.append("prepositions (in, on, at, etc.)")
    
    if dominant_errors:
        details.append("To improve fast, focus on fixing your mistakes with: " + ", ".join(dominant_errors) + ".")

    if complex_ratio >= 0.35 and raw_band >= 5.0:
        details.append("You used complex sentences beautifully. This is key for a high score!")
    elif complex_ratio >= 0.15 and raw_band >= 5.0:
        details.append("You used some complex sentences, but try to use more. Connect short sentences using words like 'because', 'although', or 'which'.")
    elif raw_band < 5.0:
        details.append("Your sentences are too short and simple. To get a higher score, practice linking them together with words like 'although', 'even if', or 'whereas'.")

    return _join_sentences(base, details)


# Pronunciation Feedback

def _pronunciation_feedback(band: float, f: dict) -> str:
    """Generate Pronunciation examiner commentary."""
    raw_band = _score_pronunciation(f)
    pitch_range = f.get("pitch_range_hz", 0)
    rhythm_cv = f.get("rhythm_regularity_cv", 1)
    avg_chunk = f.get("avg_words_per_chunk", 0)

    if raw_band >= 8.5:
        base = "Perfect pronunciation! Your accent, stress, and rhythm sound completely natural and effortless to understand."
    elif raw_band >= 7.5:
        base = "Great pronunciation! You use natural rhythm and stress, making you very easy to understand."
    elif raw_band >= 6.5:
        base = "Good job. You are generally easy to understand, though you sometimes mispronounce words or have flat intonation."
    elif raw_band >= 5.5:
        base = "Your pronunciation is okay, but it requires some effort from the listener to understand you. You might be struggling with word stress or speaking too flatly."
    elif raw_band >= 4.5:
        base = "Your pronunciation makes it hard to catch what you're saying at times. Focus on pronouncing the ends of your words clearly."
    elif raw_band >= 3.5:
        base = "It is very hard to understand your speech due to pronunciation issues. Try listening to native speakers and imitating how they say words."
    else:
        base = "Your speech is extremely difficult to understand. Heavy pronunciation practice is needed."

    details = []

    if pitch_range >= 100:
        details.append("Your voice goes up and down naturally (great intonation), making you sound engaging!")
    elif pitch_range >= 60:
        details.append("Your voice is a bit flat. Try putting more emotion and emphasis on important words to sound more natural.")
    elif pitch_range > 0:
        details.append("You speak in a very monotone, robotic voice. You need to practice making your voice go up and down to highlight important words.")

    if rhythm_cv < 0.3:
        details.append("Your speaking rhythm is steady and easy to follow.")
    elif rhythm_cv < 0.6:
        details.append("Your rhythm is a bit choppy. Try to flow from one word to the next more smoothly.")
    else:
        details.append("Your rhythm is very irregular. Try shadowing (repeating exactly after) a native speaker to get the feel of English rhythm.")

    if avg_chunk >= 7:
        details.append("You group words together perfectly without weird pauses in the middle of sentences.")
    elif avg_chunk >= 4:
        details.append("Try to speak in longer, connected phrases rather than stopping after every few words.")
    elif avg_chunk > 0:
        details.append("You are speaking word-by-word. Focus on linking words together into full phrases.")

    return _join_sentences(base, details)


# Utility

def _join_sentences(base: str, details: list[str]) -> str:
    """Combine base feedback with specific detail sentences into a paragraph."""
    if not details:
        return base
    return base + " " + " ".join(details)




def evaluate_pipeline(transcript: str, segments: list, word_timestamps: list, audio_path: str, audio_array, sample_rate: int, lang: str='en-GB', question: str=None, part: int=1) -> dict:
    import concurrent.futures
    t0 = time.time()
    audio_duration = len(audio_array) / sample_rate
    if not transcript:
        print('WARNING: Whisper returned an empty transcript. Returning 0.0 scores.', file=sys.stderr)
        feedback_msg = 'No speech was detected in your recording. Please ensure your microphone is working and speak clearly.'
        return {'fluency': 0.0, 'lexical': 0.0, 'grammar': 0.0, 'pronunciation': 0.0, 'overall': 0.0, 'user_input': '', 'features': {}, 'topic_relevance': 0.0 if question else None, 'feedback': {'fluency': feedback_msg, 'lexical': feedback_msg, 'grammar': feedback_msg, 'pronunciation': feedback_msg}, '_meta': {'audio_file': os.path.basename(audio_path), 'grammar_lang': lang, 'processing_time_sec': round(time.time() - t0, 2)}}
    print(f'\n  Transcript preview: "{transcript[:120]}…"\n', file=sys.stderr)
    print('[4/5] Extracting features (parallel)…', file=sys.stderr)

    def _run_fluency():
        print('    → Fluency & Coherence…', file=sys.stderr)
        return analyze_fluency(transcript, segments, word_timestamps, audio_duration)

    def _run_lexical():
        print('    → Lexical Resource…', file=sys.stderr)
        return analyze_lexical(transcript)

    def _run_grammar():
        print('    → Grammatical Range & Accuracy…', file=sys.stderr)
        return analyze_grammar(transcript, lang=lang)

    def _run_pronunciation():
        print('    → Pronunciation…', file=sys.stderr)
        return analyze_pronunciation(audio_path, word_timestamps, segments, audio_array=audio_array, sample_rate=sample_rate)

    def _run_relevance():
        if question:
            print('    → Topic Relevance…', file=sys.stderr)
            return analyze_relevance(transcript, question)
        return {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        f_fluency = pool.submit(_run_fluency)
        f_lexical = pool.submit(_run_lexical)
        f_grammar = pool.submit(_run_grammar)
        f_pronunciation = pool.submit(_run_pronunciation)
        f_relevance = pool.submit(_run_relevance)
        fluency_feats = f_fluency.result()
        lexical_feats = f_lexical.result()
        grammar_feats = f_grammar.result()
        pronunciation_feats = f_pronunciation.result()
        relevance_feats = f_relevance.result()
    print('[5/5] Computing IELTS bands…', file=sys.stderr)
    bands = compute_bands(fluency_feats, lexical_feats, grammar_feats, pronunciation_feats, relevance_feats, part)
    examiner_feedback = generate_feedback(bands, fluency_feats, lexical_feats, grammar_feats, pronunciation_feats, relevance_feats, part)
    elapsed = time.time() - t0
    result = {'fluency': bands['fluency'], 'lexical': bands['lexical'], 'grammar': bands['grammar'], 'pronunciation': bands['pronunciation'], 'overall': bands['overall'], 'user_input': transcript, 'features': {'fluency': fluency_feats, 'lexical': lexical_feats, 'grammar': grammar_feats, 'pronunciation': pronunciation_feats, 'relevance': relevance_feats}, 'topic_relevance': relevance_feats.get('max_sentence_similarity', None) if question else None, 'feedback': examiner_feedback, '_meta': {'audio_file': os.path.basename(audio_path), 'grammar_lang': lang, 'processing_time_sec': round(elapsed, 2)}}
    return result