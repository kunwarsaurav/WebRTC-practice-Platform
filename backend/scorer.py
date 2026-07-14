import argparse
import math
import os
import httpx
import json
import re
import sys
import threading
_transcribe_lock = threading.Lock()
import time
from dotenv import load_dotenv
load_dotenv()
from collections import Counter, defaultdict
from typing import Any
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
import librosa
import numpy as np
import parselmouth  # type: ignore
from parselmouth.praat import call  # type: ignore




def init_models():
    print('[INIT] No heavy models to load! Using Groq API for NLP.', file=sys.stderr)

def _query_groq_llm(system_prompt: str, user_prompt: str) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {}
    
    try:
        response = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0
            },
            timeout=30.0
        )
        response.raise_for_status()
        result = response.json()
        return json.loads(result["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Groq API Error: {e}", file=sys.stderr)
        return {}

FILLER_WORDS = {'uh', 'um', 'er', 'erm', 'hmm', 'hm', 'ah', 'oh', 'like', 'you know', 'i mean', 'sort of', 'kind of', 'basically', 'literally', 'actually', 'right'}
FILLER_PHRASES = {'you know', 'i mean', 'sort of', 'kind of'}
DISCOURSE_MARKERS = {'however', 'therefore', 'moreover', 'in addition', 'furthermore', 'additionally', 'consequently', 'nevertheless', 'nonetheless', 'on the other hand', 'on the contrary', 'in contrast', 'for example', 'for instance', 'such as', 'in particular', 'firstly', 'secondly', 'finally', 'in conclusion', 'to sum up', 'in other words', 'that is to say', 'as a result', 'because of this', 'well', 'actually', 'basically', 'naturally', 'clearly', 'obviously', 'certainly', 'indeed', 'of course'}
SELF_CORRECTION_SIGNALS = {'i mean', 'sorry', 'no wait', 'actually', 'let me', 'i should say', 'what i mean is', 'or rather', 'to be more precise'}
PAUSE_THRESHOLD_SEC = 0.25
LONG_PAUSE_THRESHOLD_SEC = 2.0

def analyze_fluency(transcript: str, segments: list[dict], word_timestamps: list[dict], audio_duration: float) -> dict[str, Any]:
    text_lower = transcript.lower()
    total_speaking_time = _speaking_time(word_timestamps)
    word_count = len(word_timestamps)
    wpm = _words_per_minute(word_count, total_speaking_time)
    pauses = _extract_pauses(word_timestamps)
    pause_count = len(pauses)
    avg_pause_dur = sum(pauses) / len(pauses) if pauses else 0.0
    long_pause_count = sum((1 for p in pauses if p >= LONG_PAUSE_THRESHOLD_SEC))
    filler_count, filler_positions = _count_fillers(word_timestamps)
    repetition_count = _count_repetitions(word_timestamps)
    self_correction_count = _count_self_corrections(text_lower)
    dm_count, dm_variety, dm_used = _count_discourse_markers(text_lower)
    actual_speaking_time = sum((w.get('end', 0) - w.get('start', 0) for w in word_timestamps if 'start' in w and 'end' in w))
    continuity_ratio = actual_speaking_time / total_speaking_time if total_speaking_time > 0 else 0.0
    return {'total_speaking_time_sec': round(total_speaking_time, 2), 'word_count': word_count, 'words_per_minute': round(wpm, 1), 'pause_count': pause_count, 'avg_pause_duration_sec': round(avg_pause_dur, 3), 'long_pause_count': long_pause_count, 'filler_count': filler_count, 'repetition_count': repetition_count, 'self_correction_count': self_correction_count, 'discourse_marker_count': dm_count, 'discourse_marker_variety': dm_variety, 'discourse_markers_used': dm_used, 'speech_continuity_ratio': round(continuity_ratio, 3)}

def _speaking_time(word_timestamps: list[dict]) -> float:
    if not word_timestamps:
        return 0.0
    starts = [w['start'] for w in word_timestamps if 'start' in w]
    ends = [w['end'] for w in word_timestamps if 'end' in w]
    if not starts or not ends:
        return 0.0
    return max(ends) - min(starts)

def _words_per_minute(word_count: int, speaking_time_sec: float) -> float:
    if speaking_time_sec <= 0:
        return 0.0
    return word_count / (speaking_time_sec / 60.0)

def _extract_pauses(word_timestamps: list[dict]) -> list[float]:
    pauses = []
    for i in range(1, len(word_timestamps)):
        prev_end = word_timestamps[i - 1].get('end', None)
        curr_start = word_timestamps[i].get('start', None)
        if prev_end is not None and curr_start is not None:
            gap = curr_start - prev_end
            if gap >= PAUSE_THRESHOLD_SEC:
                pauses.append(gap)
    return pauses

def _count_fillers(word_timestamps: list[dict]) -> tuple[int, list[int]]:
    count = 0
    positions = []
    words = [w.get('word', '').lower().strip(',.!?') for w in word_timestamps]
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            bigram = words[i] + ' ' + words[i + 1]
            if bigram in FILLER_PHRASES:
                count += 1
                positions.append(i)
                i += 2
                continue
        if words[i] in FILLER_WORDS:
            count += 1
            positions.append(i)
        i += 1
    return (count, positions)

def _count_repetitions(word_timestamps: list[dict]) -> int:
    words = [w.get('word', '').lower().strip(',.!?') for w in word_timestamps if w.get('word', '').strip()]
    STOP = {'the', 'a', 'an', 'is', 'it', 'i', 'in', 'on', 'at', 'to', 'of', 'and', 'or', 'but', 'so', 'for', 'with', 'was', 'are'}
    content_words = [w for w in words if w not in STOP and len(w) > 2]
    count = 0
    for i in range(1, len(content_words)):
        if content_words[i] == content_words[i - 1]:
            count += 1
        elif i >= 2 and content_words[i] == content_words[i - 2]:
            count += 1
    return count

def _count_self_corrections(text_lower: str) -> int:
    count = 0
    for phrase in SELF_CORRECTION_SIGNALS:
        count += text_lower.count(phrase)
    return count

def _count_discourse_markers(text_lower: str) -> tuple[int, int, list[str]]:
    used = []
    total = 0
    for marker in DISCOURSE_MARKERS:
        if ' ' in marker:
            occ = text_lower.count(marker)
        else:
            occ = len(re.findall('\\b' + re.escape(marker) + '\\b', text_lower))
        if occ > 0:
            used.append(marker)
            total += occ
    return (total, len(used), sorted(used))
CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV'}
ADVANCED_VOCAB_ZIPF_THRESHOLD = 4.0
MATTR_WINDOW = 50
PARAPHRASE_PHRASES = {'in other words', 'that is to say', 'put differently', 'what i mean is', 'to put it another way', 'in short', 'to clarify', 'that means', 'which means'}
STRONG_COLLOCATIONS = {('make', 'decision'), ('make', 'progress'), ('make', 'difference'), ('make', 'contribution'), ('make', 'effort'), ('make', 'mistake'), ('take', 'advantage'), ('take', 'responsibility'), ('take', 'action'), ('take', 'part'), ('take', 'place'), ('take', 'approach'), ('give', 'opportunity'), ('give', 'priority'), ('give', 'impression'), ('pay', 'attention'), ('pay', 'price'), ('pay', 'role'), ('raise', 'awareness'), ('raise', 'concern'), ('raise', 'question'), ('carry', 'out'), ('carry', 'responsibility'), ('come', 'across'), ('come', 'conclusion'), ('deal', 'with'), ('deal', 'issue'), ('look', 'forward'), ('look', 'issue'), ('play', 'role'), ('play', 'part'), ('face', 'challenge'), ('face', 'consequence'), ('meet', 'demand'), ('meet', 'need'), ('reach', 'goal'), ('reach', 'conclusion'), ('reach', 'agreement'), ('pose', 'threat'), ('pose', 'challenge'), ('build', 'relationship'), ('build', 'community'), ('develop', 'skill'), ('develop', 'understanding'), ('provide', 'opportunity'), ('provide', 'support'), ('provide', 'access'), ('gain', 'experience'), ('gain', 'knowledge'), ('gain', 'access'), ('achieve', 'goal'), ('achieve', 'balance'), ('achieve', 'success'), ('address', 'issue'), ('address', 'problem'), ('address', 'concern'), ('have', 'impact'), ('have', 'effect'), ('have', 'influence'), ('significant', 'impact'), ('significant', 'role'), ('significant', 'increase'), ('major', 'challenge'), ('major', 'issue'), ('major', 'factor'), ('crucial', 'role'), ('crucial', 'factor'), ('key', 'factor'), ('key', 'role'), ('key', 'issue'), ('vital', 'role'), ('vital', 'importance'), ('widespread', 'use'), ('widespread', 'concern'), ('growing', 'concern'), ('growing', 'number'), ('growing', 'demand'), ('increasing', 'number'), ('increasing', 'pressure'), ('strong', 'influence'), ('strong', 'evidence'), ('critical', 'thinking'), ('critical', 'issue'), ('global', 'problem'), ('global', 'issue'), ('global', 'community'), ('positive', 'impact'), ('negative', 'impact'), ('long', 'term'), ('short', 'term'), ('high', 'quality'), ('high', 'standard'), ('social', 'media'), ('social', 'issue'), ('social', 'problem'), ('social', 'responsibility'), ('social', 'inequality'), ('climate', 'change'), ('climate', 'crisis'), ('public', 'transport'), ('public', 'health'), ('public', 'sector'), ('economic', 'growth'), ('economic', 'development'), ('economic', 'impact'), ('human', 'rights'), ('human', 'nature'), ('human', 'development'), ('mental', 'health'), ('mental', 'wellbeing'), ('living', 'standard'), ('standard', 'living'), ('quality', 'life'), ('way', 'life'), ('government', 'policy'), ('education', 'system'), ('health', 'care'), ('job', 'opportunity'), ('work', 'life'), ('life', 'expectancy'), ('natural', 'environment'), ('natural', 'resource'), ('population', 'growth'), ('urban', 'development'), ('technological', 'advancement'), ('scientific', 'research'), ('financial', 'support'), ('financial', 'crisis')}

def analyze_lexical(transcript: str) -> dict[str, Any]:
    if not transcript.strip(): return _empty_lexical()
    
    sys_prompt = """You are an expert IELTS examiner analyzing a speaker's lexical resource.
Return a JSON object with the exact keys:
{
  "total_words": 0,
  "unique_words": 0,
  "type_token_ratio": 0.0,
  "moving_avg_ttr": 0.0,
  "lexical_density": 0.0,
  "advanced_vocab_count": 0,
  "advanced_vocab_ratio": 0.0,
  "advanced_vocab_examples": ["word1"],
  "high_repetition_words": ["word2"],
  "repetition_frequency": 0,
  "paraphrase_indicator": false,
  "paraphrase_count": 0,
  "avg_word_length": 0.0,
  "collocations_detected": 0
}
Ensure all values are appropriate numerical/boolean types."""
    
    res = _query_groq_llm(sys_prompt, transcript)
    if not res: return _empty_lexical()
    
    return {
        'total_words': res.get('total_words', len(transcript.split())),
        'unique_words': res.get('unique_words', 0),
        'type_token_ratio': float(res.get('type_token_ratio', 0.0)),
        'moving_avg_ttr': float(res.get('moving_avg_ttr', 0.0)),
        'lexical_density': float(res.get('lexical_density', 0.0)),
        'advanced_vocab_count': int(res.get('advanced_vocab_count', 0)),
        'advanced_vocab_ratio': float(res.get('advanced_vocab_ratio', 0.0)),
        'advanced_vocab_examples': res.get('advanced_vocab_examples', []),
        'high_repetition_words': res.get('high_repetition_words', []),
        'repetition_frequency': int(res.get('repetition_frequency', 0)),
        'paraphrase_indicator': bool(res.get('paraphrase_indicator', False)),
        'paraphrase_count': int(res.get('paraphrase_count', 0)),
        'avg_word_length': float(res.get('avg_word_length', 0.0)),
        'collocations_detected': int(res.get('collocations_detected', 0))
    }

def _empty_lexical() -> dict[str, Any]:
    return {'total_words': 0, 'unique_words': 0, 'type_token_ratio': 0.0, 'moving_avg_ttr': 0.0, 'lexical_density': 0.0, 'advanced_vocab_count': 0, 'advanced_vocab_ratio': 0.0, 'advanced_vocab_examples': [], 'high_repetition_words': [], 'repetition_frequency': 0, 'paraphrase_indicator': False, 'paraphrase_count': 0, 'avg_word_length': 0.0, 'collocations_detected': 0}

TENSE_RULE_IDS = {'ENGLISH_WORD_REPEAT_RULE', 'PAST_TENSE_WITH_WOULD', 'VERB_TENSE', 'PERFECT_TENSE'}
AGREEMENT_RULE_IDS = {'AGREEMENT_SENT_START', 'PRP_VB', 'DOES_X', 'HE_VERB_AGR', 'SV_AGREEMENT'}
ARTICLE_RULE_IDS = {'EN_A_VS_AN', 'THE_SUPERLATIVE', 'MISSING_ARTICLE', 'ARTICLE_MISSING', 'ARTICLE_REDUNDANT'}
PREPOSITION_RULE_IDS = {'AT_THE_WEEKEND', 'PREPOSITION_AFTER', 'ON_THE_WAY', 'IN_TIME_PERIOD', 'PREP_REDUNDANT'}

def analyze_grammar(transcript: str, lang: str='en-GB') -> dict[str, Any]:
    if not transcript.strip(): return _empty_grammar()

    sys_prompt = """You are an expert IELTS examiner analyzing a speaker's grammar.
Return a JSON object with the exact keys:
{
  "total_grammar_errors": 0,
  "errors_per_100_words": 0.0,
  "tense_errors": 0,
  "agreement_errors": 0,
  "article_errors": 0,
  "preposition_errors": 0,
  "other_errors": 0,
  "error_examples": [{"rule": "RULE", "message": "msg", "context": "ctx"}],
  "sentence_count": 0,
  "avg_sentence_length": 0.0,
  "subordinate_clause_count": 0,
  "subordinate_clause_freq": 0.0,
  "complex_sentence_ratio": 0.0,
  "compound_sentence_ratio": 0.0,
  "simple_sentence_ratio": 0.0,
  "sentence_variety_score": 0.0
}"""
    
    res = _query_groq_llm(sys_prompt, transcript)
    if not res: return _empty_grammar()
    
    return {
        'total_grammar_errors': int(res.get('total_grammar_errors', 0)),
        'errors_per_100_words': float(res.get('errors_per_100_words', 0.0)),
        'tense_errors': int(res.get('tense_errors', 0)),
        'agreement_errors': int(res.get('agreement_errors', 0)),
        'article_errors': int(res.get('article_errors', 0)),
        'preposition_errors': int(res.get('preposition_errors', 0)),
        'other_errors': int(res.get('other_errors', 0)),
        'error_examples': res.get('error_examples', []),
        'sentence_count': int(res.get('sentence_count', 0)),
        'avg_sentence_length': float(res.get('avg_sentence_length', 0.0)),
        'subordinate_clause_count': int(res.get('subordinate_clause_count', 0)),
        'subordinate_clause_freq': float(res.get('subordinate_clause_freq', 0.0)),
        'complex_sentence_ratio': float(res.get('complex_sentence_ratio', 0.0)),
        'compound_sentence_ratio': float(res.get('compound_sentence_ratio', 0.0)),
        'simple_sentence_ratio': float(res.get('simple_sentence_ratio', 0.0)),
        'sentence_variety_score': float(res.get('sentence_variety_score', 0.0))
    }

def _empty_grammar() -> dict[str, Any]:
    return {'total_grammar_errors': 0, 'errors_per_100_words': 0.0, 'tense_errors': 0, 'agreement_errors': 0, 'article_errors': 0, 'preposition_errors': 0, 'other_errors': 0, 'error_examples': [], 'sentence_count': 0, 'avg_sentence_length': 0.0, 'subordinate_clause_count': 0, 'subordinate_clause_freq': 0.0, 'complex_sentence_ratio': 0.0, 'compound_sentence_ratio': 0.0, 'simple_sentence_ratio': 0.0, 'sentence_variety_score': 0.0}

PITCH_FLOOR_HZ = 75.0
PITCH_CEILING_HZ = 500.0
CHUNK_PAUSE_THRESHOLD_SEC = 0.4
LONG_WORD_DURATION_SEC = 0.6

def analyze_pronunciation(audio_path: str, word_timestamps: list[dict], segments: list[dict], audio_array=None, sample_rate=None) -> dict[str, Any]:
    prosodic = _extract_prosodic_features(audio_path, audio_array, sample_rate)
    rhythmic = _extract_rhythmic_features(word_timestamps)
    return {**prosodic, **rhythmic}

def _extract_prosodic_features(audio_path: str, audio_array=None, sample_rate=None) -> dict[str, Any]:
    try:
        if audio_array is not None and sample_rate is not None:
            sound = parselmouth.Sound(audio_array, sampling_frequency=sample_rate)
        else:
            sound = parselmouth.Sound(audio_path)
    except Exception as e:
        return _empty_prosodic(f'Audio load error: {e}')
    pitch_obj = sound.to_pitch(time_step=0.01, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ)
    pitch_values = pitch_obj.selected_array['frequency']
    voiced_pitch = pitch_values[pitch_values > 0]
    if len(voiced_pitch) < 5:
        mean_pitch = 0.0
        pitch_range = 0.0
        pitch_std = 0.0
    else:
        mean_pitch = float(np.mean(voiced_pitch))
        pitch_range = float(np.max(voiced_pitch) - np.min(voiced_pitch))
        pitch_std = float(np.std(voiced_pitch))
    intensity_obj = sound.to_intensity(minimum_pitch=75.0, time_step=0.01)
    intensity_values = intensity_obj.values.T.flatten()
    intensity_values = intensity_values[intensity_values > 0]
    if len(intensity_values) < 5:
        mean_intensity = 0.0
        intensity_variation = 0.0
    else:
        mean_intensity = float(np.mean(intensity_values))
        intensity_variation = float(np.std(intensity_values))
    speaking_rate = _estimate_speaking_rate(audio_path, audio_array, sample_rate)
    voiced_segment_durations = _voiced_segment_durations(pitch_obj)
    if len(voiced_segment_durations) > 2:
        cv = float(np.std(voiced_segment_durations) / np.mean(voiced_segment_durations))
        rhythm_regularity = round(cv, 4)
    else:
        rhythm_regularity = 0.0
    return {'mean_pitch_hz': round(mean_pitch, 2), 'pitch_range_hz': round(pitch_range, 2), 'pitch_std_dev_hz': round(pitch_std, 2), 'mean_intensity_db': round(mean_intensity, 2), 'intensity_variation_db': round(intensity_variation, 2), 'speaking_rate_syl_per_sec': round(speaking_rate, 2), 'rhythm_regularity_cv': rhythm_regularity}

def _estimate_speaking_rate(audio_path: str, audio_array=None, sample_rate=None) -> float:
    try:
        if audio_array is not None and sample_rate is not None:
            y, sr = (audio_array, sample_rate)
        else:
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration <= 0:
            return 0.0
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time', pre_max=1, post_max=1, pre_avg=3, post_avg=3, delta=0.07, wait=0.06)
        syllable_count = len(onset_frames)
        return syllable_count / duration
    except Exception:
        return 0.0

def _voiced_segment_durations(pitch_obj) -> list[float]:
    frame_times = pitch_obj.ts()
    pitch_values = pitch_obj.selected_array['frequency']
    durations = []
    in_voiced = False
    seg_start = 0.0
    for t, p in zip(frame_times, pitch_values):
        if p > 0 and (not in_voiced):
            in_voiced = True
            seg_start = t
        elif p == 0 and in_voiced:
            in_voiced = False
            durations.append(t - seg_start)
    if in_voiced and len(frame_times) > 0:
        durations.append(frame_times[-1] - seg_start)
    return [d for d in durations if d > 0.05]

def _extract_rhythmic_features(word_timestamps: list[dict]) -> dict[str, Any]:
    if not word_timestamps:
        return _empty_rhythmic()
    word_durations = []
    for w in word_timestamps:
        s = w.get('start')
        e = w.get('end')
        if s is not None and e is not None and (e > s):
            word_durations.append(e - s)
    if not word_durations:
        return _empty_rhythmic()
    duration_variance = float(np.var(word_durations))
    long_words = sum((1 for d in word_durations if d >= LONG_WORD_DURATION_SEC))
    long_word_ratio = long_words / len(word_durations)
    chunks = _detect_chunks(word_timestamps)
    chunk_count = len(chunks)
    avg_words_per_chunk = np.mean([len(c) for c in chunks]) if chunks else 0.0
    return {'word_duration_variance': round(duration_variance, 4), 'long_word_ratio': round(long_word_ratio, 4), 'chunk_boundary_count': chunk_count, 'avg_words_per_chunk': round(float(avg_words_per_chunk), 2)}

def _detect_chunks(word_timestamps: list[dict]) -> list[list[dict]]:
    if not word_timestamps:
        return []
    chunks = []
    current_chunk = [word_timestamps[0]]
    for i in range(1, len(word_timestamps)):
        prev_end = word_timestamps[i - 1].get('end', 0)
        curr_start = word_timestamps[i].get('start', 0)
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

def _empty_prosodic(error: str='') -> dict[str, Any]:
    return {'mean_pitch_hz': 0.0, 'pitch_range_hz': 0.0, 'pitch_std_dev_hz': 0.0, 'mean_intensity_db': 0.0, 'intensity_variation_db': 0.0, 'speaking_rate_syl_per_sec': 0.0, 'rhythm_regularity_cv': 0.0, '_error': error}

def _empty_rhythmic() -> dict[str, Any]:
    return {'word_duration_variance': 0.0, 'long_word_ratio': 0.0, 'chunk_boundary_count': 0, 'avg_words_per_chunk': 0.0}
OFF_TOPIC_THRESHOLD = 0.35
TANGENTIAL_THRESHOLD = 0.5

def analyze_relevance(transcript: str, question: str) -> dict[str, Any]:
    if not transcript.strip() or not question.strip(): return _empty_relevance()
        
    sys_prompt = """You are an AI assistant analyzing if a spoken transcript answers the provided question.
Return a JSON object with the exact keys:
{
  "overall_similarity": 0.0,
  "max_sentence_similarity": 0.0,
  "is_off_topic": false,
  "is_tangential": false,
  "question_provided": true
}"""

    res = _query_groq_llm(sys_prompt, f"Question: {question}\\n\\nTranscript: {transcript}")
    if not res: return _empty_relevance()
    return {
        'overall_similarity': float(res.get('overall_similarity', 0.0)),
        'max_sentence_similarity': float(res.get('max_sentence_similarity', 0.0)),
        'is_off_topic': bool(res.get('is_off_topic', False)),
        'is_tangential': bool(res.get('is_tangential', False)),
        'question_provided': True
    }

def _empty_relevance() -> dict[str, Any]:
    return {'overall_similarity': 0.0, 'max_sentence_similarity': 0.0, 'is_off_topic': False, 'is_tangential': False, 'question_provided': False}


def compute_bands(fluency_features: dict, lexical_features: dict, grammar_features: dict, pronunciation_features: dict, relevance_features: dict, exam_part: int=1) -> dict[str, float]:
    fluency_band = _score_fluency(fluency_features)
    lexical_band = _score_lexical(lexical_features)
    grammar_band = _score_grammar(grammar_features)
    pronunciation_band = _score_pronunciation(pronunciation_features)
    is_off_topic = relevance_features.get('is_off_topic', False)
    is_tangential = relevance_features.get('is_tangential', False)
    if is_off_topic:
        fluency_band = 1.0
        lexical_band = 1.0
        grammar_band = 1.0
        pronunciation_band = 1.0
        overall = 1.0
    else:
        if is_tangential:
            fluency_band = min(float(fluency_band), 5.0)
            lexical_band = min(float(lexical_band), 5.0)
        word_count = fluency_features.get('word_count', 0)
        duration = fluency_features.get('total_speaking_time_sec', 0)
        if exam_part == 1:
            if duration < 10 or word_count < 20:
                fluency_band = min(float(fluency_band), 6.0)
                lexical_band = min(float(lexical_band), 6.0)
                grammar_band = min(float(grammar_band), 6.0)
            elif duration < 14 or word_count < 30:
                fluency_band = min(float(fluency_band), 7.0)
                lexical_band = min(float(lexical_band), 7.0)
        elif exam_part == 2:
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
            if duration < 15 or word_count < 30:
                fluency_band = min(float(fluency_band), 5.5)
                lexical_band = min(float(lexical_band), 5.5)
                grammar_band = min(float(grammar_band), 5.5)
            elif duration < 25 or word_count < 50:
                fluency_band = min(float(fluency_band), 6.5)
                lexical_band = min(float(lexical_band), 6.5)
        pitch_range = pronunciation_features.get('pitch_range_hz', 0)
        pitch_std = pronunciation_features.get('pitch_std_dev_hz', 0)
        fillers = fluency_features.get('filler_count', 0)
        word_count = fluency_features.get('word_count', 0)
        density = lexical_features.get('lexical_density', 0)
        words_100 = max(word_count, 1) / 100.0
        filler_rate = fillers / words_100
        if word_count > 30:
            is_flat = pitch_range < 60 or pitch_std < 15
            is_too_fluent = filler_rate < 1.0
            is_written_style = density >= 0.48
            if is_flat and is_too_fluent and is_written_style:
                fluency_band = min(float(fluency_band), 5.0)
                lexical_band = min(float(lexical_band), 5.0)
                relevance_features['is_scripted'] = True
        overall = _compute_overall(fluency_band, lexical_band, grammar_band, pronunciation_band)
    return {'fluency': float(fluency_band), 'lexical': lexical_band, 'grammar': grammar_band, 'pronunciation': pronunciation_band, 'overall': overall}

def _score_fluency(f: dict) -> float:
    wpm = f.get('words_per_minute', 0)
    long_pauses = f.get('long_pause_count', 99)
    pause_count = f.get('pause_count', 99)
    fillers = f.get('filler_count', 99)
    reps = f.get('repetition_count', 99)
    corrections = f.get('self_correction_count', 0)
    dm_variety = f.get('discourse_marker_variety', 0)
    dm_count = f.get('discourse_marker_count', 0)
    continuity = f.get('speech_continuity_ratio', 0)
    word_count = f.get('word_count', 0)
    words_100 = max(word_count, 1) / 100.0
    filler_rate = fillers / words_100
    rep_rate = reps / words_100
    corr_rate = corrections / words_100
    b9 = _evidence_score([wpm >= 130, long_pauses == 0, filler_rate < 1.0, rep_rate < 1.0, corr_rate < 1.0, dm_variety >= 6, continuity >= 0.9])
    b8 = _evidence_score([wpm >= 120, long_pauses <= 1, filler_rate < 2.0, rep_rate < 2.0, dm_variety >= 5, continuity >= 0.85])
    b7 = _evidence_score([wpm >= 100, long_pauses <= 2, filler_rate < 4.0, rep_rate < 4.0, dm_variety >= 3, continuity >= 0.78])
    b6 = _evidence_score([wpm >= 85, long_pauses <= 4, filler_rate < 7.0, rep_rate < 6.0, dm_variety >= 2, continuity >= 0.68])
    b5 = _evidence_score([wpm >= 70, filler_rate >= 5.0 or rep_rate >= 5.0, long_pauses <= 6, continuity >= 0.55])
    b4 = _evidence_score([wpm >= 50, long_pauses <= 10, continuity >= 0.4])
    b3 = _evidence_score([wpm >= 30, long_pauses > 6, continuity < 0.55])
    b2 = _evidence_score([wpm < 30, continuity < 0.35])
    b1 = word_count > 0 and word_count < 15
    if word_count == 0:
        return 0.0
    if b1:
        return 1.0
    band = _select_band(b2, b3, b4, b5, b6, b7, b8, b9)
    band = _interpolate(band, {9: b9, 8: b8, 7: b7, 6: b6, 5: b5, 4: b4, 3: b3, 2: b2})
    avg_pause_dur = f.get('avg_pause_duration_sec', 0)
    if wpm < 90 or avg_pause_dur > 1.5:
        band = min(band, 5.5)
    return round(band * 2) / 2

def _score_lexical(f: dict) -> float:
    ttr = f.get('type_token_ratio', 0)
    mattr = f.get('moving_avg_ttr', 0)
    density = f.get('lexical_density', 0)
    adv_ratio = f.get('advanced_vocab_ratio', 0)
    adv_count = f.get('advanced_vocab_count', 0)
    rep_freq = f.get('repetition_frequency', 99)
    paraphrase = f.get('paraphrase_indicator', False)
    para_count = f.get('paraphrase_count', 0)
    avg_wlen = f.get('avg_word_length', 0)
    collocations = f.get('collocations_detected', 0)
    total_words = f.get('total_words', 0)
    rep_burden = rep_freq / max(total_words / 100, 1)
    b9 = _evidence_score([mattr >= 0.78, adv_ratio >= 0.12, density >= 0.56, para_count >= 2, collocations >= 4, rep_burden < 1.0, avg_wlen >= 5.2])
    b8 = _evidence_score([mattr >= 0.72, adv_ratio >= 0.09, density >= 0.52, paraphrase, collocations >= 3, rep_burden < 1.5, avg_wlen >= 4.8])
    b7 = _evidence_score([mattr >= 0.65, adv_ratio >= 0.06, density >= 0.48, paraphrase, collocations >= 2, rep_burden < 2.5, avg_wlen >= 4.5])
    b6 = _evidence_score([mattr >= 0.58, adv_ratio >= 0.04, density >= 0.44, rep_burden < 4.0, avg_wlen >= 4.2])
    b5 = _evidence_score([mattr >= 0.5, adv_ratio >= 0.02, density >= 0.4, rep_burden < 6.0])
    b4 = _evidence_score([mattr >= 0.4, density >= 0.35, rep_burden >= 5.0])
    b3 = _evidence_score([mattr >= 0.3, density < 0.4, adv_ratio < 0.02])
    b2 = _evidence_score([mattr < 0.3, total_words < 30])
    if total_words == 0:
        return 0.0
    if total_words < 10:
        return 1.0
    band = _select_band(b2, b3, b4, b5, b6, b7, b8, b9)
    band = _interpolate(band, {9: b9, 8: b8, 7: b7, 6: b6, 5: b5, 4: b4, 3: b3, 2: b2})
    return round(band * 2) / 2

def _score_grammar(f: dict) -> float:
    err_per_100 = f.get('errors_per_100_words', 99)
    complex_ratio = f.get('complex_sentence_ratio', 0)
    compound_ratio = f.get('compound_sentence_ratio', 0)
    simple_ratio = f.get('simple_sentence_ratio', 1)
    variety = f.get('sentence_variety_score', 0)
    avg_sent_len = f.get('avg_sentence_length', 0)
    subord_freq = f.get('subordinate_clause_freq', 0)
    sentence_count = f.get('sentence_count', 0)
    b9 = _evidence_score([err_per_100 < 1.0, complex_ratio >= 0.45, variety >= 0.9, subord_freq >= 1.0, avg_sent_len >= 14])
    b8 = _evidence_score([err_per_100 < 2.5, complex_ratio >= 0.38, variety >= 0.67, subord_freq >= 0.7, avg_sent_len >= 12])
    b7 = _evidence_score([err_per_100 < 4.0, complex_ratio >= 0.3, variety >= 0.67, subord_freq >= 0.5, avg_sent_len >= 11])
    b6 = _evidence_score([err_per_100 < 6.0, complex_ratio >= 0.2, variety >= 0.33, avg_sent_len >= 9])
    b5 = _evidence_score([err_per_100 < 9.0, complex_ratio >= 0.1, avg_sent_len >= 7])
    b4 = _evidence_score([err_per_100 < 14.0, complex_ratio < 0.15, avg_sent_len >= 5])
    b3 = _evidence_score([err_per_100 >= 12.0, complex_ratio < 0.1])
    b2 = _evidence_score([err_per_100 >= 20.0, sentence_count < 3])
    if sentence_count == 0:
        return 0.0
    if sentence_count < 2 and err_per_100 > 20:
        return 1.0
    band = _select_band(b2, b3, b4, b5, b6, b7, b8, b9)
    band = _interpolate(band, {9: b9, 8: b8, 7: b7, 6: b6, 5: b5, 4: b4, 3: b3, 2: b2})
    return round(band * 2) / 2

def _score_pronunciation(f: dict) -> float:
    pitch_range = f.get('pitch_range_hz', 0)
    pitch_std = f.get('pitch_std_dev_hz', 0)
    intensity_var = f.get('intensity_variation_db', 0)
    speaking_rate = f.get('speaking_rate_syl_per_sec', 0)
    rhythm_cv = f.get('rhythm_regularity_cv', 99)
    avg_chunk = f.get('avg_words_per_chunk', 0)
    long_word_ratio = f.get('long_word_ratio', 0)
    word_dur_var = f.get('word_duration_variance', 0)
    rhythm_score = max(0.0, 1.0 - min(rhythm_cv, 1.0))
    b9 = _evidence_score([pitch_range >= 120, pitch_std >= 35, intensity_var >= 6.0, speaking_rate >= 4.0, rhythm_score >= 0.8, avg_chunk >= 7])
    b8 = _evidence_score([pitch_range >= 95, pitch_std >= 28, intensity_var >= 5.0, speaking_rate >= 3.5, rhythm_score >= 0.72, avg_chunk >= 6])
    b7 = _evidence_score([pitch_range >= 75, pitch_std >= 22, intensity_var >= 4.0, speaking_rate >= 3.0, rhythm_score >= 0.62, avg_chunk >= 5])
    b6 = _evidence_score([pitch_range >= 55, pitch_std >= 16, intensity_var >= 3.0, speaking_rate >= 2.5, rhythm_score >= 0.5, avg_chunk >= 4])
    b5 = _evidence_score([pitch_range >= 40, pitch_std >= 10, speaking_rate >= 2.0, rhythm_score >= 0.38])
    b4 = _evidence_score([pitch_range >= 25, speaking_rate >= 1.5, rhythm_score >= 0.25])
    b3 = _evidence_score([pitch_range < 30, rhythm_score < 0.35])
    b2 = _evidence_score([pitch_range < 15, speaking_rate < 1.5, rhythm_score < 0.2])
    if speaking_rate == 0 and pitch_range == 0:
        return 0.0
    band = _select_band(b2, b3, b4, b5, b6, b7, b8, b9)
    band = _interpolate(band, {9: b9, 8: b8, 7: b7, 6: b6, 5: b5, 4: b4, 3: b3, 2: b2})
    if rhythm_score < 0.4:
        band = min(band, 6.0)
    return round(band * 2) / 2

def _compute_overall(f: float, l: float, g: float, p: float) -> float:
    mean = (f + l + g + p) / 4.0
    return math.floor(mean * 2 + 0.5) / 2.0

def _evidence_score(conditions: list[bool]) -> float:
    if not conditions:
        return 0.0
    return sum((1 for c in conditions if c)) / len(conditions)

def _select_band(b2, b3, b4, b5, b6, b7, b8, b9) -> float:
    scores = [(9, b9), (8, b8), (7, b7), (6, b6), (5, b5), (4, b4), (3, b3), (2, b2)]
    for band, score in scores:
        if score >= 0.7:
            return float(band)
    return 2.0

def _interpolate(band: float, scores: dict[int, float]) -> float:
    current_score = scores.get(int(band), 0.5)
    above_score = scores.get(int(band) + 1, 0.0)
    below_score = scores.get(int(band) - 1, 0.0)
    if current_score >= 0.65 and above_score >= 0.4:
        band += 0.5
    elif current_score <= 0.55 and below_score >= 0.55:
        band -= 0.5
    return max(1.0, min(9.0, band))

def generate_feedback(bands: dict[str, float], fluency_features: dict, lexical_features: dict, grammar_features: dict, pronunciation_features: dict, relevance_features: dict, exam_part: int=1) -> dict[str, str]:
    is_off_topic = relevance_features.get('is_off_topic', False)
    is_tangential = relevance_features.get('is_tangential', False)
    is_scripted = relevance_features.get('is_scripted', False)
    return {'fluency': _fluency_feedback(bands['fluency'], fluency_features, is_off_topic, is_tangential, is_scripted, exam_part), 'lexical': _lexical_feedback(bands['lexical'], lexical_features, is_off_topic, is_tangential, is_scripted), 'grammar': _grammar_feedback(bands['grammar'], grammar_features, is_tangential), 'pronunciation': _pronunciation_feedback(bands['pronunciation'], pronunciation_features)}

def _fluency_feedback(band: float, f: dict, is_off_topic: bool, is_tangential: bool, is_scripted: bool=False, exam_part: int=1) -> str:
    raw_band = _score_fluency(f)
    wpm = f.get('words_per_minute', 0)
    long_pauses = f.get('long_pause_count', 0)
    fillers = f.get('filler_count', 0)
    reps = f.get('repetition_count', 0)
    dm_variety = f.get('discourse_marker_variety', 0)
    dm_used = f.get('discourse_markers_used', [])
    word_count = f.get('word_count', 0)
    if raw_band >= 8.5:
        base = 'Excellent job! You speak very naturally and smoothly, connecting your ideas perfectly without needing to stop and think.'
    elif raw_band >= 7.5:
        base = 'Great flow! You sound confident and your ideas connect well. When you did pause, it felt like a natural break rather than searching for words.'
    elif raw_band >= 6.5:
        base = 'Good effort! You kept the conversation going well. You hesitated and repeated yourself a few times, but overall it was easy to follow.'
    elif raw_band >= 5.5:
        base = 'You kept speaking, which is good, but you paused and repeated yourself quite a bit. Try to link your ideas more smoothly.'
    elif raw_band >= 4.5:
        base = 'You seem to be pausing often to search for words or fix mistakes. To get a higher score, focus on keeping your sentences flowing, even if you make a small mistake.'
    elif raw_band >= 3.5:
        base = 'Your speech has a lot of long pauses and repetitions, making it hard to follow your main point. Try practicing speaking out loud without stopping.'
    elif raw_band >= 2.5:
        base = 'You are pausing too much, which makes it difficult to understand your ideas. Try focusing on speaking continuously about simple topics first.'
    else:
        base = "It was very difficult to follow your speech because of the long stops. Don't worry—start by practicing short, simple sentences without pausing."
    details = []
    duration = f.get('total_speaking_time_sec', 0)
    if exam_part == 1:
        if duration < 10 or word_count < 20:
            details.append('Your answer was way too short! For Part 1, aim to speak for 15-20 seconds (about 2-4 sentences).')
        elif duration < 14 or word_count < 30:
            details.append('Your answer was slightly short. Try to add one more detail to reach 15-20 seconds.')
    elif exam_part == 2:
        if duration < 30 or word_count < 60:
            details.append('Your answer was much too short. In Part 2, you need to speak continuously for 1 to 2 minutes.')
        elif duration < 60 or word_count < 120:
            details.append('Try to push yourself to speak for the full 1 to 2 minutes in Part 2.')
    elif exam_part == 3:
        if duration < 15 or word_count < 30:
            details.append('This was too short for Part 3. You need to explain your ideas deeply for 30-45 seconds.')
        elif duration < 25 or word_count < 50:
            details.append('Try to expand your reasoning more to hit that 30-45 second mark for Part 3.')
    if wpm > 0:
        if wpm >= 130:
            details.append('Your speaking speed is fast and natural.')
        elif wpm >= 100:
            details.append('Your speaking pace is great for IELTS.')
        elif wpm >= 75:
            details.append('You are speaking a bit slowly. Try picking up the pace slightly to sound more natural.')
        else:
            details.append('Your speaking speed is very slow. Practicing faster delivery will help your score.')
    if long_pauses > 0:
        details.append(f"I noticed {long_pauses} long pause{('s' if long_pauses != 1 else '')} (over 2 seconds). To score higher, try using filler phrases like 'Well, let me think...' instead of staying silent.")
    if fillers > 5:
        details.append(f"You used filler words ('um', 'ah', 'like') {fillers} times. Try to reduce these by taking a deep breath instead!")
    if dm_variety >= 4:
        examples = ', '.join((f"'{m}'" for m in dm_used[:3]))
        details.append(f'Awesome! You used great linking words like {examples} to connect your ideas.')
    elif dm_variety >= 2:
        details.append("You used a few linking words, but try to use more variety (like 'however', 'moreover', 'for instance') to connect your thoughts better.")
    elif dm_variety == 0:
        details.append("You didn't use any linking words! To boost your score, start using words like 'however', 'therefore', and 'for example' to connect your sentences.")
    if is_off_topic:
        details.append('Note: It seems your response to at least one question was off-topic. Make sure you listen carefully and answer the exact question asked!')
    elif is_tangential:
        details.append('Note: Your response was partially off-topic or tangential to the question asked. Your score was penalized as a result.')
    if is_scripted:
        details.append('Note: Your response sounded like it was read from a textbook or memorised. IELTS examiners heavily penalise scripted speech. Try to sound more spontaneous and natural!')
    return _join_sentences(base, details)

def _lexical_feedback(band: float, f: dict, is_off_topic: bool, is_tangential: bool, is_scripted: bool=False) -> str:
    raw_band = _score_lexical(f)
    mattr = f.get('moving_avg_ttr', 0)
    adv_ratio = f.get('advanced_vocab_ratio', 0)
    adv_examples = f.get('advanced_vocab_examples', [])
    rep_freq = f.get('repetition_frequency', 0)
    paraphrase = f.get('paraphrase_indicator', False)
    para_count = f.get('paraphrase_count', 0)
    density = f.get('lexical_density', 0)
    collocations = f.get('collocations_detected', 0)
    if raw_band >= 8.5:
        base = 'Fantastic vocabulary! You used a wide range of precise and advanced words perfectly. You sounded very natural.'
    elif raw_band >= 7.5:
        base = 'Great vocabulary! You used some impressive words and expressions naturally to explain your ideas.'
    elif raw_band >= 6.5:
        base = 'Good job. You have enough vocabulary to discuss things clearly, but sometimes you used the wrong word or expression. Trying to learn exact word pairings (collocations) will help you improve.'
    elif raw_band >= 5.5:
        base = 'Your vocabulary is okay for simple topics, but you struggle when talking about complex ideas. You tend to repeat the same words.'
    elif raw_band >= 4.5:
        base = 'You used very basic vocabulary and repeated the same simple words often. You need to learn more synonyms to express yourself better.'
    elif raw_band >= 3.5:
        base = 'Your vocabulary is very limited, which makes it hard for you to discuss unfamiliar topics. Try learning topic-specific vocabulary lists.'
    else:
        base = 'You only used very basic, isolated words. You need to focus on learning more English vocabulary to build full sentences.'
    details = []
    if adv_ratio >= 0.08:
        ex_str = f" (like {', '.join(adv_examples[:3])})" if adv_examples else ''
        details.append(f'You used some excellent, high-level vocabulary{ex_str}! Keep it up.')
    elif adv_ratio >= 0.04:
        details.append('You used a few advanced words. To reach a Band 7+, try to incorporate even more sophisticated or less common words.')
    else:
        details.append("You mostly relied on basic, everyday words. Try learning 'less common' vocabulary to impress the examiner.")
    if mattr >= 0.65:
        details.append('You did a great job avoiding repetition by using lots of different words.')
    elif mattr >= 0.5:
        details.append('You repeated some words quite a bit. Try to use synonyms (different words with the same meaning).')
    else:
        details.append('You repeated the same words over and over. Expanding your vocabulary will fix this.')
    if paraphrase:
        details.append('Great job paraphrasing! Rephrasing ideas instead of getting stuck is exactly what examiners look for.')
    else:
        details.append("Try to use phrases like 'in other words' or 'what I mean is' to explain yourself if you get stuck.")
    if collocations >= 3:
        details.append('You naturally grouped words together (collocations) very well.')
    if is_off_topic:
        details.append('Note: Because your response drifted off-topic, your vocabulary score was penalized.')
    elif is_tangential:
        details.append('Note: Because your response was somewhat off-topic or tangential, your vocabulary score was penalized.')
    if is_scripted:
        details.append('Note: Because your response sounded memorised or read, your vocabulary score was capped. Examiners expect natural, spoken language rather than perfectly written text.')
    return _join_sentences(base, details)

def _grammar_feedback(band: float, f: dict, is_tangential: bool=False) -> str:
    raw_band = _score_grammar(f)
    err_per_100 = f.get('errors_per_100_words', 0)
    complex_ratio = f.get('complex_sentence_ratio', 0)
    tense_err = f.get('tense_errors', 0)
    agreement_err = f.get('agreement_errors', 0)
    article_err = f.get('article_errors', 0)
    prep_err = f.get('preposition_errors', 0)
    if raw_band >= 8.5:
        base = 'Flawless grammar! You used a wide mix of complex sentences with almost zero mistakes.'
    elif raw_band >= 7.5:
        base = 'Excellent grammar. You confidently used complex sentences and your mistakes were very rare and minor.'
    elif raw_band >= 6.5:
        base = "Good grammar! You mixed simple and complex sentences well. You made some mistakes, but they didn't make you hard to understand."
    elif raw_band >= 5.5:
        base = 'You do well with basic sentences, but you make quite a few errors when you try to use complex, longer sentences. Focus on practicing sentence structures.'
    elif raw_band >= 4.5:
        base = 'You mostly use very simple, short sentences. Your grammar mistakes are frequent enough that it sometimes causes confusion.'
    elif raw_band >= 3.5:
        base = 'You made a lot of grammar errors that made it hard to understand your points. Try to master basic, simple sentences before moving on.'
    else:
        base = 'Your grammar is very limited. Focused study on the basic rules of English grammar is highly recommended.'
    details = []
    if err_per_100 < 2 and raw_band >= 5.0:
        details.append('You made almost no mistakes. Fantastic accuracy!')
    elif err_per_100 < 5 and raw_band >= 5.0:
        details.append('Your accuracy is decent, but you still make noticeable errors. Proofread your speech mentally to catch them.')
    elif raw_band < 5.0:
        details.append('You make a lot of grammatical errors. Try to slow down and think about your grammar while speaking.')
    dominant_errors = []
    if tense_err > 0:
        dominant_errors.append('verb tenses (past/present/future)')
    if agreement_err > 0:
        dominant_errors.append('matching singular/plural subjects and verbs')
    if article_err > 0:
        dominant_errors.append("using 'a', 'an', or 'the' correctly")
    if prep_err > 0:
        dominant_errors.append('prepositions (in, on, at, etc.)')
    if dominant_errors:
        details.append('To improve fast, focus on fixing your mistakes with: ' + ', '.join(dominant_errors) + '.')
    if complex_ratio >= 0.35 and raw_band >= 5.0:
        details.append('You used complex sentences beautifully. This is key for a high score!')
    elif complex_ratio >= 0.15 and raw_band >= 5.0:
        details.append("You used some complex sentences, but try to use more. Connect short sentences using words like 'because', 'although', or 'which'.")
    elif raw_band < 5.0:
        details.append("Your sentences are too short and simple. To get a higher score, practice linking them together with words like 'although', 'even if', or 'whereas'.")
    return _join_sentences(base, details)

def _pronunciation_feedback(band: float, f: dict) -> str:
    raw_band = _score_pronunciation(f)
    pitch_range = f.get('pitch_range_hz', 0)
    rhythm_cv = f.get('rhythm_regularity_cv', 1)
    avg_chunk = f.get('avg_words_per_chunk', 0)
    if raw_band >= 8.5:
        base = 'Perfect pronunciation! Your accent, stress, and rhythm sound completely natural and effortless to understand.'
    elif raw_band >= 7.5:
        base = 'Great pronunciation! You use natural rhythm and stress, making you very easy to understand.'
    elif raw_band >= 6.5:
        base = 'Good job. You are generally easy to understand, though you sometimes mispronounce words or have flat intonation.'
    elif raw_band >= 5.5:
        base = 'Your pronunciation is okay, but it requires some effort from the listener to understand you. You might be struggling with word stress or speaking too flatly.'
    elif raw_band >= 4.5:
        base = "Your pronunciation makes it hard to catch what you're saying at times. Focus on pronouncing the ends of your words clearly."
    elif raw_band >= 3.5:
        base = 'It is very hard to understand your speech due to pronunciation issues. Try listening to native speakers and imitating how they say words.'
    else:
        base = 'Your speech is extremely difficult to understand. Heavy pronunciation practice is needed.'
    details = []
    if pitch_range >= 100:
        details.append('Your voice goes up and down naturally (great intonation), making you sound engaging!')
    elif pitch_range >= 60:
        details.append('Your voice is a bit flat. Try putting more emotion and emphasis on important words to sound more natural.')
    elif pitch_range > 0:
        details.append('You speak in a very monotone, robotic voice. You need to practice making your voice go up and down to highlight important words.')
    if rhythm_cv < 0.3:
        details.append('Your speaking rhythm is steady and easy to follow.')
    elif rhythm_cv < 0.6:
        details.append('Your rhythm is a bit choppy. Try to flow from one word to the next more smoothly.')
    else:
        details.append('Your rhythm is very irregular. Try shadowing (repeating exactly after) a native speaker to get the feel of English rhythm.')
    if avg_chunk >= 7:
        details.append('You group words together perfectly without weird pauses in the middle of sentences.')
    elif avg_chunk >= 4:
        details.append('Try to speak in longer, connected phrases rather than stopping after every few words.')
    elif avg_chunk > 0:
        details.append('You are speaking word-by-word. Focus on linking words together into full phrases.')
    return _join_sentences(base, details)

def _join_sentences(base: str, details: list[str]) -> str:
    if not details:
        return base
    return base + ' ' + ' '.join(details)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='IELTS Speaking Scorer', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  python main.py --audio sample_audio/test.wav\n  python main.py --audio interview.mp3 --output results.json\n  python main.py --audio test.wav --lang en-US --device cpu\n        ')
    p.add_argument('--audio', required=True, help='Path to the audio file (.wav, .mp3, .m4a, .flac)')
    p.add_argument('--output', default=None, help='Optional: Save JSON output to this file path (default: stdout only)')
    p.add_argument('--question', default=None, help='Optional: The question/topic prompt given to the speaker (for relevance checking)')
    p.add_argument('--lang', default='en-GB', help='LanguageTool language code for grammar checking (default: en-GB)')
    p.add_argument('--part', type=int, choices=[1, 2, 3], required=True, help='The part of the IELTS speaking exam (1, 2, or 3) to apply correct length penalties.')
    return p

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