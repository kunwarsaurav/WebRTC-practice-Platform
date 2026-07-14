import os
import re

with open('scorer.py', 'r', encoding='utf-8') as f:
    code = f.read()

# We will just write a whole new scorer.py to keep it clean.
new_code = '''import argparse
import math
import os
import re
import sys
import threading
import time
import json
import httpx
from collections import Counter, defaultdict
from typing import Any
from dotenv import load_dotenv

import librosa
import numpy as np
import parselmouth

load_dotenv()

_transcribe_lock = threading.Lock()

def init_models():
    print('[INIT] No heavy models to load! Using Groq API for NLP.', file=sys.stderr)

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
            occ = len(re.findall(r'\\b' + re.escape(marker) + r'\\b', text_lower))
        if occ > 0:
            used.append(marker)
            total += occ
    return (total, len(used), sorted(used))

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

'''

with open('backend/new_scorer.py', 'w', encoding='utf-8') as f:
    f.write(new_code)
