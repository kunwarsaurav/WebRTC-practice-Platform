import numpy as np
from scorer import evaluate_pipeline, init_models
import sys
import json

init_models()

sample_rate = 16000
audio_duration = 15.0
# Create dummy audio array of 15 seconds
audio_array = np.zeros(int(sample_rate * audio_duration), dtype=np.float32)

transcript = "Well, I think the most important thing is that, you know, people should try to learn how to deal with complex issues. In addition, they need to communicate effectively because communication is essential."
segments = [
    {"start": 0.0, "end": 5.0, "text": "Well, I think the most important thing is that,"},
    {"start": 5.0, "end": 10.0, "text": "you know, people should try to learn how to deal with complex issues."},
    {"start": 10.0, "end": 15.0, "text": "In addition, they need to communicate effectively because communication is essential."}
]
word_timestamps = [
    {"start": 0.1, "end": 0.5, "word": "Well"},
    {"start": 0.6, "end": 0.8, "word": "I"},
    {"start": 0.9, "end": 1.2, "word": "think"},
    {"start": 1.3, "end": 1.4, "word": "the"},
    {"start": 1.5, "end": 2.0, "word": "most"},
    {"start": 2.1, "end": 2.8, "word": "important"},
    {"start": 2.9, "end": 3.4, "word": "thing"},
    {"start": 3.5, "end": 3.7, "word": "is"},
    {"start": 3.8, "end": 4.1, "word": "that"},
    {"start": 5.1, "end": 5.5, "word": "you"},
    {"start": 5.6, "end": 6.0, "word": "know"},
    {"start": 6.1, "end": 6.5, "word": "people"},
    {"start": 6.6, "end": 7.0, "word": "should"},
    {"start": 7.1, "end": 7.5, "word": "try"},
    {"start": 7.6, "end": 7.8, "word": "to"},
    {"start": 7.9, "end": 8.3, "word": "learn"},
    {"start": 8.4, "end": 8.6, "word": "how"},
    {"start": 8.7, "end": 8.8, "word": "to"},
    {"start": 8.9, "end": 9.2, "word": "deal"},
    {"start": 9.3, "end": 9.5, "word": "with"},
    {"start": 9.6, "end": 10.0, "word": "complex"},
    {"start": 10.1, "end": 10.5, "word": "issues"},
    {"start": 10.6, "end": 10.8, "word": "In"},
    {"start": 10.9, "end": 11.5, "word": "addition"},
    {"start": 11.6, "end": 11.9, "word": "they"},
    {"start": 12.0, "end": 12.3, "word": "need"},
    {"start": 12.4, "end": 12.6, "word": "to"},
    {"start": 12.7, "end": 13.5, "word": "communicate"},
    {"start": 13.6, "end": 14.2, "word": "effectively"},
    {"start": 14.3, "end": 14.6, "word": "because"},
    {"start": 14.7, "end": 15.5, "word": "communication"},
    {"start": 15.6, "end": 15.8, "word": "is"},
    {"start": 15.9, "end": 16.5, "word": "essential"}
]

question = "How important is communication?"

try:
    report = evaluate_pipeline(
        transcript=transcript,
        segments=segments,
        word_timestamps=word_timestamps,
        audio_path="dummy.wav",
        audio_array=audio_array,
        sample_rate=sample_rate,
        lang="en-US",
        question=question,
        part=3
    )
    print("SUCCESS!")
    print(json.dumps(report['feedback'], indent=2))
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
