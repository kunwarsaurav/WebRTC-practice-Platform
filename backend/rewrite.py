import os, re

with open('scorer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove heavy imports
content = re.sub(r'import spacy\n', '', content)
content = re.sub(r'import language_tool_python\n', '', content)
content = re.sub(r'from wordfreq import zipf_frequency\n', '', content)
content = re.sub(r'import httpx\n', '', content)

# 2. Add httpx and json if not present (they are, but just to be sure)
content = content.replace('import os\n', 'import os\nimport httpx\nimport json\n')

# 3. Replace init_models and remove _get_nlp, _get_lt, _get_st_model
init_code_replacement = '''
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
'''

# Find init_models and replace it
content = re.sub(r'def init_models\(\):.*?print\(\'\[INIT\] All models successfully loaded into RAM/VRAM!\', file=sys\.stderr\)', init_code_replacement, content, flags=re.DOTALL)

# Remove cosine_similarity, _get_nlp, _get_lt, _get_st_model and their globals
content = re.sub(r'def cosine_similarity.*?return np\.dot\(a, b\.T\) / np\.clip\(norms, 1e-10, None\)', '', content, flags=re.DOTALL)
content = re.sub(r'_nlp = None\n\ndef _get_nlp\(\):.*?return _nlp\n', '', content, flags=re.DOTALL)
content = re.sub(r'_lt_tool = None\n\ndef _get_lt\(lang: str=\'en-GB\'\):.*?return _lt_tool\n', '', content, flags=re.DOTALL)
content = re.sub(r'_st_model = None\n\ndef _get_st_model\(\):.*?return _st_model\n', '', content, flags=re.DOTALL)

# 4. Replace analyze_lexical
lexical_new = '''def analyze_lexical(transcript: str) -> dict[str, Any]:
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
'''

content = re.sub(r'def analyze_lexical\(transcript: str\) -> dict\[str, Any\]:.*?def _empty_features\(\) -> dict\[str, Any\]:\n    return \{.*?\}', lexical_new, content, flags=re.DOTALL)

# 5. Replace analyze_grammar
grammar_new = '''def analyze_grammar(transcript: str, lang: str='en-GB') -> dict[str, Any]:
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
'''

content = re.sub(r'def analyze_grammar\(transcript: str, lang: str=\'en-GB\'\) -> dict\[str, Any\]:.*?def _empty_features\(\) -> dict\[str, Any\]:\n    return \{.*?\}', grammar_new, content, flags=re.DOTALL)

# 6. Replace analyze_relevance
relevance_new = '''def analyze_relevance(transcript: str, question: str) -> dict[str, Any]:
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
'''

content = re.sub(r'def analyze_relevance\(transcript: str, question: str\) -> dict\[str, Any\]:.*?def _empty_features\(\) -> dict\[str, Any\]:\n    return \{.*?\}', relevance_new, content, flags=re.DOTALL)

# Fix empty dictionary returning references
content = content.replace('_empty_features()', '{}')

with open('scorer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Scorer rewritten successfully.")
