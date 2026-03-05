# Compatibility fix for scipy import issue with gensim
import sys
import scipy.sparse

# Monkey patch scipy.linalg to include triu from scipy.sparse
if not hasattr(scipy.linalg, 'triu'):
    scipy.linalg.triu = scipy.sparse.triu

import os
from nltk.data import find
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import gc
import torch

from .Diffractor.Diffractor import Lists, Diffractor
# DPMLM and DPPrompt imported lazily in get_dpmlm_model() and get_dp_prompt_model()
# so that running only Diffractor does not require wn, spacy, transformers, etc.
import nltk

# Global model cache to avoid reloading models
_model_cache = {}
_model_lock = threading.Lock()

def clear_model_cache():
    """Clear the model cache to free up memory."""
    global _model_cache
    with _model_lock:
        for model in _model_cache.values():
            if hasattr(model, 'model') and hasattr(model.model, 'cpu'):
                model.model.cpu()
            del model
        _model_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

def get_dp_prompt_model(model_checkpoint="google/flan-t5-large"):
    """Get or create a cached DPPrompt model instance."""
    from .PrivFill.LLMDP import DPPrompt
    with _model_lock:
        if model_checkpoint not in _model_cache:
            _model_cache[model_checkpoint] = DPPrompt(model_checkpoint=model_checkpoint)
        return _model_cache[model_checkpoint]

def get_dpmlm_model(model_name="roberta-base"):
    """Get or create a cached DPMLM model instance (loads from HF once, then reused)."""
    from .DPMLM.DPMLM import DPMLM
    cache_key = f"dpmlm_{model_name}"
    with _model_lock:
        if cache_key not in _model_cache:
            _model_cache[cache_key] = DPMLM(MODEL=model_name)
        return _model_cache[cache_key]

def diff_privacy_diffractor(text_with_pii, epsilon, language="en"):
    try:
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    try:
        # Get the absolute path to the data directory
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Diffractor/data')
        
        # Use only numberbatch model (we have numberbatch-en-19.08_filtered.txt).
        # Other models (glove-twitter-200, etc.) require extra files from Diffractor Google Drive.
        lists = Lists(
            home=data_dir,
            model_names=['conceptnet-numberbatch-19-08-300'],
        )
        diff = Diffractor(
            L=lists,
            gamma=5,
            epsilon=epsilon,
            rep_stop=False, 
            method="geometric"
        )
        text_lower_case = text_with_pii.lower()
        perturbed_text, num_perturbed, num_diff, total, support, changes = diff.rewrite(text_lower_case)
        diff.cleanup()
        
        # Clear memory
        del lists, diff
        gc.collect()
        
        return ' '.join(perturbed_text)
    except Exception as e:
        print(f"Error in diff_privacy_diffractor: {e}")
        # Return original text if processing fails
        return text_with_pii

def diff_privacy_dp_prompt(text_with_pii, epsilon, language="en"):
    try:
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        find(f'tokenizers/punkt/{language}')
    except LookupError:
        nltk.download('punkt')
        # Download additional language data if needed
        if language == "de":
            nltk.download('punkt_german')
    
    sentences = nltk.sent_tokenize(text_with_pii)
    # For German language, use a multilingual model
    if language == "de":
        model_checkpoint = "google/mt5-base"
    else:
        model_checkpoint = "google/flan-t5-large"
    
    dpprompt = get_dp_prompt_model(model_checkpoint)  # Use cached model
    text_pii_dp_dp_prompt = dpprompt.privatize_dp(sentences, epsilon=epsilon)
    return ' '.join(text_pii_dp_dp_prompt)

def diff_privacy_dp_prompt_batch(text_with_pii, epsilon_values=[150, 200, 250]):
    """
    Process text with multiple epsilon values using a single model instance.
    This is much more efficient than calling diff_privacy_dp_prompt multiple times.
    """
    try:
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    sentences = nltk.sent_tokenize(text_with_pii)
    
    # Use cached model for efficiency
    dpprompt = get_dp_prompt_model()  # Use cached model
    
    results = {}
    for epsilon in epsilon_values:
        text_pii_dp_dp_prompt = dpprompt.privatize_dp(sentences, epsilon=epsilon)
        results[f'epsilon_{epsilon}'] = ' '.join(text_pii_dp_dp_prompt)
    
    return results

def diff_privacy_dpmlm(text_with_pii, epsilon, language="en"):
    try:
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        find(f'tokenizers/punkt/{language}')
    except LookupError:
        nltk.download('punkt')
        # Download additional language data if needed
        if language == "de":
            nltk.download('punkt_german')
    
    sentences = nltk.sent_tokenize(text_with_pii)
    
    # Use cached DPMLM so we load from HF only once
    if language == "de":
        dpmlm = get_dpmlm_model("dbmdz/bert-base-german-cased")
    else:
        dpmlm = get_dpmlm_model("roberta-base")
    
    perturbed_sentences = []
    
    for sentence in sentences:
        # If sentence is too long, break it into chunks
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) > 75:  # Conservative estimate for model's token limit
            # Process in fixed-size chunks
            chunk_size = 75
            chunks = []
            
            # Split into chunks
            for i in range(0, len(tokens), chunk_size):
                chunk = tokens[i:i + chunk_size]
                chunks.append(' '.join(chunk))
            
            # Process each chunk
            chunk_results = []
            for chunk in chunks:
                perturbed_chunk, perturbed, total = dpmlm.dpmlm_rewrite(chunk, epsilon=epsilon)
                chunk_results.append(perturbed_chunk)
            
            # Join chunk results and append to perturbed_sentences
            perturbed_sentences.append(' '.join(chunk_results))
        else:
            # Process normally if sentence is not too long
            perturbed_sentence, perturbed, total = dpmlm.dpmlm_rewrite(sentence, epsilon=epsilon)
            perturbed_sentences.append(perturbed_sentence)
    
    text_pii_dpmlm = ' '.join(perturbed_sentences)
    return text_pii_dpmlm

if __name__ == "__main__":

    # text = "This is a sample text containing sensitive information like a phone number 123-456-7890."
    # text = "David"
    # text = "Arthur Hailey: King of the bestsellers Novelist Arthur Hailey, who has died at the age of 84, was known for his bestselling page-turners exploring the inner workings of various industries, from the hotels to high finance. Born in Luton, Bedfordshire, on 5 April 1920, Hailey was the only child of working class parents, They could not afford to keep him in school beyond the age of 14. He served as a pilot with the Royal Air Force during World War II, flying fighter planes to the Middle East. It was an occupation that was later to feature in his authorial debut, the television screenplay Flight into Danger. Hailey emigrated to Canada in 1947, where he eventually became a citizen. He wanted to be a writer from an early age, but did not take it up professionally until his mid-thirties, when he was inspired to write his first screenplay while on a return flight to Toronto."
    text = "Ireland 19-13 England Ireland consigned England to their third straight Six Nations defeat with a stirring victory at Lansdowne Road. A second-half try from captain Brian O'Driscoll and 14 points from Ronan O'Gara kept Ireland on track for their first Grand Slam since 1948. England scored first through Martin Corry but had tries from Mark Cueto and Josh Lewsey disallowed. Andy Robinson's men have now lost nine of their last 14 matches since the 2003 World Cup final. The defeat also heralded England's worst run in the championship since 1987. Ireland last won the title, then the Five Nations, in 1985, but 20 years on they share top spot in the table on maximum points with Wales."
    
    """
    # Diffractor
    print("Initial text:", text)
    epsilon = 1.5
    result_diffractor = diff_privacy_diffractor(text, epsilon) 
    print("Epsilon:", epsilon)
    print("Perturbed Text:", result_diffractor)
    """ 
    """   
    # DP Prompt
    print("Initial text:", text)
    epsilon = 1000
    result_dp_prompt = diff_privacy_dp_prompt(text, epsilon)
    print("Epsilon:", epsilon)
    print("Perturbed Text:")
    sentences = nltk.sent_tokenize(result_dp_prompt)
    for sentence in sentences:
        print(sentence)
    """
    
    # DP MLM
    print("Initial text:", text)
    epsilon = 100
    result_dpmlm = diff_privacy_dpmlm(text, epsilon)
    print("Epsilon:", epsilon)
    print("Perturbed Text:", result_dpmlm)