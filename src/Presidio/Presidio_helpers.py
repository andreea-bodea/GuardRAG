"""
Helper methods for the Presidio Streamlit app
"""
from typing import List, Optional, Tuple
import logging

# Make streamlit optional - only needed for Streamlit app, not for server-side processing
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    # Create no-op decorators if streamlit is not available
    # These decorators just return the function unchanged (no caching)
    def noop_decorator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class StreamlitStub:
        @staticmethod
        def cache_resource(*args, **kwargs):
            return noop_decorator
        
        @staticmethod
        def cache_data(*args, **kwargs):
            return noop_decorator
        
        class cache_resource_static:
            @staticmethod
            def clear():
                pass
        
        class cache_data_static:
            @staticmethod
            def clear():
                pass
        
        @staticmethod
        def error(*args, **kwargs):
            print(f"STREAMLIT ERROR (streamlit not installed): {args}")
    
    st = StreamlitStub()
    st.cache_resource = StreamlitStub.cache_resource
    st.cache_data = StreamlitStub.cache_data
    st.cache_resource_static = StreamlitStub.cache_resource_static
    st.cache_data_static = StreamlitStub.cache_data_static
    STREAMLIT_AVAILABLE = False

from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerResult,
    RecognizerRegistry,
)
from presidio_analyzer.nlp_engine import NlpEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from Presidio.Presidio_OpenAI import (
    call_completion_model,
    OpenAIParams,
    create_prompt,
)
from Presidio.Presidio_NLP_engine import (
    create_nlp_engine_with_spacy,
    create_nlp_engine_with_flair
)
from presidio_analyzer.nlp_engine import NlpEngineProvider

logger = logging.getLogger("presidio-streamlit")

@st.cache_resource(show_spinner=False)
def nlp_engine_and_registry(language: str = "en") -> Tuple[NlpEngine, RecognizerRegistry]:
    """Create the NLP Engine + registry based on language."""
    import spacy
    
    # Force cache clear for different languages to ensure we get the right model
    # Note: This is only needed for Streamlit caching, not for server-side processing
    # if STREAMLIT_AVAILABLE:
    #     st.cache_resource.clear()
    
    if language == "de":
        # Check if German model is installed, if not, install it
        try:
            model_name = "de_core_news_lg"
            if not spacy.util.is_package(model_name):
                print(f"Installing German model: {model_name}")
                spacy.cli.download(model_name)
                print("German model installed successfully")
            
            # Create specific German NLP configuration
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "de", "model_name": model_name}],
                "ner_model_configuration": {
                    "model_to_presidio_entity_mapping": {
                        "PER": "PERSON",
                        "LOC": "LOCATION",
                        "ORG": "ORGANIZATION",
                        "MISC": "GENERIC_PII", 
                        "DATE": "DATE_TIME",
                        "TIME": "DATE_TIME",
                    }
                }
            }
            
            nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(nlp_engine=nlp_engine)
            return nlp_engine, registry
            
        except Exception as e:
            if STREAMLIT_AVAILABLE:
                st.error(f"Error loading German model: {str(e)}. Falling back to English.")
            print(f"Error loading German model: {str(e)}. Falling back to English.")
            language = "en"  # Fall back to English
    
    # Default to English
    return create_nlp_engine_with_flair()

@st.cache_resource(show_spinner=False, hash_funcs={"builtins.str": lambda s: s})
def analyzer_engine(language: str = "en") -> AnalyzerEngine:
    """Create the Analyzer Engine instance with proper language support."""
    nlp_engine, registry = nlp_engine_and_registry(language=language)
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
    return analyzer

@st.cache_resource
def anonymizer_engine():
    """Return AnonymizerEngine."""
    return AnonymizerEngine()

@st.cache_data
def get_supported_entities():
    """Return supported entities from the Analyzer Engine."""
    return analyzer_engine().get_supported_entities() + ["GENERIC_PII"]

@st.cache_data(show_spinner=False)
def analyze(text: str, language: str = "en", **kwargs):
    """Analyze input using Analyzer engine and input arguments (kwargs)."""
    # The analyzer will use all entities by default when none are specified
    return analyzer_engine(language=language).analyze(text=text, language=language, **kwargs)

def anonymize(
    text: str,
    operator: str,
    analyze_results: List[RecognizerResult],
):
    """Anonymize identified input using Presidio Anonymizer.
    :param text: Full text
    :param operator: Operator name (redact/replace/synthesize/highlight)
    :param analyze_results: list of results from presidio analyzer engine
    """
    if operator == "highlight":
        operator_config = {"lambda": lambda x: x}
        operator = "custom"
    elif operator == "synthesize":
        operator_config = None
        operator = "replace"
    else:
        operator_config = None

    res = anonymizer_engine().anonymize(
        text,
        analyze_results,
        operators={"DEFAULT": OperatorConfig(operator, operator_config)},
    )
    return res

def annotate(text: str, analyze_results: List[RecognizerResult]):
    """Highlight the identified PII entities on the original text
    :param text: Full text
    :param analyze_results: list of results from presidio analyzer engine
    """
    tokens = []
    results = anonymize(
        text=text,
        operator="highlight",
        analyze_results=analyze_results,
    )

    # Sort by start index
    results = sorted(results.items, key=lambda x: x.start)
    for i, res in enumerate(results):
        if i == 0:
            tokens.append(text[: res.start])

        # Append entity text and entity type
        tokens.append((text[res.start : res.end], res.entity_type))

        # If another entity coming i.e. we're not at the last results element, add text up to next entity
        if i != len(results) - 1:
            tokens.append(text[res.end : results[i + 1].start])
        # If no more entities coming, add all remaining text
        else:
            tokens.append(text[res.end :])
    return tokens

def create_fake_data(
    text: str,
    analyze_results: List[RecognizerResult],
    openai_params: OpenAIParams,
    language: str = "en",
) -> str:
    """Creates a synthetic version of the text using OpenAI APIs"""
    if not openai_params.openai_key:
        return "Please provide your OpenAI key"
    results = anonymize(text=text, operator="replace", analyze_results=analyze_results)
    prompt = create_prompt(results.text, language=language)
    return call_completion_model(prompt=prompt, openai_params=openai_params)

@st.cache_data
def call_openai_api(
    prompt: str, openai_model_name: str, openai_deployment_name: Optional[str] = None
) -> str:
    """Call OpenAI API with the given prompt and model
    :param prompt: The prompt to send to OpenAI
    :param openai_model_name: The name of the OpenAI model to use
    :param openai_deployment_name: Optional deployment name for Azure OpenAI
    """
    fake_data = call_completion_model(
        prompt, model=openai_model_name, deployment_id=openai_deployment_name
    )
    return fake_data
