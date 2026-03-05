import json
import os
import dotenv
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Data.Database_management import insert_partial_record, add_data

st_logger = logging.getLogger('Data_loader ')
st_logger.setLevel(logging.INFO)
dotenv.load_dotenv()

# Lazy imports - only import when needed
def _import_presidio():
    from Presidio.Presidio_helpers import analyze, anonymize, create_fake_data, analyzer_engine
    from Presidio.Presidio_OpenAI import OpenAIParams
    return analyze, anonymize, create_fake_data, analyzer_engine, OpenAIParams

def _import_pinecone():
    from RAG.Pinecone_LlamaIndex import loadDataPinecone
    return loadDataPinecone

def _import_dp():
    from Differential_privacy.DP import diff_privacy_dp_prompt, diff_privacy_dp_prompt_batch, diff_privacy_diffractor, diff_privacy_dpmlm
    return diff_privacy_dp_prompt, diff_privacy_dp_prompt_batch, diff_privacy_diffractor, diff_privacy_dpmlm

def split_text_into_chunks(text, max_words):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i + max_words])
        chunks.append(chunk) 
    return chunks

def anonymize_presidio(text_with_pii):
    analyze, anonymize, create_fake_data, analyzer_engine, OpenAIParams = _import_presidio()
    
    st_logger.info(f"Presidio text analysis started on the text: {text_with_pii}")
    analyzer_instance = analyzer_engine()
    st_analyze_results = analyze(
        text=text_with_pii,
        language="en",
        score_threshold=0.5,
        allow_list=[],
    )
    st_logger.info(f"Presidio text analysis completed")

    results_as_dicts = [result.to_dict() for result in st_analyze_results]
    results_json = json.dumps(results_as_dicts, indent=2)

    st_logger.info(f"Presidio text anonymization started.")
    text_pii_deleted = anonymize(
        text=text_with_pii,
        operator="redact",
        analyze_results=st_analyze_results,
    )
    st_logger.info(f"Text with PII deleted: {text_pii_deleted}")

    text_pii_labeled = anonymize(
        text=text_with_pii,
        operator="replace",
        analyze_results=st_analyze_results,
    )
    st_logger.info(f"Text with PII labeled: {text_pii_labeled}")

    open_ai_params = OpenAIParams(
        openai_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo-instruct",
        api_base=None,
        deployment_id="",
        api_version=None,
        api_type="openai",
    )
    # Reduced chunk size to avoid OpenAI token limit (4097 max tokens)
    # With max_tokens=2048 for completion, we need prompt to be < 2049 tokens
    # Using 1200 words per chunk to leave room for prompt instructions and anonymized text
    text_chunks = split_text_into_chunks(text_with_pii, max_words=1200)
    text_pii_synthetic_list = []
    for chunk in text_chunks:
        st_analyze_chunk_results = analyze(
            text=chunk,
            language="en",
            score_threshold=0.5,
            allow_list=[],
        )
        text_chunk_pii_synthetic = create_fake_data(
            chunk,
            st_analyze_chunk_results,
            open_ai_params,
        )
        text_pii_synthetic_list.append(text_chunk_pii_synthetic)
    text_pii_synthetic = ' '.join(text_pii_synthetic_list)
    st_logger.info(f"Synthetic data created: {text_pii_synthetic}")

    return {
        'text_pii_deleted': text_pii_deleted.text,
        'text_pii_labeled': text_pii_labeled.text,
        'text_pii_synthetic': text_pii_synthetic,
        'details': results_json
    }

def anonymize_diffractor(text_with_pii):
    _, _, diff_privacy_diffractor, _ = _import_dp()
    
    text_pii_dp_diffractor1 = diff_privacy_diffractor(text_with_pii, epsilon=1)
    st_logger.info(f"text_pii_dp_diffractor1: {text_pii_dp_diffractor1}")
    text_pii_dp_diffractor2 = diff_privacy_diffractor(text_with_pii, epsilon=2)
    st_logger.info(f"text_pii_dp_diffractor2: {text_pii_dp_diffractor2}")
    text_pii_dp_diffractor3 = diff_privacy_diffractor(text_with_pii, epsilon=3)
    st_logger.info(f"text_pii_dp_diffractor3: {text_pii_dp_diffractor3}")
    
    return {
        'text_pii_dp_diffractor1': text_pii_dp_diffractor1,
        'text_pii_dp_diffractor2': text_pii_dp_diffractor2,
        'text_pii_dp_diffractor3': text_pii_dp_diffractor3
    }

def anonymize_dp_prompt(text_with_pii):
    _, diff_privacy_dp_prompt_batch, _, _ = _import_dp()
    
    # Use batch processing for better performance
    results = diff_privacy_dp_prompt_batch(text_with_pii, epsilon_values=[150, 200, 250])
    
    text_pii_dp_dp_prompt1 = results['epsilon_150']
    text_pii_dp_dp_prompt2 = results['epsilon_200']
    text_pii_dp_dp_prompt3 = results['epsilon_250']
    
    st_logger.info(f"text_pii_dp_dp_prompt1: {text_pii_dp_dp_prompt1}")
    st_logger.info(f"text_pii_dp_dp_prompt2: {text_pii_dp_dp_prompt2}")
    st_logger.info(f"text_pii_dp_dp_prompt3: {text_pii_dp_dp_prompt3}")
    
    return {
        'text_pii_dp_dp_prompt1': text_pii_dp_dp_prompt1,
        'text_pii_dp_dp_prompt2': text_pii_dp_dp_prompt2,
        'text_pii_dp_dp_prompt3': text_pii_dp_dp_prompt3
    }

def anonymize_dpmlm(text_with_pii):
    _, _, _, diff_privacy_dpmlm = _import_dp()
    
    text_pii_dp_dpmlm1 = diff_privacy_dpmlm(text_with_pii, epsilon=50)
    st_logger.info(f"text_pii_dp_dpmlm1: {text_pii_dp_dpmlm1}")
    text_pii_dp_dpmlm2 = diff_privacy_dpmlm(text_with_pii, epsilon=75)
    st_logger.info(f"text_pii_dp_dpmlm2: {text_pii_dp_dpmlm2}")
    text_pii_dp_dpmlm3 = diff_privacy_dpmlm(text_with_pii, epsilon=100)
    st_logger.info(f"text_pii_dp_dpmlm3: {text_pii_dp_dpmlm3}")
    
    return {
        'text_pii_dp_dpmlm1': text_pii_dp_dpmlm1,
        'text_pii_dp_dpmlm2': text_pii_dp_dpmlm2,
        'text_pii_dp_dpmlm3': text_pii_dp_dpmlm3
    }

def save_to_database_text_with_pii(table_name, file_name, file_hash, file_bytes, text_with_pii):
    from Data.Database_management import retrieve_record_by_hash
    
    # Check if record already exists
    existing_record = retrieve_record_by_hash(table_name, file_hash)
    if existing_record is not None and existing_record.get('text_with_pii') is not None:
        st_logger.info(f"Original text already exists in database for {file_name} {file_hash}, skipping insert")
        return
    
    try:
        insert_partial_record(
            table_name, 
            file_name, 
            file_hash, 
            file_bytes, 
            text_with_pii=text_with_pii
        )
        st_logger.info(f"Original text inserted into the database: {file_name} {file_hash}")
    except Exception as e:
        st_logger.warning(f"Failed to insert original text into database for {file_name}: {str(e)}")
        raise

def save_to_database(table_name, file_hash, anonymized_data, method_name=None):
    """
    Generic function to save anonymized data to SQL database.
    
    Parameters:
    - table_name: Name of the database table
    - file_hash: Hash identifying the record
    - anonymized_data: Dictionary with anonymized data (keys match database column names)
    - method_name: Optional name for logging (e.g., 'Presidio', 'Diffractor')
    """
    add_data(table_name, file_hash, **anonymized_data)
    method_str = f"{method_name} " if method_name else ""
    st_logger.info(f"{method_str} data inserted into the database: {file_hash}")

def save_to_vector_database_text_with_pii(index_name, file_name, file_hash, text_with_pii):
    try:
        loadDataPinecone = _import_pinecone()
        loadDataPinecone(
            index_name=index_name,
            text=text_with_pii,
            file_name=file_name,
            file_hash=file_hash,
            text_type="text_with_pii",
            skip_if_exists=True
        )
        st_logger.info(f"Original text processed in Pinecone: {file_name} {file_hash}")
    except Exception as e:
        st_logger.warning(f"Failed to insert original text into Pinecone for {file_name}: {str(e)}. Data is still saved in database.")

def save_to_vector_database(index_name, file_name, file_hash, anonymized_data, method_name=None):
    """
    Generic function to save anonymized data to vector database (Pinecone).
    
    Parameters:
    - index_name: Name of the Pinecone index
    - file_name: Name of the file
    - file_hash: Hash identifying the record
    - anonymized_data: Dictionary with anonymized data (keys are used as text_type)
    - method_name: Optional name for logging (e.g., 'Presidio', 'Diffractor')
    
    Note: The 'details' key is excluded from vector database storage as it's metadata.
    """
    try:
        loadDataPinecone = _import_pinecone()
        for text_type, text_value in anonymized_data.items():
            # Skip 'details' as it's metadata, not text to be indexed
            if text_type != 'details' and text_value:
                loadDataPinecone(
                    index_name=index_name,
                    text=text_value,
                    file_name=file_name,
                    file_hash=file_hash,
                    text_type=text_type,
                    skip_if_exists=True
                )
        method_str = f"{method_name} " if method_name else ""
        st_logger.info(f"{method_str}data processed in Pinecone: {file_name} {file_hash}")
    except Exception as e:
        method_str = f"{method_name} " if method_name else ""
        st_logger.warning(f"Failed to insert {method_str}data into Pinecone for {file_name}: {str(e)}. Data is still saved in database.")

def load_data_text_with_pii(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes):
    save_to_database_text_with_pii(table_name, file_name, file_hash, file_bytes, text_with_pii)
    # save_to_vector_database_text_with_pii(index_name, file_name, file_hash, text_with_pii)

def load_data_presidio(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes):
    anonymized_data = anonymize_presidio(text_with_pii)
    save_to_database(table_name, file_hash, anonymized_data, method_name="Presidio")
    # save_to_vector_database(index_name, file_name, file_hash, anonymized_data, method_name="Presidio")

def load_data_diffractor(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes):
    anonymized_data = anonymize_diffractor(text_with_pii)
    save_to_database(table_name, file_hash, anonymized_data, method_name="Diffractor")
    #save_to_vector_database(index_name, file_name, file_hash, anonymized_data, method_name="Diffractor")

def load_data_dp_prompt(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes):
    anonymized_data = anonymize_dp_prompt(text_with_pii)
    save_to_database(table_name, file_hash, anonymized_data, method_name="DP PROMPT")
    # save_to_vector_database(index_name, file_name, file_hash, anonymized_data, method_name="DP PROMPT")

def load_data_dpmlm(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes):
    anonymized_data = anonymize_dpmlm(text_with_pii)
    save_to_database(table_name, file_hash, anonymized_data, method_name="DPMLM")
    # save_to_vector_database(index_name, file_name, file_hash, anonymized_data, method_name="DPMLM")

def load_data_all(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes):
    load_data_text_with_pii(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes)
    load_data_presidio(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes)
    load_data_diffractor(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes)
    load_data_dp_prompt(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes)
    load_data_dpmlm(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes)
    st_logger.info("All data inserted into database and Pinecone.")