
import logging
import gc
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Data.Database_management import retrieve_responses_by_name_and_question, insert_responses_preprocessed
from RAG.Response_generation import get_all_questions
from Presidio.Presidio import delete_pii, label_pii, replace_pii
from Differential_privacy.DP import diff_privacy_dp_prompt, diff_privacy_diffractor, diff_privacy_dpmlm

st_logger = logging.getLogger('Response_postprocessor ')
st_logger.setLevel(logging.INFO)

questions = get_all_questions()
question_utility = questions[0]
question_untargeted_attack = questions[1]

def postprocess(postprocessing_table_name, responses_table_name, file_name, question):
    database_file = retrieve_responses_by_name_and_question(responses_table_name, file_name, question)
    
    if database_file is None:
        st_logger.error(f"No record found in {responses_table_name} for file {file_name} and question {question}")
        return
        
    response_with_pii = database_file['response_with_pii']
    st_logger.info(f"{file_name} - response_with_pii: {response_with_pii}")

    response_postprocessed_pii_deleted = delete_pii(response_with_pii)
    st_logger.info(f"{file_name} - response_postprocessed_pii_deleted: {response_postprocessed_pii_deleted}")
    response_postprocessed_pii_labeled = label_pii(response_with_pii)
    st_logger.info(f"{file_name} - response_postprocessed_pii_labeled: {response_postprocessed_pii_labeled}")
    response_postprocessed_pii_synthetic = replace_pii(response_with_pii)
    st_logger.info(f"{file_name} - response_postprocessed_pii_synthetic: {response_postprocessed_pii_synthetic}")

    response_postprocessed_pii_diffractor1 = diff_privacy_diffractor(response_with_pii, epsilon = 1)
    st_logger.info(f"{file_name} - response_postprocessed_pii_diffractor1: {response_postprocessed_pii_diffractor1}")    
    response_postprocessed_pii_diffractor2 = diff_privacy_diffractor(response_with_pii, epsilon = 2)
    st_logger.info(f"{file_name} - response_postprocessed_pii_diffractor2: {response_postprocessed_pii_diffractor2}")    
    response_postprocessed_pii_diffractor3 = diff_privacy_diffractor(response_with_pii, epsilon = 3)
    st_logger.info(f"{file_name} - response_postprocessed_pii_diffractor3: {response_postprocessed_pii_diffractor3}") 
        
    response_postprocessed_pii_dp_prompt1 = diff_privacy_dp_prompt(response_with_pii, epsilon = 150)
    st_logger.info(f"{file_name} - response_postprocessed_pii_dp_prompt1: {response_postprocessed_pii_dp_prompt1}")
    response_postprocessed_pii_dp_prompt2 = diff_privacy_dp_prompt(response_with_pii, epsilon = 200)
    st_logger.info(f"{file_name} - response_postprocessed_pii_dp_prompt2: {response_postprocessed_pii_dp_prompt2}")
    response_postprocessed_pii_dp_prompt3 = diff_privacy_dp_prompt(response_with_pii, epsilon = 250)
    st_logger.info(f"{file_name} - response_postprocessed_pii_dp_prompt3: {response_postprocessed_pii_dp_prompt3}")   
    
    response_postprocessed_pii_dpmlm1 = diff_privacy_dpmlm(response_with_pii, epsilon = 50)
    st_logger.info(f"{file_name} - response_postprocessed_pii_dpmlm1: {response_postprocessed_pii_dpmlm1}")
    response_postprocessed_pii_dpmlm2 = diff_privacy_dpmlm(response_with_pii, epsilon = 75)
    st_logger.info(f"{file_name} - response_postprocessed_pii_dpmlm2: {response_postprocessed_pii_dpmlm2}")
    response_postprocessed_pii_dpmlm3 = diff_privacy_dpmlm(response_with_pii, epsilon = 100)
    st_logger.info(f"{file_name} - response_postprocessed_pii_dpmlm3: {response_postprocessed_pii_dpmlm3}")

    insert_responses_preprocessed(
        table_name=postprocessing_table_name,
        file_name=file_name,
        question=question,
        response_with_pii=response_with_pii,
        response_postprocessed_pii_deleted=response_postprocessed_pii_deleted,
        response_postprocessed_pii_labeled=response_postprocessed_pii_labeled,
        response_postprocessed_pii_synthetic=response_postprocessed_pii_synthetic,
        response_postprocessed_pii_diffractor1=response_postprocessed_pii_diffractor1,
        response_postprocessed_pii_diffractor2=response_postprocessed_pii_diffractor2,
        response_postprocessed_pii_diffractor3=response_postprocessed_pii_diffractor3,
        response_postprocessed_pii_dp_prompt1=response_postprocessed_pii_dp_prompt1,
        response_postprocessed_pii_dp_prompt2=response_postprocessed_pii_dp_prompt2,
        response_postprocessed_pii_dp_prompt3=response_postprocessed_pii_dp_prompt3,
        response_postprocessed_pii_dpmlm1=response_postprocessed_pii_dpmlm1,
        response_postprocessed_pii_dpmlm2=response_postprocessed_pii_dpmlm2,
        response_postprocessed_pii_dpmlm3=response_postprocessed_pii_dpmlm3,
        evaluation=None
    )
    
    # Clear memory after processing
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def postprocess_all(postprocessing_table_name, responses_table_name, file_name_pattern, first, last, question): 
    for i in range(first, last+1):  # FOR EACH DATABASE FILE
        if responses_table_name == "enron_responses2" and i == 61: continue
        file_name = file_name_pattern.format(i)
        postprocess(postprocessing_table_name, responses_table_name, file_name, question)

if __name__ == "__main__":
    pass
    # postprocess_all(postprocessing_table_name="bbc_responses_postprocessed", responses_table_name="bbc_responses2", file_name_pattern="BBC_{}", first=171, last=200, question=question_untargeted_attack)
    # postprocess_all(postprocessing_table_name="bbc_responses_postprocessed2", responses_table_name="bbc_responses2", file_name_pattern="BBC_{}", first=98, last=200, question=question_utility)

    # postprocess_all(postprocessing_table_name="enron_responses_postprocessed", responses_table_name="enron_responses2", file_name_pattern="Enron_{}", first=44, last=44, question=question_utility)
    # postprocess_all(postprocessing_table_name="enron_responses_postprocessed_101_201", responses_table_name="enron_responses3", file_name_pattern="Enron_{}", first=, last=, question=question_untargeted_attack)