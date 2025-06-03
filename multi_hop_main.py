from utils import process_context, load_json_questions, extract_filepaths
from QAeval_api_call import claude, gemini, gpt_4o, gpt_o1, summary_df
import pandas as pd
import os
import concurrent.futures
 
def generate_prompts(paper_file_path, json_file_path):
    prompts = list()
    num_qs = list()
    contexts = list()
    if len(paper_file_path) != len(json_file_path):
        raise ValueError('Length is not the same')
    for paper, json in zip(paper_file_path, json_file_path):
        print(json)
        if not (os.path.exists(paper) and os.path.exists(json)):
            print(f"File path: {paper} or {json} does not exist")
            continue
        context = process_context(paper)
        output_text = "CONTEXT:"+context+"\n\nQ&A DATASET:"
        q = 0
        for pair in load_json_questions(json):
            output_text = output_text + str(pair) + "\n\n"
            q+=1
        prompts.append(output_text)
        num_qs.append(q)
        contexts.append(context)
    return prompts, num_qs, contexts

QA_claude_20241017_df = pd.read_excel('inputs/QA_claude_20241017.xlsx')
# The code executes sequentially according to the order specified in the Excel sheet. 
# It requires a directory named 'multi-hop-data', which has been omitted due to copyright restrictions held by the original publishers.

json_file_paths = []
paper_file_paths = [path for path in extract_filepaths('inputs/multi-hop-data') if path.count('/') == 3]
paper_file_paths_copy = paper_file_paths.copy()
for path in paper_file_paths_copy:
    json_file_path = f'inputs/multi-hop-dataset/{path.split("/")[-1]}_multi-hop.json'
    if os.path.exists(json_file_path):
        json_file_paths.append(json_file_path)
    else:
        paper_file_paths.remove(path)
prompts, num_qs, contexts = generate_prompts(paper_file_paths, json_file_paths)

classification_prompt = """
    You have been provided with a context from which question-answer pairs have been generated. 
    Note that the questions are multi hop, meaning that the answer can come from many different 
    parts of the paper combined, including both the main manuscript (MS) and supplementary information (SI).
    Your task is to classify these pairs according to the following criteria:

    Classification Criteria:
    1. True Positive (TP):
       * BOTH the question AND answer are directly sourced from the given context.
       * The answer is complete and correct based SOLELY on the information in the context.
       * No additional knowledge or inference beyond what is explicitly stated in the context is required.

    2. False Positive (FP):
       * BOTH the question AND answer appear to be sourced from the given context.
       * However, the answer is incorrect, incomplete, or requires inference beyond the explicit information provided.

    3. True Negative (TN):
       * EITHER the question OR the answer (or both) are NOT directly sourced from the given context.
       * The answer may be correct based on general knowledge, but cannot be fully validated using only the provided context.

    4. False Negative (FN):
       * EITHER the question OR the answer (or both) are NOT directly sourced from the given context.
       * The answer is incorrect based on general knowledge or principles.

    Important Guidelines:
    1. Context Boundaries: 
       * Exclude any references or citations from consideration as part of the context.

    2. Direct Sourcing:
       * "Sourced from the context" means the information is explicitly stated, not inferred or derived.
       * If any inference or additional knowledge is required, the pair is not considered directly sourced.

    3. Completeness:
       * For TP classification, ensure ALL information required for the answer is explicitly stated in the context.
       * Partial matches or answers requiring additional inference should be classified as FP, TN or FN.

    4. Specificity:
       * Pay close attention to specific details, numbers, and phrasings in both questions and answers.
       * Minor discrepancies may change the classification.

    5. General Knowledge:
       * Be cautious with answers that seem correct but rely on general knowledge not provided in the context.
       * These should typically be classified as TN, not TP.

    6. Inference and Reasoning:
       * Questions requiring reasoning or inference beyond explicitly stated facts should not be classified as TP, even if the reasoning seems sound.

    7. Precision in Language:
       * Be wary of absolute terms like "always," "never," or "only" in questions or answers. Verify such claims are explicitly supported by the context for TP classification.

    8. Numeric Values:
       * For questions involving calculations or numeric values, ensure all required numbers and operations are explicitly provided in the context for TP classification.

    9. Chemical Formulas and Structures:
       * For questions about chemical formulas or structures, ensure the exact information is provided in the context. Do not rely on chemical knowledge to infer details not explicitly stated.

    10. Experimental Procedures:
        * For questions about experimental procedures or synthesis, all steps should be explicitly described in the context for a TP classification.

    11. Material Properties:
        * When classifying questions about material properties (e.g., surface area, gas uptake), ensure the specific values and conditions are explicitly stated in the context.

    12. Comparative Statements:
        * For questions comparing different materials or properties, ensure the context explicitly provides the comparison. Do not rely on calculations or inferences not directly stated.

    13. Hypothetical Scenarios:
        * Questions asking about hypothetical situations or changes to experimental conditions should be classified as TN unless the context explicitly discusses such scenarios.

    14. Mechanism and Theoretical Explanations:
        * Be cautious with answers that provide mechanisms or theoretical explanations. Ensure these are explicitly stated in the context, not inferred based on chemical knowledge.

    15. Implicit Information:
        * Avoid classifying as TP any question-answer pairs that rely on information that seems obvious or implicit but is not explicitly stated in the context.

    16. Strict Interpretation:
        * When evaluating question-answer pairs, adopt a very strict interpretation of what constitutes "directly sourced" information. If there's any doubt, lean towards classifying as FP or TN rather than TP.

    17. Context Verification:
        * For each classification, explicitly reference the relevant part of the context that supports your decision. This ensures a thorough check against the provided information.

    18. Mathematical Operations:
        * For questions requiring simple mathematical operations (e.g., averaging, ratios), classify as TP only if ALL required values are explicitly stated in the context AND the operation is trivial.
        * For more complex calculations or those requiring multiple steps, classify as TN or FP unless the context explicitly provides the calculated result.

    19. Partial Information:
        * If a question-answer pair contains some information from the context but also includes additional unsupported claims or details, classify as FP rather than TP.

    20. Time Sensitivity:
        * Be aware of the potential for time-sensitive information. If a question-answer pair relies on information that may change over time (e.g., "current" record holders, latest discoveries), ensure the context explicitly supports the claim for the relevant time period.

    21. Structural Inferences:
        * For questions about molecular or crystal structures, ensure that all structural details are explicitly stated in the context. Do not rely on chemical knowledge to infer structural information not directly provided.

    22. Charge Balance:
        * When dealing with questions about ionic compounds or charge states, ensure that the context explicitly states the charge information. Do not make assumptions about charge balance based on chemical knowledge.

    23. Coordination Numbers:
        * For questions about metal coordination, ensure that the context explicitly states the coordination number and geometry. Do not infer coordination information based on general chemistry principles.

    24. Strict Adherence to Context:
        * Even if an answer seems logically correct or chemically sound, if it relies on any information or reasoning not explicitly provided in the context, classify it as TN or FP.

    25. Multiple Supporting Statements:
        * For a TP classification, ensure that all parts of the answer are supported by explicit statements in the context. If only part of the answer is directly supported, classify as FP.

    26. Keyword Matching:
        * Be cautious of answers that use keywords from the context but apply them incorrectly or out of context. Ensure the full meaning and application of terms match the context exactly.

    27. Scope of Questions:
        * Ensure that the scope of the question matches the information provided in the context. If a question asks for information beyond what is explicitly stated, classify as TN or FN.

    28. Emphasis on TN Classification:
        * When in doubt between FP and TN, lean towards classifying as TN. This is especially important for answers that seem plausible but require any degree of inference or additional knowledge beyond the context.

    Task:
    Evaluate each question-answer pair carefully against these criteria. Provide a brief but specific explanation for your classification, referencing exact phrases or sentences from the context that informed your decision.

    Remember: Accuracy in classification relies on strict adherence to what is explicitly stated in the context. Avoid making assumptions or inferences beyond the provided information. When in doubt, err on the side of caution and classify as TN rather than FP or TP.
"""

folder_name = 'Multi-Hop-Individual-LLM-Output'
os.makedirs(folder_name, exist_ok=True)
runs = 1
step = 10  # Use 'step' consistently

for i in range(0, len(prompts), step):
    start = i
    # Set 'end' to either i + step or the total number of prompts if fewer remain.
    end = i + step if (i + step) <= len(prompts) else len(prompts)
    reruns = 3  # Maximum number of reattempts for this batch

    while True:
        print(f"Processing prompts [{start}:{end}]")
        # Launch all four API calls concurrently for this batch.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(gpt_4o, classification_prompt, start, end, prompts, num_qs, f"{folder_name}/4o.xlsx", runs=runs),
                executor.submit(gemini, classification_prompt, start, end, prompts, num_qs, f"{folder_name}/gemini.xlsx", runs=runs),
                executor.submit(claude, classification_prompt, start, end, prompts, num_qs, f"{folder_name}/claude.xlsx", runs=runs),
                executor.submit(gpt_o1, classification_prompt, start, end, prompts, num_qs, f"{folder_name}/o1.xlsx", runs=runs)
            ]
            concurrent.futures.wait(futures)
        
        print(f"Remaining reruns: {reruns}")
        
        try:
            file_paths = [
                f"{folder_name}/4o.xlsx", 
                f"{folder_name}/gemini.xlsx", 
                f"{folder_name}/claude.xlsx", 
                f"{folder_name}/o1.xlsx"
            ]
            # For each file, check if it exists and if the number of sheets is at least 'end'
            sheet_counts = []
            for fp in file_paths:
                if os.path.exists(fp):
                    sheets = pd.ExcelFile(fp).sheet_names
                    sheet_counts.append(len(sheets) >= end)
                else:
                    sheet_counts.append(False)
            
            if all(sheet_counts):
                print("All sheets processed successfully for this batch.")
                break  # Exit the while loop for this batch
            
            if reruns <= 0:
                print("Maximum reruns reached for this batch.")
                break
            
            reruns -= 1
        except Exception as e:
            print(f"Error during sheet verification: {e}")

file_name = 'summary.xlsx'
summary_df(folder_name, file_name, len(num_qs), runs=3)
