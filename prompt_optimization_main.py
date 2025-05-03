from utils import generate_prompts_by_df, generate_prompts, calculate_cumulative_statistics
from QAeval_api_call import claude, gemini, gpt_4o, gpt_o1, summary_df, get_revised_classification_prompt
import pandas as pd
import os
import concurrent.futures

doi_ground_truth = {'../wiley/10.1002_anie.200351546': ['TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TN',
  'TP',
  'TP',
  'TP',
  'FP',
  'TP'],
 '../wiley/10.1002_adfm.202008499': ['TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP'],
 '../wiley/10.1002_adfm.202203745': ['TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'FP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'FP',
  'TP',
  'TP',
  'TP'],
 '../wiley/10.1002_anie.202306048': ['TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'FP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'FP'],
 '../wiley/10.1002_anie.202009613': ['TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'FP',
  'TP',
  'TP'],
 '../wiley/10.1002_anie.200462126': ['TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'FP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'FP',
  'FP',
  'TP',
  'TP',
  'TP'],
 '../wiley/10.1002_anie.201308220': ['TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP'],
 '../wiley/10.1002_anie.202305942': ['TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP'],
 '../wiley/10.1002_anie.200600376': ['TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP'],
 '../wiley/10.1002_adma.200904238': ['TP',
  'TP',
  'FP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'FP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP',
  'TP']}

single_hop_DOIs = pd.read_excel("inputs/single-hop-DOIs.xlsx")
# The code executes sequentially according to the order specified in the Excel sheet. 
# It requires a directory named 'prompt-optimization-data', which has been omitted due to copyright restrictions held by the original publishers.
single_hop_DOIs['DOI'] = single_hop_DOIs['DOI'].apply(lambda x: "."+x)
single_hop_DOIs['File_Path_json'] = "inputs/single-hop-dataset/"+single_hop_DOIs['DOI'].str.split("/").str[-1] + "_single-hop.json"
single_hop_DOIs['File_Path_paper'] = "inputs/prompt-optimization-data"+single_hop_DOIs['DOI'].str[2:]
prompts, num_qs, contexts, ground_truth = generate_prompts_by_df(single_hop_DOIs.drop([1, 6, 7, 8], axis=0).iloc[0:6], doi_ground_truth)
nchem_p, nchem_n, nchem_c, nchem_g = generate_prompts(['inputs/prompt-optimization-data/nchem.834'],
                                   ['inputs/single-hop-dataset/nchem.834_single_hop.json'], 
                                  {'nchem.834_single_hop.json':[
                                        "TP", "TP", "TP", "TP", "TP", "TP", "TN", "TN", "TN", "TP",
                                        "TN", "TN", "TN", "TP", "TP", "TP", "TP", "TN", "TN", "TN"
                                    ]})
prompts.append(nchem_p[0])
num_qs.append(nchem_n[0])
contexts.append(nchem_c[0])
ground_truth.append(nchem_g[0])
    
classification_prompt = """
    Question-Answer Pair Classification Prompt
    You have been provided with a context from which question-answer pairs have been generated. Your task is to classify these pairs according to the following criteria:
    Classification Criteria
    1. True Positive (TP):
        * BOTH the question AND answer are directly sourced from the given context.
        * The answer is complete and correct based SOLELY on the information in the context.
        * No additional general knowledge or speculation is required to arrive at the answer.
        * All information needed to answer the question is explicitly stated in the context.
    2. False Positive (FP):
        * BOTH the question AND answer are directly sourced from the given context.
        * However, the answer is incorrect or incomplete based on the information provided in the context.
        * No inference or additional knowledge beyond the context is needed to determine the incorrectness.
    3. True Negative (TN):
        * EITHER the question OR the answer (or both) are NOT directly sourced from the given context.
        * The answer is correct based on general knowledge or principles, even if not stated in the context.
    4. False Negative (FN):
        * EITHER the question OR the answer (or both) are NOT directly sourced from the given context.
        * The answer is incorrect based on general knowledge or principles.
        * It must be explicitly stated that the question CANNOT be accurately answered based on general principles or knowledge.
    Important Guidelines
    1. Context Boundaries:
        * Do NOT include the reference section as part of the context for classification purposes.
    2. Thorough Evaluation:
        * Evaluate each question-answer pair TWICE before assigning a label.
        * If your initial evaluation is not TP, re-evaluate to ensure accuracy.
    3. Reasoning Questions:
        * For reasoning questions, ensure ALL information required for the answer is explicitly stated in the context before assigning a TP label.
        * If any part of the reasoning relies on general knowledge or principles not stated in the context, classify as TN or FN.
    4. Speculative Answers:
        * Answers based on general knowledge or speculation, rather than direct information from the context, should be classified as TN or FN, not TP or FP.
    5. Language Indicators:
        * Be cautious with phrases like "not explicitly stated in the context" or "based on general chemistry knowledge" in your explanations. These usually indicate the pair should NOT be classified as TP or FP.
    6. Superlatives:
        * Exercise extra caution when evaluating Q&A pairs containing superlatives like "always," "best," or "worst."
    7. Partial Context Match:
        * If only part of the question or answer is found in the context, but additional information or reasoning is required, classify as TN or FN, not TP or FP.
    8. Scientific Principles:
        * If an answer requires application of general scientific principles not explicitly stated in the context, classify as TN or FN, even if the principles are correct.
    9. Checking FN Correctness:
        * When classifying a pair as FN, always verify that the answer is indeed incorrect based on general principles or knowledge.
        * Explicitly state that the question cannot be accurately answered based on general principles or knowledge.
    10. Direct Sourcing:
        * "Sourced from the context" means the information is explicitly stated in the context, not inferred or derived from it.
        * If any inference or additional knowledge is required, the pair is not considered directly sourced from the context.
    11. Comparisons:
        * For questions involving comparisons, ensure that all elements of the comparison are explicitly stated in the context for a TP or FP classification.
    Task
    Please identify all the Q&A pairs that have been classified as TP, FP, TN, or FN. Provide a brief explanation for each classification, ensuring your reasoning aligns with the criteria and guidelines above.
    Remember: The key to accurate classification is determining whether BOTH the question AND answer are directly and fully sourced from the given context, without requiring any additional knowledge or reasoning. Be especially vigilant about distinguishing between information explicitly stated in the context and information that requires inference or additional knowledge.
    """

revision_prompt = """
    
    You are a Classification Prompt Expert specializing in optimizing evaluation prompts for 
    question-answer assessment. Your goal is to improve classification accuracy and catching rate 
    across diverse academic papers.
    

    INPUT:
    1. Average_accuracy：
       - Average accuracy of evaluation matches ground truth
    2. Average_total_catch:
       - Percentage of evaluation matches the ground truth of FP, TN, FN
    3. Original classification prompt
    4. Set of Q&A examples with:
       - Question and answer
       - LLM classifications
       - Ground truth classifications
       - Context used for classification
    5. Any correct classifications made by LLMs (to learn from successful cases)
    6. Prompts and performence of the trial with the best performence and the previous trial (if any)

    No explanatory text, analysis, or commentary should be included. Deliver only the enhanced prompt 
    text optimized for accurate classification.
    
    Note: Your revised prompt should emphasize specific evaluation criteria while maintaining flexibility for different academic contexts.
"""

# --- Configuration Parameters ---
MAX_TRIALS = 15
RUNS = 1
MAX_RERUNS = 5
MIN_CATCH_THRESHOLD = 0.95  # Minimum average total catch to consider the prompt acceptable
GOOD_PROMPT_THRESHOLD = 0.98  # If reached, we stop early
OUTPUT_MODEL_FILES = {
    "gpt_4o": "4o.xlsx",
    "gemini": "gemini.xlsx",
    "claude": "claude.xlsx",
    "gpt_o1": "o1.xlsx"
}
# (Assumes that functions: gpt_4o, gemini, claude, gpt_o1, summary_df, calculate_cumulative_statistics, 
# and global variables 'prompts', 'num_qs', 'ground_truth', 'contexts', 'claude_client', 'claude_model',
# 'revision_prompt', and 'client' are defined elsewhere.)

# --- Main Orchestration Loop ---
trial = 0
while trial < MAX_TRIALS:
    trial += 1
    folder_name = f"trial {trial}"  
    os.makedirs(folder_name, exist_ok=True)
    
    # If not the first trial, use the revised classification prompt from the previous trial.
    if trial > 1:
        prev_prompt_file = os.path.join(f"trial {trial-1}", "revised classification prompt.txt")
        try:
            with open(prev_prompt_file, "r") as file:
                classification_prompt = file.read()
        except Exception as e:
            print(f"Error reading previous classification prompt: {e}")
            classification_prompt = "Default classification prompt"  # Fallback

    # --- Run all models concurrently until we obtain the expected number of sheets ---
    reruns = MAX_RERUNS
    while True:
        print(f"\nTrial {trial}: Running models concurrently (reruns remaining: {reruns})...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(gpt_4o, classification_prompt, prompts, num_qs, os.path.join(folder_name, OUTPUT_MODEL_FILES["gpt_4o"]), runs=RUNS),
                executor.submit(gemini, classification_prompt, prompts, num_qs, os.path.join(folder_name, OUTPUT_MODEL_FILES["gemini"]), runs=RUNS),
                executor.submit(claude, classification_prompt, prompts, num_qs, os.path.join(folder_name, OUTPUT_MODEL_FILES["claude"]), runs=RUNS),
                executor.submit(gpt_o1, classification_prompt, prompts, num_qs, os.path.join(folder_name, OUTPUT_MODEL_FILES["gpt_o1"]), runs=RUNS)
            ]
            # Wait until all model tasks complete.
            concurrent.futures.wait(futures)

        # --- Check if the outputs are complete ---
        try:
            file_paths = [os.path.join(folder_name, OUTPUT_MODEL_FILES[model]) for model in OUTPUT_MODEL_FILES]
            sheet_counts = [len(pd.ExcelFile(fp).sheet_names) for fp in file_paths]
            expected_sheet_count = RUNS * len(prompts)
            print(f"Sheet counts: {sheet_counts}, Expected: {expected_sheet_count}")
            if all(count == expected_sheet_count for count in sheet_counts):
                # Reset reruns and break out of the inner loop.
                reruns = MAX_RERUNS
                break
            elif reruns <= 0:
                print("Maximum reruns reached; moving on.")
                reruns = MAX_RERUNS
                break
            else:
                reruns -= 1
        except Exception as e:
            print(f"Error checking output files: {e}")
            break

    # --- Summarize results from the models ---
    summary_file_path = os.path.join(folder_name, "summary.xlsx")
    dfs = summary_df(folder_name, "summary.xlsx", ground_truth, runs=RUNS)
    cumulative_statistics = calculate_cumulative_statistics(dfs)
    average_total_catch = cumulative_statistics.get('Cumulative Avg Non-TP Catching Rate', 0)
    average_accuracy = cumulative_statistics.get('Cumulative Avg Accuracy', 0)

    # Write summary statistics to files.
    for stat_name, stat_value in [("average_total_catch", average_total_catch), ("average_accuracy", average_accuracy)]:
        for ext in [f" {stat_value}.txt", ".txt"]:
            stat_file = os.path.join(folder_name, f"{stat_name}{ext}")
            with open(stat_file, "w") as file:
                file.write(f"{stat_name}: {stat_value}")

    # If the catch rate is high enough, possibly stop trials.
    if average_total_catch >= MIN_CATCH_THRESHOLD:
        if trial >= 5 or average_total_catch >= GOOD_PROMPT_THRESHOLD:
            print("Found a good prompt. Stopping trials.")
            break

    # --- Generate combined mismatch details ---
    combined_mismatch_dict = {}
    for key, df in dfs.items():
        # Assumes the sheet name follows a pattern and the DOI is the third token.
        DOI = key.split(' ')[2]
        if DOI not in combined_mismatch_dict:
            combined_mismatch_dict[DOI] = []
        # Compare each row's evaluation with the ground truth.
        for _, row in df[df['evaluation'] != df['ground_truth']].iterrows():
            question = row['question']
            answer = row['answer']
            ground_truth_eval = row['ground_truth']
            model_results = []
            for model in ['gpt', 'gemini', 'claude', 'gpt_o1']:
                model_evaluation = row.get(f'evaluation_{model}', 'N/A')
                model_explanation = row.get(f'explanation_{model}', 'N/A')
                is_match = "Match" if model_evaluation == ground_truth_eval else "Mismatch"
                model_results.append(
                    f"Model: {model}\nEvaluation: {model_evaluation}\nExplanation: {model_explanation}\nResult: {is_match}\n"
                )
            full_string = (
                f"Question: {question}\nAnswer: {answer}\nGround Truth: {ground_truth_eval}\n\n" +
                "".join(model_results)
            )
            combined_mismatch_dict[DOI].append(full_string)

    # --- Build final revision input string ---
    output_strings = []
    for DOI, evaluation_list in combined_mismatch_dict.items():
        # Assumes 'contexts' is a list and DOI can be evaluated to an index.
        context = contexts[eval(DOI)]
        header = f"--- Start of Evaluations Result for {DOI} ---"
        performance = f"---Average accuracy: {average_accuracy},  Average Total Catch: {average_total_catch}---"
        context_section = f"Context:\n{context}"
        evaluations = "Evaluations:\n" + "\n\n".join(evaluation_list)
        footer = f"--- End of Evaluations Result for {DOI} ---"
        output_strings.append(f"{header}\n{performance}\n{context_section}\n\n{evaluations}\n{footer}\n")
    final_output_string = "\n\n".join(output_strings)
    formatted_input = f"Current Classification Prompt: {classification_prompt}\n\n" + final_output_string

    # Write the revision input to a file.
    with open(os.path.join(folder_name, "revision input.txt"), "w") as file:
        file.write(final_output_string)

    # --- Determine best and previous run inputs for prompt revision ---
    best_input = ""
    previous_input = ""
    best_input_trial = None
    if trial > 1:
        best_input_catch_rate = 0
        for i in range(1, trial):
            try:
                with open(f"trial {i}/average_total_catch.txt", "r") as file:
                    input_catch_rate = float(file.read().split(": ")[-1])
                if input_catch_rate > best_input_catch_rate:
                    best_input_catch_rate = input_catch_rate
                    best_input_trial = i
            except Exception as e:
                print(f"Error reading average_total_catch from trial {i}: {e}")
        if best_input_trial is not None:
            with open(f"trial {best_input_trial}/revised classification prompt.txt", "r") as file:
                best_input = file.read()
            best_input = f"--- Start of Best Run Information ---\nPrompt: {best_input}"
            with open(f"trial {best_input_trial}/average_accuracy.txt", "r") as file:
                best_input += f"\n{file.read()}"
            with open(f"trial {best_input_trial}/average_total_catch.txt", "r") as file:
                best_input += f"\n{file.read()}\n--- End of Best Run Information"
        if trial > 2 and (trial - 1) != best_input_trial:
            with open(f"trial {trial-1}/revised classification prompt.txt", "r") as file:
                previous_input = file.read()
            previous_input = f"--- Start of Previous Run Information ---\nPrompt: {previous_input}"
            with open(f"trial {trial-1}/average_accuracy.txt", "r") as file:
                previous_input += f"\n{file.read()}"
            with open(f"trial {trial-1}/average_total_catch.txt", "r") as file:
                previous_input += f"\n{file.read()}\n--- End of Previous Run Information"
    
    formatted_input = formatted_input + "\n\n" + best_input + "\n\n" + previous_input

    # Write out prompt summary files.
    with open(os.path.join(folder_name, "prompt used.txt"), "w") as file:
        file.write(f"best: {best_input_trial}, previous: {trial - 1}")
    with open(os.path.join(folder_name, "full revision input.txt"), "w") as file:
        file.write(formatted_input)
    
    get_revised_classification_prompt(formatted_input, revision_prompt, folder_name)
