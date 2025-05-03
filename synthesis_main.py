from utils import process_context, walk_and_create_dataframe
from synthesis_api_call import claude, gemini, gpt_4o, gpt_o1, summary_df
import pandas as pd
import os
import concurrent.futures

def generate_prompts(df):
    prompts = list()
    contexts = list()
    for _, row in df.iterrows():
        print(_)
        paper = row['Paper File Path']
        sc_data = row['materials']
        if not (os.path.exists(paper)):
            print(f"file path: {paper} does not exist")
            # prompts.append('')
            # contexts.append('')
            continue
        try:
            context = process_context(paper)
        except Exception as e:
            print(e)
            # prompts.append('')
            # contexts.append('')
            continue
        output_text = "CONTEXT:"+context+"\n\nSynthesis Conditions Data:"+sc_data
        prompts.append(output_text)
        contexts.append(context)
    return prompts, contexts

base_path = 'inputs/synthesis-conditions-analysis' 
doi_dataframe = walk_and_create_dataframe(base_path)
# The code executes sequentially according to the order specified in the DataFrame. 
# It requires a directory named 'synthesis-condition-data', which has been omitted due to copyright restrictions held by the original publishers.
doi_dataframe['Paper File Path'] = 'inputs/synthesis-condition-data/' + doi_dataframe['Paper File Path']

prompts, contexts = generate_prompts(doi_dataframe)

check_prompt = """
Please evaluate ALL the synthesis conditions extracted from the provided context, focusing on Metal-Organic 
Frameworks (MOFs). Analyze according to these three criteria:
Criterion 1: Completeness
* Have synthesis conditions been included for all MOFs mentioned in the context?
Criterion 2: Data Type
* Verify that ONLY synthesis information has been extracted
* Ensure all experimental characterization data (like XRD patterns, surface area measurements, spectroscopic data) 
has been excluded from the extraction
Criterion 3: Accuracy
* Confirm that all synthesis conditions details have been extracted and that they are correctly matched to their 
corresponding MOFs

For each criterion, output Y if fully met for ALL MOFs, or N if not met for ANY MOF
"""

# Create output folder if it doesn't exist
folder_name = 'Synthesis-Condition-Individual-LLM-Output'
os.makedirs(folder_name, exist_ok=True)

# Settings
runs = 1        # Number of runs per prompt batch
step = 5        # Number of prompts to process per batch
max_reruns = 3  # Maximum number of rerun attempts per batch

# no_need list can be used to track sheets that failed irrecoverably
no_need = []

# Process prompts in batches of "step"
for i in range(0, len(prompts), step):
    start = i
    # Adjust 'end' so that if we exceed the number of prompts, we add one extra (as per original logic)
    end = i + step if i + step <= len(prompts) else len(prompts) + 1

    # Set number of rerun attempts for this batch
    reruns = max_reruns

    # While loop to re-run batch if not all sheets are generated
    while True:
        print(f'Processing batch prompts[{start}:{end}]')
        # Execute all four functions concurrently for this prompt batch
        with concurrent.futures.ThreadPoolExecutor() as executor:
            print('Launching concurrent tasks...')
            futures = [
                executor.submit(gpt_4o, check_prompt, start, end, prompts, f'{folder_name}/4o.xlsx', runs=runs),
                executor.submit(gemini, check_prompt, start, end, prompts, f'{folder_name}/gemini.xlsx', runs=runs),
                executor.submit(claude, check_prompt, start, end, prompts, f'{folder_name}/claude.xlsx', runs=runs),
                executor.submit(gpt_o1, check_prompt, start, end, prompts, f'{folder_name}/o1.xlsx', runs=runs)
            ]
            # Wait until all tasks are complete
            concurrent.futures.wait(futures)

        # After the concurrent execution, check if all expected sheets exist in the output files
        try:
            # Build list of expected sheet names for this batch
            expected_sheets = [f'{t} loop {j} DOI' for t in range(runs) for j in range(len(prompts[start:end]))]
            # Remove any sheets already marked as "failed" (no_need)
            expected_sheets = [sheet for sheet in expected_sheets if sheet not in no_need]

            # List of output Excel file paths
            file_paths = [
                f'{folder_name}/4o.xlsx',
                f'{folder_name}/gemini.xlsx',
                f'{folder_name}/claude.xlsx',
                f'{folder_name}/o1.xlsx'
            ]
            # Check for each file whether it contains at least as many sheets as expected
            sheets_complete = []
            for file_path in file_paths:
                if os.path.exists(file_path):
                    excel_file = pd.ExcelFile(file_path)
                    # Count sheets for this batch; using condition ">= end" as in your original code
                    sheets_complete.append(len(excel_file.sheet_names) >= end)
                else:
                    sheets_complete.append(False)

            # If all files have produced the required number of sheets, move on to the next batch
            if all(sheets_complete):
                print(f"Batch [{start}:{end}] processed successfully.")
                break

            # If maximum reruns reached, exit the loop for this batch
            if reruns <= 0:
                print(f"Maximum reruns reached for batch [{start}:{end}].")
                break

            # Otherwise, decrement reruns and try again
            reruns -= 1
            print(f"Retrying batch [{start}:{end}], remaining attempts: {reruns}")
        except Exception as e:
            print(f"Error checking Excel files: {e}")

file_name = 'summary.xlsx'
ground_truth_df = doi_dataframe 
run = 1
dfs = summary_df(folder_name, file_name, run)
