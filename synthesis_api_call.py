import pandas as pd
import time
import os
from pydantic import BaseModel, Field
from typing import Literal
from anthropic import BadRequestError, RateLimitError
from openai import BadRequestError, LengthFinishReasonError
import google.generativeai as genai
from openai import OpenAI
import google.generativeai as genai
import anthropic
from utils import weighted_mode, claude_info, gemini_info, gpt_4o_info, gpt_o1_info

def claude_output_to_df(output):
    """
    Convert the Anthropic API response to a pandas DataFrame.
    
    This function first attempts to extract the data from output.content[0].input.
    If that fails, it falls back to evaluating the output (expecting a 'pair' key).
    The resulting DataFrame columns are prefixed with 'claude_'.
    """
    try:
        df = pd.DataFrame([output.content[0].input])
    except Exception as e:
        print("Standard parsing failed, using fallback parsing. Error:", e)
        df = pd.DataFrame(eval(output)['pair'])
    df.columns = ['claude_' + col for col in df.columns]
    return df

def claude(check_prompt, start, end, prompts, output_file_name, need_regenerate=None, runs=3):
    """
    Process a subset of prompts using the Anthropic Claude API and write the resulting classification
    outputs to an Excel file. This function uses a while loop to iteratively reattempt processing for
    any missing sheets (i.e. regeneration).
    
    Parameters:
        check_prompt (str): The system prompt guiding the evaluation.
        start (int): Start index in the prompts list.
        end (int): End index (exclusive) in the prompts list.
        prompts (list): List of prompt strings.
        num_qs (list): Expected number of question-answer pairs for each prompt.
        output_file_name (str): Path to the Excel file for saving outputs.
        need_regenerate (list, optional): List of sheet names that need regeneration.
        runs (int, optional): Number of runs/loops per prompt batch.
        
    Returns:
        The final call to claude is replaced by iterative regeneration; when complete, the function exits.
    """
    # Initialize API keys and client
    
    claude_key, claude_model = claude_info()
    claude_client = anthropic.Anthropic(api_key=claude_key)
    
    # Define expected sheet names for all runs and prompt indices.
    need_regenerate_list = [f'{t} loop {i} DOI' for t in range(runs) for i in range(len(prompts[start:end]))]
    file_path = output_file_name
    
    # If a need_regenerate list is provided and is empty, nothing needs to be done.
    if isinstance(need_regenerate, list) and len(need_regenerate) == 0:
        print("claude finished")
        return
    
    # List to record sheets that fail unrecoverably.
    no_need = []
    max_iterations = 3  # Maximum number of regeneration iterations
    iteration = 0

    # Outer while loop for regeneration attempts
    while iteration < max_iterations:
        iteration += 1
        print(f"\n\n--- Iteration {iteration} of Claude regeneration loop ---")
        for t in range(runs):
            print(f"\nProcessing run {t} loop")
            for i in range(start, end):
                # Skip if prompt is invalid (e.g., NaN check)
                if prompts[i] != prompts[i]:
                    continue
                sheet_name = f'{t} loop {i} DOI'
                # If the output file exists and this sheet is already present, skip processing.
                if os.path.exists(file_path):
                    excel_file = pd.ExcelFile(file_path)
                    if sheet_name in excel_file.sheet_names:
                        continue
                # If a need_regenerate list is provided, only process the sheets in that list.
                if need_regenerate is not None and sheet_name not in need_regenerate:
                    continue

                prompt = prompts[i]
                retries = 5
                waiting_time = 60  # seconds
                print(f"Processing sheet: {sheet_name}")
                pass_this = False

                # Inner loop: try calling the API until success or retries are exhausted.
                while retries > 0:
                    try:
                        result = claude_client.messages.create(
                            model=claude_model,  
                            max_tokens=8192,
                            system=check_prompt,
                            tools=[
                                {
                                    "name": "synthesis_condition_checker",
                                    "description": "Evaluate synthesis conditions using well-structured JSON.",
                                    "input_schema": {
                                        "type": "object",
                                        "properties": {
                                            "all_synthesis_conditions": {
                                                "type": "string",
                                                "description": "string containing all input synthesis conditions exactly as provided"
                                            },
                                            "criterion_1": {
                                                "type": "string",
                                                "enum": ["Y", "N"],
                                                "description": "criterion 1 check result"
                                            },
                                            "criterion_1_explanation": {
                                                "type": "string",
                                                "description": "explanation of criterion 1 check result"
                                            },
                                            "criterion_2": {
                                                "type": "string",
                                                "enum": ["Y", "N"],
                                                "description": "criterion 2 check result"
                                            },
                                            "criterion_2_explanation": {
                                                "type": "string",
                                                "description": "explanation of criterion 2 check result"
                                            },
                                            "criterion_3": {
                                                "type": "string",
                                                "enum": ["Y", "N"],
                                                "description": "criterion 3 check result"
                                            },
                                            "criterion_3_explanation": {
                                                "type": "string",
                                                "description": "explanation of criterion 3 check result"
                                            }
                                        },
                                        "required": [
                                            "all_synthesis_conditions",
                                            "criterion_1",
                                            "criterion_1_explanation",
                                            "criterion_2", 
                                            "criterion_2_explanation",
                                            "criterion_3",
                                            "criterion_3_explanation"
                                        ]
                                    }
                                }
                            ],
                            tool_choice={"type": "tool", "name": "synthesis_condition_checker"},
                            messages=[{"role": "user", "content": prompt}]
                        )
                        result_df = claude_output_to_df(result)
                        if result_df.shape[1] == 7 and len(result_df) >= 1:
                            break # Exit retry loop upon success.
                    except (RateLimitError, BadRequestError, Exception) as e:
                        print(f"claude Exception on sheet {sheet_name}: {e}. Retrying after {waiting_time} seconds...")
                        time.sleep(waiting_time)
                        retries -= 1

                # If retries are exhausted, mark this sheet as "failed" and move on.
                if retries <= 0:
                    print(f"{sheet_name}: Exhausted retries, marking as no need.")
                    no_need.append(sheet_name)
                    continue

                print(f"Successfully processed sheet: {sheet_name}")
                # Write the resulting DataFrame to the Excel file.
                if os.path.exists(file_path):
                    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    with pd.ExcelWriter(file_path, mode='w', engine='openpyxl') as writer:
                        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Update the list of sheets that still need regeneration.
        if os.path.exists(file_path):
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            need_regenerate = [sheet for sheet in need_regenerate_list if sheet not in sheet_names]
            # Exclude sheets that have already failed unrecoverably.
            if no_need:
                need_regenerate = [sheet for sheet in need_regenerate if sheet not in no_need]
            if len(need_regenerate) == 0:
                print("claude finished: All sheets have been processed successfully.")
                break
            else:
                print("Sheets still needing regeneration:", need_regenerate)
        else:
            print("Output file not found, regenerating all sheets.")
    print("Exiting Claude regeneration loop after maximum iterations.")

class gemini_synthesis_condition_check(BaseModel):
    """
    Data model for evaluating synthesis conditions using the Gemini LLM API.
    
    This class captures the evaluation results for synthesis conditions extracted from scientific literature.
    It assesses the provided conditions based on three distinct criteria, indicating whether each criterion is
    met ('Y') or not met ('N'), along with explanations for each decision.
    """
    all_synthesis_conditions: str = Field(
        description="MAKE SURE YOU INCLUDE THIS:string containing all input synthesis conditions exactly as provided"
    )
    criterion_1: Literal["Y", "N"] = Field(
        description="criterion 1 check result"
    )
    criterion_1_explanation: str = Field(
        description="explanation of criterion 1 check result"
    )
    criterion_2: Literal["Y", "N"] = Field(
        description="criterion 2 check result"
    )
    criterion_2_explanation: str = Field(
        description="explanation of criterion 2 check result"
    )
    criterion_3: Literal["Y", "N"] = Field(
        description="criterion 3 check result"
    )
    criterion_3_explanation: str = Field(
        description="explanation of criterion 3 check result"
    )

def gemini_output_to_df(output):
    """
    Convert the Gemini API response into a pandas DataFrame.
    The output (a JSON string) is evaluated and converted into a DataFrame.
    If the 'evaluation' column is missing, it is added with a default value.
    Finally, all column names are prefixed with 'gemini_'.
    """
    df = pd.DataFrame([eval(output.parts[0].text)])
    df.columns = ['gemini_' + col for col in df.columns]
    return df

def gemini(check_prompt, start, end, prompts, output_file_name, need_regenerate=None, runs=3):
    """
    Process a subset of prompts using the Gemini API and write results to an Excel file.
    This function uses a while loop to iteratively reattempt processing of any missing sheets.
    
    Parameters:
        check_prompt (str): The system prompt guiding the evaluation.
        start (int): Start index in the prompts list.
        end (int): End index (exclusive) in the prompts list.
        prompts (list): List of prompt strings.
        num_qs (list): Expected number of Q&A pairs per prompt.
        output_file_name (str): Path to the output Excel file.
        need_regenerate (list, optional): List of sheet names that need regeneration.
        runs (int, optional): Number of runs (loops) per prompt.
    
    Returns:
        None. Outputs are written directly to the specified Excel file.
    """
    # Initialize API keys and client
    gemini_api_key, gemini_model = gemini_info()
    genai.configure(api_key=gemini_api_key)

    # Create the expected list of sheet names based on runs and prompt count.
    need_regenerate_list = [f'{t} loop {i} DOI' for t in range(runs) for i in range(len(prompts[start:end]))]
    file_path = output_file_name

    if isinstance(need_regenerate, list) and len(need_regenerate) == 0:
        print("gemini finished")
        return

    max_iterations = 3  # Maximum regeneration attempts
    iteration = 0

    # Outer while loop for regeneration attempts
    while iteration < max_iterations:
        iteration += 1
        print(f"\n\n--- Iteration {iteration} of Gemini regeneration loop ---")
        for t in range(runs):
            print(f"\nProcessing run {t} loop")
            for i in range(start, end):
                # Skip invalid prompt entries (NaN check)
                if prompts[i] != prompts[i]:
                    continue
                sheet_name = f'{t} loop {i} DOI'
                # Skip if sheet already exists in the output file
                if os.path.exists(file_path):
                    excel_file = pd.ExcelFile(file_path)
                    if sheet_name in excel_file.sheet_names:
                        continue
                # If a need_regenerate list is provided, only process sheets in that list.
                if need_regenerate is not None and sheet_name not in need_regenerate:
                    continue

                prompt = prompts[i]
                retries = 3
                waiting_time = 5  # seconds

                while retries > 0:
                    try:
                        print(f"Gemini processing sheet: {sheet_name}")
                        gemini_client = genai.GenerativeModel(
                            model_name=gemini_model,  # Ensure gemini_model is defined in your environment
                            system_instruction=check_prompt
                        )
                        result = gemini_client.generate_content(
                            prompt,
                            generation_config=genai.GenerationConfig(
                                response_mime_type="application/json", 
                                response_schema=gemini_synthesis_condition_check
                            ),
                        )
                        result_df = gemini_output_to_df(result)
                        if result_df.shape[1] == 7 and len(result_df) >= 1:
                            break # Exit retry loop upon success.
                    except Exception as e:
                        print(f"Gemini Error on sheet {sheet_name}: {e}. Retrying...")
                        if retries > 0:
                            print(f"Retrying after {waiting_time} seconds...")
                            time.sleep(waiting_time)
                            retries -= 1
                if retries <= 0:
                    print(f"Sheet {sheet_name} failed after retries.")
                    continue

                print(f"Successfully processed sheet: {sheet_name}")
                # Write the result DataFrame to the Excel file
                if os.path.exists(file_path):
                    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    with pd.ExcelWriter(file_path, mode='w', engine='openpyxl') as writer:
                        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Update the list of sheets that still need regeneration.
        if os.path.exists(file_path):
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            need_regenerate = [sheet for sheet in need_regenerate_list if sheet not in sheet_names]
            if len(need_regenerate) == 0:
                print("gemini finished: All sheets have been processed successfully.")
                break
            else:
                print("Sheets still needing regeneration:", need_regenerate)
        else:
            print("Output file not found, regenerating all sheets.")
    print("Exiting Gemini regeneration loop after maximum iterations.")

class gpt_synthesis_condition_check(BaseModel):
    """
    Data model for evaluating synthesis conditions using the OpenAI GPT LLM API.
    
    This class captures evaluation results for synthesis conditions extracted from scientific papers.
    It assesses these conditions based on three criteria, clearly marking each criterion as met ('Y') or not met ('N'),
    and includes a detailed explanation of each determination.
    """
    all_synthesis_conditions: str = Field(
        description="string containing all input synthesis conditions exactly as provided"
    )
    criterion_1: Literal["Y", "N"] = Field(
        description="criterion 1 check result"
    )
    criterion_1_explanation: str = Field(
        description="explanation of criterion 1 check result"
    )
    criterion_2: Literal["Y", "N"] = Field(
        description="criterion 2 check result"
    )
    criterion_2_explanation: str = Field(
        description="explanation of criterion 2 check result"
    )
    criterion_3: Literal["Y", "N"] = Field(
        description="criterion 3 check result"
    )
    criterion_3_explanation: str = Field(
        description="explanation of criterion 3 check result"
    )

def gpt_output_to_df(output):
    """
    Convert the GPT output (a string representing a dictionary) into a pandas DataFrame.
    Each column is prefixed with an underscore.
    """
    df = pd.DataFrame([eval(output)])
    df.columns = ['gpt_' + col for col in df.columns]
    return df

def gpt_4o(check_prompt, start, end, prompts, output_file_name, need_regenerate=None, runs=1):
    """
    Process a subset of prompts using the GPT-4o model and write results to an Excel file.
    Uses a while loop to iteratively reattempt processing for any missing sheets.
    
    Parameters:
        check_prompt (str): The system prompt guiding the GPT‑4O evaluation.
        start (int): Starting index in the prompts list.
        end (int): Ending index (exclusive) in the prompts list.
        prompts (list): List of prompt strings.
        num_qs (list): Expected number of Q&A pairs per prompt.
        output_file_name (str): Path to the Excel file for saving outputs.
        need_regenerate (list, optional): List of sheet names that need regeneration.
        runs (int, optional): Number of runs/loops per prompt.
        
    Returns:
        None. Outputs are written directly to the specified Excel file.
    """
    # Initialize API keys and client
    openai_api_key, gpt_4o_model = gpt_4o_info()
    openai_client = OpenAI(api_key=openai_api_key)
    
    # Generate the expected list of sheet names based on runs and number of prompts.
    need_regenerate_list = [f'{t} loop {i} DOI' for t in range(runs) for i in range(len(prompts[start:end]))]
    file_path = output_file_name

    # If need_regenerate is provided and empty, nothing to do.
    if isinstance(need_regenerate, list) and len(need_regenerate) == 0:
        print("4o finished")
        return

    max_iterations = 3  # Maximum number of regeneration iterations
    iteration = 0
    no_need = []  # List to track sheets that failed irrecoverably

    # Outer while loop for regeneration attempts
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} of GPT‑4O regeneration loop ---")
        for t in range(runs):
            print(f"\nProcessing run {t} loop")
            for i in range(start, end):
                # Skip if prompt is invalid (e.g., NaN check)
                if prompts[i] != prompts[i]:
                    continue
                sheet_name = f'{t} loop {i} DOI'
                # If the output file exists and this sheet is already present, skip it.
                if os.path.exists(file_path):
                    excel_file = pd.ExcelFile(file_path)
                    if sheet_name in excel_file.sheet_names:
                        continue
                # If a need_regenerate list is provided, process only those sheets.
                if need_regenerate is not None and sheet_name not in need_regenerate:
                    continue

                prompt = prompts[i]
                retries = 5
                waiting_time = 60  # seconds
                print(f"Processing sheet: {sheet_name}")
                pass_this = False

                # Inner loop: attempt API call with retries.
                while retries > 0:
                    try:
                        completion = openai_client.beta.chat.completions.parse(
                            model=gpt_4o_model,
                            messages=[
                                {"role": "system", "content": check_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            response_format=gpt_synthesis_condition_check,
                        )
                        result = completion.choices[0].message.content
                        result_df = gpt_output_to_df(result)
                        if len(result_df) >= 1:
                            break  # Exit retry loop upon success.
                    except (BadRequestError, LengthFinishReasonError) as e:
                        pass_this = True
                        print(f"{sheet_name}: Error encountered: {e}")
                        break
                    except Exception as e:
                        print(f"{sheet_name}: Exception encountered: {e}. Retrying after {waiting_time} seconds...")
                        time.sleep(waiting_time)
                        retries -= 1

                # If the API call failed irrecoverably or retries exhausted, mark the sheet.
                if pass_this or retries <= 0:
                    print(f"{sheet_name}: Marked as failed (no need to retry further).")
                    no_need.append(sheet_name)
                    continue

                print(f"Successfully processed sheet: {sheet_name}")
                # Write the DataFrame to the Excel file (append if file exists, otherwise create new).
                if os.path.exists(file_path):
                    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    with pd.ExcelWriter(file_path, mode='w', engine='openpyxl') as writer:
                        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Update the need_regenerate list based on the sheets produced.
        if os.path.exists(file_path):
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            need_regenerate = [sheet for sheet in need_regenerate_list if sheet not in sheet_names]
            # Exclude sheets that have been marked as failed.
            if no_need:
                need_regenerate = [sheet for sheet in need_regenerate if sheet not in no_need]
            if len(need_regenerate) == 0:
                print("4o finished: All sheets processed successfully.")
                break
            else:
                print("Sheets still needing regeneration:", need_regenerate)
        else:
            print("Output file not found. Regenerating all sheets.")
    print("Exiting GPT‑4o regeneration loop after maximum iterations.")

def gpt_o1_output_to_df(output):
    """
    Convert the GPT-o1 output (a string representing a dictionary) into a pandas DataFrame.
    Each column is prefixed with an underscore.
    """
    df = pd.DataFrame([eval(output)])
    df.columns = ['gpt_o1_' + col for col in df.columns]
    return df

def gpt_o1(check_prompt, start, end, prompts, output_file_name, need_regenerate=None, runs=3):
    """
    Process a subset of prompts using the GPT-o1 model and write results to an Excel file.
    This function uses an outer while loop to reattempt processing of any missing sheets (i.e. regeneration).
    
    Parameters:
        check_prompt (str): The system prompt that guides GPT-o1 evaluation.
        start (int): Start index in the prompts list.
        end (int): End index (exclusive) in the prompts list.
        prompts (list): List of prompt strings.
        num_qs (list): Expected number of Q&A pairs per prompt.
        output_file_name (str): Path to the Excel file for saving outputs.
        need_regenerate (list, optional): List of sheet names that need regeneration.
        runs (int, optional): Number of runs/loops per prompt.
        
    Returns:
        None. Outputs are written directly to the specified Excel file.
    """
    # Initialize API keys and client
    openai_api_key, gpt_o1_model = gpt_o1_info()
    _, gpt_4o_model = gpt_4o_info()
    openai_client = OpenAI(api_key=openai_api_key)
    
    # Instruction to ensure proper output formatting
    o1_list_output_prompt = (
        'Please format your answers to a dictionary in string following this structure: '
        '{"all_synthesis_conditions": "string containing all input synthesis conditions exactly as provided", '
        '"criterion_1": "Y or N", '
        '"criterion_1_explanation": "string: explanation of criterion 1 check result", '
        '"criterion_2": "Y or N", '
        '"criterion_2_explanation": "string: explanation of criterion 2 check result", '
        '"criterion_3": "Y or N", '
        '"criterion_3_explanation": "string: explanation of criterion 3 check result"}'
    )
    
    # Generate the expected list of sheet names based on the number of runs and prompts.
    need_regenerate_list = [f'{t} loop {i} DOI' for t in range(runs) for i in range(len(prompts[start:end]))]
    file_path = output_file_name

    # If a need_regenerate list is provided and empty, nothing needs to be done.
    if isinstance(need_regenerate, list) and len(need_regenerate) == 0:
        print("o1 finished")
        return

    max_iterations = 3  # Maximum number of regeneration attempts
    iteration = 0
    no_need = []  # Track sheets that fail irrecoverably

    # Outer while loop for regeneration attempts
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} of GPT-o1 regeneration loop ---")
        for t in range(runs):
            print(f"\nProcessing run {t} loop")
            for i in range(start, end):
                # Skip invalid prompt entries (NaN check)
                if prompts[i] != prompts[i]:
                    continue
                sheet_name = f'{t} loop {i} DOI'
                print(f"Processing sheet: {sheet_name}")
                
                # Skip if the sheet already exists in the output file.
                if os.path.exists(file_path):
                    excel_file = pd.ExcelFile(file_path)
                    if sheet_name in excel_file.sheet_names:
                        continue
                
                # If need_regenerate is specified, process only those sheets.
                if need_regenerate is not None and sheet_name not in need_regenerate:
                    continue

                prompt = prompts[i]
                retries = 5
                pass_this = False
                output_df = None

                # Inner loop: try the API call with multiple retries.
                while retries > 0:
                    print(f"GPT-o1 retries remaining for {sheet_name}: {retries}")
                    retries -= 1
                    if retries <= 0:
                        break
                    try:
                        # Combine the check prompt, the prompt itself, and additional output instructions.
                        input_prompt = check_prompt + prompt + o1_list_output_prompt
                        completion = openai_client.chat.completions.create(
                            model=gpt_o1_model,
                            messages=[{"role": "user", "content": input_prompt}],
                        )
                        print("GPT-o1 response received")
                        output = completion.choices[0].message.content
                        if len(output) <= 0:
                            pass_this = True
                            break
                        output_df = gpt_o1_output_to_df(output)
                        if len(output_df) >= 1:
                            break
                    except (BadRequestError, LengthFinishReasonError) as e:
                        pass_this = True
                        print(f"{sheet_name}: BadRequestError encountered: {e}")
                        break
                    except Exception as e:
                        print(f"{sheet_name}: Exception encountered: {e}")
                        try:
                            # Attempt to fix broken JSON output using the GPT-fix model.
                            fixed_completion = openai_client.beta.chat.completions.parse(
                                model=gpt_4o_model,
                                messages=[
                                    {"role": "system", "content": "This is a broken JSON file that contains a list of all_synthesis_conditions, criterion_1, criterion_1_explanation, criterion_2, criterion_2_explanation, criterion_3, criterion_3_explanation"},
                                    {"role": "user", "content": output}
                                ],
                                response_format=gpt_synthesis_condition_check,
                            )
                            output = fixed_completion.choices[0].message.content
                            output_df = gpt_o1_output_to_df(output)
                            break
                        except Exception as fix_e:
                            print(f"{sheet_name}: Cannot fix broken output: {fix_e}")
                
                # If the API call failed irrecoverably, mark the sheet and continue.
                if pass_this or output_df is None:
                    print(f"{sheet_name}: Marked as failed.")
                    no_need.append(sheet_name)
                    continue

                print(f"Successfully processed sheet: {sheet_name}")
                # Write the DataFrame to the Excel file (append if file exists, else create new).
                if os.path.exists(file_path):
                    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                        output_df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    with pd.ExcelWriter(file_path, mode='w', engine='openpyxl') as writer:
                        output_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Update the list of sheets that still need regeneration.
        if os.path.exists(file_path):
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            need_regenerate = [sheet for sheet in need_regenerate_list if sheet not in sheet_names]
            # Remove sheets that have been marked as failed.
            for failed_sheet in no_need:
                if failed_sheet in need_regenerate:
                    need_regenerate.remove(failed_sheet)
            if len(need_regenerate) == 0:
                print("o1 finished: All sheets processed successfully.")
                break
            else:
                print("Sheets still needing regeneration:", need_regenerate)
        else:
            print("Output file not found. Regenerating all sheets.")
    print("Exiting GPT-o1 regeneration loop after maximum iterations.")

def summary_df(folder_name, file_name, runs=3):
    """
    Create a summary by merging results from multiple Excel files and computing evaluation criteria.
    
    This function reads sheets from four Excel files (4o.xlsx, gemini.xlsx, claude.xlsx, o1.xlsx)
    that are expected to have names in the format '{outer} loop {sheet_index} DOI'. It extracts
    the first-row data (excluding the first column) from each sheet, merges these into a single
    DataFrame, computes evaluation criteria using weighted_mode, and writes the summary into a
    designated summary Excel file.
    
    Parameters:
        folder_name (str): Folder containing the source Excel files.
        file_name (str): Name of the summary Excel file to write.
        runs (int, optional): Number of outer iterations/loops. Default is 3.
    
    Returns:
        dict: A dictionary mapping sheet names to their summary DataFrames.
    """
    def add_col_prefix(df, prefix):
        """Prefix all column names in the DataFrame with the provided prefix."""
        df.columns = [prefix + col for col in df.columns]
        return df

    def reorder_columns(df):
        """
        Reorder DataFrame columns in a specific pattern:
        - Last two columns first,
        - Then the five columns preceding those,
        - Followed by the remaining columns.
        """
        cols = df.columns.tolist()
        new_order = cols[-2:] + cols[-7:-2] + cols[:-7]
        return df[new_order]

    outputs = {}
    # Loop over each "run" (outer loop)
    for outer in range(runs):
        exit_loop = False
        # Loop over a fixed range of sheet indices (inner loop)
        for sheet_index in range(100):
            sheet_name = f'{outer} loop {sheet_index} DOI'
            # Initialize flags for tracking successful reading from each file
            gpt_4o_pass = gemini_pass = claude_pass = gpt_o1_pass = False
            try:
                # Read and process each file's sheet
                gpt_4o_df = pd.read_excel(f'{folder_name}/4o.xlsx', sheet_name=sheet_name)
                gpt_4o_pass = True

                gemini_df = pd.read_excel(f'{folder_name}/gemini.xlsx', sheet_name=sheet_name)
                gemini_pass = True

                # For claude, no prefix function is applied in this version.
                claude_df = pd.read_excel(f'{folder_name}/claude.xlsx', sheet_name=sheet_name)
                claude_pass = True

                gpt_o1_df = pd.read_excel(f'{folder_name}/o1.xlsx', sheet_name=sheet_name)
                gpt_o1_pass = True

                # Combine the results from all sources: extract first row (excluding first column) from each
                combined_data = {}
                for df in [gpt_4o_df, gemini_df, claude_df, gpt_o1_df]:
                    # Starting from index 1 (skip first column)
                    for col in df.columns[1:]:
                        combined_data[col] = df[col].iloc[0]
                output = pd.DataFrame([combined_data])

                # Compute evaluation criteria using weighted_mode function
                output['criterion_1'] = weighted_mode(
                    output,
                    ['gpt_criterion_1', 'gemini_criterion_1', 'claude_criterion_1', 'gpt_o1_criterion_1'],
                    [0.23, 0.23, 0.23, 0.3]
                )
                output['criterion_2'] = weighted_mode(
                    output,
                    ['gpt_criterion_2', 'gemini_criterion_2', 'claude_criterion_2', 'gpt_o1_criterion_2'],
                    [0.23, 0.23, 0.23, 0.3]
                )
                output['criterion_3'] = weighted_mode(
                    output,
                    ['gpt_criterion_3', 'gemini_criterion_3', 'claude_criterion_3', 'gpt_o1_criterion_3'],
                    [0.23, 0.23, 0.23, 0.3]
                )

                # Reorder columns according to the specified pattern
                # output = reorder_columns(output)
                outputs[sheet_name] = output

                # Write the summary DataFrame to the summary Excel file
                summary_path = f'{folder_name}/{file_name}'
                if os.path.exists(summary_path):
                    with pd.ExcelWriter(summary_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                        output.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    with pd.ExcelWriter(summary_path, mode='w', engine='openpyxl') as writer:
                        output.to_excel(writer, sheet_name=sheet_name, index=False)

            except ValueError as e:
                print(e)
                print(f'Need to regenerate sheet: {sheet_name} | Flags: '
                      f'gpt_4o_pass={gpt_4o_pass}, gemini_pass={gemini_pass}, '
                      f'claude_pass={claude_pass}, gpt_o1_pass={gpt_o1_pass}')
            except Exception as e:
                print(f"Error processing sheet {sheet_name}: {e}")
                exit_loop = True
        if exit_loop:
            break
    return outputs
