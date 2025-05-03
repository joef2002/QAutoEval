import pandas as pd
import time
import os
from pydantic import BaseModel
from anthropic import BadRequestError, RateLimitError
from openai import BadRequestError, LengthFinishReasonError
import google.generativeai as genai
from openai import OpenAI
import google.generativeai as genai
import anthropic
from utils import weighted_mode, merged_df_v2, claude_info, gemini_info, gpt_4o_info, gpt_o1_info

class question_answer_pairs(BaseModel):
    """
    Data model for representing and evaluating multiple question-answer pairs.

    This class encapsulates a collection of individual question-answer pairs, 
    along with their types, evaluations, and explanations. It also provides an overall 
    evaluation summary for the entire set of pairs.
    """
    class Single_Pair(BaseModel):
        """
        Represents a single question-answer pair with evaluation details.

        Attributes:
            question (str): The text of the question.
            answer (str): The text of the answer.
            question_type (str): Category or type of the question.
            evaluation (str): Evaluation result (e.g., TP, FP, TN, FN).
            explanation (str): Explanation for the evaluation result.
        """
        question: str
        answer: str
        question_type: str
        evaluation: str
        explanation: str
    pair:list[Single_Pair]
    question_anwer_pairs_evaluation: str

def claude_output_to_df(output):
    """
    Convert the Anthropic API response into a pandas DataFrame.
    Attempts to extract the 'pair' field from the response content; if that fails, falls back to eval.
    Then, standardizes the evaluation values and renames columns with a 'claude_' prefix.
    """
    try:
        df = pd.DataFrame(output.content[0].input['pair'])
    except Exception as e:
        print('Standard parsing failed, fixed by fallback:', e)
        df = pd.DataFrame(eval(output)['pair'])
    df['evaluation'] = df['evaluation'].replace({
        r'True\s+Positive\b.*': 'TP',
        r'False\s+Positive\b.*': 'FP',
        r'False\s+Negative\b.*': 'FN',
        r'True\s+Negative\b.*': 'TN'
    }, regex=True)
    df = df.rename(columns={
        'evaluation': 'evaluation_claude',
        'explanation': 'explanation_claude',
        'revised_evaluation': 'revised_evaluation_claude',
        'reason': 'reason_claude'
    })
    return df

def claude(classification_prompt, start, end, prompts, num_qs, output_file_name, need_regenerate=None, runs=3):
    """
    Process a subset of prompts using the Anthropic Claude API and write the resulting classification
    outputs to an Excel file. This function uses a while loop to iteratively reattempt processing for
    any missing sheets (i.e. regeneration).
    
    Parameters:
        classification_prompt (str): The system prompt guiding the evaluation.
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

    openai_api_key, gpt_4o_model = gpt_4o_info()
    openai_client = OpenAI(api_key=openai_api_key)
    
    # Build the complete list of expected sheet names.
    need_regenerate_list = [f'{t} loop {j} DOI' for t in range(runs) for j in range(len(prompts[start:end]))]
    # If need_regenerate is provided and empty, there is nothing to do.
    if isinstance(need_regenerate, list) and len(need_regenerate) == 0:
        print('claude finished')
        return

    max_iterations = 3  # Maximum outer regeneration iterations
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} of Claude regeneration loop ---")
        for t in range(runs):
            print(f"\nProcessing run {t} loop")
            for i in range(start, end):
                # Skip if no questions expected for this prompt.
                if num_qs[i] == 0:
                    continue
                sheet_name = f'{t} loop {i} DOI'
                # Skip if this sheet already exists in the output file.
                if os.path.exists(output_file_name):
                    excel_file = pd.ExcelFile(output_file_name)
                    if sheet_name in excel_file.sheet_names:
                        continue
                # If need_regenerate list is provided, process only sheets in that list.
                if need_regenerate is not None and sheet_name not in need_regenerate:
                    continue

                prompt = prompts[i]
                retries = 5
                waiting_time = 60  # seconds
                # Outer loop for retrying the API call.
                while retries > 0:
                    try:
                        result = claude_client.messages.create(
                            model=claude_model,
                            max_tokens=8192,
                            system=classification_prompt,
                            tools=[
                                {
                                    "name": "question_answer_pair_classifier",
                                    "description": "Classification result of each question-answer pair using well-structured JSON.",
                                    "input_schema": {
                                        "type": "object",
                                        "properties": {
                                            "pair": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "question": {
                                                            "type": "string",
                                                            "description": "The question being evaluated"
                                                        },
                                                        "answer": {
                                                            "type": "string",
                                                            "description": "The answer to the question"
                                                        },
                                                        "question_type": {
                                                            "type": "string",
                                                            "description": "The type or category of the question"
                                                        },
                                                        "evaluation": {
                                                            "type": "string",
                                                            "description": "Evaluation classification of the question answer pair (i.e., TP, FP, TN, FN)"
                                                        },
                                                        "explanation": {
                                                            "type": "string",
                                                            "description": "Explanation for the evaluation"
                                                        }
                                                    },
                                                    "required": ["question", "answer", "question_type", "evaluation", "explanation"]
                                                },
                                                "description": "List of question-answer pairs with their evaluations"
                                            },
                                        },
                                        "required": ["pair"]
                                    }
                                }
                            ],
                            tool_choice={"type": "tool", "name": "question_answer_pair_classifier"},
                            messages=[{"role": "user", "content": prompt}]
                        )
                        # Attempt to convert the response to a DataFrame.
                        result_df = claude_output_to_df(result)
                        if len(result_df) != num_qs[i] or result_df.shape[1] != 5:
                            print(f"Claude: Sheet {sheet_name}: Expected {num_qs[i]} pairs but got {len(result_df)}")
                        else:
                            break # Successful call; exit retry loop.
                    except ValueError as e:
                        try:
                            # Attempt to fix broken JSON using a GPT-fix call.
                            gpt_fix = openai_client.beta.chat.completions.parse(
                                model=gpt_4o_model,
                                messages=[
                                    {"role": "system", "content": "This is a broken JSON file that contains a list of questions, answers, and additional metadata. Please fix the JSON structure and remove any extra quote characters."},
                                    {"role": "user", "content": result.content[0].input['pair']}
                                ],
                                response_format=question_answer_pairs,
                            )
                            result = gpt_fix.choices[0].message.content
                            result_df = claude_output_to_df(result)
                            if len(result_df) != num_qs[i] or result_df.shape[1] != 5:
                                print(f"GPT 4o(Claude): Sheet {sheet_name}: Expected {num_qs[i]} pairs but got {len(result_df)}")
                            else:
                                break
                        except Exception as fix_e:
                            print('Unable to fix JSON output:', fix_e)
                            break
                    except (RateLimitError, BadRequestError, Exception) as e:
                        print(f"claude Exception on sheet {sheet_name}: {e}. Retrying after {waiting_time} seconds...")
                        time.sleep(waiting_time)
                        retries -= 1
                # Exit inner while if retries are exhausted.
                if retries <= 0:
                    print(f"Sheet {sheet_name} failed after retries.")
                    continue
                # Write the result DataFrame to the Excel file.
                if os.path.exists(output_file_name):
                    with pd.ExcelWriter(output_file_name, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    with pd.ExcelWriter(output_file_name, mode='w', engine='openpyxl') as writer:
                        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
        # End of processing for current iteration.
        # Update need_regenerate: check which expected sheets are still missing.
        if os.path.exists(output_file_name):
            excel_file = pd.ExcelFile(output_file_name)
            existing_sheets = excel_file.sheet_names
            new_need_regenerate = [sheet for sheet in need_regenerate_list if sheet not in existing_sheets]
        else:
            new_need_regenerate = need_regenerate_list
        if len(new_need_regenerate) == 0:
            print("claude finished: All sheets processed successfully.")
            break
        else:
            print("Sheets still needing regeneration:", new_need_regenerate)
            need_regenerate = new_need_regenerate
    print("Exiting claude regeneration loop after maximum iterations.")

def gemini_output_to_df(output):
    """
    Convert the Gemini API response to a standardized pandas DataFrame.
    
    The function attempts to evaluate the text response from the first part of the output,
    extract the "pair" field, and convert it into a DataFrame. If the "pair" key is missing,
    it falls back to evaluating the entire text. It then:
      - Ensures an "explanation" column exists (defaulting to "none" if absent).
      - Standardizes the evaluation strings using regular expressions.
      - Renames columns to include a 'gemini_' prefix.
    
    Parameters:
        output: The response object from the Gemini API, which must have a 'parts' attribute 
                where the first element's text is a JSON-like string.
                
    Returns:
        pd.DataFrame: A DataFrame with the standardized question-answer pair information.
    """
    try:
        # Try to parse the response and extract the 'pair' field.
        data = eval(output.parts[0].text)
        df = pd.DataFrame(data['pair'])
    except Exception as e:
        # Fallback: evaluate the entire text if 'pair' is missing.
        df = pd.DataFrame(eval(output.parts[0].text))
    
    # Ensure 'explanation' exists.
    if 'explanation' not in df.columns:
        df['explanation'] = ['none' for _ in range(len(df))]
    
    # Standardize evaluation strings.
    df['evaluation'] = df['evaluation'].replace({
        r'True\s+Positive\b.*': 'TP',
        r'False\s+Positive\b.*': 'FP',
        r'False\s+Negative\b.*': 'FN',
        r'True\s+Negative\b.*': 'TN'
    }, regex=True)
    
    # Rename columns to add a gemini prefix.
    df = df.rename(columns={
        'evaluation': 'evaluation_gemini',
        'explanation': 'explanation_gemini',
        'revised_evaluation': 'revised_evaluation_gemini',
        'reason': 'reason_gemini'
    })
    return df

def gemini(classification_prompt, start, end, prompts, num_qs, output_file_name, need_regenerate=None, runs=3):
    """
    Process a subset of prompts using the Gemini API and write results to an Excel file.
    This function uses a while loop to iteratively reattempt processing of any missing sheets.
    
    Parameters:
        classification_prompt (str): The system prompt for classification.
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

    # Build the list of expected sheet names for all runs within [start:end]
    need_regenerate_list = [f'{t} loop {j} DOI' for t in range(runs) for j in range(len(prompts[start:end]))]
    if isinstance(need_regenerate, list) and len(need_regenerate) == 0:
        print("gemini finished")
        return

    max_iterations = 3  # Maximum number of outer iterations for regeneration
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Gemini regeneration iteration {iteration} ---")
        
        # Process each run and each prompt in the given range.
        for t in range(runs):
            print(f"\nProcessing run {t} loop")
            for i in range(start, end):
                # Skip prompts with zero expected Q&A pairs.
                if num_qs[i] == 0:
                    continue
                sheet_name = f'{t} loop {i} DOI'
                # Skip sheet if it already exists.
                if os.path.exists(output_file_name):
                    excel_file = pd.ExcelFile(output_file_name)
                    if sheet_name in excel_file.sheet_names:
                        continue
                # Process only if the sheet is in the regeneration list (if provided)
                if need_regenerate is not None and sheet_name not in need_regenerate:
                    continue
                prompt = prompts[i]
                retries = 5
                waiting_time = 5  # seconds

                # Inner loop: retry API call until a valid response is obtained or retries are exhausted.
                while retries > 0:
                    try:
                        print(f"gemini processing sheet: {sheet_name}")
                        gemini_client = genai.GenerativeModel(
                            model_name=gemini_model,
                            system_instruction=classification_prompt
                        )
                        result = gemini_client.generate_content(
                            prompt,
                            generation_config=genai.GenerationConfig(
                                response_mime_type="application/json",
                                response_schema=question_answer_pairs
                            ),
                        )
                        df_result = gemini_output_to_df(result)
                        if len(df_result) != num_qs[i] or df_result.shape[1] != 5:
                            print(f"Gemini: Sheet {sheet_name}: Expected {num_qs[i]} pairs but got {len(df_result)}")
                            retries -= 1
                            if retries < 0:
                                print("Too many retries for this sheet.")
                                break
                        else:
                            break  # Successful response.
                    except Exception as e:
                        print(f"gemini Error on sheet {sheet_name}: {e}. Retrying after {waiting_time} seconds...")
                        time.sleep(waiting_time)
                        retries -= 1
                # Skip this sheet if retries are exhausted.
                if retries <= 0:
                    continue

                result_df = gemini_output_to_df(result)
                # Write the result DataFrame to the output Excel file.
                if os.path.exists(output_file_name):
                    with pd.ExcelWriter(output_file_name, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    with pd.ExcelWriter(output_file_name, mode='w', engine='openpyxl') as writer:
                        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"Finished processing sheet: {sheet_name}")

        # Update regeneration list: determine which expected sheets are still missing.
        if os.path.exists(output_file_name):
            excel_file = pd.ExcelFile(output_file_name)
            existing_sheets = excel_file.sheet_names
            new_need_regenerate = [sheet for sheet in need_regenerate_list if sheet not in existing_sheets]
        else:
            new_need_regenerate = need_regenerate_list

        if len(new_need_regenerate) == 0:
            print("gemini finished: All sheets processed successfully.")
            break
        else:
            print("Sheets still needing regeneration:", new_need_regenerate)
            need_regenerate = new_need_regenerate

    print("Exiting Gemini regeneration loop after maximum iterations.")
    return

def gpt_output_to_df(output):
    """
    Convert the GPT-4o output (a string representing a dictionary) into a pandas DataFrame.
    Each column is prefixed with an underscore.
    """
    df = pd.DataFrame(eval(output)['pair'])
    df = df.rename(columns={'evaluation': 'evaluation_gpt', 'explanation': 'explanation_gpt'})
    return df

def gpt_4o(classification_prompt, start, end, prompts, num_qs, output_file_name, need_regenerate=None, runs=1):
    """
    Process a subset of prompts using the GPT-4o model and write results to an Excel file.
    Uses a while loop to iteratively reattempt processing for any missing sheets.
    
    Parameters:
        classification_prompt (str): The system prompt for GPT-4o evaluation.
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
    
    # Build the complete list of expected sheet names for the current batch.
    need_regenerate_list = [f'{t} loop {j} DOI' for t in range(runs) for j in range(len(prompts[start:end]))]
    file_path = output_file_name
    # If need_regenerate is provided and empty, nothing to do.
    if isinstance(need_regenerate, list) and len(need_regenerate) == 0:
        print("GPT-4o finished")
        return

    max_iterations = 3  # Maximum number of outer iterations for regeneration
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- GPT-4o regeneration iteration {iteration} ---")
        no_need = []  # Track sheets that have encountered irrecoverable errors
        # Process each run and prompt within the given range.
        for t in range(runs):
            print(f"\nProcessing run {t} loop")
            for i in range(start, end):
                # Skip prompts with zero expected Q&A pairs.
                if num_qs[i] == 0:
                    continue
                sheet_name = f'{t} loop {i} DOI'
                # Skip if this sheet already exists in the output file.
                if os.path.exists(file_path):
                    excel_file = pd.ExcelFile(output_file_name)
                    if sheet_name in excel_file.sheet_names:
                        continue
                # Skip if this sheet is not in the regeneration list.
                if sheet_name not in need_regenerate_list:
                    continue

                prompt = prompts[i]
                retries = 5
                print(f"Processing sheet: {sheet_name}")
                bad_request = False
                result_df = None

                # Inner loop: attempt the API call with retries.
                while retries > 0:
                    try:
                        completion = openai_client.beta.chat.completions.parse(
                            model=gpt_4o_model,
                            messages=[
                                {"role": "system", "content": classification_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            response_format=question_answer_pairs,
                        )
                    except (BadRequestError, LengthFinishReasonError) as e:
                        bad_request = True
                        print(f"{sheet_name}: BadRequestError: {e}")
                        break
                    # Process the response.
                    result = completion.choices[0].message.content
                    result_df = gpt_output_to_df(result)
                    if len(result_df) == num_qs[i]:
                        break
                    print(f"GPT 4o: Sheet {sheet_name}: Expected {num_qs[i]} pairs but got {len(result_df)}")
                    retries -= 1
                    
                    if retries <= 0:
                        print(f"{sheet_name}: Too many retries")
                        break

                if bad_request or result_df is None:
                    no_need.append(sheet_name)
                    continue

                print(f"Finished processing sheet: {sheet_name}")
                # Write the result DataFrame to the output Excel file.
                if os.path.exists(file_path):
                    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    with pd.ExcelWriter(file_path, mode='w', engine='openpyxl') as writer:
                        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
        # End of one outer iteration: update the regeneration list.
        if os.path.exists(file_path):
            excel_file = pd.ExcelFile(file_path)
            existing_sheets = excel_file.sheet_names
            new_need_regenerate = [sheet for sheet in need_regenerate_list if sheet not in existing_sheets]
        else:
            new_need_regenerate = need_regenerate_list.copy()

        # Remove sheets that had irrecoverable errors.
        for sheet in no_need:
            if sheet in new_need_regenerate:
                new_need_regenerate.remove(sheet)
        if len(new_need_regenerate) == 0:
            print("GPT-4o finished: All sheets processed successfully.")
            break
        else:
            print("Sheets still needing regeneration:", new_need_regenerate)
            need_regenerate = new_need_regenerate

    print("Exiting GPT-4o regeneration loop after maximum iterations.")

def gpt_o1_output_to_df(output):
    """
    Convert the GPT-o1 API response (a string representing a dictionary) 
    into a standardized pandas DataFrame with prefixed column names.
    """
    df = pd.DataFrame(eval(output))
    df.columns = [col + '_gpt_o1' for col in df.columns]
    df['evaluation_gpt_o1'] = df['evaluation_gpt_o1'].replace({
        r'True\s+Positive\b.*': 'TP',
        r'False\s+Positive\b.*': 'FP',
        r'False\s+Negative\b.*': 'FN',
        r'True\s+Negative\b.*': 'TN'
    }, regex=True)
    df = df.rename(columns={
        'evaluation': 'evaluation_gpt_o1',
        'explanation': 'explanation_gpt_o1',
        'revised_evaluation': 'revised_evaluation_gpt_o1',
        'reason': 'reason_gpt_o1'
    })
    return df

def gpt_o1(classification_prompt, start, end, prompts, num_qs, output_file_name, need_regenerate=None, runs=3):
    """
    Process a subset of prompts using the GPT-o1 model and write results to an Excel file.
    This function uses an outer while loop to reattempt processing of any missing sheets (i.e. regeneration).
    
    Parameters:
        classification_prompt (str): System prompt for GPT-o1 evaluation.
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
    openai_client = OpenAI(api_key=openai_api_key)

    # Instruction appended to each prompt.
    o1_list_output_prompt = (
        'Please format all answers to the following questions in a list, where each question answer pair in the list '
        'follows this structure: {"question": "string", "answer": "string", "question_type": "factual/reasoning/True or False", '
        '"evaluation": "TP/FP/FN/TN", "explanation": "string"}'
    )
    
    # Generate the full list of expected sheet names.
    expected_sheets = [f'{t} loop {j} DOI' for t in range(runs) for j in range(len(prompts[start:end]))]
    
    # Initialize regeneration list if not provided.
    if need_regenerate is None:
        need_regenerate = expected_sheets.copy()
    # If regeneration list is empty, nothing to do.
    if isinstance(need_regenerate, list) and len(need_regenerate) == 0:
        print("GPT-o1 finished")
        return
    
    max_iterations = 3  # Maximum outer iterations for regeneration.
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- GPT-o1 regeneration iteration {iteration} ---")
        no_need = []  # Track sheets that fail irrecoverably.
        for t in range(runs):
            print(f"\nProcessing run {t} loop")
            for i in range(start, end):
                # Skip if no Q&A pairs are expected.
                if num_qs[i] == 0:
                    continue
                sheet_name = f'{t} loop {i} DOI'
                # Skip if sheet already exists in the output file.
                if os.path.exists(output_file_name):
                    excel_file = pd.ExcelFile(output_file_name)
                    if sheet_name in excel_file.sheet_names:
                        continue
                # Process only if the sheet is in the regeneration list.
                if sheet_name not in need_regenerate:
                    continue
                prompt = prompts[i]
                retries = 5
                print(f"Processing sheet: {sheet_name}")
                bad_request = False
                output_df = None
                while retries > 0:
                    try:
                        input_prompt = classification_prompt + prompt + o1_list_output_prompt
                        completion = openai_client.chat.completions.create(
                            model=gpt_o1_model,
                            messages=[{"role": "user", "content": input_prompt}],
                        )
                        print("GPT-o1 response received for", sheet_name)
                        output = completion.choices[0].message.content
                        # Check that the output evaluates to a list.
                        evaluated_output = eval(output)
                        if type(evaluated_output) != list:
                            raise TypeError("Output is not a list")
                        else:
                            length = len(evaluated_output)
                        if length == num_qs[i]:
                            output_df = gpt_o1_output_to_df(output)
                            break  # Successful response.
                    except (BadRequestError, LengthFinishReasonError) as e:
                        bad_request = True
                        print(f"{sheet_name}: BadRequestError: {e}")
                        break
                    except Exception as e:
                        print(f"{sheet_name}: Error: {e}")
                    retries -= 1
                    if retries <= 0:
                        print(f"{sheet_name}: Too many retries.")
                        break
                if bad_request or output_df is None:
                    no_need.append(sheet_name)
                    continue
                print(f"Finished processing sheet: {sheet_name}")
                # Write the result DataFrame to the Excel file.
                if os.path.exists(output_file_name):
                    with pd.ExcelWriter(output_file_name, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                        output_df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    with pd.ExcelWriter(output_file_name, mode='w', engine='openpyxl') as writer:
                        output_df.to_excel(writer, sheet_name=sheet_name, index=False)
        # End of one outer iteration. Update regeneration list.
        if os.path.exists(output_file_name):
            excel_file = pd.ExcelFile(output_file_name)
            existing_sheets = excel_file.sheet_names
            new_need_regenerate = [sheet for sheet in expected_sheets if sheet not in existing_sheets]
        else:
            new_need_regenerate = expected_sheets.copy()
        # Remove sheets that encountered irrecoverable errors.
        for sheet in no_need:
            if sheet in new_need_regenerate:
                new_need_regenerate.remove(sheet)
        if len(new_need_regenerate) == 0:
            print("GPT-o1 finished: All sheets processed successfully.")
            break
        else:
            print("Sheets still needing regeneration:", new_need_regenerate)
            need_regenerate = new_need_regenerate

    print("Exiting GPT-o1 regeneration loop after maximum iterations.")

def summary_df(folder_name, file_name, num, runs=3):
    """
    Generate a summary by merging sheets from multiple Excel files.
    
    The function attempts to read sheets from "4o.xlsx", "gemini.xlsx", "claude.xlsx", and "o1.xlsx"
    located in folder_name. Each sheet is identified by a name in the format "{run} loop {index} DOI".
    The data from these sheets is merged using merged_df_v2 and processed to compute an overall 
    evaluation (via weighted_mode). The resulting summary for each sheet is written to a summary Excel 
    file (file_name). If some sheets are missing or fail to load, the function reattempts processing 
    them using a while loop until all expected sheets are processed or the maximum iterations is reached.
    
    Assumes that:
      - Functions 'merged_df_v2' and 'weighted_mode' are defined and available.
    
    Parameters:
        folder_name (str): Folder containing the Excel files.
        file_name (str): Name of the summary Excel file to be created.
        runs (int, optional): Number of outer runs/loops used for generating sheets.
    
    Returns:
        dict: A dictionary mapping sheet names to their merged summary DataFrames.
    """
    # Build the complete list of expected sheet names.
    expected_sheets = [f'{r} loop {i} DOI' for r in range(runs) for i in range(num)]
    dfs = {}
    summary_path = os.path.join(folder_name, file_name)

    for sheet_name in expected_sheets:
        # Skip if the summary file already contains this sheet.
        if os.path.exists(summary_path):
            excel_file = pd.ExcelFile(summary_path)
            if sheet_name in excel_file.sheet_names:
                continue
        # Parse the inner index from the sheet name ("{run} loop {index} DOI")
        try:
            parts = sheet_name.split(" loop ")
            inner_str = parts[1].split(" DOI")[0]
            inner = int(inner_str)
        except Exception as e:
            print(f"Error parsing sheet name {sheet_name}: {e}")
            continue
        try:
            # Read the corresponding sheet from each source Excel file.
            gpt_4o_df = pd.read_excel(os.path.join(folder_name, "4o.xlsx"), sheet_name=sheet_name)
            gemini_df = pd.read_excel(os.path.join(folder_name, "gemini.xlsx"), sheet_name=sheet_name)
            claude_df = pd.read_excel(os.path.join(folder_name, "claude.xlsx"), sheet_name=sheet_name)
            gpt_o1_df = pd.read_excel(os.path.join(folder_name, "o1.xlsx"), sheet_name=sheet_name)
            
            # Merge and process the dataframes.
            questions = gpt_4o_df['question']
            merged = merged_df_v2([gpt_4o_df, gemini_df, claude_df, gpt_o1_df], questions)
            output = merged[['question', 'answer0', 'question_type0', 'evaluation_gpt', 'explanation_gpt', 
                             'evaluation_gemini', 'explanation_gemini', 'evaluation_claude', 'explanation_claude', 
                             'evaluation_gpt_o1', 'explanation_gpt_o1']].copy()
            output['evaluation'] = weighted_mode(
                output, 
                ['evaluation_gpt', 'evaluation_gemini', 'evaluation_claude', 'evaluation_gpt_o1'],
                [0.23, 0.23, 0.23, 0.3]
            )
            output_df = output.rename(columns={'answer0': 'answer', 'question_type0': 'question_type'})
            dfs[sheet_name] = output_df
            
            # Write the merged DataFrame to the summary Excel file.
            if os.path.exists(summary_path):
                with pd.ExcelWriter(summary_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                    output_df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                with pd.ExcelWriter(summary_path, mode='w', engine='openpyxl') as writer:
                    output_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Processed sheet: {sheet_name}")
        except ValueError as e:
            print(f"Need to regenerate {sheet_name} (ValueError): {e}")
            continue
        except Exception as e:
            print(f"Error processing {sheet_name}: {e}")
            continue
    return dfs

def get_revised_classification_prompt(formatted_input, revision_prompt, folder_name):
    """
    Request a revised classification prompt from the Anthropic Claude API,
    save the returned prompt to a file, and return the revised prompt.

    Parameters:
        formatted_input (str): The input string containing classification mismatch details.
        revision_prompt (str): The system prompt guiding the revision.
        folder_name (str): The folder where the revised prompt file will be saved.

    Returns:
        str: The revised classification prompt.
    """
    claude_key, claude_model = claude_info()
    claude_client = anthropic.Anthropic(api_key=claude_key)

    response = claude_client.messages.create(
        model=claude_model,
        max_tokens=8192,
        system=revision_prompt,
        tools=[{
            "name": "revised_prompt_generator",
            "description": "Generate a revised classification prompt based on analysis of mismatches",
            "input_schema": {
                "type": "object",
                "properties": {
                    "revised_prompt": {
                        "type": "string",
                        "description": "The complete revised classification prompt"
                    }
                },
                "required": ["revised_prompt"]
            }
        }],
        tool_choice={"type": "tool", "name": "revised_prompt_generator"},
        messages=[{
            "role": "user",
            "content": f"Please analyze these classification mismatches and generate a revised prompt:\n\n{formatted_input}"
        }]
    )

    classification_prompt = response.content[0].input['revised_prompt']
    with open(os.path.join(folder_name, "revised classification prompt.txt"), "w") as file:
        file.write(classification_prompt)
    return classification_prompt
