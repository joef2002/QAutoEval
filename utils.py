import pandas as pd
import os
import json
import numpy as np
import docx
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import subprocess
from io import StringIO
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

def claude_info():
    """
    Replace 'INSERT ANTHROPIC API KEY' with your actual API key.
    """
    claude_key = 'INSERT ANTHROPIC API KEY'
    claude_model = 'claude-3-5-sonnet-20240620'
    return claude_key, claude_model 

def gemini_info():
    """
    Replace 'INSERT GEMINI API KEY' with your actual API key.
    """
    gemini_api_key = "INSERT GEMINI API KEY"
    gemini_model = 'gemini-1.5-pro-002'
    return gemini_api_key, gemini_model

def gpt_4o_info():
    """
    Replace 'INSERT OPENAI API KEY' with your actual API key.
    """
    openai_api_key = 'INSERT OPENAI API KEY'
    openai_4o_model = "gpt-4o-2024-08-06"
    return openai_api_key, openai_4o_model

def gpt_o1_info():
    """
    Replace 'INSERT OPENAI API API KEY' with your actual API key.
    """
    openai_api_key = 'INSERT OPENAI API KEY'
    openai_o1_model = 'o1-2024-12-17'
    return openai_api_key, openai_o1_model

def extract_text_from_pdf(pdf_path):
    """Extracts the text from a PDF file and returns it as a string.

    Parameters:
    pdf_path (str): The file path to the PDF file.

    Returns:
    str: The extracted text.
    """
    with open(pdf_path, 'rb') as fh:
        # Create a PDF resource manager object that stores shared resources
        rsrcmgr = PDFResourceManager()

        # Create a StringIO object to store the extracted text
        output = StringIO()

        # Create a TextConverter object to convert PDF pages to text
        device = TextConverter(rsrcmgr, output, laparams=LAParams())

        # Create a PDF page interpreter object
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        # Process each page contained in the PDF document
        pageCounter = 1 
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True): 
            try: 
                interpreter.process_page(page) 
            except: 
                with open("fails.txt", 'a', encoding='utf-8') as f:
                    f.write("page " + str(pageCounter) + " failed to process") 
            pageCounter += 1


        # Get the extracted text as a string and close the StringIO object
        text = output.getvalue()
        output.close()

        # Close the PDF file and text converter objects
        device.close()

    # Remove ^L page breaks from the text
    text = text.replace('\x0c', '\n')

    # Write the extracted text to the output file
    with open("full.txt", 'w', encoding='utf-8') as f:
        f.write(text)

#     print(f"Text extracted from file and written to full.txt")

    return text

def load_json_questions(file_path):
    try:
        with open(file_path, 'r') as file:
            output = json.load(file)
            possible_keys = ['qas', 'Q&A', 'QAs', 'questions', 'data', 'dataset']
            found_key = None 
            for key in possible_keys:
                if key in output:
                    found_key = key
                    break
            if found_key is None:
                return output
            else:
                return output[key]
                
    except json.JSONDecodeError as e:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Skip the first line
            lines = file.readlines()[1:]  # Read from the second line onwards
            json_data = ''.join(lines)    # Join the remaining lines into a single string
            if json_data[-1] != '}':
                json_data = json_data + '}'
            output = json.loads(json_data)  # Parse the JSON data
            possible_keys = ['qas', 'Q&A', 'QAs', 'questions', 'data', 'dataset']
            found_key = None 
            for key in possible_keys:
                if key in output:
                    found_key = key
                    break
            if found_key is None:
                return output
            else:
                return output[key]
    except Exception as e:
        print(f"An error occurred: {e}")

def process_docx(file_path):
    doc = docx.Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

def process_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return ET.tostring(root, encoding='utf8').decode('utf8')

def process_xhtml(file_path):
    with open(file_path, 'r') as file:
        soup = BeautifulSoup(file, 'lxml')
        return soup.get_text()

# Main processing function to handle different file types
def process_file(file_path):
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext.lower() == '.docx':  # Assuming .doc is handled like .docx
        return process_docx(file_path)
    elif ext.lower() == '.doc':
        return extract_text_from_doc(file_path)
    elif ext.lower() == '.xml':
        return process_xml(file_path)
    elif ext.lower() == '.xhtml':
        return process_xhtml(file_path)
    else:
        return None

def process_context(doi_dir):
    combined_text = ""

    # Process files in the DOI directory
    for root, dirs, files in os.walk(doi_dir):
        for file in files:
            if file.endswith(('.pdf', '.docx', '.doc', '.xml', '.xhtml')):
                file_path = os.path.join(root, file)
                text = process_file(file_path)
                if text is not None:
                    combined_text += text + " "

    return combined_text


def extract_text_from_doc(file_path):
    
    command = ['antiword', file_path]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        text = result.stdout.decode('utf-8')
        return text
    except subprocess.CalledProcessError as e:
        print(f"Error during file processing: {e}")
        return None

def extract_filepaths(root_folder):
    # List to store the full folder paths
    folder_paths = []
    
    # Walk through all directories in the root folder
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Split the path to count depth, ensure it's deeper than one level (i.e., root/folder/subfolder)
        relative_path = os.path.relpath(dirpath, root_folder)
        if os.sep in relative_path:  # Only include paths that are deeper than the root
            folder_paths.append(dirpath)
    
    return folder_paths

from collections import Counter
def weighted_mode(df, col_names, col_weights):
    result = []

    # Iterate over each row in the dataframe
    for _, row in df[col_names].iterrows():
        # Create a Counter object to count occurrences of each value
        value_counts = Counter(row)

        # Find the most common values
        most_common = value_counts.most_common()

        # Check if there's a tie in the mode
        max_count = most_common[0][1]
        tied_modes = [val for val, count in most_common if count == max_count]

        if len(tied_modes) == 1:
            # If no tie, append the mode directly
            result.append(tied_modes[0])
        else:
            # If tie, break the tie based on the weight
            weighted_choice = None
            max_weight = -1

            # Go through the tied modes and choose the one with the highest weight
            for value in tied_modes:
                weight = sum([col_weights[i] for i, col in enumerate(col_names) if row[col] == value])
                if weight > max_weight:
                    max_weight = weight
                    weighted_choice = value

            result.append(weighted_choice)

    # Return the result as a pandas Series for consistency
    return result

def merged_df_v2(df:list, questions:list):
    """outter merge of dfs in the list by col_lst"""
    length = len(df[0])
    if not all([len(i) == length for i in df]):
        raise ValueError("length doesn't match")
    merge_df = df[0]
    suffix_i = 0
    suffix_j = 1
    for i in df[1:]:
        merge_df = merge_df.merge(i, left_index=True, right_index=True, how='outer', suffixes = (str(suffix_i), str(suffix_j)))
        suffix_i+=2
        suffix_j+=2
    if len(merge_df) != length:
        raise ValueError("output length error")
    # merge_df = merge_df.rename(columns={'question0':'question'})
    merge_df['question'] = pd.Categorical(merge_df['question'], categories=questions, ordered=True)
    merge_df = merge_df.sort_values('question').reset_index(drop=True)
    return merge_df

def walk_and_create_dataframe(base_path):
    """
    Walk through the folder structure of base_path and create a dataframe with folder names (k folders) as DOI.

    Args:
        base_path (str): The root folder to start walking from.

    Returns:
        pd.DataFrame: A dataframe with a column "DOI" containing the k folder names.
    """
    doi_list = []
    file_path_list = []
    materials_list = []
    criterion_3_list = []
    criterion_2_list = []
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            k_folder_path = os.path.join(root, dir_name)
            excel_files = [file for file in os.listdir(k_folder_path) if file.endswith('.xlsx')]
            if excel_files:
                excel_path = os.path.join(k_folder_path, excel_files[0])
                try:
                    data = pd.read_excel(excel_path)
                    materials_json = transform_to_nested_json(data)
                    c_1, c_2 = criterion(data)
                    materials_list.append(materials_json)
                    criterion_3_list.append(c_1)
                    criterion_2_list.append(c_2)
                    relative_path = os.path.relpath(k_folder_path, base_path)
                    file_path_list.append(relative_path)
                    doi_list.append(dir_name)
                except Exception as e:
                    print(f"Error processing file {excel_path}: {e}")

    df = pd.DataFrame({
        'DOI': doi_list,
        'Paper File Path': file_path_list,
        'materials': materials_list,
        'Ground Truth Criterion 2': criterion_2_list,
        'Ground Truth Criterion 3': criterion_3_list,
    })
    return df

def transform_to_nested_json(data):
    result = []
    for _, row in data.iterrows():
        material = {}
        for col in data.columns:
            if col.lower() not in ["criterion 1", "criterion 2"]:
                material[col] = row[col]
        result.append(material)
    return json.dumps(result, indent=4)

def criterion(data):
    c_1 = 'Y' if sum(data['Criterion 1'] == 'Y') == len(data) else 'N'
    c_2 = 'Y' if sum(data['Criterion 2'] == 'N') == len(data) else 'N'
    return c_1, c_2

def generate_prompts_by_df(df, doi_ground_truth):
    prompts = list()
    num_qs = list()
    contexts = list()
    ground_truth = list()
    for idx, row in df.iterrows():
        paper = row['File_Path_paper']
        json = row['File_Path_json']
        doi = row['DOI']
        if not (os.path.exists(paper) and os.path.exists(json)):
            print(f"file path: {paper} or {json} does not exist")
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
        ground_truth.append(doi_ground_truth[doi])
    return prompts, num_qs, contexts, ground_truth

def calculate_cumulative_statistics(dataframes_dict):
    results = []

    # Loop through the dictionary and calculate metrics for each DataFrame
    for name, df in dataframes_dict.items():
        # Accuracy as a decimal
        accuracy = np.mean(df['evaluation'] == df['ground_truth'])

        # True Positive Catch Rate as a decimal
        true_positive_total = sum(df['ground_truth'] == 'TP')
        true_positive_catch = (sum((df['ground_truth'] == 'TP') & (df['evaluation'] == 'TP')) / true_positive_total) if true_positive_total != 0 else -1

        # Calculate overall non-TP catch rate using the corrected method as a decimal
        total_catch_rate = calculate_non_tp_catch_rate(df)

        # Append results for this DataFrame
#         if total_catch_rate != -1 and true_positive_catch != -1:
        results.append((accuracy, true_positive_catch, total_catch_rate))

    # Calculate cumulative averages in decimal format
    cumulative_avg_accuracy = np.mean([res[0] for res in results])
    cumulative_avg_tp_rate = np.mean([res[1] for res in results if res[1] != -1])
    cumulative_avg_non_tp_catch_rate = np.mean([res[2] for res in results if res[2] != -1])

    # Create a final dictionary with cumulative statistics
    cumulative_statistics = {
        'Cumulative Avg Accuracy': cumulative_avg_accuracy,
        'Cumulative Avg TP Rate': cumulative_avg_tp_rate,
        'Cumulative Avg Non-TP Catching Rate': cumulative_avg_non_tp_catch_rate
    }

    return cumulative_statistics

def calculate_non_tp_catch_rate(results_df):
    # Calculate the total number of non-TP instances
    non_tp_instances = results_df[(results_df['ground_truth'] == 'FP') | 
                                  (results_df['ground_truth'] == 'TN') | 
                                  (results_df['ground_truth'] == 'FN')]
    total_non_tp_instances = len(non_tp_instances)

    # Correctly identified non-TP instances
    correctly_caught_non_tp = sum(
        ((results_df['ground_truth'] == 'FP') & (results_df['evaluation'] == 'FP')) |
        ((results_df['ground_truth'] == 'TN') & (results_df['evaluation'] == 'TN')) |
        ((results_df['ground_truth'] == 'FN') & (results_df['evaluation'] == 'FN'))
    )
    
    # Overall non-TP catch rate as a decimal
    total_catch = correctly_caught_non_tp / total_non_tp_instances if total_non_tp_instances != 0 else -1
    return total_catch
