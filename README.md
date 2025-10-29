<p align="center">
<img src="./docs/images/logo.png" width="50%" height="150%">
</p>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17479256.svg)](https://doi.org/10.5281/zenodo.17479256)

## Overview

This project provides an evaluation framework for automated classification of question–answer pairs and analysis of synthesis conditions extracted from scientific literature. It leverages multiple large language model (LLM) APIs—including Anthropic Claude, OpenAI (GPT-4o and GPT-o1), and Google Generative AI—to assess whether Q&A pairs are correctly generated and whether synthesis conditions meet defined criteria.

The framework supports both multi-hop and single-hop Q&A evaluation pipelines, as well as prompt optimization for improved classification accuracy.

## Features

- **Multi-Hop Q&A Evaluation:**  
  Processes research papers and associated multi-hop Q&A datasets by extracting context, generating prompts, and using concurrent API calls for classification and evaluation.

- **Single-Hop Q&A Evaluation:**  
  Processes research papers and associated single-hop Q&A datasets by extracting context, generating prompts, and using concurrent API calls for classification and evaluation.

- **Prompt Optimization:**  
  Iteratively refines single-hop Q&A evaluation classification prompts based on ground truth comparisons to improve evaluation performance.

- **Synthesis Conditions Evaluation:**  
  Extracts synthesis conditions from scientific papers and assesses them against criteria for completeness, exclusivity, and accuracy.

  For a more detailed description of the evaluation criteria used and how the dataset is generated(along with the code), please refer to our previous work: [RetChemQA](https://github.com/nakulrampal/RetChemQA) and [Comparison-of-LLMs](https://github.com/Sya-bitsea/Comparison-of-LLMs)

## Quick Start

If you're on **macOS or Linux**, a convenience launcher script—`Launcher(Mac or Linux).command`—is provided at the repository root. Double click on it, and it will create a virtual environment, install dependencies, and start the app, all in one go. 

> **Note:** The first time you run the script, installation may take a while because all dependencies must be downloaded.

Under the hood, the script:

1. Creates an isolated Python environment in `.venv` (if it doesn't already exist).
2. Upgrades `pip` and installs packages listed in `requirements.txt`.
3. Launches the QAutoEval application (`QAutoEval_app.py`), forwarding any additional command‑line arguments you supply.

**Explore with example data:** Inside `docs/example_data/` you'll find three folders—`single_hop`, `multi_hop`, and `synthesis_condition`. Each folder contains the manuscript (`MS`), supplementary information (`SI`), and a dataset for one representative DOI, so you can exercise every pipeline right after installation.

**Windows users** can follow the manual installation in next section. (We are working on App version for Windows!)

## Manual Installation

### Prerequisites

- **Python:** Version 3.8 or higher (Recommended 3.10.14, please avoid using Python 3.12 to prevent dependency errors) 
- **Virtual Environment:** Recommended (e.g., `venv` or `conda`)

### Setup

   ```bash
   python -m venv .venv               # Create virtual environment (optional but recommended)
   source .venv/bin/activate          # On Windows use `.venv\Scripts\activate`
   pip install -r requirements.txt    # Install dependencies
   ```

### Configuration
Before running the pipelines, configure the following:

**API Keys and Models:**  
Insert API keys for Anthropic Claude, OpenAI, and Google Generative AI on the top of the [utils.py](utils.py), please note that OpenAI API key need to be inserted twice, for both GPT 4o and GPT o1. 

Please take a note that model name changes and sometime it can cause an error.

**Data Directories:**
The code executes sequentially according to the order specified in the Excel sheet. It requires four directory named 'single-hop-data', 'multi-hop-data', 'prompt-optimization-data', and 'synthesis-condition-data', which have been omitted due to copyright restrictions held by the original publishers. Within the those four 'data' directory, subdirectories are named after each publisher. Each publisher directory contains folders named according to the DOI, and these DOI folders hold the corresponding manuscript, supplementary materials, and other related data. QAutoEval agent was used to evaluate all the data in the folder 'single-hop-dataset', 'multi-hop-dataset' and 'synthesis-conditions-analysis' to produce Figure 5 in the paper. For additional details (such as publisher information and evaluation distribution), refer to the file 'single-hop-DOIs.xlsx' and 'QA_claude_20241017.xlsx', which inlcude the human evaluation for each question and answer pair of each DOI.

*********

## Contributing
Contributions are welcome! If you’d like to contribute:

Fork the repository.
Create a new branch for your feature or bug fix.
Ensure your code adheres to the project’s style guidelines.
Submit a pull request with a detailed description of your changes.

## Issues
If there is any issues with the code, please report to [Issues](https://github.com/joef2002/QAutoEval/issues)

## License
This project is licensed under the MIT License.

## Acknowledgements
Thanks to the API providers (Anthropic, OpenAI, Google) for their services.
Special thanks to the developers of the libraries used in this project.

We acknowledge the financial support from the following sources:
1. Bakar Institute of Digital Materials for the Planet (BIDMaP)
