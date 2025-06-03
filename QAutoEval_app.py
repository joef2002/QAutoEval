import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
import threading
import pandas as pd
import concurrent.futures
import sys
from PIL import Image, ImageTk

# Import the necessary functions from the existing modules
from utils import weighted_mode
from QAeval_api_call import (question_answer_pairs, claude_output_to_df, gemini_output_to_df, gpt_output_to_df)
import anthropic
import google.generativeai as genai
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal

class QAEvaluationApp:
    def __init__(self, root):
        self.root = root

        # Set window icon
        try:
            # Load the icon image
            icon_image = Image.open('docs/images/icon.png')
            
            # Define target icon size (common sizes: 16, 32, 48, 64)
            icon_size = 48
            
            # Calculate the scaling to fit within the square while maintaining aspect ratio
            original_width, original_height = icon_image.size
            scale = min(icon_size / original_width, icon_size / original_height)
            
            # Calculate new dimensions
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Resize the image maintaining aspect ratio
            resized_image = icon_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create a square canvas with transparent background
            square_image = Image.new('RGBA', (icon_size, icon_size), (0, 0, 0, 0))
            
            # Calculate position to center the resized image
            paste_x = (icon_size - new_width) // 2
            paste_y = (icon_size - new_height) // 2
            
            # Paste the resized image onto the square canvas
            square_image.paste(resized_image, (paste_x, paste_y), resized_image if resized_image.mode == 'RGBA' else None)
            
            # Convert to PhotoImage for tkinter
            self.icon_photo = ImageTk.PhotoImage(square_image)
            # Set as window icon
            self.root.iconphoto(True, self.icon_photo)
        except Exception as e:
            print(f"Could not load window icon: {e}")
            # Optionally, you can try to use a .ico file as fallback
            try:
                self.root.iconbitmap('docs/images/icon.ico')  # if you have an .ico version
            except:
                pass  # Use default icon if both fail

        if sys.platform.startswith("win"):
            # Windows native "maximised"
            self.root.state("zoomed")
        elif sys.platform.startswith("linux"):
            # Most modern X11/Wayland builds recognise -zoomed
            self.root.attributes("-zoomed", True)
        else:  # macOS fallback – emulate maximise by sizing to the screen
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            self.root.geometry(f"{w}x{h}+0+0")

        self.root.title("QAutoEval - Multi-Mode Evaluation System")
        self.root.geometry("1000x800")
        
        # Initialize variables
        self.manuscript_path = tk.StringVar()
        self.supplement_path = tk.StringVar()
        self.dataset_path = tk.StringVar()  # Changed from qa_dataset_path to be more generic
        
        # API Keys
        self.claude_key = tk.StringVar()
        self.gemini_key = tk.StringVar()
        self.openai_key = tk.StringVar()
        self.openai_o1_key = tk.StringVar()
        
        # Main evaluation type
        self.evaluation_type = tk.StringVar(value="qa")  # "qa" or "synthesis"
        
        # Q&A specific variables
        self.evaluation_mode = tk.StringVar(value="single-hop")
        
        # Results storage
        self.evaluation_results = None
        
        # UI state
        self.results_visible = False
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights so the UI scales dynamically
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Store reference to main frame for dynamic grid management
        self.main_frame = main_frame
        
        # Title and evaluation type selection
        self.setup_header(main_frame)
        
        # Mode selection (only for Q&A) - row 1
        self.setup_mode_selection(main_frame)
        
        # File uploads - rows 2-4
        self.setup_file_uploads(main_frame)

        # Control buttons - row 5
        self.button_frame = ttk.Frame(main_frame)
        self.button_frame.grid(row=5, column=0, columnspan=3, pady=20)
        ttk.Button(self.button_frame, text="Start Evaluation", command=self.start_evaluation, width=15).grid(
            row=0, column=0, padx=(0, 10)
        )
        ttk.Button(self.button_frame, text="Start New Run", command=self.start_new_run, width=15).grid(row=0, column=1)

        # Progress and status - rows 6-7
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        self.status_label = ttk.Label(main_frame, text="Ready to evaluate")
        self.status_label.grid(row=7, column=0, columnspan=3, pady=5)

        # Results frame - row 8 (this will be the expanding row)
        self.results_frame = ttk.LabelFrame(main_frame, text="Evaluation Results", padding="10")
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(2, weight=1)
        
        # Initialize UI state and set up dynamic grid weights
        self.update_ui_for_evaluation_type()
    
    def setup_header(self, parent):
        # Header frame
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        header_frame.columnconfigure(1, weight=1)  # Make middle column expandable
        
        # Logo frame (left side)
        logo_frame = ttk.Frame(header_frame)
        logo_frame.grid(row=0, column=0, padx=(0, 20))
        
        # Load and display logo
        try:
            # Calculate size to match button dimensions (width=20 chars ≈ 160px, height=2 lines ≈ 50px)
            container_width = 160
            container_height = 50

            # Use PIL for better image handling
            logo_image = Image.open('docs/images/logo.png')
            
            # Resize image while maintaining aspect ratio
            logo_image.thumbnail((container_width, container_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.logo_photo = ImageTk.PhotoImage(logo_image)
            
            # Create logo label with the same dimensions as the Q&A button
            logo_label = tk.Label(
                logo_frame, 
                image=self.logo_photo,
                width=container_width,
                height=container_height
            )
            logo_label.pack()
                
        except Exception as e:
            # Fallback if image can't be loaded - show placeholder
            print(f"Could not load logo: {e}")
            placeholder_label = tk.Label(
                logo_frame,
                text="QAutoEval\nLogo",
                width=20,  # Same as Q&A button
                height=2,  # Same as Q&A button
                font=("Arial", 10, "bold"),
                relief="raised",
                borderwidth=1,
                bg="lightgray",
                fg="darkblue"
            )
            placeholder_label.pack()
        
        # Evaluation type buttons (center)
        eval_buttons_frame = ttk.Frame(header_frame)
        eval_buttons_frame.grid(row=0, column=1)
        
        # Q&A Evaluation button
        self.qa_button = tk.Button(eval_buttons_frame, text="Q&A Pairs Evaluation", 
                                command=lambda: self.set_evaluation_type("qa"),
                                font=("Arial", 12, "bold"), width=20, height=2)
        self.qa_button.grid(row=0, column=0, padx=(0, 5))
        
        # Synthesis Condition Evaluation button
        self.synthesis_button = tk.Button(eval_buttons_frame, text="Synthesis Condition Evaluation", 
                                        command=lambda: self.set_evaluation_type("synthesis"),
                                        font=("Arial", 12, "bold"), width=25, height=2)
        self.synthesis_button.grid(row=0, column=1, padx=(5, 0))
        
        # Settings button (top-right)
        settings_btn = ttk.Button(header_frame, text="Settings", command=self.open_settings, width=10)
        settings_btn.grid(row=0, column=2, sticky=tk.E)
        
        # Update button colors
        self.update_button_colors()
    
    def set_evaluation_type(self, eval_type):
        """Set the evaluation type and update UI accordingly"""
        self.evaluation_type.set(eval_type)
        self.update_button_colors()
        self.update_ui_for_evaluation_type()
        self.hide_results()  # Hide any existing results
        self.update_status("Ready to evaluate")
        self.update_progress(0)
    
    def update_button_colors(self):
        """Update button selection status using highlight borders only"""
        try:
            if self.evaluation_type.get() == "qa":
                # Q&A button - selected state (highlighted border)
                self.qa_button.config(
                    bg="SystemButtonFace",  # Use system default
                    fg="SystemButtonText",  # Use system default
                    relief="raised", 
                    highlightbackground="#1f5582",  # Blue highlight for selection
                    highlightcolor="#1f5582",
                    highlightthickness=3,
                    borderwidth=3,
                    state='normal'
                )
                # Synthesis button - unselected state (normal border)
                self.synthesis_button.config(
                    bg="SystemButtonFace",  # Use system default
                    fg="SystemButtonText",  # Use system default
                    relief="raised", 
                    highlightbackground="SystemButtonFace",  # Normal highlight
                    highlightcolor="SystemButtonFace",
                    highlightthickness=1,
                    borderwidth=1,
                    state='normal'
                )
            else:
                # Q&A button - unselected state (normal border)
                self.qa_button.config(
                    bg="SystemButtonFace",  # Use system default
                    fg="SystemButtonText",  # Use system default
                    relief="raised", 
                    highlightbackground="SystemButtonFace",  # Normal highlight
                    highlightcolor="SystemButtonFace",
                    highlightthickness=1,
                    borderwidth=1,
                    state='normal'
                )
                # Synthesis button - selected state (highlighted border)
                self.synthesis_button.config(
                    bg="SystemButtonFace",  # Use system default
                    fg="SystemButtonText",  # Use system default
                    relief="raised", 
                    highlightbackground="#1f5582",  # Blue highlight for selection
                    highlightcolor="#1f5582",
                    highlightthickness=3,
                    borderwidth=3,
                    state='normal'
                )
            
            # Force update the display
            self.qa_button.update_idletasks()
            self.synthesis_button.update_idletasks()
            
        except Exception as e:
            print(f"Button highlight update failed: {e}")
            # Fallback: use relief and borderwidth only
            try:
                if self.evaluation_type.get() == "qa":
                    self.qa_button.config(relief="sunken", borderwidth=3)
                    self.synthesis_button.config(relief="raised", borderwidth=1)
                else:
                    self.qa_button.config(relief="raised", borderwidth=1)
                    self.synthesis_button.config(relief="sunken", borderwidth=3)
            except:
                pass
    
    def update_ui_for_evaluation_type(self):
        """Update UI elements based on the selected evaluation type and manage grid layout dynamically"""
        if self.evaluation_type.get() == "qa":
            # Show mode selection for Q&A
            if hasattr(self, 'mode_frame'):
                self.mode_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
            # Update dataset label
            if hasattr(self, 'dataset_label'):
                self.dataset_label.config(text="Q&A Dataset (JSON):")
            # Adjust grid layout - file uploads start at row 2
            self._update_file_upload_positions(start_row=2)
            # Set results frame row to 8 (after mode selection)
            results_row = 8
        else:
            # Hide mode selection for synthesis
            if hasattr(self, 'mode_frame'):
                self.mode_frame.grid_remove()
            # Update dataset label
            if hasattr(self, 'dataset_label'):
                self.dataset_label.config(text="Synthesis Condition Dataset (JSON):")
            # Adjust grid layout - file uploads start at row 1 (no mode selection)
            self._update_file_upload_positions(start_row=1)
            # Set results frame row to 7 (no mode selection)
            results_row = 7
        
        # Configure the expanding row dynamically
        if hasattr(self, 'main_frame'):
            # Clear all row weights first
            for i in range(20):  # Clear up to row 20
                self.main_frame.rowconfigure(i, weight=0)
            # Set the results row to expand
            self.main_frame.rowconfigure(results_row, weight=1)
        
        # Store the results row for later use
        self.results_row = results_row
    
    def show_info_popup(self):
        """Custom pop-up without the default icon, with wrapping so full text shows."""
        popup = tk.Toplevel(self.root)
        popup.title("Need More Detail?")
        popup.transient(self.root)
        popup.resizable(False, False)

        popup.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 250,
            self.root.winfo_rooty() + 150
        ))

        ttk.Label(
            popup,
            text=(
                "For a more detailed summary sheet with each LLM's individual "
                "evaluation and explanations, please export the results to an "
                "Excel file."
            ),
            wraplength=340,
            justify="left",
            padding=20
        ).pack()

        ttk.Button(popup, text="OK", command=popup.destroy).pack(pady=(0, 12))
    
    def setup_mode_selection(self, parent):
        # Mode selection frame (only for Q&A)
        self.mode_frame = ttk.LabelFrame(parent, text="Evaluation Mode", padding="10")
        self.mode_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Radio buttons for mode selection
        single_hop_radio = ttk.Radiobutton(self.mode_frame, text="Single-hop Q&A Evaluation", 
                                          variable=self.evaluation_mode, value="single-hop",
                                          command=self.on_mode_change)
        single_hop_radio.grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        multi_hop_radio = ttk.Radiobutton(self.mode_frame, text="Multi-hop Q&A Evaluation", 
                                         variable=self.evaluation_mode, value="multi-hop",
                                         command=self.on_mode_change)
        multi_hop_radio.grid(row=0, column=1, sticky=tk.W)
        
        # Mode description
        self.mode_description = ttk.Label(self.mode_frame, text="", foreground="blue", font=("Arial", 9))
        self.mode_description.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
        
        # Initialize description
        self.on_mode_change()
    
    def on_mode_change(self):
        """Update the mode description when selection changes"""
        if self.evaluation_mode.get() == "single-hop":
            description = "Single-hop mode: Evaluates Q&A pairs that require direct information from the context without reasoning chains."
        else:
            description = "Multi-hop mode: Evaluates Q&A pairs that require reasoning across multiple pieces of information or inference steps."
        
        self.mode_description.config(text=description)
    
    def _update_file_upload_positions(self, start_row):
        """Update positions of file upload elements dynamically"""
        if hasattr(self, 'manuscript_label'):
            self.manuscript_label.grid(row=start_row, column=0, sticky=tk.W, pady=5)
        if hasattr(self, 'manuscript_entry'):
            self.manuscript_entry.grid(row=start_row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 5))
        if hasattr(self, 'ms_btn_frame'):
            self.ms_btn_frame.grid(row=start_row, column=2, pady=5)
            
        if hasattr(self, 'supplement_label'):
            self.supplement_label.grid(row=start_row+1, column=0, sticky=tk.W, pady=5)
        if hasattr(self, 'supplement_entry'):
            self.supplement_entry.grid(row=start_row+1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 5))
        if hasattr(self, 'supp_btn_frame'):
            self.supp_btn_frame.grid(row=start_row+1, column=2, pady=5)
            
        if hasattr(self, 'dataset_label'):
            self.dataset_label.grid(row=start_row+2, column=0, sticky=tk.W, pady=5)
        if hasattr(self, 'dataset_entry'):
            self.dataset_entry.grid(row=start_row+2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 5))
        if hasattr(self, 'dataset_btn_frame'):
            self.dataset_btn_frame.grid(row=start_row+2, column=2, pady=5)
            
        # Update button frame position
        if hasattr(self, 'button_frame'):
            self.button_frame.grid(row=start_row+3, column=0, columnspan=3, pady=20)
            
        # Update progress bar position  
        if hasattr(self, 'progress_bar'):
            self.progress_bar.grid(row=start_row+4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
            
        # Update status label position
        if hasattr(self, 'status_label'):
            self.status_label.grid(row=start_row+5, column=0, columnspan=3, pady=5)
    
    def setup_file_uploads(self, parent):
        # Manuscript upload
        self.manuscript_label = ttk.Label(parent, text="MS:")
        self.manuscript_entry = ttk.Entry(parent, textvariable=self.manuscript_path, width=40)
        
        # Manuscript buttons frame
        self.ms_btn_frame = ttk.Frame(parent)
        ttk.Button(self.ms_btn_frame, text="Browse", command=lambda: self.browse_file(self.manuscript_path), width=10).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(self.ms_btn_frame, text="Clear", command=lambda: self.manuscript_path.set(""), width=8).grid(row=0, column=1)
        
        # Supplement upload
        self.supplement_label = ttk.Label(parent, text="SI:")
        self.supplement_entry = ttk.Entry(parent, textvariable=self.supplement_path, width=40)
        
        # Supplement buttons frame
        self.supp_btn_frame = ttk.Frame(parent)
        ttk.Button(self.supp_btn_frame, text="Browse", command=lambda: self.browse_file(self.supplement_path), width=10).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(self.supp_btn_frame, text="Clear", command=lambda: self.supplement_path.set(""), width=8).grid(row=0, column=1)
        
        # Dataset upload (generic label that changes based on evaluation type)
        self.dataset_label = ttk.Label(parent, text="Q&A Dataset (JSON):")
        self.dataset_entry = ttk.Entry(parent, textvariable=self.dataset_path, width=40)
        
        # Dataset buttons frame
        self.dataset_btn_frame = ttk.Frame(parent)
        ttk.Button(self.dataset_btn_frame, text="Browse", command=lambda: self.browse_json_file(self.dataset_path), width=10).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(self.dataset_btn_frame, text="Clear", command=lambda: self.dataset_path.set(""), width=8).grid(row=0, column=1)
        
        # Initial positioning (will be updated by update_ui_for_evaluation_type)
        self._update_file_upload_positions(start_row=2)
    
    def get_classification_prompt(self):
        """Return the appropriate classification prompt based on the evaluation type and mode"""
        if self.evaluation_type.get() == "qa":
            if self.evaluation_mode.get() == "single-hop":
                return self.get_single_hop_prompt()
            else:
                return self.get_multi_hop_prompt()
        else:
            return self.get_synthesis_prompt()
    
    def get_synthesis_prompt(self):
        """Synthesis condition evaluation prompt"""
        return """
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
    
    def get_single_hop_prompt(self):
        """Single-hop Q&A evaluation prompt"""
        return """
    You have been provided with a context from which question-answer pairs have been generated. Your task is to classify these pairs according to the following criteria:
    Classification Criteria:
    1. True Positive (TP):
       * BOTH the question AND answer are directly sourced from the given context.
       * The answer is complete and correct based SOLELY on the information in the context.
       * No additional knowledge or inference beyond what is explicitly stated in the context is required.
    2. False Positive (FP):
       * BOTH the question AND answer appear to be sourced from the given context.
       * However, the answer is incorrect or incomplete.
       * No additional knowledge or inference beyond what is explicitly stated in the context is required.
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
       * Partial matches should be classified as FP.
       * Answers requiring additional inference should be classified as TN or FN.
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
        * When evaluating question-answer pairs, adopt a very strict interpretation of what constitutes "directly sourced" information. If there's any doubt, lean towards classifying as TN or FN rather than TP or FP.
    17. Context Verification:
        * For each classification, explicitly reference the relevant part of the context that supports your decision. This ensures a thorough check against the provided information.
    18. Mathematical Operations:
        * For questions requiring simple mathematical operations (e.g., averaging, ratios), classify as TP only if ALL required values are explicitly stated in the context AND the operation is trivial.
        * For more complex calculations or those requiring multiple steps, classify as TN or FP unless the context explicitly provides the calculated result.
    19. Partial Information:
        * If a question-answer pair contains some information from the context but also includes additional unsupported claims or details, classify as TN rather than TP.
    20. Time Sensitivity:
        * Be aware of the potential for time-sensitive information. If a question-answer pair relies on information that may change over time (e.g., "current" record holders, latest discoveries), ensure the context explicitly supports the claim for the relevant time period.
    21. Structural Inferences:
        * For questions about molecular or crystal structures, ensure that all structural details are explicitly stated in the context. Do not rely on chemical knowledge to infer structural information not directly provided.
    22. Charge Balance:
        * When dealing with questions about ionic compounds or charge states, ensure that the context explicitly states the charge information. Do not make assumptions about charge balance based on chemical knowledge.
    23. Coordination Numbers:
        * For questions about metal coordination, ensure that the context explicitly states the coordination number and geometry. Do not infer coordination information based on general chemistry principles.
    24. Strict Adherence to Context:
        * Even if an answer seems logically correct or chemically sound, if it relies on any information or reasoning not explicitly provided in the context, classify it as TN or FN.
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
    
    def get_multi_hop_prompt(self):
        """Multi-hop Q&A evaluation prompt"""
        return """
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
    
    def browse_file(self, var):
        filetypes = [
            ("All supported", "*.pdf;*.docx;*.doc;*.xml;*.xhtml"),
            ("PDF files", "*.pdf"),
            ("Word files", "*.docx;*.doc"),
            ("XML files", "*.xml;*.xhtml"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            var.set(filename)
    
    def browse_json_file(self, var):
        filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            var.set(filename)
    
    def open_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("API Settings")
        settings_window.geometry("700x400")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Center the window
        settings_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 200, self.root.winfo_rooty() + 100))
        
        frame = ttk.Frame(settings_window, padding="20")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        settings_window.columnconfigure(0, weight=1)
        settings_window.rowconfigure(0, weight=1)
        
        ttk.Label(frame, text="API Keys Configuration", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Claude API Key
        ttk.Label(frame, text="Claude API Key:").grid(row=2, column=0, sticky=tk.W, pady=5)
        claude_entry = ttk.Entry(frame, textvariable=self.claude_key, show="*", width=50)
        claude_entry.grid(row=2, column=1, pady=5, padx=(10, 0))
        
        # Gemini API Key
        ttk.Label(frame, text="Gemini API Key:").grid(row=3, column=0, sticky=tk.W, pady=5)
        gemini_entry = ttk.Entry(frame, textvariable=self.gemini_key, show="*", width=50)
        gemini_entry.grid(row=3, column=1, pady=5, padx=(10, 0))
        
        # OpenAI API Key (for GPT-4o)
        ttk.Label(frame, text="OpenAI API Key (GPT-4o):").grid(row=4, column=0, sticky=tk.W, pady=5)
        openai_entry = ttk.Entry(frame, textvariable=self.openai_key, show="*", width=50)
        openai_entry.grid(row=4, column=1, pady=5, padx=(10, 0))
        
        # OpenAI O1 API Key (can use same key)
        ttk.Label(frame, text="OpenAI API Key (GPT-o1):").grid(row=5, column=0, sticky=tk.W, pady=5)
        openai_o1_entry = ttk.Entry(frame, textvariable=self.openai_o1_key, show="*", width=50)
        openai_o1_entry.grid(row=5, column=1, pady=5, padx=(10, 0))
        
        # Save button
        save_btn = ttk.Button(frame, text="Save", command=settings_window.destroy, width=12)
        save_btn.grid(row=7, column=0, columnspan=2, pady=20)
    
    def start_new_run(self):
        """Clear all file inputs but keep API keys and evaluation settings for both Q&A and synthesis evaluations"""
        self.manuscript_path.set("")
        self.supplement_path.set("")
        self.dataset_path.set("")
        self.evaluation_results = None
        self.hide_results()
        
        eval_type_text = "Q&A" if self.evaluation_type.get() == "qa" else "synthesis condition"
        self.update_status(f"Ready for new {eval_type_text} evaluation")
        self.update_progress(0)
    
    def load_dataset(self, file_path):
        """Load dataset based on evaluation type"""
        if self.evaluation_type.get() == "qa":
            return self.load_qa_dataset(file_path)
        else:
            return self.load_synthesis_dataset(file_path)
    
    def load_qa_dataset(self, file_path):
        """Robust JSON loading for Q&A datasets"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return self.extract_qa_pairs(data)
        except json.JSONDecodeError:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                start_idx = content.find('{')
                if start_idx > 0:
                    content = content[start_idx:]
                
                end_idx = content.rfind('}')
                if end_idx != -1:
                    content = content[:end_idx + 1]
                
                data = json.loads(content)
                return self.extract_qa_pairs(data)
            except Exception as e:
                raise ValueError(f"Could not parse Q&A JSON file: {str(e)}")
    
    def load_synthesis_dataset(self, file_path):
        """Load synthesis condition dataset"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # For synthesis, we expect an array of materials with synthesis conditions
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Look for common keys that might contain the synthesis data
                possible_keys = ['materials', 'synthesis_conditions', 'data', 'conditions']
                for key in possible_keys:
                    if key in data:
                        return data[key]
                # If no standard key found, return the data as-is
                return data
            else:
                raise ValueError("Synthesis dataset should be a JSON array or object")
                
        except Exception as e:
            raise ValueError(f"Could not parse synthesis condition JSON file: {str(e)}")
    
    def extract_qa_pairs(self, data):
        """Extract Q&A pairs from various JSON structures"""
        if isinstance(data, list):
            return data
        
        possible_keys = ['questions', 'qas', 'Q&A', 'QAs', 'data', 'dataset', 'pairs']
        
        for key in possible_keys:
            if key in data:
                return data[key]
        
        return data
    
    def process_file_content(self, file_path):
        """Process content from a single file (PDF, DOCX, DOC, XML, XHTML)"""
        try:
            if not os.path.exists(file_path):
                return ""
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        return text
                except ImportError:
                    print("PyPDF2 not installed. Please install it to process PDF files.")
                    return ""
                except Exception as e:
                    print(f"Error reading PDF {file_path}: {e}")
                    return ""
            
            elif file_ext in ['.docx', '.doc']:
                try:
                    import docx
                    doc = docx.Document(file_path)
                    text = ""
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                    return text
                except ImportError:
                    print("python-docx not installed. Please install it to process Word files.")
                    return ""
                except Exception as e:
                    print(f"Error reading Word document {file_path}: {e}")
                    return ""
            
            elif file_ext in ['.xml', '.xhtml']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    
                    import re
                    text = re.sub(r'<[^>]+>', '', content)
                    text = re.sub(r'\s+', ' ', text).strip()
                    return text
                except Exception as e:
                    print(f"Error reading XML file {file_path}: {e}")
                    return ""
            
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        return file.read()
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    return ""
        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return ""
    
    def validate_inputs(self):
        """Enhanced validation requiring all API keys and providing guidance for synthesis evaluation"""
        try:
            # Check if manuscript file is provided and exists
            if not self.manuscript_path.get():
                raise ValueError("Please select a manuscript file.")
            
            if not os.path.exists(self.manuscript_path.get()):
                raise ValueError("Manuscript file does not exist.")
            
            # Check if dataset is provided and exists
            if not self.dataset_path.get():
                dataset_type = "Q&A dataset" if self.evaluation_type.get() == "qa" else "synthesis condition dataset"
                raise ValueError(f"Please select a {dataset_type} file.")
            
            if not os.path.exists(self.dataset_path.get()):
                dataset_type = "Q&A dataset" if self.evaluation_type.get() == "qa" else "synthesis condition dataset"
                raise ValueError(f"{dataset_type.title()} file does not exist.")

            # For synthesis evaluation, recommend supplement file if not provided
            if self.evaluation_type.get() == "synthesis" and not self.supplement_path.get():
                result = messagebox.askyesno(
                    "Supplement Information Recommended", 
                    "For synthesis condition evaluation, it's highly recommended to include "
                    "the Supplement Information (SI) file as it often contains detailed "
                    "synthesis procedures.\n\n"
                    "Do you want to continue without the SI file?"
                )
                if not result:
                    return False

            # Check that ALL API keys are provided
            missing_keys = []
            
            if not self.claude_key.get().strip():
                missing_keys.append("Claude API Key")
            
            if not self.gemini_key.get().strip():
                missing_keys.append("Gemini API Key")
            
            if not self.openai_key.get().strip():
                missing_keys.append("OpenAI API Key (GPT-4o)")
            
            if not self.openai_o1_key.get().strip():
                missing_keys.append("OpenAI API Key (GPT-o1)")
            
            if missing_keys:
                missing_keys_text = "\n• ".join(missing_keys)
                raise ValueError(f"Please Insert API key(s) in the setting. All API keys are required for evaluation. Missing:\n\n• {missing_keys_text}\n\n")

            # Test if the dataset can be loaded
            dataset = self.load_dataset(self.dataset_path.get())

            if not dataset or len(dataset) == 0:
                dataset_type = "Q&A pairs" if self.evaluation_type.get() == "qa" else "synthesis conditions"
                raise ValueError(f"No {dataset_type} found in the dataset file.\n\nPlease check that your JSON file contains valid {dataset_type}.")
            
            return True
            
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))
            return False
        except Exception as e:
            messagebox.showerror("Unexpected Error", f"An unexpected error occurred during validation:\n\n{str(e)}")
            return False
    
    def start_evaluation(self):
        """Start evaluation with enhanced error handling"""
        try:
            if not self.validate_inputs():
                return
            
            # Show that evaluation is starting
            eval_type_text = "Q&A" if self.evaluation_type.get() == "qa" else "synthesis condition"
            mode_text = ""
            if self.evaluation_type.get() == "qa":
                mode_text = f" ({self.evaluation_mode.get()})"
            
            self.update_status(f"Starting {eval_type_text} evaluation{mode_text}...")
            self.update_progress(5)
            
            # Start evaluation in a separate thread to prevent GUI freezing
            thread = threading.Thread(target=self.run_evaluation)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start evaluation:\n\n{str(e)}")
            self.update_status("Ready to evaluate")
            self.update_progress(0)
    
    def run_evaluation(self):
        try:
            eval_type_text = "Q&A" if self.evaluation_type.get() == "qa" else "synthesis condition"
            mode_text = ""
            if self.evaluation_type.get() == "qa":
                mode_text = f" ({self.evaluation_mode.get()})"
            
            self.update_status(f"Loading manuscript and supplement files for {eval_type_text} evaluation{mode_text}...")
            self.update_progress(10)
            
            # Process context from manuscript and supplement files
            context = ""
            files_loaded = []
            
            if self.manuscript_path.get():
                manuscript_content = self.process_file_content(self.manuscript_path.get())
                if manuscript_content:
                    context += manuscript_content + " "
                    files_loaded.append("manuscript")
                
            if self.supplement_path.get() and os.path.exists(self.supplement_path.get()):
                supplement_content = self.process_file_content(self.supplement_path.get())
                if supplement_content:
                    context += supplement_content + " "
                    files_loaded.append("supplement")
            
            # Update status to show what files were loaded
            files_text = " and ".join(files_loaded) if files_loaded else "manuscript"
            self.update_status(f"Loaded {files_text}. Loading {eval_type_text} dataset...")
            self.update_progress(20)
            
            # Load dataset
            dataset = self.load_dataset(self.dataset_path.get())
            if not dataset:
                raise ValueError(f"No {eval_type_text} data found in the dataset")
            
            # Prepare prompt based on evaluation type
            if self.evaluation_type.get() == "qa":
                prompt = f"CONTEXT:{context}\n\nQ&A DATASET:"
                for pair in dataset:
                    prompt += str(pair) + "\n\n"
            else:
                # For synthesis conditions, convert to JSON string
                synthesis_data = json.dumps(dataset, indent=4)
                prompt = f"CONTEXT:{context}\n\nSynthesis Conditions Data:{synthesis_data}"
            
            self.update_status(f"Running {eval_type_text} LLM evaluations{mode_text} using {files_text}...")
            self.update_progress(30)
            
            # Get the appropriate classification prompt
            classification_prompt = self.get_classification_prompt()
            
            # Run evaluations with all 4 LLMs concurrently
            self.update_status(f"Running all LLM evaluations concurrently ({eval_type_text}{mode_text})...")
            self.update_progress(30)
            
            # Use ThreadPoolExecutor to run all models simultaneously
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                
                # Submit all available models
                if self.claude_key.get().strip():
                    if self.evaluation_type.get() == "qa":
                        futures['claude'] = executor.submit(self.run_claude_qa_evaluation, classification_prompt, prompt, len(dataset))
                    else:
                        futures['claude'] = executor.submit(self.run_claude_synthesis_evaluation, classification_prompt, prompt)
                
                if self.gemini_key.get().strip():
                    if self.evaluation_type.get() == "qa":
                        futures['gemini'] = executor.submit(self.run_gemini_qa_evaluation, classification_prompt, prompt, len(dataset))
                    else:
                        futures['gemini'] = executor.submit(self.run_gemini_synthesis_evaluation, classification_prompt, prompt)
                
                if self.openai_key.get().strip():
                    if self.evaluation_type.get() == "qa":
                        futures['gpt4o'] = executor.submit(self.run_gpt_qa_evaluation, classification_prompt, prompt, len(dataset))
                    else:
                        futures['gpt4o'] = executor.submit(self.run_gpt_synthesis_evaluation, classification_prompt, prompt)
                
                if self.openai_o1_key.get().strip():
                    if self.evaluation_type.get() == "qa":
                        futures['gpt_o1'] = executor.submit(self.run_gpt_o1_qa_evaluation, classification_prompt, prompt, len(dataset))
                    else:
                        futures['gpt_o1'] = executor.submit(self.run_gpt_o1_synthesis_evaluation, classification_prompt, prompt)
                
                # Wait for all to complete and collect results
                results = []
                successful_models = []
                failed_models = []
                
                for model_name, future in futures.items():
                    try:
                        result = future.result(timeout=600)  # 10 minute timeout per model
                        if result is not None:
                            if self.evaluation_type.get() == "qa" and len(result) == len(dataset):
                                results.append((model_name, result))
                                model_display_name = {
                                    'claude': 'Claude',
                                    'gemini': 'Gemini', 
                                    'gpt4o': 'GPT-4o',
                                    'gpt_o1': 'GPT-o1'
                                }.get(model_name, model_name)
                                successful_models.append(model_display_name)
                            elif self.evaluation_type.get() == "synthesis" and len(result) >= 1:
                                results.append((model_name, result))
                                model_display_name = {
                                    'claude': 'Claude',
                                    'gemini': 'Gemini', 
                                    'gpt4o': 'GPT-4o',
                                    'gpt_o1': 'GPT-o1'
                                }.get(model_name, model_name)
                                successful_models.append(model_display_name)
                            else:
                                failed_models.append(model_name)
                                print(f"{model_name} evaluation failed or returned incorrect number of results")
                        else:
                            failed_models.append(model_name)
                            print(f"{model_name} evaluation failed or returned None")
                    except Exception as e:
                        failed_models.append(model_name)
                        print(f"{model_name} evaluation failed with error: {e}")
            
            self.update_progress(80)
            
            if not results:
                error_msg = "No LLM evaluations were successful."
                if failed_models:
                    error_msg += f"\n\nFailed models: {', '.join(failed_models)}"
                error_msg += "\n\nPlease check:\n• Your API keys are valid\n• You have sufficient API credits\n• Your internet connection is stable"
                raise ValueError(error_msg)
            
            # Show which models were successful
            success_msg = f"Successfully completed evaluations with: {', '.join(successful_models)}"
            if failed_models:
                success_msg += f"\nNote: {', '.join(failed_models)} failed to provide complete evaluations."
            print(success_msg)
            
            self.update_status("Processing results...")
            
            # Combine results and calculate final evaluation
            if self.evaluation_type.get() == "qa":
                final_results = self.combine_qa_results(results, dataset)
            else:
                final_results = self.combine_synthesis_results(results)
            
            self.evaluation_results = final_results
            
            self.update_progress(100)
            success_status = f"{eval_type_text.title()} evaluation{mode_text} completed! Used models: {', '.join(successful_models)}"
            if failed_models:
                success_status += f" (Failed: {', '.join(failed_models)})"
            self.update_status(success_status)
            
            # Show results in the same window
            self.root.after(0, self.show_results_inline)
            
        except ValueError as e:
            # Known validation errors
            self.root.after(0, lambda: messagebox.showerror("Evaluation Error", str(e)))
            self.update_status("Evaluation failed")
            self.update_progress(0)
        except Exception as e:
            # Unexpected errors
            error_msg = f"An unexpected error occurred during evaluation:\n\n{str(e)}\n\nPlease check your inputs and try again."
            self.root.after(0, lambda: messagebox.showerror("Unexpected Error", error_msg))
            self.update_status("Evaluation failed")
            self.update_progress(0)
    
    # Q&A Evaluation Methods
    def run_claude_qa_evaluation(self, classification_prompt, prompt, num_qs):
        retries = 5
        
        while retries > 0:
            try:
                if not self.claude_key.get().strip():
                    return None
                    
                claude_client = anthropic.Anthropic(api_key=self.claude_key.get().strip())
                
                result = claude_client.messages.create(
                    model='claude-3-5-sonnet-20240620',
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
                                                "question": {"type": "string"},
                                                "answer": {"type": "string"},
                                                "question_type": {"type": "string"},
                                                "evaluation": {"type": "string"},
                                                "explanation": {"type": "string"}
                                            },
                                            "required": ["question", "answer", "question_type", "evaluation", "explanation"]
                                        }
                                    }
                                },
                                "required": ["pair"]
                            }
                        }
                    ],
                    tool_choice={"type": "tool", "name": "question_answer_pair_classifier"},
                    messages=[{"role": "user", "content": prompt}]
                )
                
                df = claude_output_to_df(result)
                
                if len(df) != num_qs:
                    print(f"Claude evaluation count mismatch: expected {num_qs}, got {len(df)}")
                    retries -= 1
                    if retries > 0:
                        print(f"Retrying Claude evaluation, {retries} attempts remaining...")
                        continue
                    return None
                
                return df
                
            except Exception as e:
                print(f"Claude evaluation failed: {e}. Retrying...")
                retries -= 1
                if retries <= 0:
                    print("Claude evaluation failed after all retries")
                    return None
        
        return None
    
    # Synthesis Evaluation Methods
    def run_claude_synthesis_evaluation(self, classification_prompt, prompt):
        retries = 5
        
        while retries > 0:
            try:
                if not self.claude_key.get().strip():
                    return None
                    
                claude_client = anthropic.Anthropic(api_key=self.claude_key.get().strip())
                
                result = claude_client.messages.create(
                    model='claude-3-5-sonnet-20240620',
                    max_tokens=8192,
                    system=classification_prompt,
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
                
                # Convert Claude synthesis output to dataframe
                try:
                    df = pd.DataFrame([result.content[0].input])
                except Exception as e:
                    print("Standard parsing failed, using fallback parsing. Error:", e)
                    df = pd.DataFrame(eval(result)['pair'])
                df.columns = ['claude_' + col for col in df.columns]
                
                if df.shape[1] == 7 and len(df) >= 1:
                    return df
                else:
                    print(f"Claude synthesis evaluation returned unexpected format")
                    retries -= 1
                    if retries > 0:
                        print(f"Retrying Claude synthesis evaluation, {retries} attempts remaining...")
                        continue
                    return None
                
            except Exception as e:
                print(f"Claude synthesis evaluation failed: {e}. Retrying...")
                retries -= 1
                if retries <= 0:
                    print("Claude synthesis evaluation failed after all retries")
                    return None
        
        return None
    
    def run_gemini_synthesis_evaluation(self, classification_prompt, prompt):
        # Define Gemini synthesis condition schema
        class gemini_synthesis_condition_check(BaseModel):
            all_synthesis_conditions: str = Field(
                description="MAKE SURE YOU INCLUDE THIS:string containing all input synthesis conditions exactly as provided"
            )
            criterion_1: Literal["Y", "N"] = Field(description="criterion 1 check result")
            criterion_1_explanation: str = Field(description="explanation of criterion 1 check result")
            criterion_2: Literal["Y", "N"] = Field(description="criterion 2 check result")
            criterion_2_explanation: str = Field(description="explanation of criterion 2 check result")
            criterion_3: Literal["Y", "N"] = Field(description="criterion 3 check result")
            criterion_3_explanation: str = Field(description="explanation of criterion 3 check result")
        
        retries = 5
        
        while retries > 0:
            try:
                if not self.gemini_key.get().strip():
                    return None
                    
                genai.configure(api_key=self.gemini_key.get().strip())
                gemini_client = genai.GenerativeModel(
                    model_name='gemini-1.5-pro-latest',
                    system_instruction=classification_prompt
                )
                
                result = gemini_client.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=gemini_synthesis_condition_check
                    ),
                )
                
                # Convert Gemini synthesis output to dataframe
                df = pd.DataFrame([eval(result.parts[0].text)])
                df.columns = ['gemini_' + col for col in df.columns]
                
                if df.shape[1] == 7 and len(df) >= 1:
                    return df
                else:
                    print(f"Gemini synthesis evaluation returned unexpected format")
                    retries -= 1
                    if retries > 0:
                        print(f"Retrying Gemini synthesis evaluation, {retries} attempts remaining...")
                        continue
                    return None
                
            except Exception as e:
                print(f"Gemini synthesis evaluation failed: {e}. Retrying...")
                retries -= 1
                if retries <= 0:
                    print("Gemini synthesis evaluation failed after all retries")
                    return None
        
        return None
    
    def run_gpt_synthesis_evaluation(self, classification_prompt, prompt):
        # Define GPT synthesis condition schema
        class gpt_synthesis_condition_check(BaseModel):
            all_synthesis_conditions: str = Field(description="string containing all input synthesis conditions exactly as provided")
            criterion_1: Literal["Y", "N"] = Field(description="criterion 1 check result")
            criterion_1_explanation: str = Field(description="explanation of criterion 1 check result")
            criterion_2: Literal["Y", "N"] = Field(description="criterion 2 check result")
            criterion_2_explanation: str = Field(description="explanation of criterion 2 check result")
            criterion_3: Literal["Y", "N"] = Field(description="criterion 3 check result")
            criterion_3_explanation: str = Field(description="explanation of criterion 3 check result")
        
        retries = 5
        
        while retries > 0:
            try:
                if not self.openai_key.get().strip():
                    return None
                    
                openai_client = OpenAI(api_key=self.openai_key.get().strip())
                
                completion = openai_client.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {"role": "system", "content": classification_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=gpt_synthesis_condition_check,
                )
                
                result = completion.choices[0].message.content
                
                # Convert GPT synthesis output to dataframe
                df = pd.DataFrame([eval(result)])
                df.columns = ['gpt_' + col for col in df.columns]
                
                if len(df) >= 1:
                    return df
                else:
                    print(f"GPT-4o synthesis evaluation returned unexpected format")
                    retries -= 1
                    if retries > 0:
                        print(f"Retrying GPT-4o synthesis evaluation, {retries} attempts remaining...")
                        continue
                    return None
                
            except Exception as e:
                print(f"GPT-4o synthesis evaluation failed: {e}. Retrying...")
                retries -= 1
                if retries <= 0:
                    print("GPT-4o synthesis evaluation failed after all retries")
                    return None
        
        return None
    
    def run_gpt_o1_synthesis_evaluation(self, classification_prompt, prompt):
        # O1 specific output prompt for synthesis
        o1_synthesis_output_prompt = (
            'Please format your answers to a dictionary in string following this structure: '
            '{"all_synthesis_conditions": "string containing all input synthesis conditions exactly as provided", '
            '"criterion_1": "Y or N", '
            '"criterion_1_explanation": "string: explanation of criterion 1 check result", '
            '"criterion_2": "Y or N", '
            '"criterion_2_explanation": "string: explanation of criterion 2 check result", '
            '"criterion_3": "Y or N", '
            '"criterion_3_explanation": "string: explanation of criterion 3 check result"}'
        )
        
        retries = 5
        
        while retries > 0:
            try:
                if not self.openai_o1_key.get().strip():
                    return None
                    
                openai_client = OpenAI(api_key=self.openai_o1_key.get().strip())
                
                input_prompt = classification_prompt + prompt + o1_synthesis_output_prompt
                completion = openai_client.chat.completions.create(
                    model='o1-preview-2024-09-12',
                    messages=[{"role": "user", "content": input_prompt}],
                )
                
                output = completion.choices[0].message.content
                
                if len(output) <= 0:
                    print(f"GPT-o1 synthesis evaluation returned empty output")
                    retries -= 1
                    if retries > 0:
                        print(f"Retrying GPT-o1 synthesis evaluation, {retries} attempts remaining...")
                        continue
                    return None
                
                # Convert GPT-o1 synthesis output to dataframe
                df = pd.DataFrame([eval(output)])
                df.columns = ['gpt_o1_' + col for col in df.columns]
                
                if len(df) >= 1:
                    return df
                else:
                    print(f"GPT-o1 synthesis evaluation returned unexpected format")
                    retries -= 1
                    if retries > 0:
                        print(f"Retrying GPT-o1 synthesis evaluation, {retries} attempts remaining...")
                        continue
                    return None
                
            except Exception as e:
                print(f"GPT-o1 synthesis evaluation failed: {e}. Retrying...")
                retries -= 1
                if retries <= 0:
                    print("GPT-o1 synthesis evaluation failed after all retries")
                    return None
        
        return None
    
    # Additional Q&A evaluation methods (implement similar to synthesis but for Q&A)
    def run_gemini_qa_evaluation(self, classification_prompt, prompt, num_qs):
        retries = 5
        
        while retries > 0:
            try:
                if not self.gemini_key.get().strip():
                    return None
                    
                genai.configure(api_key=self.gemini_key.get().strip())
                gemini_client = genai.GenerativeModel(
                    model_name='gemini-1.5-pro-latest',
                    system_instruction=classification_prompt
                )
                
                result = gemini_client.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=question_answer_pairs
                    ),
                )
                
                df = gemini_output_to_df(result)
                
                if len(df) != num_qs:
                    print(f"Gemini evaluation count mismatch: expected {num_qs}, got {len(df)}")
                    retries -= 1
                    if retries > 0:
                        print(f"Retrying Gemini evaluation, {retries} attempts remaining...")
                        continue
                    return None
                
                return df
                
            except Exception as e:
                print(f"Gemini evaluation failed: {e}. Retrying...")
                retries -= 1
                if retries <= 0:
                    print("Gemini evaluation failed after all retries")
                    return None
        
        return None
    
    def run_gpt_qa_evaluation(self, classification_prompt, prompt, num_qs):
        retries = 5
        
        while retries > 0:
            try:
                if not self.openai_key.get().strip():
                    return None
                    
                openai_client = OpenAI(api_key=self.openai_key.get().strip())
                
                completion = openai_client.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {"role": "system", "content": classification_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=question_answer_pairs,
                )
                
                result = completion.choices[0].message.content
                df = gpt_output_to_df(result)
                
                if len(df) != num_qs:
                    print(f"GPT-4o evaluation count mismatch: expected {num_qs}, got {len(df)}")
                    retries -= 1
                    if retries > 0:
                        print(f"Retrying GPT-4o evaluation, {retries} attempts remaining...")
                        continue
                    return None
                
                return df
                
            except Exception as e:
                print(f"GPT-4o evaluation failed: {e}. Retrying...")
                retries -= 1
                if retries <= 0:
                    print("GPT-4o evaluation failed after all retries")
                    return None
        
        return None
    
    def run_gpt_o1_qa_evaluation(self, classification_prompt, prompt, num_qs):
        retries = 5
        
        o1_list_output_prompt = (
            'Please format all answers to the following questions in a list, where each question answer pair in the list '
            'follows this structure: {"question": "string", "answer": "string", "question_type": "factual/reasoning/True or False", '
            '"evaluation": "TP/FP/FN/TN", "explanation": "string"}'
        )
        
        while retries > 0:
            try:
                if not self.openai_o1_key.get().strip():
                    return None
                    
                openai_client = OpenAI(api_key=self.openai_o1_key.get().strip())
                
                input_prompt = classification_prompt + prompt + o1_list_output_prompt
                completion = openai_client.chat.completions.create(
                    model='o1-preview-2024-09-12',
                    messages=[{"role": "user", "content": input_prompt}],
                )
                
                output = completion.choices[0].message.content
                
                evaluated_output = eval(output)
                if type(evaluated_output) != list:
                    raise TypeError("Output is not a list")
                
                if len(evaluated_output) != num_qs:
                    print(f"GPT-o1 evaluation count mismatch: expected {num_qs}, got {len(evaluated_output)}")
                    retries -= 1
                    if retries > 0:
                        print(f"Retrying GPT-o1 evaluation, {retries} attempts remaining...")
                        continue
                    return None
                
                df = pd.DataFrame(evaluated_output)
                df.columns = [col + '_gpt_o1' if col != 'question' and col != 'answer' else col for col in df.columns]
                if 'evaluation_gpt_o1' in df.columns:
                    df['evaluation_gpt_o1'] = df['evaluation_gpt_o1'].replace({
                        r'True\s+Positive\b.*': 'TP',
                        r'False\s+Positive\b.*': 'FP',
                        r'False\s+Negative\b.*': 'FN',
                        r'True\s+Negative\b.*': 'TN'
                    }, regex=True)
                
                return df
                
            except Exception as e:
                print(f"GPT-o1 evaluation failed: {e}. Retrying...")
                retries -= 1
                if retries <= 0:
                    print("GPT-o1 evaluation failed after all retries")
                    return None
        
        return None
    
    def combine_qa_results(self, results, qa_pairs):
        """Combine Q&A evaluation results from multiple models"""
        if len(results) == 1:
            df = results[0][1].copy()
            eval_col = None
            for col in df.columns:
                if 'evaluation' in col:
                    eval_col = col
                    break
            
            if eval_col:
                df['final_evaluation'] = df[eval_col]
            else:
                df['final_evaluation'] = 'TN'
        else:
            # Multiple models, combine using weighted voting
            base_df = results[0][1].copy()
            
            # Find and rename evaluation columns properly
            base_eval_col = None
            for col in base_df.columns:
                if 'evaluation' in col:
                    base_eval_col = col
                    break
            
            base_model_name = results[0][0]
            if base_eval_col:
                base_df[f'evaluation_{base_model_name}'] = base_df[base_eval_col]
                if base_eval_col != f'evaluation_{base_model_name}':
                    base_df = base_df.drop(columns=[base_eval_col])
            
            # Add results from other models
            for model_name, model_df in results[1:]:
                if len(model_df) != len(base_df):
                    raise ValueError(f"Model result length mismatch: {model_name} has {len(model_df)} results, expected {len(base_df)}")
                
                model_eval_col = None
                for col in model_df.columns:
                    if 'evaluation' in col:
                        model_eval_col = col
                        break
                
                if model_eval_col:
                    base_df[f'evaluation_{model_name}'] = model_df[model_eval_col]
                
                # Add explanation if available
                model_expl_col = None
                for col in model_df.columns:
                    if 'explanation' in col:
                        model_expl_col = col
                        break
                
                if model_expl_col:
                    base_df[f'explanation_{model_name}'] = model_df[model_expl_col]
            
            # Calculate weighted mode
            eval_columns = []
            weights = []
            
            model_weights = {'gpt4o': 0.23, 'claude': 0.23, 'gemini': 0.23, 'gpt_o1': 0.3}
            
            for model_name, _ in results:
                col_name = f'evaluation_{model_name}'
                if col_name in base_df.columns:
                    eval_columns.append(col_name)
                    weights.append(model_weights.get(model_name, 0.25))
            
            if eval_columns:
                base_df['final_evaluation'] = self.custom_weighted_mode(base_df, eval_columns, weights)
            else:
                base_df['final_evaluation'] = ['TN'] * len(base_df)
            
            df = base_df
        
        # Ensure required columns exist
        required_columns = ['question', 'answer', 'final_evaluation']
        for col in required_columns:
            if col not in df.columns:
                if col == 'question':
                    df['question'] = [pair.get('question', f'Question {i+1}') for i, pair in enumerate(qa_pairs)]
                elif col == 'answer':
                    df['answer'] = [pair.get('answer', f'Answer {i+1}') for i, pair in enumerate(qa_pairs)]
        
        return df
    
    def combine_synthesis_results(self, results):
        """Combine synthesis condition evaluation results from multiple models"""
        if len(results) == 1:
            # Only one model, use its results directly
            df = results[0][1].copy()
            return df
        else:
            # Multiple models, combine using weighted voting
            base_df = results[0][1].copy()
            
            # Add results from other models
            for model_name, model_df in results[1:]:
                if len(model_df) != len(base_df):
                    raise ValueError(f"Model result length mismatch: {model_name} has {len(model_df)} results, expected {len(base_df)}")
                
                # Add all columns from this model
                for col in model_df.columns:
                    base_df[col] = model_df[col]
            
            # Calculate weighted mode for each criterion
            model_weights = {'gpt': 0.23, 'claude': 0.23, 'gemini': 0.23, 'gpt_o1': 0.3}
            
            # For each criterion, find the weighted mode
            for criterion_num in [1, 2, 3]:
                criterion_cols = []
                weights = []
                
                for model_name, _ in results:
                    col_name = f'{model_name}_criterion_{criterion_num}'
                    if col_name in base_df.columns:
                        criterion_cols.append(col_name)
                        weights.append(model_weights.get(model_name, 0.25))
                
                if criterion_cols:
                    base_df[f'criterion_{criterion_num}'] = weighted_mode(base_df, criterion_cols, weights)
                else:
                    base_df[f'criterion_{criterion_num}'] = ['N'] * len(base_df)
            
            return base_df
    
    def custom_weighted_mode(self, df, eval_columns, weights):
        """Calculate weighted mode with GPT-o1 as tiebreaker"""
        final_evaluations = []
        
        for idx, row in df.iterrows():
            vote_counts = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
            gpt_o1_vote = None
            
            for col, weight in zip(eval_columns, weights):
                vote = row[col]
                if pd.notna(vote) and vote in vote_counts:
                    vote_counts[vote] += weight
                    
                    if 'gpt_o1' in col:
                        gpt_o1_vote = vote
            
            max_score = max(vote_counts.values())
            winners = [eval_type for eval_type, score in vote_counts.items() if score == max_score]
            
            if len(winners) == 1:
                final_evaluation = winners[0]
            else:
                if gpt_o1_vote and gpt_o1_vote in winners:
                    final_evaluation = gpt_o1_vote
                else:
                    final_evaluation = winners[0]
            
            final_evaluations.append(final_evaluation)
        
        return final_evaluations
    
    def show_results_inline(self):
        """Display results in the main window"""
        if self.evaluation_results is None:
            return
        
        # Use the dynamically calculated results row
        results_row = getattr(self, 'results_row', 8)
        self.results_frame.grid(row=results_row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(20, 0))
        self.results_visible = True
        
        # Clear any existing content
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Performance stats
        self.show_performance_stats_inline(self.results_frame)
        
        # Details section
        details_frame = ttk.Frame(self.results_frame)
        details_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(20, 0))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(1, weight=1)
        
        # Controls frame
        controls_frame = ttk.Frame(details_frame)
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        controls_frame.columnconfigure(2, weight=1)
        
        # Details toggle
        self.details_visible = tk.BooleanVar()
        details_btn = ttk.Checkbutton(
            controls_frame,
            text="Show Individual Evaluation",
            variable=self.details_visible,
            command=lambda: self.toggle_details_inline(details_frame),
        )
        details_btn.grid(row=0, column=0, sticky=tk.W)
        
        # Information button
        info_btn = ttk.Button(
            controls_frame,
            text="\u2139",       
            width=1,
            command=self.show_info_popup
        )
        info_btn.grid(row=0, column=1, padx=(10, 5), sticky=tk.E)
        
        # Export to Excel button
        export_btn = ttk.Button(
            controls_frame, text="Export to Excel", command=self.export_to_excel, width=15
        )
        export_btn.grid(row=0, column=3, sticky=tk.E)
        
        # Details content (initially hidden)
        self.details_content = ttk.Frame(details_frame)
    
    def show_performance_stats_inline(self, parent):
        stats_frame = ttk.LabelFrame(parent, text="Performance Statistics", padding="10")
        stats_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        df = self.evaluation_results
        
        if self.evaluation_type.get() == "qa":
            self.show_qa_stats(stats_frame, df)
        else:
            self.show_synthesis_stats(stats_frame, df)
    
    def show_qa_stats(self, parent, df):
        total_pairs = len(df)
        
        # Count evaluations
        tp_count = sum(df['final_evaluation'] == 'TP')
        fp_count = sum(df['final_evaluation'] == 'FP')
        tn_count = sum(df['final_evaluation'] == 'TN')
        fn_count = sum(df['final_evaluation'] == 'FN')
        
        # Calculate accuracy
        accuracy = (tp_count / total_pairs * 100) if total_pairs > 0 else 0
        
        # Add evaluation mode info
        mode_text = "Single-hop" if self.evaluation_mode.get() == "single-hop" else "Multi-hop"
        ttk.Label(parent, text=f"Q&A Evaluation Mode: {mode_text}", font=("Arial", 12, "bold"), foreground="blue").grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 15))
        
        # Accuracy
        ttk.Label(parent, text="Accuracy (TP %):", font=("Arial", 14, "bold")).grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        accuracy_label = tk.Label(parent, text=f"{accuracy:.1f}%", font=("Arial", 16, "bold"), fg="red")
        accuracy_label.grid(row=1, column=1, sticky=tk.W, pady=(0, 10))
        
        # Total
        ttk.Label(parent, text="Total Q&A Pairs:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Label(parent, text=str(total_pairs)).grid(row=2, column=1, sticky=tk.W, pady=(0, 10))
        
        # TP, FP
        ttk.Label(parent, text="True Positive (TP):", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Label(parent, text=str(tp_count)).grid(row=3, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(parent, text="False Positive (FP):", font=("Arial", 10, "bold")).grid(row=3, column=2, sticky=tk.W, padx=(20, 10))
        ttk.Label(parent, text=str(fp_count)).grid(row=3, column=3, sticky=tk.W)
        
        # TN, FN
        ttk.Label(parent, text="True Negative (TN):", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Label(parent, text=str(tn_count)).grid(row=4, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(parent, text="False Negative (FN):", font=("Arial", 10, "bold")).grid(row=4, column=2, sticky=tk.W, padx=(20, 10))
        ttk.Label(parent, text=str(fn_count)).grid(row=4, column=3, sticky=tk.W)
    
    def show_synthesis_stats(self, parent, df):
        # Add evaluation type info
        ttk.Label(parent, text="Synthesis Condition Evaluation", font=("Arial", 12, "bold"), foreground="blue").grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 15))
        
        # Define criterion descriptions
        criterion_descriptions = {
            1: {
                'Y': 'Synthesis conditions have been included for all MOFs mentioned.',
                'N': 'Synthesis conditions have not been included for all MOFs mentioned.'
            },
            2: {
                'Y': 'Only synthesis information has been extracted.',
                'N': 'Extraction contains more than just synthesis information.'
            },
            3: {
                'Y': 'All synthesis-condition details have been extracted and correctly matched to their MOFs.',
                'N': 'Some synthesis-condition details are missing or not correctly matched to their MOFs.'
            }
        }
        
        # Show criterion results with detailed descriptions
        for i in [1, 2, 3]:
            criterion_col = f'criterion_{i}'
            if criterion_col in df.columns:
                result = df[criterion_col].iloc[0] if len(df) > 0 else 'N/A'
                
                # Criterion header
                ttk.Label(parent, text=f"C{i}:", font=("Arial", 12, "bold")).grid(row=i, column=0, sticky=tk.W, pady=5)
                
                # Color code the result
                color = "green" if result == 'Y' else "red" if result == 'N' else "black"
                result_label = tk.Label(parent, text=result, font=("Arial", 14, "bold"), fg=color)
                result_label.grid(row=i, column=1, sticky=tk.W, pady=5, padx=(10, 10))
                
                # Show appropriate description based on result
                if result in criterion_descriptions[i]:
                    description = criterion_descriptions[i][result]
                    ttk.Label(parent, text=description, 
                             font=("Arial", 10), wraplength=500).grid(row=i, column=2, sticky=tk.W, pady=5)
    
    def toggle_details_inline(self, parent):
        if self.details_visible.get():
            self.show_details_inline(parent)
        else:
            self.hide_details_inline()
    
    def show_details_inline(self, parent):
        self.details_content.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        self.details_content.columnconfigure(0, weight=1)
        self.details_content.rowconfigure(0, weight=1)
        
        # Create scrollable text widget
        canvas = tk.Canvas(self.details_content, height=300)
        scrollbar = ttk.Scrollbar(self.details_content, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        if self.evaluation_type.get() == "qa":
            self.show_qa_details(scrollable_frame)
        else:
            self.show_synthesis_details(scrollable_frame)
        
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    def show_qa_details(self, parent):
        # Add Q&A details
        for idx, row in self.evaluation_results.iterrows():
            qa_frame = ttk.LabelFrame(parent, text=f"Q&A Pair {idx + 1}", padding="5")
            qa_frame.grid(row=idx, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
            qa_frame.columnconfigure(0, weight=1)
            
            # Question
            ttk.Label(qa_frame, text="Question:", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky=tk.W)
            q_text = tk.Text(qa_frame, height=2, wrap=tk.WORD, font=("Arial", 9))
            q_text.insert("1.0", row['question'])
            q_text.config(state=tk.DISABLED)
            q_text.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
            
            # Answer
            ttk.Label(qa_frame, text="Answer:", font=("Arial", 9, "bold")).grid(row=2, column=0, sticky=tk.W)
            a_text = tk.Text(qa_frame, height=2, wrap=tk.WORD, font=("Arial", 9))
            a_text.insert("1.0", row['answer'])
            a_text.config(state=tk.DISABLED)
            a_text.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
            
            # Evaluation
            eval_text = f"Evaluation: {row['final_evaluation']}"
            if 'explanation' in row and pd.notna(row['explanation']):
                eval_text += f"\nExplanation: {row['explanation']}"
            
            ttk.Label(qa_frame, text="LLM Evaluation:", font=("Arial", 9, "bold")).grid(row=4, column=0, sticky=tk.W)
            eval_label = ttk.Label(qa_frame, text=eval_text, font=("Arial", 9), wraplength=600)
            eval_label.grid(row=5, column=0, sticky=tk.W)
    
    def show_synthesis_details(self, parent):
        # Add synthesis details
        df = self.evaluation_results
        
        synthesis_frame = ttk.LabelFrame(parent, text="Synthesis Condition Evaluation Details", padding="5")
        synthesis_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        synthesis_frame.columnconfigure(0, weight=1)
        
        # Show detailed evaluations for each criterion
        for i in [1, 2, 3]:
            criterion_frame = ttk.LabelFrame(synthesis_frame, text=f"Criterion {i}", padding="5")
            criterion_frame.grid(row=i+1, column=0, sticky=(tk.W, tk.E), pady=5)
            # Don't make columns expandable to reduce gap
            criterion_frame.columnconfigure(0, weight=0)
            criterion_frame.columnconfigure(1, weight=0)
            
            # Show results from each model
            row_idx = 0
            for model in ['claude', 'gemini', 'gpt', 'gpt_o1']:
                result_col = f'{model}_criterion_{i}'
                
                if result_col in df.columns and len(df) > 0:
                    result = df[result_col].iloc[0]
                    
                    # Model name
                    ttk.Label(criterion_frame, text=f"{model.upper()}:", font=("Arial", 9, "bold")).grid(row=row_idx, column=0, sticky=tk.W)
                    
                    # Result with color coding - reduced padding
                    color = "green" if result == 'Y' else "red" if result == 'N' else "black"
                    result_label = tk.Label(criterion_frame, text=result, font=("Arial", 9, "bold"), fg=color)
                    result_label.grid(row=row_idx, column=1, sticky=tk.W, padx=(10, 0))
                    
                    row_idx += 1
    
    def hide_details_inline(self):
        if hasattr(self, 'details_content'):
            self.details_content.grid_remove()
    
    def export_to_excel(self):
        """Export evaluation results to an Excel file"""
        if self.evaluation_results is None:
            messagebox.showwarning(
                "No Results", "No evaluation results to export. Please run an evaluation first."
            )
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="Save Evaluation Results",
        )

        if not filename:
            return

        export_df = self.prepare_export_data()

        try:
            export_df.to_excel(filename, index=False)
            messagebox.showinfo(
                "Export Successful", f"Evaluation results exported successfully."
            )
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results:\n\n{str(e)}")
    
    def prepare_export_data(self):
        """Prepare the evaluation data for Excel export"""
        df = self.evaluation_results.copy()
        
        if self.evaluation_type.get() == "qa":
            return self.prepare_qa_export_data(df)
        else:
            return self.prepare_synthesis_export_data(df)
    
    def prepare_qa_export_data(self, df):
        """Prepare Q&A evaluation data for export"""
        required_columns = [
            'question', 'answer', 'question_type',
            'evaluation_gpt', 'explanation_gpt',
            'evaluation_gemini', 'explanation_gemini',
            'evaluation_claude', 'explanation_claude',
            'evaluation_gpt_o1', 'explanation_gpt_o1',
            'evaluation'
        ]
        
        export_df = pd.DataFrame()
        
        for col in required_columns:
            if col == 'question':
                export_df[col] = df['question'] if 'question' in df.columns else 'N/A'
            elif col == 'answer':
                export_df[col] = df['answer'] if 'answer' in df.columns else 'N/A'
            elif col == 'question_type':
                found_qtype = False
                for potential_col in ['question_type', 'question_type_claude', 'question_type_gemini', 'question_type_gpt4o', 'question_type_gpt_o1']:
                    if potential_col in df.columns:
                        export_df[col] = df[potential_col]
                        found_qtype = True
                        break
                if not found_qtype:
                    export_df[col] = 'N/A'
            elif col == 'evaluation_gpt':
                export_df[col] = df['evaluation_gpt4o'] if 'evaluation_gpt4o' in df.columns else 'N/A'
            elif col == 'explanation_gpt':
                export_df[col] = df['explanation_gpt4o'] if 'explanation_gpt4o' in df.columns else 'N/A'
            elif col == 'evaluation_gemini':
                export_df[col] = df['evaluation_gemini'] if 'evaluation_gemini' in df.columns else 'N/A'
            elif col == 'explanation_gemini':
                export_df[col] = df['explanation_gemini'] if 'explanation_gemini' in df.columns else 'N/A'
            elif col == 'evaluation_claude':
                export_df[col] = df['evaluation_claude'] if 'evaluation_claude' in df.columns else 'N/A'
            elif col == 'explanation_claude':
                export_df[col] = df['explanation_claude'] if 'explanation_claude' in df.columns else 'N/A'
            elif col == 'evaluation_gpt_o1':
                export_df[col] = df['evaluation_gpt_o1'] if 'evaluation_gpt_o1' in df.columns else 'N/A'
            elif col == 'explanation_gpt_o1':
                export_df[col] = df['explanation_gpt_o1'] if 'explanation_gpt_o1' in df.columns else 'N/A'
            elif col == 'evaluation':
                export_df[col] = df['final_evaluation'] if 'final_evaluation' in df.columns else 'N/A'
        
        return export_df
    
    def prepare_synthesis_export_data(self, df):
        """Prepare synthesis condition evaluation data for export with specific column order"""
        # Define the exact column order required for synthesis evaluation
        required_columns = [
            'criterion_1',
            'criterion_2', 
            'criterion_3',
            'gpt_criterion_1',
            'gpt_criterion_1_explanation',
            'gpt_criterion_2',
            'gpt_criterion_2_explanation', 
            'gpt_criterion_3',
            'gpt_criterion_3_explanation',
            'gemini_criterion_1',
            'gemini_criterion_1_explanation',
            'gemini_criterion_2',
            'gemini_criterion_2_explanation',
            'gemini_criterion_3', 
            'gemini_criterion_3_explanation',
            'claude_criterion_1',
            'claude_criterion_1_explanation',
            'claude_criterion_2',
            'claude_criterion_2_explanation',
            'claude_criterion_3',
            'claude_criterion_3_explanation',
            'gpt_o1_criterion_1',
            'gpt_o1_criterion_1_explanation',
            'gpt_o1_criterion_2',
            'gpt_o1_criterion_2_explanation',
            'gpt_o1_criterion_3',
            'gpt_o1_criterion_3_explanation'
        ]
        
        # Create export dataframe with only the required columns
        export_df = pd.DataFrame()
        
        for col in required_columns:
            if col in df.columns:
                export_df[col] = df[col]
            else:
                # Fill missing columns with 'N/A'
                export_df[col] = 'N/A'
        
        return export_df
        
    def hide_results(self):
        """Hide the results section"""
        if hasattr(self, 'results_frame'):
            self.results_frame.grid_remove()
        self.results_visible = False
    
    def update_status(self, message):
        self.root.after(0, lambda: self.status_label.config(text=message))
    
    def update_progress(self, value):
        self.root.after(0, lambda: self.progress_var.set(value))

def main():
    root = tk.Tk()
    app = QAEvaluationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()