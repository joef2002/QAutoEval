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

        self.root.title("GenAI for Reticular Chemistry")
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
        
        # Main evaluation type - now includes 'generation' as default
        self.evaluation_type = tk.StringVar(value="generation")  # "generation", "qa", or "synthesis"
        
        # Q&A specific variables
        self.evaluation_mode = tk.StringVar(value="single-hop")
        
        # Dataset generation specific variables
        self.dataset_type = tk.StringVar(value="single-hop-qa")  # "single-hop-qa", "multi-hop-qa", "synthesis-condition"
        
        # Results storage
        self.evaluation_results = None
        self.generation_results = None  # For dataset generation results
        
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
        
        # Mode selection (for Q&A evaluation and dataset generation) - row 1
        self.setup_mode_selection(main_frame)
        
        # File uploads - rows 2-4
        self.setup_file_uploads(main_frame)

        # Control buttons - row 5
        self.button_frame = ttk.Frame(main_frame)
        self.button_frame.grid(row=5, column=0, columnspan=3, pady=20)
        
        # Start button (text changes based on mode)
        self.start_button = ttk.Button(self.button_frame, text="Start Generation", command=self.start_operation, width=15)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(self.button_frame, text="Start New Run", command=self.start_new_run, width=15).grid(row=0, column=1)

        # Progress and status - rows 6-7
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        self.status_label = ttk.Label(main_frame, text="Ready to generate dataset")
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
        
        # Logo frame (left side) - now contains single logo that changes
        logo_frame = ttk.Frame(header_frame)
        logo_frame.grid(row=0, column=0, padx=(0, 20))
        
        # Store logo frame reference for later updates
        self.logo_frame = logo_frame
        
        # Load both logos but don't display them yet
        try:
            # Calculate size to match button dimensions (width=20 chars ≈ 160px, height=2 lines ≈ 50px)
            container_width = 140
            container_height = 50
            self.container_width = container_width
            self.container_height = container_height

            # Load retchemqa logo for generation mode
            try:
                retchemqa_image = Image.open('docs/images/retchemqa-logo_720.png')
                retchemqa_image.thumbnail((container_width, container_height), Image.Resampling.LANCZOS)
                self.retchemqa_photo = ImageTk.PhotoImage(retchemqa_image)
            except Exception as e:
                print(f"Could not load retchemqa logo: {e}")
                self.retchemqa_photo = None

            # Load main logo for evaluation modes  
            try:
                logo_image = Image.open('docs/images/logo.png')
                logo_image.thumbnail((container_width, container_height), Image.Resampling.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo_image)
            except Exception as e:
                print(f"Could not load main logo: {e}")
                self.logo_photo = None
                
        except Exception as e:
            print(f"Error in logo setup: {e}")
            self.retchemqa_photo = None
            self.logo_photo = None
        
        # Create the logo label (will be updated based on mode)
        self.logo_label = None
        self.update_logo_display()
        
        # Evaluation type buttons (center)
        eval_buttons_frame = ttk.Frame(header_frame)
        eval_buttons_frame.grid(row=0, column=1)
        
        # Dataset Generation button (FIRST)
        self.generation_button = tk.Button(eval_buttons_frame, text="Dataset Generation", 
                                        command=lambda: self.set_evaluation_type("generation"),
                                        font=("Arial", 12, "bold"), width=18, height=2)
        self.generation_button.grid(row=0, column=0, padx=(0, 5))
        
        # Q&A Evaluation button
        self.qa_button = tk.Button(eval_buttons_frame, text="Q&A Pairs Evaluation", 
                                command=lambda: self.set_evaluation_type("qa"),
                                font=("Arial", 12, "bold"), width=20, height=2)
        self.qa_button.grid(row=0, column=1, padx=(5, 0))
        
        # Synthesis Condition Evaluation button
        self.synthesis_button = tk.Button(eval_buttons_frame, text="Synthesis Condition Evaluation", 
                                        command=lambda: self.set_evaluation_type("synthesis"),
                                        font=("Arial", 12, "bold"), width=25, height=2)
        self.synthesis_button.grid(row=0, column=2, padx=(5, 0))
        
        # Settings button (top-right)
        settings_btn = ttk.Button(header_frame, text="Settings", command=self.open_settings, width=10)
        settings_btn.grid(row=0, column=2, sticky=tk.E)
        
        # Update button colors
        self.update_button_colors()

    def update_logo_display(self):
        """Update which logo is displayed based on the current evaluation type"""
        # Remove existing logo label if it exists
        if hasattr(self, 'logo_label') and self.logo_label:
            self.logo_label.destroy()
            self.logo_label = None
        
        # Make sure we have the logo frame
        if not hasattr(self, 'logo_frame'):
            return
        
        # Determine which logo to show based on evaluation type
        if self.evaluation_type.get() == "generation":
            # Show retchemqa logo for generation mode
            if hasattr(self, 'retchemqa_photo') and self.retchemqa_photo:
                self.logo_label = tk.Label(
                    self.logo_frame, 
                    image=self.retchemqa_photo,
                    width=self.container_width,
                    height=self.container_height,
                    compound='center'
                )
                self.logo_label.grid(row=0, column=0)
            else:
                # Fallback placeholder for retchemqa logo
                self.logo_label = tk.Label(
                    self.logo_frame,
                    text="RetChemQA\nLogo",
                    width=20,
                    height=2,
                    font=("Arial", 10, "bold"),
                    relief="raised",
                    borderwidth=1,
                    bg="lightblue",
                    fg="darkblue",
                    justify='center'
                )
                self.logo_label.grid(row=0, column=0)
        else:
            # Show main logo for evaluation modes (qa and synthesis)
            if hasattr(self, 'logo_photo') and self.logo_photo:
                self.logo_label = tk.Label(
                    self.logo_frame, 
                    image=self.logo_photo,
                    width=self.container_width,
                    height=self.container_height,
                    compound='center'
                )
                self.logo_label.grid(row=0, column=0)
            else:
                # Fallback placeholder for main logo
                self.logo_label = tk.Label(
                    self.logo_frame,
                    text="QAutoEval\nLogo",
                    width=20,
                    height=2,
                    font=("Arial", 10, "bold"),
                    relief="raised",
                    borderwidth=1,
                    bg="lightgray",
                    fg="darkblue",
                    justify='center'
                )
                self.logo_label.grid(row=0, column=0)

    def set_evaluation_type(self, eval_type):
        """Set the evaluation type and update UI accordingly"""
        self.evaluation_type.set(eval_type)
        self.update_button_colors()
        self.update_logo_display()  # Update logo when evaluation type changes
        self.update_ui_for_evaluation_type()
        self.hide_results()  # Hide any existing results
        
        # Update status and button text based on type
        if eval_type == "generation":
            self.update_status("Ready to generate dataset")
            self.start_button.config(text="Start Generation")
        else:
            self.update_status("Ready to evaluate")
            self.start_button.config(text="Start Evaluation")
        self.update_progress(0)
    
    def update_button_colors(self):
        """Update button selection status using highlight borders only"""
        try:
            # Reset all buttons to unselected state
            for button in [self.generation_button, self.qa_button, self.synthesis_button]:
                button.config(
                    bg="SystemButtonFace",
                    fg="SystemButtonText",
                    relief="raised", 
                    highlightbackground="SystemButtonFace",
                    highlightcolor="SystemButtonFace",
                    highlightthickness=1,
                    borderwidth=1,
                    state='normal'
                )
            
            # Set selected button
            selected_button = None
            if self.evaluation_type.get() == "generation":
                selected_button = self.generation_button
            elif self.evaluation_type.get() == "qa":
                selected_button = self.qa_button
            elif self.evaluation_type.get() == "synthesis":
                selected_button = self.synthesis_button
            
            if selected_button:
                selected_button.config(
                    highlightbackground="#1f5582",
                    highlightcolor="#1f5582",
                    highlightthickness=3,
                    borderwidth=3
                )
            
            # Force update the display
            for button in [self.generation_button, self.qa_button, self.synthesis_button]:
                button.update_idletasks()
            
        except Exception as e:
            print(f"Button highlight update failed: {e}")
            # Fallback: use relief and borderwidth only
            try:
                for button in [self.generation_button, self.qa_button, self.synthesis_button]:
                    button.config(relief="raised", borderwidth=1)
                
                if self.evaluation_type.get() == "generation":
                    self.generation_button.config(relief="sunken", borderwidth=3)
                elif self.evaluation_type.get() == "qa":
                    self.qa_button.config(relief="sunken", borderwidth=3)
                elif self.evaluation_type.get() == "synthesis":
                    self.synthesis_button.config(relief="sunken", borderwidth=3)
            except:
                pass
    
    def update_ui_for_evaluation_type(self):
        """Update UI elements based on the selected evaluation type and manage grid layout dynamically"""
        # Update mode widgets first
        self.update_mode_widgets()
        
        if self.evaluation_type.get() == "qa":
            # Show mode selection for Q&A
            if hasattr(self, 'mode_frame'):
                self.mode_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
            # Update dataset label
            if hasattr(self, 'dataset_label'):
                self.dataset_label.config(text="Q&A Dataset (JSON)*:")
            # Show dataset file selection
            if hasattr(self, 'dataset_label'):
                self.dataset_label.grid(row=4, column=0, sticky=tk.W, pady=5)
                self.dataset_entry.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 5))
                self.dataset_btn_frame.grid(row=4, column=2, pady=5)
            # Adjust grid layout - file uploads start at row 2
            self._update_file_upload_positions(start_row=2)
            # Set results frame row to 8 (after mode selection)
            results_row = 8
        elif self.evaluation_type.get() == "synthesis":
            # Hide mode selection for synthesis
            if hasattr(self, 'mode_frame'):
                self.mode_frame.grid_remove()
            # Update dataset label
            if hasattr(self, 'dataset_label'):
                self.dataset_label.config(text="Synthesis Condition Dataset (JSON)*:")
            # Show dataset file selection
            if hasattr(self, 'dataset_label'):
                self.dataset_label.grid(row=3, column=0, sticky=tk.W, pady=5)
                self.dataset_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 5))
                self.dataset_btn_frame.grid(row=3, column=2, pady=5)
            # Adjust grid layout - file uploads start at row 1 (no mode selection)
            self._update_file_upload_positions(start_row=1)
            # Set results frame row to 7 (no mode selection)
            results_row = 7
        elif self.evaluation_type.get() == "generation":
            # Show dataset type selection for generation
            if hasattr(self, 'mode_frame'):
                self.mode_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
            # Hide dataset file selection for generation
            if hasattr(self, 'dataset_label'):
                self.dataset_label.grid_remove()
                self.dataset_entry.grid_remove()
                self.dataset_btn_frame.grid_remove()
            # Adjust grid layout - file uploads start at row 2
            self._update_file_upload_positions(start_row=2, hide_dataset=True)
            # Set results frame row to 7 (after dataset type selection, no dataset file)
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
    
    def show_custom_messagebox(self, title, message, msg_type="info"):
        """Show a custom messagebox with proper text wrapping and sizing"""
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.transient(self.root)
        popup.resizable(True, True)
        popup.grab_set()

        # Calculate window size based on message length
        lines = message.count('\n') + 1
        estimated_width = min(max(len(message) // lines * 8, 400), 800)
        estimated_height = min(max(lines * 25 + 100, 150), 600)
        
        popup.geometry(f"{estimated_width}x{estimated_height}")
        
        # Center the window
        popup.geometry("+%d+%d" % (
            self.root.winfo_rootx() + (self.root.winfo_width() - estimated_width) // 2,
            self.root.winfo_rooty() + (self.root.winfo_height() - estimated_height) // 2
        ))
        
        # Main frame with padding
        main_frame = ttk.Frame(popup, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Scrollable text area
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Message label with proper wrapping
        msg_label = ttk.Label(
            scrollable_frame,
            text=message,
            wraplength=estimated_width - 80,
            justify="left",
            font=("Arial", 10)
        )
        msg_label.pack(fill=tk.BOTH, expand=True, pady=10)
        
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        # OK button
        ok_button = ttk.Button(button_frame, text="OK", command=popup.destroy)
        ok_button.pack()
        
        # Focus on OK button
        ok_button.focus_set()
        popup.bind('<Return>', lambda e: popup.destroy())
        popup.bind('<Escape>', lambda e: popup.destroy())
    
    def show_info_popup(self):
        """Custom pop-up for additional information"""
        self.show_custom_messagebox(
            "Need More Detail?",
            "For a more detailed summary sheet with each LLM's individual "
            "evaluation and explanations, please export the results to an "
            "Excel file."
        )
    
    def setup_mode_selection(self, parent):
        # Mode selection frame (for Q&A evaluation and dataset generation)
        self.mode_frame = ttk.LabelFrame(parent, text="Evaluation Mode", padding="10")
        self.mode_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Different radio buttons based on evaluation type
        self.mode_widgets = {}  # Store references to mode widgets
        
        # Q&A Evaluation radio buttons
        self.mode_widgets['qa_single'] = ttk.Radiobutton(self.mode_frame, text="Single-hop Q&A Evaluation", 
                                      variable=self.evaluation_mode, value="single-hop",
                                      command=self.on_mode_change)
        
        self.mode_widgets['qa_multi'] = ttk.Radiobutton(self.mode_frame, text="Multi-hop Q&A Evaluation", 
                                     variable=self.evaluation_mode, value="multi-hop",
                                     command=self.on_mode_change)
        
        # Dataset Generation radio buttons
        self.mode_widgets['gen_single'] = ttk.Radiobutton(self.mode_frame, text="Single-hop Q&A Generation", 
                                       variable=self.dataset_type, value="single-hop-qa",
                                       command=self.on_dataset_type_change)
        
        self.mode_widgets['gen_multi'] = ttk.Radiobutton(self.mode_frame, text="Multi-hop Q&A Generation", 
                                      variable=self.dataset_type, value="multi-hop-qa",
                                      command=self.on_dataset_type_change)
        
        self.mode_widgets['gen_synthesis'] = ttk.Radiobutton(self.mode_frame, text="Synthesis Condition Generation", 
                                          variable=self.dataset_type, value="synthesis-condition",
                                          command=self.on_dataset_type_change)
        
        # Mode description
        self.mode_description = ttk.Label(self.mode_frame, text="", foreground="blue", font=("Arial", 9))
        self.mode_description.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
        
        # Initialize description and layout
        self.update_mode_widgets()
    
    def update_mode_widgets(self):
        """Update which mode widgets are visible based on evaluation type"""
        # Hide all mode widgets first
        for widget in self.mode_widgets.values():
            widget.grid_remove()
        
        if self.evaluation_type.get() == "qa":
            # Show Q&A evaluation options
            self.mode_frame.config(text="Evaluation Mode")
            self.mode_widgets['qa_single'].grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
            self.mode_widgets['qa_multi'].grid(row=0, column=1, sticky=tk.W)
            self.on_mode_change()
        elif self.evaluation_type.get() == "generation":
            # Show dataset generation options
            self.mode_frame.config(text="Generation Mode")
            self.mode_widgets['gen_single'].grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
            self.mode_widgets['gen_multi'].grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
            self.mode_widgets['gen_synthesis'].grid(row=0, column=2, sticky=tk.W)
            self.on_dataset_type_change()
    
    def on_mode_change(self):
        """Update the mode description when Q&A evaluation selection changes"""
        if self.evaluation_mode.get() == "single-hop":
            description = "Single-hop mode: Evaluates Q&A pairs that require direct information from the context without reasoning chains."
        else:
            description = "Multi-hop mode: Evaluates Q&A pairs that require reasoning across multiple pieces of information or inference steps."
        
        self.mode_description.config(text=description)
    
    def on_dataset_type_change(self):
        """Update the description when dataset generation type changes"""
        if self.dataset_type.get() == "single-hop-qa":
            description = "Generate 20 single-hop Q&A pairs with a balanced mix of question types that require direct information from the manuscript without reasoning chains."
        elif self.dataset_type.get() == "multi-hop-qa":
            description = "Generate 20 multi-hop Q&A pairs with a balanced mix of question types that require reasoning across multiple pieces of information or inference steps."
        else:  # synthesis-condition
            description = "Generate synthesis condition dataset by extracting all material synthesis procedures from the research papers (excluding characterization data)."
        
        self.mode_description.config(text=description)
    
    def _update_file_upload_positions(self, start_row, hide_dataset=False):
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
            
        # Dataset row - only show if not hidden
        dataset_row = start_row + 2
        if not hide_dataset:
            if hasattr(self, 'dataset_label'):
                self.dataset_label.grid(row=dataset_row, column=0, sticky=tk.W, pady=5)
            if hasattr(self, 'dataset_entry'):
                self.dataset_entry.grid(row=dataset_row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 5))
            if hasattr(self, 'dataset_btn_frame'):
                self.dataset_btn_frame.grid(row=dataset_row, column=2, pady=5)
            button_row = start_row + 3
        else:
            button_row = start_row + 2
            
        # Update button frame position
        if hasattr(self, 'button_frame'):
            self.button_frame.grid(row=button_row, column=0, columnspan=3, pady=20)
            
        # Update progress bar position  
        if hasattr(self, 'progress_bar'):
            self.progress_bar.grid(row=button_row+1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
            
        # Update status label position
        if hasattr(self, 'status_label'):
            self.status_label.grid(row=button_row+2, column=0, columnspan=3, pady=5)
    
    def setup_file_uploads(self, parent):
        # Manuscript upload
        self.manuscript_label = ttk.Label(parent, text="MS*:")
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
        self.dataset_label = ttk.Label(parent, text="Q&A Dataset (JSON)*:")
        self.dataset_entry = ttk.Entry(parent, textvariable=self.dataset_path, width=40)
        
        # Dataset buttons frame
        self.dataset_btn_frame = ttk.Frame(parent)
        ttk.Button(self.dataset_btn_frame, text="Browse", command=lambda: self.browse_json_file(self.dataset_path), width=10).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(self.dataset_btn_frame, text="Clear", command=lambda: self.dataset_path.set(""), width=8).grid(row=0, column=1)
        
        # Initial positioning (will be updated by update_ui_for_evaluation_type)
        self._update_file_upload_positions(start_row=2)
    
    def start_operation(self):
        """Start either evaluation or generation based on current mode"""
        if self.evaluation_type.get() == "generation":
            self.start_generation()
        else:
            self.start_evaluation()
    
    def start_generation(self):
        """Start dataset generation"""
        try:
            if not self.validate_generation_inputs():
                return
            
            # Show that generation is starting
            dataset_type_text = self.get_dataset_type_display_name()
            
            self.update_status(f"Starting {dataset_type_text} dataset generation...")
            self.update_progress(5)
            
            # Start generation in a separate thread to prevent GUI freezing
            thread = threading.Thread(target=self.run_generation)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            error_msg = (
                "Failed to start generation:\n\n"
                f"{str(e)}\n\n"
                "Please check your inputs and try again."
            )
            self.show_custom_messagebox("Generation Error", error_msg, "error")
            self.update_status("Ready to generate dataset")
            self.update_progress(0)
    
    def get_dataset_type_display_name(self):
        """Get display name for current dataset type"""
        type_map = {
            "single-hop-qa": "Single-hop Q&A",
            "multi-hop-qa": "Multi-hop Q&A",
            "synthesis-condition": "Synthesis Condition"
        }
        return type_map.get(self.dataset_type.get(), "Dataset")
    
    def validate_generation_inputs(self):
        """Validate inputs for dataset generation"""
        try:
            # Check if manuscript file is provided and exists
            if not self.manuscript_path.get():
                self.show_custom_messagebox("File Required", "Please select a manuscript file.", "error")
                return False
            
            if not os.path.exists(self.manuscript_path.get()):
                self.show_custom_messagebox("File Not Found", "Manuscript file does not exist.", "error")
                return False
            
            # For generation, supplement is optional but recommended
            # No popup needed - just proceed without SI if not provided
            
            # Check that Gemini API key is provided (only Gemini is used for generation)
            if not self.gemini_key.get().strip():
                error_msg = (
                    "Please enter your Gemini API key in Settings.\n\n"
                    "Gemini is required for dataset generation.\n\n"
                    "You can access Settings by clicking the 'Settings' button in the top-right corner."
                )
                self.show_custom_messagebox("Gemini API Key Required", error_msg, "error")
                return False
            
            return True
            
        except ValueError as e:
            self.show_custom_messagebox("Validation Error", str(e), "error")
            return False
        except Exception as e:
            error_msg = (
                "An unexpected error occurred during validation:\n\n"
                f"{str(e)}\n\n"
                "Please check your inputs and try again."
            )
            self.show_custom_messagebox("Unexpected Error", error_msg, "error")
            return False
    
    def run_generation(self):
        """Run the dataset generation process"""
        try:
            dataset_type_text = self.get_dataset_type_display_name()
            
            self.update_status(f"Loading manuscript and supplement files for {dataset_type_text} generation...")
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
            self.update_status(f"Loaded {files_text}. Generating {dataset_type_text} dataset...")
            self.update_progress(30)
            
            # Generate dataset using Gemini
            generation_prompt = self.get_generation_prompt()
            
            self.update_status(f"Running Gemini for {dataset_type_text} dataset generation using {files_text}...")
            self.update_progress(50)
            
            # Run generation with retries for Q&A to ensure 20 pairs
            generated_data = self.run_gemini_generation(generation_prompt, context)
            
            if generated_data is None:
                error_msg = (
                    "Dataset generation failed.\n\n"
                    "Please check:\n"
                    "• Your Gemini API key is valid\n"
                    "• You have sufficient API credits\n"
                    "• Your internet connection is stable\n"
                    "• The manuscript contains sufficient content for generation"
                )
                raise ValueError(error_msg)
            
            self.update_progress(90)
            
            # Store results
            self.generation_results = generated_data
            
            self.update_progress(100)
            self.update_status(f"{dataset_type_text} dataset generation completed successfully!")
            
            # Show results in the same window
            self.root.after(0, self.show_generation_results_inline)
            
        except ValueError as e:
            # Known validation errors
            self.root.after(0, lambda: self.show_custom_messagebox("Generation Error", str(e), "error"))
            self.update_status("Generation failed")
            self.update_progress(0)
        except Exception as e:
            # Unexpected errors
            error_msg = (
                "An unexpected error occurred during generation:\n\n"
                f"{str(e)}\n\n"
                "Please check your inputs and try again."
            )
            self.root.after(0, lambda: self.show_custom_messagebox("Unexpected Error", error_msg, "error"))
            self.update_status("Generation failed")
            self.update_progress(0)
    
    def get_generation_prompt(self):
        """Get the appropriate generation prompt based on dataset type"""
        if self.dataset_type.get() == "single-hop-qa":
            return self.get_single_hop_generation_prompt()
        elif self.dataset_type.get() == "multi-hop-qa":
            return self.get_multi_hop_generation_prompt()
        else:  # synthesis-condition
            return self.get_synthesis_generation_prompt()
    
    def get_single_hop_generation_prompt(self):
        """Single-hop Q&A generation prompt"""
        return """You are a single hop Question and Answering (Q&A) dataset generation agent. A single hop question and answer set is one that requires a single step of reasoning. You are required to go through the given text and identify the synthesis conditions and based on those synthesis conditions develop a set of 20 Q&As. There may be information about the synthesis conditions of more than one material in the text. For example, you may come across a series of different materials such as ZIF-1, ZIF-2, .... ZIF-12. Please try to diversify the types of questions that you include. Please also try to include a question for each material you come across in the paper. Please feel free to include labels that are also used in some of the most widely used Q&A datasets e.g., the question, the answer, the difficulty level, and the type of question. the different types of questions are factual, reasoning (single step reasoning), and True or False. Please generate 6 'factual' type questions, 7 'reasoning' type questions, and 7 True or False type questions.

Generate a single hop .json file for the following text. Please include questions of different types including factual (6 questions), single-step reasoning (7 questions), and True or False (7 questions).

Format each question-answer pair as a JSON object with the following structure:
{
    "question": "your question here",
    "answer": "your answer here",
    "question_type": "factual/reasoning/True or False",
    "difficulty_level": "easy/medium/hard"
}

Generate exactly 20 such pairs and return them as a JSON array."""
    
    def get_multi_hop_generation_prompt(self):
        """Multi-hop Q&A generation prompt"""
        return """You are a multi-hop Question and Answering (Q&A) dataset generation agent. A multi-hop Q&A is one that requires multi-step reasoning to come to an answer (this information can come from any part of the paper, both MS and SI). To give you more details: A multi-hop Q&A will always involve going through multiple parts of the paper to come to an answer. This may include different paragraphs, different pages, and also different documents (i.e., the manuscript and the supplementary information). You are required to go through the given text and identify the synthesis conditions and based on those synthesis conditions develop a set of multi-hop (questions that require multiple steps of reasoning) 20 Q&As for each DOI. There may be information about the synthesis conditions of more than one material in the text. For example, you may come across a series of different materials such as ZIF-1, ZIF-2, .... ZIF-12. Please diversify the type of questions to encompass different ideas and materials. Please feel free to include labels that are also used in some of the most widely used Q&A dataset for e.g., the question, the answer, the difficulty level, and the type of question. The different types of questions are factual, reasoning (single step reasoning), and True or False. Please generate 6 'factual' type questions, 7 'reasoning' type questions, and 7 True or False type questions. For factual questions, please try to be creative with the questions as it should require information from different parts of the text to answer.

Generate a multi-hop Q&A json file for the following text. Please include questions of different types including factual (6 questions), single-step reasoning (7 questions), and True or False (7 questions).

Format each question-answer pair as a JSON object with the following structure:
{
    "question": "your question here",
    "answer": "your answer here",
    "question_type": "factual/reasoning/True or False",
    "difficulty_level": "easy/medium/hard"
}

Generate exactly 20 such pairs and return them as a JSON array."""
    
    def get_synthesis_generation_prompt(self):
        """Synthesis condition generation prompt"""
        return """You are a synthesis condition classification agent. You are required to go through the given text and identify the synthesis conditions for each and every material given in the paper (both MS and SI, if available). There may be information about the synthesis condition of more than one material. Please make sure to separate these materials when generating the .json file. For each material try to classify the conditions under different labels such as temperature, solvents, the amount of each solvent (this is important), equipment, chemicals used, time, washing method, drying method, yield, etc. This is not an exhaustive list of labels, please feel free to add more labels as required. Some synthesis conditions may involve multiple steps, please take that into account. Please do not include any experimental characterization data such as those from Powder X-Ray Diffraction (PXRD), Infrared (IR) spectra, adsorption isotherms, thermogravimetric analysis (TGA), nuclear magnetic resonance (NMR) experiments, etc. (this is not an exhaustive list and there may be other characterization techniques) including information about its properties. I reiterate, please do not include any experimental characterization data.

Generate a machine readable .json file containing the synthesis conditions for the following text.

Format the output as a JSON array where each object represents a material and its synthesis conditions:
{
    "material_name": "name of the material",
    "synthesis_conditions": {
        "temperature": "synthesis temperature",
        "solvents": "solvents used and their amounts",
        "chemicals_used": "precursor chemicals and amounts",
        "time": "reaction time",
        "equipment": "equipment used",
        "washing_method": "washing procedure",
        "drying_method": "drying procedure",
        "yield": "synthesis yield",
        "additional_steps": "any additional synthesis steps"
    }
}

Extract synthesis conditions for all materials found in the context."""
    
    def run_gemini_generation(self, prompt, context):
        """Run Gemini for dataset generation with retries for Q&A only"""
        try:
            if not self.gemini_key.get().strip():
                return None
                
            genai.configure(api_key=self.gemini_key.get().strip())
            
            # Create the generation prompt based on dataset type
            if self.dataset_type.get() == "single-hop-qa":
                user_prompt = f"Generate a single hop .json file for the following text. Please include questions of different types including factual (6 questions), single-step reasoning (7 questions), and True or False (7 questions): {context}"
            elif self.dataset_type.get() == "multi-hop-qa":
                user_prompt = f"Generate a multi-hop Q&A json file for the following text. Please include questions of different types including factual (6 questions), single-step reasoning (7 questions), and True or False (7 questions): {context}"
            else:  # synthesis-condition
                user_prompt = f"Generate a machine readable .json file containing the synthesis conditions for the following text: {context}"
            
            # Use the system prompt as system instruction
            gemini_client = genai.GenerativeModel(
                model_name='gemini-1.5-pro-latest',
                system_instruction=prompt
            )
            
            # For Q&A generation, retry until we get exactly 20 pairs
            if self.dataset_type.get() in ["single-hop-qa", "multi-hop-qa"]:
                max_attempts = 5
                for attempt in range(max_attempts):
                    try:
                        result = gemini_client.generate_content(
                            user_prompt,
                            generation_config=genai.GenerationConfig(
                                response_mime_type="application/json"
                            ),
                        )
                        
                        response_text = result.candidates[0].content.parts[0].text
                        generated_data = json.loads(response_text)
                        
                        # Check if we got exactly 20 pairs
                        if isinstance(generated_data, list) and len(generated_data) == 20:
                            # Validate that each pair has required fields
                            valid_pairs = []
                            for pair in generated_data:
                                if isinstance(pair, dict) and 'question' in pair and 'answer' in pair:
                                    # Ensure all required fields are present
                                    if 'question_type' not in pair:
                                        pair['question_type'] = 'factual'
                                    if 'difficulty_level' not in pair:
                                        pair['difficulty_level'] = 'medium'
                                    valid_pairs.append(pair)
                            
                            if len(valid_pairs) == 20:
                                return valid_pairs
                        
                        print(f"Attempt {attempt + 1}: Got {len(generated_data) if isinstance(generated_data, list) else 0} pairs instead of 20. Retrying...")
                        
                    except json.JSONDecodeError as e:
                        print(f"Attempt {attempt + 1}: JSON parsing failed: {e}. Retrying...")
                    except Exception as e:
                        print(f"Attempt {attempt + 1}: Generation failed: {e}. Retrying...")
                
                # If all attempts failed, return None
                print("Failed to generate exactly 20 Q&A pairs after all attempts")
                return None
            
            else:
                # For synthesis condition generation, single attempt - no need to check results
                result = gemini_client.generate_content(
                    user_prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json"
                    ),
                )
                
                response_text = result.candidates[0].content.parts[0].text
                generated_data = json.loads(response_text)
                
                # Return whatever is generated for synthesis conditions
                return generated_data
                
        except Exception as e:
            print(f"Gemini generation failed: {e}")
            return None
    
    def show_generation_results_inline(self):
        """Display generation results in the main window"""
        if self.generation_results is None:
            return
        
        # Use the dynamically calculated results row
        results_row = getattr(self, 'results_row', 7)
        self.results_frame.config(text="Dataset Generation Results")
        self.results_frame.grid(row=results_row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(20, 0))
        self.results_visible = True
        
        # Clear any existing content
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Generation stats
        self.show_generation_stats_inline(self.results_frame)
        
        # JSON display and export section
        json_frame = ttk.Frame(self.results_frame)
        json_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(20, 0))
        json_frame.columnconfigure(0, weight=1)
        json_frame.rowconfigure(1, weight=1)
        
        # Controls frame
        controls_frame = ttk.Frame(json_frame)
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        controls_frame.columnconfigure(1, weight=1)
        
        # Show JSON toggle
        self.json_visible = tk.BooleanVar()
        json_btn = ttk.Checkbutton(
            controls_frame,
            text="Show Generated JSON",
            variable=self.json_visible,
            command=lambda: self.toggle_json_display(json_frame),
        )
        json_btn.grid(row=0, column=0, sticky=tk.W)
        
        # Export JSON button
        export_btn = ttk.Button(
            controls_frame, text="Export JSON", 
            command=lambda: self.export_generation_json(), width=15
        )
        export_btn.grid(row=0, column=2, sticky=tk.E)
        
        print("Export JSON button created and connected")  # Debug print
        
        # JSON content (initially hidden)
        self.json_content = ttk.Frame(json_frame)
    
    def show_generation_stats_inline(self, parent):
        """Show generation statistics"""
        stats_frame = ttk.LabelFrame(parent, text="Generation Statistics", padding="10")
        stats_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        dataset_type_text = self.get_dataset_type_display_name()
        
        # Dataset type info
        ttk.Label(stats_frame, text=f"Dataset Type: {dataset_type_text}", 
                 font=("Arial", 12, "bold"), foreground="blue").grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))
        
        # Generation stats (remove success indicator)
        if isinstance(self.generation_results, list):
            count = len(self.generation_results)
            ttk.Label(stats_frame, text="Generated Items:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, pady=5)
            ttk.Label(stats_frame, text=str(count)).grid(row=1, column=1, sticky=tk.W, pady=5, padx=(20, 0))
            
            if self.dataset_type.get() in ["single-hop-qa", "multi-hop-qa"]:
                # Show Q&A specific stats - count actual generated question types
                question_types = {}
                difficulty_levels = {}
                
                # Count each question type and difficulty level from actual results
                for item in self.generation_results:
                    if isinstance(item, dict):
                        # Count question types
                        if 'question_type' in item and item['question_type']:
                            qtype = str(item['question_type']).strip()
                            question_types[qtype] = question_types.get(qtype, 0) + 1
                        else:
                            # Handle missing question_type
                            question_types['unspecified'] = question_types.get('unspecified', 0) + 1
                        
                        # Count difficulty levels
                        if 'difficulty_level' in item and item['difficulty_level']:
                            difficulty = str(item['difficulty_level']).strip()
                            difficulty_levels[difficulty] = difficulty_levels.get(difficulty, 0) + 1
                        else:
                            # Handle missing difficulty_level
                            difficulty_levels['unspecified'] = difficulty_levels.get('unspecified', 0) + 1
                
                # Display actual question type counts
                if question_types:
                    ttk.Label(stats_frame, text="Question Types:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=5)
                    # Sort for consistent display
                    sorted_types = sorted(question_types.items())
                    type_text = ", ".join([f"{k}: {v}" for k, v in sorted_types])
                    ttk.Label(stats_frame, text=type_text).grid(row=2, column=1, sticky=tk.W, pady=5, padx=(20, 0))
                
                # Display actual difficulty level counts
                if difficulty_levels:
                    ttk.Label(stats_frame, text="Difficulty Levels:", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky=tk.W, pady=5)
                    # Sort for consistent display
                    sorted_difficulties = sorted(difficulty_levels.items())
                    difficulty_text = ", ".join([f"{k}: {v}" for k, v in sorted_difficulties])
                    ttk.Label(stats_frame, text=difficulty_text).grid(row=3, column=1, sticky=tk.W, pady=5, padx=(20, 0))
                
                # Add note if distribution doesn't match expected
                expected_total = 20
                actual_total = sum(question_types.values())
                if actual_total != expected_total:
                    note_label = ttk.Label(stats_frame, text=f"Note: Generated {actual_total} items (expected {expected_total})", 
                                         font=("Arial", 9), foreground="orange")
                    note_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)
            else:
                # Show synthesis specific stats
                material_count = 0
                if isinstance(self.generation_results, list):
                    material_count = len([item for item in self.generation_results if isinstance(item, dict) and ('material_name' in item or 'mof_name' in item)])
                elif isinstance(self.generation_results, dict):
                    material_count = 1
                
                ttk.Label(stats_frame, text="Materials Found:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=5)
                ttk.Label(stats_frame, text=str(material_count)).grid(row=2, column=1, sticky=tk.W, pady=5, padx=(20, 0))
    
    def toggle_json_display(self, parent):
        """Toggle JSON display visibility"""
        if self.json_visible.get():
            self.show_json_display(parent)
        else:
            self.hide_json_display()
    
    def show_json_display(self, parent):
        """Show the generated JSON data"""
        self.json_content.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        self.json_content.columnconfigure(0, weight=1)
        self.json_content.rowconfigure(0, weight=1)
        
        # Create scrollable text widget for JSON
        canvas = tk.Canvas(self.json_content, height=400)
        scrollbar = ttk.Scrollbar(self.json_content, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add JSON text
        json_text = tk.Text(scrollable_frame, wrap=tk.WORD, font=("Courier", 10))
        json_text.pack(fill=tk.BOTH, expand=True)
        
        # Insert formatted JSON
        formatted_json = json.dumps(self.generation_results, indent=2)
        json_text.insert("1.0", formatted_json)
        json_text.config(state=tk.DISABLED)  # Make read-only
        
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    def hide_json_display(self):
        """Hide the JSON display"""
        if hasattr(self, 'json_content'):
            self.json_content.grid_remove()
    
    def export_generation_json(self):
        """Export generated dataset to JSON file"""
        try:
            print("Export JSON button clicked")  # Debug print
            
            if self.generation_results is None:
                print("No generation results found")  # Debug print
                self.show_custom_messagebox(
                    "No Results", 
                    "No generation results to export.\n\nPlease run a generation first.",
                    "warning"
                )
                return
            
            print(f"Generation results found: {len(self.generation_results) if isinstance(self.generation_results, list) else 'single item'}")  # Debug print
            
            # Default filename based on dataset type
            dataset_type_text = self.get_dataset_type_display_name().lower().replace(" ", "_").replace("-", "_")
            default_filename = f"{dataset_type_text}_dataset.json"
            
            print(f"Opening file dialog with default name: {default_filename}")  # Debug print
            
            # Make sure the dialog appears on top
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.attributes('-topmost', False)
            
            filename = filedialog.asksaveasfilename(
                parent=self.root,
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Generated Dataset",
                initialvalue=default_filename
            )
            
            print(f"User selected filename: {filename}")  # Debug print
            
            if not filename:
                print("User cancelled file dialog")  # Debug print
                return
            
            print("Attempting to save file...")  # Debug print
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.generation_results, f, indent=2, ensure_ascii=False)
            
            print(f"File saved successfully: {filename}")  # Debug print
            
            success_msg = (
                "Generated dataset exported successfully!\n\n"
                f"File saved to:\n{filename}"
            )
            self.show_custom_messagebox("Export Successful", success_msg, "info")
            
        except Exception as e:
            print(f"Export error occurred: {str(e)}")  # Debug print
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            error_msg = (
                "Failed to export dataset:\n\n"
                f"{str(e)}\n\n"
                "Please check that:\n"
                "• You have write permissions to the selected location\n"
                "• The file is not currently open in another program\n"
                "• You have sufficient disk space"
            )
            self.show_custom_messagebox("Export Error", error_msg, "error")
    
    # [Keep all existing methods unchanged...]
    
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
        """Clear all file inputs but keep API keys and evaluation settings for all modes"""
        self.manuscript_path.set("")
        self.supplement_path.set("")
        if self.evaluation_type.get() != "generation":  # Only clear dataset path for evaluation modes
            self.dataset_path.set("")
        self.evaluation_results = None
        self.generation_results = None
        self.hide_results()
        
        if self.evaluation_type.get() == "generation":
            self.update_status("Ready for new dataset generation")
            self.start_button.config(text="Start Generation")
        else:
            eval_type_text = "Q&A" if self.evaluation_type.get() == "qa" else "synthesis condition"
            self.update_status(f"Ready for new {eval_type_text} evaluation")
            self.start_button.config(text="Start Evaluation")
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
                self.show_custom_messagebox("File Required", "Please select a manuscript file.", "error")
                return False
            
            if not os.path.exists(self.manuscript_path.get()):
                self.show_custom_messagebox("File Not Found", "Manuscript file does not exist.", "error")
                return False
            
            # Check if dataset is provided and exists
            if not self.dataset_path.get():
                dataset_type = "Q&A dataset" if self.evaluation_type.get() == "qa" else "synthesis condition dataset"
                self.show_custom_messagebox("Dataset Required", f"Please select a {dataset_type} file.", "error")
                return False
            
            if not os.path.exists(self.dataset_path.get()):
                dataset_type = "Q&A dataset" if self.evaluation_type.get() == "qa" else "synthesis condition dataset"
                self.show_custom_messagebox("Dataset Not Found", f"{dataset_type.title()} file does not exist.", "error")
                return False

            # For synthesis evaluation, supplement is recommended but not required
            # No popup needed - just proceed without SI if not provided

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
                error_msg = (
                    "Please enter API key(s) in Settings. All API keys are required for evaluation.\n\n"
                    f"Missing:\n• {missing_keys_text}\n\n"
                    "You can access Settings by clicking the 'Settings' button in the top-right corner."
                )
                self.show_custom_messagebox("API Keys Required", error_msg, "error")
                return False

            # Test if the dataset can be loaded
            dataset = self.load_dataset(self.dataset_path.get())

            if not dataset or len(dataset) == 0:
                dataset_type = "Q&A pairs" if self.evaluation_type.get() == "qa" else "synthesis conditions"
                error_msg = (
                    f"No {dataset_type} found in the dataset file.\n\n"
                    f"Please check that your JSON file contains valid {dataset_type}.\n\n"
                    "The file should be properly formatted JSON with the expected structure."
                )
                self.show_custom_messagebox("Dataset Error", error_msg, "error")
                return False
            
            return True
            
        except ValueError as e:
            self.show_custom_messagebox("Validation Error", str(e), "error")
            return False
        except Exception as e:
            error_msg = (
                "An unexpected error occurred during validation:\n\n"
                f"{str(e)}\n\n"
                "Please check your inputs and try again."
            )
            self.show_custom_messagebox("Unexpected Error", error_msg, "error")
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
            error_msg = (
                "Failed to start evaluation:\n\n"
                f"{str(e)}\n\n"
                "Please check your inputs and try again."
            )
            self.show_custom_messagebox("Evaluation Error", error_msg, "error")
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
                error_msg += (
                    "\n\nPlease check:\n"
                    "• Your API keys are valid\n"
                    "• You have sufficient API credits\n"
                    "• Your internet connection is stable"
                )
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
            self.root.after(0, lambda: self.show_custom_messagebox("Evaluation Error", str(e), "error"))
            self.update_status("Evaluation failed")
            self.update_progress(0)
        except Exception as e:
            # Unexpected errors
            error_msg = (
                "An unexpected error occurred during evaluation:\n\n"
                f"{str(e)}\n\n"
                "Please check your inputs and try again."
            )
            self.root.after(0, lambda: self.show_custom_messagebox("Unexpected Error", error_msg, "error"))
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
        self.results_frame.config(text="Evaluation Results")
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
            self.show_custom_messagebox(
                "No Results", 
                "No evaluation results to export.\n\nPlease run an evaluation first.",
                "warning"
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
            success_msg = (
                "Evaluation results exported successfully!\n\n"
                f"File saved to:\n{filename}"
            )
            self.show_custom_messagebox("Export Successful", success_msg, "info")
        except Exception as e:
            error_msg = (
                "Failed to export results:\n\n"
                f"{str(e)}\n\n"
                "Please check that:\n"
                "• You have write permissions to the selected location\n"
                "• The file is not currently open in another program\n"
                "• You have sufficient disk space"
            )
            self.show_custom_messagebox("Export Error", error_msg, "error")
    
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