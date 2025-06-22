import re
import csv
import tkinter as tk
from tkinter import filedialog
import os
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import json
from typing import List, Dict, Tuple, Any, Optional, Union

# feature_parser.py
import tkinter as tk
from tkinter import filedialog

def select_feature_files():
    """
    Opens a file dialog to allow the user to select multiple .feature files.

    Initializes a hidden Tkinter root window, opens the file dialog,
    and returns a list of paths of the selected files. If the user cancels the
    dialog, an empty list is returned.

    Returns:
        list[str]: A list of full paths to the selected .feature files,
                   or an empty list if the user cancels the dialog.
    """
    # Initialize a Tkinter root window.
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Open the file selection dialog for multiple files
    # IMPORTANT: Using askopenfilenames (plural)
    file_paths_tuple = filedialog.askopenfilenames(
        title="Select .feature files",  # Updated title for clarity
        filetypes=[("feature files", "*.feature"), ("All files", "*.*")]
    )

    # Convert the tuple of paths to a list, which is often more convenient
    # If file_paths_tuple is empty, list() will return an empty list.
    return list(file_paths_tuple)



# --- Gherkin Parsing ---

# def parse_feature_file(filename: str) -> Dict[str, List[Tuple[str, str, Optional[str], Optional[str]]]]:
#     """
#     Parses a Gherkin .feature file to extract scenarios and steps.
#     Extracts 'value' and 'latency' as strings if present in the step text.
#     """
#     scenario_steps: Dict[str, List[Tuple[str, str, Optional[str], Optional[str]]]] = {}
#     current_scenario: Optional[str] = None
#     # Track the type of the most recent main step (Given, When, Then) for 'And' steps
#     current_step_context: Optional[str] = None
#     step_sequence_counter: int = 0 # Counter for steps within a context (Given_1, Given_2, etc.)

#     try:
#         with open(filename, 'r', encoding='utf-8') as file:
#             for line_number, line in enumerate(file, 1):
#                 stripped_line = line.strip()

#                 # Ignore comments and empty lines
#                 if not stripped_line or stripped_line.startswith('#'):
#                     continue

#                 # Scenario or Scenario Outline title
#                 if stripped_line.startswith('Scenario:'):
#                     current_scenario = stripped_line.split('Scenario:')[1].strip()
#                     scenario_steps[current_scenario] = []
#                     current_step_context = None # Reset context for new scenario
#                     step_sequence_counter = 0
#                     continue
#                 if stripped_line.startswith('Scenario Outline:'):
#                      current_scenario = stripped_line.split('Scenario Outline:')[1].strip()
#                      # Note: This parser doesn't handle Examples tables for Scenario Outlines.
#                      # It just gets the outline text.
#                      scenario_steps[current_scenario] = []
#                      current_step_context = None # Reset context for new scenario
#                      step_sequence_counter = 0
#                      continue

#                 # Check for step lines (Given, When, Then, And)
#                 step_match = re.match(r'(' + '|'.join(STEP_KEYWORDS) + r')\s+(.*)', stripped_line)

#                 if step_match:
#                     step_keyword = step_match.group(1)
#                     step_text = step_match.group(2).strip()

#                     # Extract value and latency as raw strings if present
#                     value = None
#                     latency = None
#                     value_match = VALUE_REGEX.search(step_text)
#                     if value_match:
#                          value = value_match.group(1) # Capture the raw string matched by \S+

#                     latency_match = LATENCY_REGEX.search(step_text)
#                     if latency_match:
#                          latency = latency_match.group(1) # Capture the raw string matched by \S+

#                     # Determine step type and sequence
#                     if step_keyword in ['Given', 'When', 'Then']:
#                         current_step_context = step_keyword # Set new context
#                         step_sequence_counter = 1 # Start counter for this context
#                         step_type = f"{step_keyword}_{step_sequence_counter}"
#                     elif step_keyword in 'And':
#                          if current_step_context:
#                               step_sequence_counter += 1 # Increment counter within current context
#                               # Use the last seen main keyword for 'And'/'But'
#                               step_type = f"{current_step_context}_{step_sequence_counter}"
#                          else:
#                               # Handle 'And' without a preceding Given/When/Then
#                               step_type = f"{step_keyword}_1" # Treat as the first step of its kind
#                               print(f"Warning: '{step_keyword}' step found without preceding Given/When/Then in {filename} line {line_number}. Treating as first step.")
#                               # current_step_context = step_keyword
#                               # step_sequence_counter = 1 # Reset counter for this new context
#                     else:
#                          # Should not happen with the regex, but as a fallback
#                          step_type = f"{step_keyword}_1"
#                          # Optionally, set context
#                          # current_step_context = step_keyword
#                          # step_sequence_counter = 1


#                     if current_scenario:
#                         # Store step type, text, extracted value (string), and extracted latency (string)
#                         scenario_steps[current_scenario].append((step_type, step_text, value, latency))
#                     else:
#                         print(f"Warning: Step '{stripped_line}' found outside of a scenario in {filename} line {line_number}.")

#         print(f"Successfully parsed {filename}")
#     except FileNotFoundError:
#         print(f"Error: Feature file not found at {filename}")
#     except Exception as e:
#         print(f"Error parsing file {filename}: {e}")

#     return scenario_steps
