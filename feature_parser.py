# feature_parser.py
import re
import tkinter as tk
from tkinter import filedialog
from typing import List, Dict, Tuple, Union, Optional

# --- Select .feature files ---
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
    file_paths_tuple = filedialog.askopenfilenames(
        title="Select .feature files",
        filetypes=[("feature files", "*.feature"), ("All files", "*.*")]
    )
    return list(file_paths_tuple)

# --- Gherkin Parsing ---
def parse_feature_file(filename: str) -> Dict[str, List[Tuple[str, str, Optional[Union[str, int]], Optional[str]]]]:
     """
     Parses a Gherkin .feature file to extract scenarios and steps.
     Ignores comments (full line or inline starting with #).
     Extracts 'value' and 'latency' as strings if present in the step text,
     with 'value' converted to int if it's a valid number (decimal or hex).
     """
     scenario_steps: Dict[str, List[Tuple[str, str, Optional[Union[str, int]], Optional[str]]]] = {}
     current_scenario: Optional[str] = None
     current_step_context: Optional[str] = None
     step_sequence_counter: int = 0

     try:
         with open(filename, 'r', encoding='utf-8') as file:
             for line_number, line in enumerate(file, 1):
                 stripped_line = line.strip()
                 # First, handle full-line comments and empty lines
                 if not stripped_line or stripped_line.startswith('#'):
                     continue
                 
                 # Now, remove any inline comments from the *rest* of the line
                 # This regex matches '#' followed by any characters to the end of the line.
                 # It replaces the comment part with an empty string.
                 # Then, re-strip to remove any trailing whitespace left by the comment removal.
                 line_without_inline_comment = re.sub(r'#.*$', '', stripped_line).strip()

                 # If the line becomes empty after removing an inline comment (e.g., "   # Only a comment"), skip it
                 if not line_without_inline_comment:
                     continue

                 # Scenario or Scenario Outline title
                 if line_without_inline_comment.startswith('Scenario:'):
                     current_scenario = line_without_inline_comment.split('Scenario:')[1].strip()
                     scenario_steps[current_scenario] = []
                     current_step_context = None
                     step_sequence_counter = 0
                     continue
                 if line_without_inline_comment.startswith('Scenario Outline:'):
                      current_scenario = line_without_inline_comment.split('Scenario Outline:')[1].strip()
                      scenario_steps[current_scenario] = []
                      current_step_context = None
                      step_sequence_counter = 0
                      continue

                 # Check for step lines (Given, When, Then, And)
                 STEP_KEYWORDS = ['Given', 'When', 'Then', 'And']
                 # Use line_without_inline_comment for matching step keywords
                 step_match = re.match(r'(' + '|'.join(STEP_KEYWORDS) + r')\s+(.*)', line_without_inline_comment)
                 if step_match:
                     step_keyword = step_match.group(1)
                     raw_step_text = step_match.group(2).strip() # This is already free of comments due to earlier processing

                     value = None
                     latency = None
                     VALUE_REGEX = re.compile(r"value\s*=\s*(\S+)", re.IGNORECASE)
                     LATENCY_REGEX = re.compile(r"latency\s*=\s*(\S+)", re.IGNORECASE)
                     
                     value_match = VALUE_REGEX.search(raw_step_text)
                     if value_match:
                        value = value_match.group(1)

                     latency_match = LATENCY_REGEX.search(raw_step_text)
                     if latency_match:
                           latency = latency_match.group(1)
                     
                     # --- Value Conversion Logic (Hex to Decimal) ---
                     value_to_store: Union[str, int, None] = value # Default to the extracted string or None
                     if value is not None:
                        # Stripping value again in case the Gherkin step had extra spaces around the value itself
                        stripped_value = value.strip() 
                        try:
                            if stripped_value.lower().startswith('0x'):
                                hex_string = stripped_value[2:]
                                value_to_store = int(hex_string, 16)
                            else:
                                value_to_store = int(stripped_value)
                        except ValueError:
                            # If conversion fails, set value to None
                            value_to_store = None
                     # --- End Value Conversion Logic --- 

                     # Determine step type and sequence
                     if step_keyword in ['Given', 'When', 'Then']:
                         current_step_context = step_keyword
                         step_sequence_counter = 1
                         step_type = f"{step_keyword}_{step_sequence_counter}"
                     elif step_keyword == 'And': # Changed from 'in' to '==' for clarity since it's a single keyword
                          if current_step_context:
                               step_sequence_counter += 1
                               step_type = f"{current_step_context}_{step_sequence_counter}"
                          else:
                               step_type = f"{step_keyword}_1"
                               print(f"Warning: '{step_keyword}' step found without preceding Given/When/Then in {filename} line {line_number}. Treating as first step.")

                     if current_scenario:
                         # Store step type, CLEANED step text, converted value, and extracted latency
                         scenario_steps[current_scenario].append((step_type, raw_step_text, value_to_store, latency))
                     else:
                         print(f"Warning: Step '{line_without_inline_comment}' found outside of a scenario in {filename} line {line_number}.")

         print(f"Successfully parsed {filename}")
     except FileNotFoundError:
         print(f"Error: Feature file not found at {filename}")
     except Exception as e:
         print(f"Error parsing file {filename}: {e}")

     return scenario_steps