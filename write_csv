import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple, Optional, Union

def write_steps_to_csv(
    steps_data: Dict[str, List[Tuple[str, str, Optional[str], Optional[str]]]],
    xml_file: str,
    csv_filename: str,
    training_data: List[Dict[str, str]],
    model_components: Optional[Tuple[SGDClassifier, TfidfVectorizer, LabelEncoder]]
):
    """
    Writes parsed Gherkin steps, extracted data, and mapped variable paths to a CSV file.
    Converts 'value' from hex to decimal if applicable.
    """
    # Unpack model components if available (can be None)
    model, vectorizer, label_encoder = model_components if model_components else (None, None, None)

    try:
        # Open the CSV file for writing
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header row
            writer.writerow(['Scenario', 'Step Type', 'Step Text', 'Value', 'Latency', 'Variable Path'])

            # Iterate through each scenario and its steps
            for scenario, step_list in steps_data.items():
                for step_type, step_text, value, latency in step_list:

                    # --- Value Conversion Logic (Hex to Decimal) ---
                    value_to_write: Union[str, int, None] = value # Default to the extracted string or None
                    if value is not None:
                        stripped_value = value.strip()
                        try:
                            # Attempt hex conversion if it starts with '0x' or '#'
                            if stripped_value.lower().startswith('0x'):
                                # Remove '0x' prefix before converting
                                hex_string = stripped_value[2:]
                                value_to_write = int(hex_string, 16)
                            elif stripped_value.startswith('#'):
                                # Remove '#' prefix before converting
                                hex_string = stripped_value[1:]
                                value_to_write = int(hex_string, 16)
                            else:
                                # Attempt decimal conversion for values without a hex prefix
                                value_to_write = int(stripped_value)
                        except ValueError:
                            # If conversion fails (for hex or decimal), keep the original string
                            # print(f"Warning: Could not convert value '{value}' to number (decimal or hex) for step '{step_text}'. Writing original string.")
                            value_to_write = value # Keep the original string
                    # --- End Value Conversion Logic ---


                    variable_path_full: Optional[str] = None
                    if step_text: # Only attempt mapping if there is step text
                         # Only attempt mapping if the XML file exists
                         if os.path.exists(xml_file):
                            # This will now return the TestbenchLabel ID
                            variable_path_full = find_label_mapping_with_model(
                                xml_file,
                                step_text,
                                model,
                                vectorizer,
                                label_encoder
                            )
                         else:
                             # Error about missing XML is printed in main
                             pass


                    # Prepare the variable path for writing
                    # Slice off the first 5 characters if it's a valid string of sufficient length
                    variable_path_to_write = ""
                    if variable_path_full is not None:
                        if len(variable_path_full) >= 5:
                            # Slice off the first 5 characters as requested
                            variable_path_to_write = variable_path_full[5:]
                        else:
                            # Path exists but is shorter than 5 characters
                            print(f"Warning: Testbench Label ID '{variable_path_full}' is less than 5 characters long. Cannot slice off the first 5. Using full ID.")
                            variable_path_to_write = variable_path_full
                    # If variable_path_full is None, variable_path_to_write remains ""


                    # Write the data row to the CSV file
                    # Ensure value_to_write is used here
                    writer.writerow([scenario, step_type, step_text, value_to_write, latency, variable_path_to_write])

        # If the loop completes without errors, print success message
        print(f"Gherkin steps, value (converted), latency, and variable paths written to {csv_filename}")

    except IOError as e:
        print(f"Error writing to CSV file {csv_filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")