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

# --- Constants ---
MODEL_FILE = 'signal_prediction_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
LABEL_ENCODER_FILE = 'label_encoder.pkl'
# Default name for .csv file
DEFAULT_CSV_OUTPUT = "gherkin_steps_with_paths.csv"

# Regular expressions for extracting value and latency
# Capture the raw string value, don't assume format (like hex) yet
VALUE_REGEX = re.compile(r"value\s*=\s*(\S+)", re.IGNORECASE)
LATENCY_REGEX = re.compile(r"latency\s*=\s*(\S+)", re.IGNORECASE)

# XML Namespace for parsing
XIL_NAMESPACE = "http://www.asam.net/XIL/Mapping/2.2.0"
# IMPORTANT: Ensure this matches the actual namespace URI from your XML.
# If your XML elements (FrameworkLabel, TestbenchLabel) do NOT have a namespace prefix (e.g., <FrameworkLabel>),
# then set NAMESPACES = {} and remove all 'ns0:' prefixes from the XPath queries below.
NAMESPACES = {'ns0': XIL_NAMESPACE} 

# Define possible step keywords for easier processing
STEP_KEYWORDS = ['Given', 'When', 'Then', 'And']


# --- ML Model Functions ---

def load_training_data(filename: str) -> List[Dict[str, str]]:
    """Loads training data from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded training data from {filename}")
        return data
    except FileNotFoundError:
        print(f"Error: Training data file not found at {filename}")
        return [] # Return empty list if file not found
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filename}: {e}")
        return [] # Return empty list if JSON is invalid
    except Exception as e:
        print(f"An unexpected error occurred loading {filename}: {e}")
        return []

def train_and_save_model(data: List[Dict[str, str]], model_file: str, vectorizer_file: str, label_encoder_file: str):
    """Trains the ML model and saves the components, handling single-sample classes."""
    if not data:
        print("Cannot train model: No training data provided.")
        return

    print("Starting model training...")
    try:
        df = pd.DataFrame(data)
        # Ensure required columns exist
        if 'sentence' not in df.columns or 'label' not in df.columns:
             print("Error: Training data must contain 'sentence' and 'label' columns.")
             return
        # Ensure there's data after checking columns
        if df.empty:
             print("Error: Training data file is empty or missing required columns.")
             return

        vectorizer = TfidfVectorizer(
        ngram_range=(1,2,3),      # Use unigrams, bigrams and trigrams
        stop_words='english',    # Remove common English stopwords
        lowercase=True           # Convert all text to lowercase
        )
        features = vectorizer.fit_transform([item['sentence'] for item in data])

        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df['label'])

        # --- Added Robustness Check for Stratified Split ---
        # Check if training is possible at all
        if features.shape[0] == 0 or len(label_encoder.classes_) == 0:
            print("Error: No training data or labels found after processing. Cannot train model.")
            return

        # Check class distribution for stratified split eligibility
        # We need at least 2 samples total, at least 2 classes total,
        # AND minimum count in any class >= 2 for stratified split.
        label_counts = pd.Series(labels).value_counts()
        can_stratify = (features.shape[0] >= 2 and
                        len(label_encoder.classes_) >= 2 and
                        (label_counts >= 2).all()) # Check if *all* classes have >= 2 samples

        if can_stratify:
            # Stratified split is possible
            print("Performing stratified train/test split.")
            X_train, X_test, y_train, y_test = train_test_split(
               features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            print(f"Training on {X_train.shape[0]} samples.")
            # Use the real SGD classifier
            sgd_classifier = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.001, random_state=42)
            sgd_classifier.fit(X_train, y_train)
        else:
            # Cannot stratify (either total samples < 2, total classes < 2,
            # or at least one class has only 1 sample). Train on all data.
            print("Warning: Cannot perform stratified split (insufficient data, classes, or single-sample class). Training on all data.")
            X_train, y_train = features, labels
            # No separate test set in this case, X_test, y_test remain None implicitly

            # Check if a real classifier is even meaningful (need at least 2 distinct classes)
            if len(label_encoder.classes_) > 1:
                 # Train a real SGD classifier on all data
                 sgd_classifier = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.001, random_state=42)
                 sgd_classifier.fit(X_train, y_train)
            else:
                 # Only 0 or 1 class - prediction will be trivial. Create a dummy.
                 # This handles the case where len(label_encoder.classes_) is 0 or 1.
                 if len(label_encoder.classes_) == 0:
                      print("Error: No labels found after processing. Cannot train model.")
                      return # Should be caught by the earlier check, but safety net
                 else: # len(label_encoder.classes_) == 1
                      print(f"Warning: Only one class ('{label_encoder.classes_[0]}') found. Using a dummy classifier that always predicts this class.")
                      class DummyClassifier:
                          # The predict method needs to return the encoded value of the single class
                          # which LabelEncoder will have encoded to 0 if there's only one class.
                          def predict(self, X):
                              return [0] * X.shape[0]
                          classes_ = label_encoder.classes_ # Attach classes for inverse_transform
                          # Dummy classifiers don't need a fit method in this simple case
                          def fit(self, X, y):
                              pass # Do nothing for dummy fit

                      sgd_classifier = DummyClassifier()
        # --- End Added Robustness Check ---


        # Save components (the trained SGDClassifier or the DummyClassifier instance)
        with open(model_file, 'wb') as f:
            pickle.dump(sgd_classifier, f)
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(label_encoder_file, 'wb') as f:
            pickle.dump(label_encoder, f)

        print("Model trained and saved successfully.")

    except Exception as e:
        print(f"Error during model training or saving: {e}")

def load_model_components(model_file: str, vectorizer_file: str, label_encoder_file: str) -> Optional[Tuple[SGDClassifier, TfidfVectorizer, LabelEncoder]]:
    """Loads saved model components."""
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(label_encoder_file, 'rb') as f:
            label_encoder = pickle.load(f)
        print("Model components loaded successfully.")
        return model, vectorizer, label_encoder
    except FileNotFoundError as e:
        print(f"Error loading model components: {e}. One or more model files not found.")
        return None
    except pickle.UnpicklingError as e:
        print(f"Error unpickling model components: {e}. Files might be corrupted or from incompatible versions.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading model components: {e}")
        return None

# --- .json file Selection ---

def select_json_file():
    """Opens a file dialog to select XML file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    json_path = filedialog.askopenfilename(
        title="Select .json file",
        filetypes=[("json files", "*.json"), ("All files", "*.*")]
    )
    return json_path

# --- .feature file Selection ---

def select_feature_files() -> List[str]:
    """Opens a file dialog to select .feature files."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(
        title="Select .feature files",
        filetypes=[("Feature files", "*.feature"), ("All files", "*.*")]
    )
    return list(file_paths) # Return list (can be empty)

# --- .xml file Selection ---

def select_xml_file():
    """Opens a file dialog to select XML file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    xml_path = filedialog.askopenfilename(
        title="Select .xml file",
        filetypes=[("xml files", "*.xml"), ("All files", "*.*")]
    )
    return xml_path

# --- Gherkin Parsing ---

def parse_feature_file(filename: str) -> Dict[str, List[Tuple[str, str, Optional[str], Optional[str]]]]:
    """
    Parses a Gherkin .feature file to extract scenarios and steps.
    Extracts 'value' and 'latency' as strings if present in the step text.
    """
    scenario_steps: Dict[str, List[Tuple[str, str, Optional[str], Optional[str]]]] = {}
    current_scenario: Optional[str] = None
    # Track the type of the most recent main step (Given, When, Then) for 'And' steps
    current_step_context: Optional[str] = None
    step_sequence_counter: int = 0 # Counter for steps within a context (Given_1, Given_2, etc.)

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                stripped_line = line.strip()

                # Ignore comments and empty lines
                if not stripped_line or stripped_line.startswith('#'):
                    continue

                # Scenario or Scenario Outline title
                if stripped_line.startswith('Scenario:'):
                    current_scenario = stripped_line.split('Scenario:')[1].strip()
                    scenario_steps[current_scenario] = []
                    current_step_context = None # Reset context for new scenario
                    step_sequence_counter = 0
                    continue
                if stripped_line.startswith('Scenario Outline:'):
                     current_scenario = stripped_line.split('Scenario Outline:')[1].strip()
                     # Note: This parser doesn't handle Examples tables for Scenario Outlines.
                     # It just gets the outline text.
                     scenario_steps[current_scenario] = []
                     current_step_context = None # Reset context for new scenario
                     step_sequence_counter = 0
                     continue

                # Check for step lines (Given, When, Then, And, But)
                step_match = re.match(r'(' + '|'.join(STEP_KEYWORDS) + r')\s+(.*)', stripped_line)

                if step_match:
                    step_keyword = step_match.group(1)
                    step_text = step_match.group(2).strip()

                    # Extract value and latency as raw strings if present
                    value = None
                    latency = None
                    value_match = VALUE_REGEX.search(step_text)
                    if value_match:
                         value = value_match.group(1) # Capture the raw string matched by \S+

                    latency_match = LATENCY_REGEX.search(step_text)
                    if latency_match:
                         latency = latency_match.group(1) # Capture the raw string matched by \S+

                    # Determine step type and sequence
                    if step_keyword in ['Given', 'When', 'Then']:
                        current_step_context = step_keyword # Set new context
                        step_sequence_counter = 1 # Start counter for this context
                        step_type = f"{step_keyword}_{step_sequence_counter}"
                    elif step_keyword in ['And', 'But']:
                         if current_step_context:
                              step_sequence_counter += 1 # Increment counter within current context
                              # Use the last seen main keyword for 'And'/'But'
                              step_type = f"{current_step_context}_{step_sequence_counter}"
                         else:
                              # Handle 'And' or 'But' without a preceding Given/When/Then
                              step_type = f"{step_keyword}_1" # Treat as the first step of its kind
                              print(f"Warning: '{step_keyword}' step found without preceding Given/When/Then in {filename} line {line_number}. Treating as first step.")
                              # Optionally, set the context here if you want subsequent 'And'/'But's to attach
                              # current_step_context = step_keyword
                              # step_sequence_counter = 1 # Reset counter for this new context
                    else:
                         # Should not happen with the regex, but as a fallback
                         step_type = f"{step_keyword}_1"
                         # Optionally, set context
                         # current_step_context = step_keyword
                         # step_sequence_counter = 1


                    if current_scenario:
                        # Store step type, text, extracted value (string), and extracted latency (string)
                        scenario_steps[current_scenario].append((step_type, step_text, value, latency))
                    else:
                        print(f"Warning: Step '{stripped_line}' found outside of a scenario in {filename} line {line_number}.")

        print(f"Successfully parsed {filename}")
    except FileNotFoundError:
        print(f"Error: Feature file not found at {filename}")
    except Exception as e:
        print(f"Error parsing file {filename}: {e}")

    return scenario_steps

# --- XML Lookup Functions ---
def get_xml_labels(xml_file: str) -> List[str]:
    """Extracts all LabelId values from FrameworkLabel elements in the XML file."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # Ensure the XPath matches your namespace prefix and element name
        labels = [elem.attrib['Id'] for elem in root.findall('.//ns0:FrameworkLabel', NAMESPACES)]
        return labels
    except FileNotFoundError:
        print(f"Error: XML file not found at '{xml_file}'")
        return []
    except ET.ParseError as e:
        print(f"Error parsing XML file '{xml_file}': {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in get_xml_labels: {e}")
        return []

# NEW FUNCTION: To get all TestbenchLabel IDs
def get_all_testbench_label_ids(xml_file: str) -> List[str]:
    """Extracts all Id values from TestbenchLabel elements in the XML file."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # Find all TestbenchLabel elements, assuming they are also namespaced
        labels = [elem.attrib['Id'] for elem in root.findall('.//ns0:TestbenchLabel', NAMESPACES)]
        return labels
    except FileNotFoundError:
        print(f"Error: XML file not found at '{xml_file}'")
        return []
    except ET.ParseError as e:
        print(f"Error parsing XML file '{xml_file}': {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in get_all_testbench_label_ids: {e}")
        return []

def fuzzy_search_xml_labels(search_text: str, target_labels: List[str], fuzzy_threshold: int = 75) -> Optional[str]:
    """
    Fuzzy matches search_text to a list of target_labels, using lowercased comparison
    but returning the original label.
    """
    if not target_labels:
        return None

    target_labels_lower = [str(label).lower() for label in target_labels]
    search_text_lower = search_text.lower()

    # Using fuzz.token_set_ratio is often better for phrases as it handles word order and extra words
    result = process.extractOne(search_text_lower, target_labels_lower, scorer=fuzz.token_set_ratio)
    
    if result:
        match_lower, score = result
        if score >= fuzzy_threshold:
            # Find the original label from the target_labels list
            try:
                idx = target_labels_lower.index(match_lower)
                return target_labels[idx]  # Return the original, case-preserved label
            except ValueError:
                # This should ideally not happen if match_lower came from target_labels_lower
                print(f"Warning: Fuzzy match '{match_lower}' not found in original lowercased list.")
                return None
    return None

# --- Mapping Logic (Fuzzy + Model) ---

def find_label_mapping_with_model(
    xml_file: str,
    step_text: str,
    model: Optional[SGDClassifier],  # Model can be None
    vectorizer: Optional[TfidfVectorizer],  # Vectorizer can be None
    label_encoder: Optional[LabelEncoder],  # LabelEncoder can be None
    framework_fuzzy_threshold: int = 75 # Threshold for step_text to FrameworkLabel
) -> Optional[str]:
    """
    Finds the corresponding Testbench variable path.
    First, identifies a FrameworkLabel using fuzzy/ML on step_text.
    Then, finds a matching TestbenchLabel using the identified FrameworkLabel ID
    by performing a second fuzzy match against all TestbenchLabels.
    Returns the TestbenchLabel Id or None.
    """
    step_text_lower = step_text.lower()
    print(f"\n[DEBUG] Processing step_text: '{step_text}' (lowercase: '{step_text_lower}')")
    
    # 1. Get all FrameworkLabels from XML (potential "source" IDs for the first stage)
    framework_labels_from_xml = get_xml_labels(xml_file)
    print(f"[DEBUG] Extracted FrameworkLabels from XML: {framework_labels_from_xml}")

    # 2. Get all TestbenchLabels from XML (potential "target" IDs for the second stage)
    all_testbench_labels_from_xml = get_all_testbench_label_ids(xml_file)
    print(f"[DEBUG] Extracted TestbenchLabels from XML: {all_testbench_labels_from_xml}")

    if not framework_labels_from_xml or not all_testbench_labels_from_xml:
        print("[DEBUG] Insufficient labels found in XML for mapping. Check XML file contents.")
        return None

    identified_framework_label: Optional[str] = None

    # --- First Stage: Identify FrameworkLabel from step_text ---
    # Try fuzzy search for FrameworkLabel
    fuzzy_result_fw = fuzzy_search_xml_labels(
        step_text_lower,
        framework_labels_from_xml,
        framework_fuzzy_threshold
    )
    
    if fuzzy_result_fw:
        identified_framework_label = fuzzy_result_fw
        print(f"[DEBUG] Fuzzy match found for FrameworkLabel: {identified_framework_label}")
    
    # If no fuzzy match, try ML prediction for FrameworkLabel
    if not identified_framework_label and model and vectorizer and label_encoder:
        # Check if the model has been fitted and has classes (prevents errors with dummy models for single-class data)
        if hasattr(model, 'predict') and hasattr(label_encoder, 'inverse_transform') and len(label_encoder.classes_) > 0:
            try:
                X = vectorizer.transform([step_text_lower])
                y_pred = model.predict(X)
                predicted_label = label_encoder.inverse_transform(y_pred)[0]
                
                if predicted_label in framework_labels_from_xml: # Ensure ML predicted label is actually in XML
                    identified_framework_label = predicted_label
                    print(f"[DEBUG] ML predicted FrameworkLabel: {identified_framework_label}")
                else:
                    print(f"[DEBUG] ML predicted label '{predicted_label}' not found in XML FrameworkLabels.")
            except Exception as e:
                print(f"Error during ML prediction for FrameworkLabel: {e}")
        else:
            print("[DEBUG] ML model/encoder not fully initialized or not suitable for prediction for FrameworkLabel. Skipping ML lookup.")

    if not identified_framework_label:
        print("[DEBUG] No FrameworkLabel identified from step_text via fuzzy or ML. Cannot proceed to TestbenchLabel lookup.")
        return None

    # --- Second Stage: Find corresponding TestbenchLabel using identified_framework_label ---
    print(f"[DEBUG] Attempting to find TestbenchLabel for identified FrameworkLabel: '{identified_framework_label}'")

    # Use fuzz.partial_ratio for this second fuzzy match
    # because the FrameworkLabel ID appears to be a substring of the TestbenchLabel ID.
    # Set a high threshold for this match, as it should be a strong semantic link.
    testbench_fuzzy_threshold = 90 # Adjust this based on how strong the match should be

    # Perform partial fuzzy matching using the identified FrameworkLabel against all TestbenchLabels
    final_testbench_label = fuzzy_search_xml_labels(
        identified_framework_label, # Use the identified FrameworkLabel as the search text
        all_testbench_labels_from_xml, # Search within all TestbenchLabels
        testbench_fuzzy_threshold,
        scorer=fuzz.partial_ratio # Use partial_ratio for substring matching
    )

    if final_testbench_label:
        print(f"[DEBUG] Found matching TestbenchLabel: {final_testbench_label}")
        return final_testbench_label
    else:
        print(f"[DEBUG] No matching TestbenchLabel found for FrameworkLabel '{identified_framework_label}'.")

    print(f"[DEBUG] No matching TestbenchLabel found for step: '{step_text}' after all attempts.")
    return None

# --- CSV Writing ---

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
    The variable path is sliced to remove the first 5 characters.
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


# --- Main Execution ---

def main():
    """Main function to orchestrate the parsing, mapping, and writing process."""

    # --- 1. Load Training Data ---
    training_data_file = select_json_file()
    if not training_data_file:
        print("No .json file selected. Exiting.")
        return
    
    print("Loading training data...")
    training_data = load_training_data(training_data_file)
    if not training_data:
        print("Exiting: Could not load training data.")
        return # Cannot proceed without training data for model/fuzzy matching

    # --- 2. Load or Train ML Model ---
    model_components = None
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE) or not os.path.exists(LABEL_ENCODER_FILE):
        print("Model files not found. Training new model...")
        train_and_save_model(training_data, MODEL_FILE, VECTORIZER_FILE, LABEL_ENCODER_FILE)
        # Attempt to load after training
        model_components = load_model_components(MODEL_FILE, VECTORIZER_FILE, LABEL_ENCODER_FILE)
    else:
        print("Loading existing model components...")
        model_components = load_model_components(MODEL_FILE, VECTORIZER_FILE, LABEL_ENCODER_FILE)

    if not model_components:
        print("Warning: Could not load or train model components. Mapping will rely solely on fuzzy matching (if training data has sentences).")
        # Continue execution, but mapping function will handle None model components

    # --- 3. Select Feature Files ---
    feature_files = select_feature_files()
    if not feature_files:
        print("No .feature files selected. Exiting.")
        return
    
    xml_file = select_xml_file()
    if not xml_file:
        print("No .xml file selected. Exiting.")
        return

    # --- 4. Specify XML Mapping File ---
    # Using the default hardcoded path. Consider adding a file dialog here too if needed.
    if not os.path.exists(xml_file):
        print(f"Error: Specified XML mapping file not found at {xml_file}. Cannot perform mapping.")
        # The mapping function will handle the missing XML file gracefully by returning None.
        # We can still write the other data (steps, values, latency) to CSV.
        pass

    # --- 5. Parse Feature Files ---
    all_steps: Dict[str, List[Tuple[str, str, Optional[str], Optional[str]]]] = {}
    print(f"Parsing {len(feature_files)} .feature files...")
    for feature_file in feature_files:
        steps = parse_feature_file(feature_file)
        # Merge steps. Note: This will overwrite scenarios if they have the same name
        # across different files. If this is not desired, the dictionary key should
        # probably be a tuple like (feature_file, scenario_name).
        for scenario_name, scenario_step_list in steps.items():
             if scenario_name in all_steps:
                  print(f"Warning: Duplicate scenario name '{scenario_name}' found in multiple files. Steps from later files will overwrite earlier ones.")
             all_steps[scenario_name] = scenario_step_list


    if not all_steps:
        print("No steps found in selected feature files. Exiting.")
        return

    # --- 6. Write Results to CSV ---
    print(f"Writing parsed steps and mapping results to {DEFAULT_CSV_OUTPUT}...")
    # Pass the xml_file path even if it doesn't exist; write_steps_to_csv handles it.
    write_steps_to_csv(all_steps, xml_file, DEFAULT_CSV_OUTPUT, training_data, model_components)

    print("\nProcessing complete.")

if __name__ == '__main__':
    main()
