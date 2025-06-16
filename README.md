# Feature-File-Parser
Python script used to convert feature requirements into test scripts for HiL test automation

# Gherkin Feature File Parser and Mapper (v1.5)

This Python script is designed to parse Gherkin `.feature` files, extract relevant information from steps (including special `value=` and `latency=` parameters), and map these steps to corresponding "Variable Paths" using a combination of fuzzy matching, a simple machine learning model, and an XML mapping file. The final output is a CSV file containing the parsed steps and their associated data.

## What it Does

The script automates the process of taking Gherkin test specifications and linking the steps to test automation variables defined in an XML format (likely related to test bench or simulation environments, given the XML structure). It does this by:

1.  **Parsing `.feature` files:** Reads one or more selected `.feature` files, identifying scenarios and individual steps (`Given`, `When`, `Then`, `And`).
2.  **Extracting Step Data:** Pulls out the main step text and specifically looks for `value=` and `latency=` parameters within the step text, extracting their raw string values. It also attempts to convert extracted `value` strings that look like hex (`0x...` or `#...`) into decimal integers.
3.  **Loading Training Data:** Reads a JSON file containing manually labelled Gherkin sentences mapped to specific "labels". This data is crucial for both the fuzzy matching and the machine learning model.
4.  **ML Model Management:**
    *   Checks if a pre-trained machine learning model (SGD Classifier) and its associated components (TF-IDF Vectorizer, Label Encoder) exist.
    *   If they don't exist or are missing, it trains a new model using the loaded training data and saves the components.
    *   If they exist, it loads the saved components.
5.  **Mapping Steps to Labels:** For each step text, it attempts to find a corresponding "label" using:
    *   **Fuzzy Matching:** Compares the step text against sentences in the training data. If a sufficiently similar match is found, it uses the label from the training data.
    *   **ML Prediction:** If fuzzy matching isn't confident or fails, it uses the trained ML model to predict a label based on the step text.
6.  **XML Lookup:** Uses the determined label (from fuzzy or ML) to find a matching "FrameworkLabel" ID in a specified XML mapping file. If a match is found, it then retrieves the associated "TestbenchLabelReference" ID from the same XML entry. This Testbench ID is considered the "Variable Path".
7.  **Generating CSV Output:** Writes all the extracted and derived information (Scenario, Step Type, Step Text, Converted Value, Latency, Variable Path) into a structured CSV file. The Variable Path is sliced, removing the first 5 characters before writing.

This process helps bridge the gap between human-readable Gherkin specifications and the technical variable names used in the test automation framework referenced by the XML mapping.

## Features

*   Parses standard Gherkin `.feature` files.
*   Identifies Scenarios, Scenario Outlines, Given, When, Then, And, But steps.
*   Extracts `value=` and `latency=` parameters from step text.
*   Automatically converts extracted hex values (`0x...`, `#...`) to decimal integers in the output.
*   Uses a JSON file for training/fuzzy matching data.
*   Implements a fuzzy matching mechanism (`fuzzywuzzy`) to find similar step sentences in the training data.
*   Trains and uses an SGD Classifier model (`scikit-learn`) to predict labels for step text.
*   Manages ML model components (training, saving, loading).
*   Looks up mapped variable paths in a specified XML file based on predicted/matched labels.
*   Outputs results to a structured CSV file.
*   Provides a basic Tkinter GUI for selecting input `.feature` files.
*   Includes robustness checks for ML training data and file handling.

## Prerequisites

*   Python 3.6 or higher (due to type hints)
*   The following Python libraries:
    *   `pandas`
    *   `scikit-learn`
    *   `fuzzywuzzy`
    *   `python-tk` (usually included with Python, but sometimes needs separate installation depending on your OS/distribution, e.g., `sudo apt-get install python3-tk` on Debian/Ubuntu)

## Setup

1.  **Install Python:** If you don't have Python installed, download it from [python.org](https://www.python.org/downloads/).
2.  **Install Dependencies:** Open your terminal or command prompt and run:
    ```bash
    pip install pandas scikit-learn fuzzywuzzy[speedup]
    ```
    *(Note: `[speedup]` attempts to install `python-Levenshtein` for faster fuzzy matching, which might require a C compiler)*
3.  **Download the Script:** Get the `Feature_File_Parser_v15.py` file.
4.  **Prepare Input Files:**
    *   **Training Data:** Ensure you have your `training_data.json` file in the location specified by `TRAINING_DATA_FILE` in the script. This file should be a JSON list of objects like `[{"sentence": "...", "label": "..."}, ...]`.
    *   **XML Mapping:** Ensure your XML mapping file is in the location specified by `DEFAULT_XML_FILE` in the script. This file should contain structure allowing lookup of `TestbenchLabelReference` based on `FrameworkLabel` IDs.
    *   **Feature Files:** Have the `.feature` files you want to parse ready.

## Configuration

Currently, the script uses hardcoded file paths for:

*   `DEFAULT_CSV_OUTPUT`: The name of the output CSV file (will be created in the directory where the script is run).
*   `MODEL_FILE`, `VECTORIZER_FILE`, `LABEL_ENCODER_FILE`: Names for the saved ML model components (will be created in the script's directory).
