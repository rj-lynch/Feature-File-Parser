import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple, Optional, Union

def write_steps_to_csv(
    csv_filename,
    mapped_scenarios,
    scenario_steps
):
    """
    Writes parsed Gherkin steps, extracted data, and mapped variable paths to a CSV file.
    Output is .csv file with collumns Scenario, Step Type, step_text, Value, Latency, Variable Path
    """
    try:
        # Open the CSV file for writing
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header row
            writer.writerow(['Scenario', 'Step Type', 'Step Text', 'Value', 'Latency', 'Variable Path'])

            # Iterate through each scenario and its steps
            for mapped_scenario in mapped_scenarios:
                    scenario_info = scenario_steps[mapped_scenario[0]]
                    # Write the data row to the CSV file
                    writer.writerow([mapped_scenario[0], mapped_scenario[1], mapped_scenario[2],scenario_info[2] , scenario_info[3], mapped_scenario[3]])

        # If the loop completes without errors, print success message
        print(f"Gherkin steps, value (converted), latency, and variable paths written to {csv_filename}")

    except IOError as e:
        print(f"Error writing to CSV file {csv_filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")