import pytest
import os
import csv
import io
from unittest.mock import mock_open, patch

# Assume the original function is in a file named 'your_module.py'
# For this example, we'll put it directly here for self-containment.
# If it's in another file, you would import it like:
# from your_module import write_steps_to_csv

# --- The function to be tested (copied for self-containment) ---
import csv
import os
from typing import List, Dict, Tuple, Optional, Union

# Note: Removed sklearn imports as they are not used in write_steps_to_csv
# and would cause an unnecessary dependency for testing this specific function.
# If they are used elsewhere in your actual module, keep them there.

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
                # Assuming mapped_scenario[0] is the scenario name, and it exists as a key in scenario_steps
                scenario_info = scenario_steps[mapped_scenario[0]]
                # Write the data row to the CSV file
                # mapped_scenario: [Scenario, Step Type, Step Text, Latency/Variable Path (used for both)]
                # scenario_info: [?, ?, Value, ...] - assuming Value is at index 2
                writer.writerow([
                    mapped_scenario[0],
                    mapped_scenario[1],
                    mapped_scenario[2],
                    scenario_info[2],
                    mapped_scenario[3], # Latency
                    mapped_scenario[3]  # Variable Path (as per original function's logic)
                ])

        # If the loop completes without errors, print success message
        print(f"Gherkin steps, value (converted), latency, and variable paths written to {csv_filename}")

    except IOError as e:
        print(f"Error writing to CSV file {csv_filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")

# --- Pytest Tests ---

@pytest.fixture
def tmp_csv_filename(tmp_path):
    """
    Pytest fixture to provide a temporary CSV file path.
    `tmp_path` is a built-in pytest fixture.
    """
    return tmp_path / "test_output.csv"

def _read_csv_content(filename):
    """Helper to read CSV content into a list of lists."""
    content = []
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                content.append(row)
    except FileNotFoundError:
        pass # Return empty content if file not found
    return content

def test_basic_write(tmp_csv_filename, capsys):
    """Test that the function correctly writes a header and multiple data rows."""
    mapped_scenarios = [
        ["Scenario A", "Given", "I have data", "10ms"],
        ["Scenario B", "When", "I process data", "20ms"],
        ["Scenario C", "Then", "I get results", "30ms"]
    ]
    scenario_steps = {
        "Scenario A": ["foo", "bar", "Value1"],
        "Scenario B": ["baz", "qux", "Value2"],
        "Scenario C": ["alpha", "beta", "Value3"]
    }

    write_steps_to_csv(tmp_csv_filename, mapped_scenarios, scenario_steps)

    expected_content = [
        ['Scenario', 'Step Type', 'Step Text', 'Value', 'Latency', 'Variable Path'],
        ['Scenario A', 'Given', 'I have data', 'Value1', '10ms', '10ms'],
        ['Scenario B', 'When', 'I process data', 'Value2', '20ms', '20ms'],
        ['Scenario C', 'Then', 'I get results', 'Value3', '30ms', '30ms']
    ]
    assert _read_csv_content(tmp_csv_filename) == expected_content

    captured = capsys.readouterr()
    assert f"Gherkin steps, value (converted), latency, and variable paths written to {tmp_csv_filename}\n" in captured.out

def test_empty_inputs(tmp_csv_filename, capsys):
    """Test that only the header is written when input data is empty."""
    mapped_scenarios = []
    scenario_steps = {}

    write_steps_to_csv(tmp_csv_filename, mapped_scenarios, scenario_steps)

    expected_content = [
        ['Scenario', 'Step Type', 'Step Text', 'Value', 'Latency', 'Variable Path']
    ]
    assert _read_csv_content(tmp_csv_filename) == expected_content

    captured = capsys.readouterr()
    assert f"Gherkin steps, value (converted), latency, and variable paths written to {tmp_csv_filename}\n" in captured.out


def test_single_scenario(tmp_csv_filename, capsys):
    """Test writing a single scenario correctly."""
    mapped_scenarios = [
        ["Single Scenario", "And", "I do something", "5ms"]
    ]
    scenario_steps = {
        "Single Scenario": ["x", "y", "SingleValue"]
    }

    write_steps_to_csv(tmp_csv_filename, mapped_scenarios, scenario_steps)

    expected_content = [
        ['Scenario', 'Step Type', 'Step Text', 'Value', 'Latency', 'Variable Path'],
        ['Single Scenario', 'And', 'I do something', 'SingleValue', '5ms', '5ms']
    ]
    assert _read_csv_content(tmp_csv_filename) == expected_content

    captured = capsys.readouterr()
    assert f"Gherkin steps, value (converted), latency, and variable paths written to {tmp_csv_filename}\n" in captured.out

def test_io_error_handling(tmp_csv_filename, capsys, mocker):
    """Test that IOError is caught and an error message is printed."""
    mapped_scenarios = [["Scenario A", "Given", "I have data", "10ms"]]
    scenario_steps = {"Scenario A": ["foo", "bar", "Value1"]}

    # Mock 'open' to raise an IOError when called
    mocker.patch('builtins.open', side_effect=IOError("Permission denied"))

    write_steps_to_csv(tmp_csv_filename, mapped_scenarios, scenario_steps)

    captured = capsys.readouterr()
    # Assert that the correct error message was printed to stderr
    assert f"Error writing to CSV file {tmp_csv_filename}: Permission denied\n" in captured.out
    # Assert that the file was not created or is empty due to the error
    assert not tmp_csv_filename.exists()

# To run these tests, save the code as a Python file (e.g., `test_csv_writer.py`)
# and run `pytest` from your terminal in the same directory.
