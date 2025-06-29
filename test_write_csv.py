# test_write_csv.py
import pytest
import csv
import os
from unittest.mock import MagicMock, call
from write_csv import write_steps_to_csv # Import the function to be tested

# --- Helper for creating mock model components ---
# These mocks are simple placeholders as the actual model logic is external
# to write_csv.py and will be mocked via find_label_mapping_with_model
@pytest.fixture
def mock_model_components():
    mock_model = MagicMock(name="SGDClassifier")
    mock_vectorizer = MagicMock(name="TfidfVectorizer")
    mock_label_encoder = MagicMock(name="LabelEncoder")
    return (mock_model, mock_vectorizer, mock_label_encoder)

# --- Test Cases for write_steps_to_csv ---

def test_write_steps_to_csv_basic_success(mocker, tmp_path, mock_model_components):
    """
    Tests successful CSV writing for a simple case with one scenario and steps.
    """
    mock_csv_file = mocker.mock_open()
    mocker.patch('builtins.open', mock_csv_file) # Mock the open function
    mock_writer = mocker.MagicMock()
    mocker.patch('csv.writer', return_value=mock_writer) # Mock csv.writer

    # Mock external dependency: find_label_mapping_with_model
    mock_find_label_mapping = mocker.patch(
        'write_csv.find_label_mapping_with_model',
        return_value="path/to/variable_123" # A typical return value
    )
    mocker.patch('os.path.exists', return_value=True) # Assume XML file exists

    steps_data = {
        "Scenario 1": [
            ("Given", "I have a step", "10", "50ms"),
            ("When", "I perform action", None, None)
        ]
    }
    xml_file = "dummy.xml"
    csv_filename = tmp_path / "output.csv"
    training_data = [] # Not directly used by write_csv, but part of signature

    write_steps_to_csv(
        steps_data,
        str(csv_filename),
        str(csv_filename), # Pass the path string
        training_data,
        mock_model_components
    )

    # Verify builtins.open was called correctly
    mock_csv_file.assert_called_once_with(str(csv_filename), 'w', newline='', encoding='utf-8')
    
    # Verify header row was written
    mock_writer.writerow.assert_any_call(['Scenario', 'Step Type', 'Step Text', 'Value', 'Latency', 'Variable Path'])

    # Verify data rows were written (and variable path was sliced)
    expected_calls = [
        call(['Scenario', 'Step Type', 'Step Text', 'Value', 'Latency', 'Variable Path']), # Header
        call(['Scenario 1', 'Given', 'I have a step', 10, '50ms', 'o/variable_123']), # Sliced 'path/t' becomes 'o/variable_123'
        call(['Scenario 1', 'When', 'I perform action', None, None, 'o/variable_123']) # Sliced 'path/t' becomes 'o/variable_123'
    ]
    mock_writer.writerow.assert_has_calls(expected_calls)
    assert mock_writer.writerow.call_count == 3 # Header + 2 data rows

    # Verify find_label_mapping_with_model was called for each step text
    mock_find_label_mapping.assert_has_calls([
        call(xml_file, "I have a step", *mock_model_components),
        call(xml_file, "I perform action", *mock_model_components)
    ])
    assert mock_find_label_mapping.call_count == 2

def test_write_steps_to_csv_multiple_scenarios(mocker, tmp_path, mock_model_components):
    """
    Tests writing multiple scenarios and their steps.
    """
    mock_csv_file = mocker.mock_open()
    mocker.patch('builtins.open', mock_csv_file)
    mock_writer = mocker.MagicMock()
    mocker.patch('csv.writer', return_value=mock_writer)
    mock_find_label_mapping = mocker.patch(
        'write_csv.find_label_mapping_with_model',
        side_effect=["path_alpha", "path_beta", "path_gamma"] # Different paths for different calls
    )
    mocker.patch('os.path.exists', return_value=True)

    steps_data = {
        "Scenario A": [("Given", "First step", "1", "10ms")],
        "Scenario B": [("When", "Second step", "2", "20ms"), ("Then", "Third step", "3", "30ms")]
    }
    csv_filename = tmp_path / "output.csv"

    write_steps_to_csv(steps_data, "dummy.xml", str(csv_filename), [], mock_model_components)

    # Verify correct number of rows written
    assert mock_writer.writerow.call_count == 1 + 3 # Header + 3 data rows

    # Verify specific rows (checking slicing and order)
    written_rows = [args[0] for args in mock_writer.writerow.call_args_list]
    assert written_rows[0] == ['Scenario', 'Step Type', 'Step Text', 'Value', 'Latency', 'Variable Path']
    assert written_rows[1] == ['Scenario A', 'Given', 'First step', 1, '10ms', 'h_alpha'] # 'path_alpha' -> 'h_alpha'
    assert written_rows[2] == ['Scenario B', 'When', 'Second step', 2, '20ms', 'h_beta']  # 'path_beta' -> 'h_beta'
    assert written_rows[3] == ['Scenario B', 'Then', 'Third step', 3, '30ms', 'h_gamma'] # 'path_gamma' -> 'h_gamma'

def test_write_steps_to_csv_empty_steps_data(mocker, tmp_path, mock_model_components):
    """
    Tests behavior with empty steps data. Should only write header.
    """
    mock_csv_file = mocker.mock_open()
    mocker.patch('builtins.open', mock_csv_file)
    mock_writer = mocker.MagicMock()
    mocker.patch('csv.writer', return_value=mock_writer)
    mock_find_label_mapping = mocker.patch('write_csv.find_label_mapping_with_model')
    mocker.patch('os.path.exists', return_value=True)

    steps_data = {} # Empty
    csv_filename = tmp_path / "output.csv"

    write_steps_to_csv(steps_data, "dummy.xml", str(csv_filename), [], mock_model_components)

    mock_csv_file.assert_called_once()
    mock_writer.writerow.assert_called_once_with(['Scenario', 'Step Type', 'Step Text', 'Value', 'Latency', 'Variable Path'])
    mock_find_label_mapping.assert_not_called() # No steps, so no mapping calls

# --- Value Conversion Tests ---

@pytest.mark.parametrize("input_value, expected_output_value", [
    ("10", 10),
    ("0x0A", 10),
    ("#A", 10),
    ("0xFF", 255),
    ("#FF", 255),
    ("0xabc", 2748),
    ("5.5", "5.5"), # Should not convert float strings
    ("hello", "hello"), # Should not convert non-numeric strings
    (None, None),
    ("", ""), # Empty string should remain empty
    ("  0x1f  ", 31), # Test stripping whitespace
    ("  123  ", 123), # Test stripping whitespace
    ("0xG", "0xG"), # Invalid hex character
    ("0x", "0x"), # Just prefix
    ("#", "#") # Just prefix
])
def test_write_steps_to_csv_value_conversion(mocker, tmp_path, mock_model_components, input_value, expected_output_value):
    """
    Tests various value conversion scenarios.
    """
    mock_writer = mocker.MagicMock()
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('csv.writer', return_value=mock_writer)
    mocker.patch('write_csv.find_label_mapping_with_model', return_value="long_path_value")
    mocker.patch('os.path.exists', return_value=True)

    steps_data = {
        "Scenario 1": [("Given", "A step with value", input_value, None)]
    }
    csv_filename = tmp_path / "output.csv"

    write_steps_to_csv(steps_data, "dummy.xml", str(csv_filename), [], mock_model_components)

    # Check the specific row where the value is
    written_row = mock_writer.writerow.call_args_list[1][0][0]
    assert written_row[3] == expected_output_value

# --- Variable Path Mapping Tests ---

def test_write_steps_to_csv_variable_path_sliced(mocker, tmp_path, mock_model_components):
    """
    Tests that variable path is sliced correctly (first 5 characters removed).
    """
    mock_writer = mocker.MagicMock()
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('csv.writer', return_value=mock_writer)
    mock_find_label_mapping = mocker.patch(
        'write_csv.find_label_mapping_with_model',
        return_value="0123456789ABCDEF" # A long path
    )
    mocker.patch('os.path.exists', return_value=True)

    steps_data = {"S1": [("G", "Step", None, None)]}
    csv_filename = tmp_path / "output.csv"

    write_steps_to_csv(steps_data, "dummy.xml", str(csv_filename), [], mock_model_components)

    written_row = mock_writer.writerow.call_args_list[1][0][0] # Get the data row
    assert written_row[5] == "56789ABCDEF" # Should be sliced

def test_write_steps_to_csv_variable_path_less_than_5_chars(mocker, tmp_path, mock_model_components, capsys):
    """
    Tests that variable path less than 5 characters is used as is, with a warning.
    """
    mock_writer = mocker.MagicMock()
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('csv.writer', return_value=mock_writer)
    mock_find_label_mapping = mocker.patch(
        'write_csv.find_label_mapping_with_model',
        return_value="abcd" # Less than 5 chars
    )
    mocker.patch('os.path.exists', return_value=True)

    steps_data = {"S1": [("G", "Step", None, None)]}
    csv_filename = tmp_path / "output.csv"

    write_steps_to_csv(steps_data, "dummy.xml", str(csv_filename), [], mock_model_components)

    written_row = mock_writer.writerow.call_args_list[1][0][0]
    assert written_row[5] == "abcd" # Should not be sliced
    captured = capsys.readouterr()
    assert "Warning: Testbench Label ID 'abcd' is less than 5 characters long. Cannot slice off the first 5. Using full ID." in captured.out

def test_write_steps_to_csv_variable_path_none(mocker, tmp_path, mock_model_components):
    """
    Tests that variable path is empty string if find_label_mapping returns None.
    """
    mock_writer = mocker.MagicMock()
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('csv.writer', return_value=mock_writer)
    mock_find_label_mapping = mocker.patch(
        'write_csv.find_label_mapping_with_model',
        return_value=None # No mapping found
    )
    mocker.patch('os.path.exists', return_value=True)

    steps_data = {"S1": [("G", "Step", None, None)]}
    csv_filename = tmp_path / "output.csv"

    write_steps_to_csv(steps_data, "dummy.xml", str(csv_filename), [], mock_model_components)

    written_row = mock_writer.writerow.call_args_list[1][0][0]
    assert written_row[5] == "" # Should be empty string

def test_write_steps_to_csv_xml_file_not_exist(mocker, tmp_path, mock_model_components):
    """
    Tests that find_label_mapping_with_model is NOT called if XML file doesn't exist.
    """
    mock_writer = mocker.MagicMock()
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('csv.writer', return_value=mock_writer)
    mock_find_label_mapping = mocker.patch('write_csv.find_label_mapping_with_model')
    mocker.patch('os.path.exists', return_value=False) # XML file does NOT exist

    steps_data = {"S1": [("G", "Step", None, None)]}
    csv_filename = tmp_path / "output.csv"

    write_steps_to_csv(steps_data, "non_existent.xml", str(csv_filename), [], mock_model_components)

    mock_find_label_mapping.assert_not_called()
    written_row = mock_writer.writerow.call_args_list[1][0][0]
    assert written_row[5] == "" # Variable path should be empty

def test_write_steps_to_csv_empty_step_text(mocker, tmp_path, mock_model_components):
    """
    Tests that find_label_mapping_with_model is NOT called if step_text is empty.
    """
    mock_writer = mocker.MagicMock()
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('csv.writer', return_value=mock_writer)
    mock_find_label_mapping = mocker.patch('write_csv.find_label_mapping_with_model')
    mocker.patch('os.path.exists', return_value=True)

    steps_data = {"S1": [("G", "", None, None)]} # Empty step text
    csv_filename = tmp_path / "output.csv"

    write_steps_to_csv(steps_data, "dummy.xml", str(csv_filename), [], mock_model_components)

    mock_find_label_mapping.assert_not_called()
    written_row = mock_writer.writerow.call_args_list[1][0][0]
    assert written_row[5] == "" # Variable path should be empty

# --- Error Handling Tests ---

def test_write_steps_to_csv_io_error(mocker, tmp_path, capsys, mock_model_components):
    """
    Tests handling of IOError during file writing.
    """
    mocker.patch('builtins.open', side_effect=IOError("Permission denied"))
    mock_find_label_mapping = mocker.patch('write_csv.find_label_mapping_with_model', return_value="path")
    mocker.patch('os.path.exists', return_value=True)

    steps_data = {"S1": [("G", "Step", None, None)]}
    csv_filename = tmp_path / "output.csv"

    write_steps_to_csv(steps_data, "dummy.xml", str(csv_filename), [], mock_model_components)

    captured = capsys.readouterr()
    assert f"Error writing to CSV file {csv_filename}: Permission denied" in captured.err # IOError goes to stderr

def test_write_steps_to_csv_unexpected_error(mocker, tmp_path, capsys, mock_model_components):
    """
    Tests handling of an unexpected Exception during processing.
    """
    mock_writer = mocker.MagicMock()
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('csv.writer', return_value=mock_writer)
    # Simulate an error during variable path mapping
    mocker.patch('write_csv.find_label_mapping_with_model', side_effect=ValueError("Mapping failed"))
    mocker.patch('os.path.exists', return_value=True)

    steps_data = {"S1": [("G", "Step", None, None)]}
    csv_filename = tmp_path / "output.csv"

    write_steps_to_csv(steps_data, "dummy.xml", str(csv_filename), [], mock_model_components)

    captured = capsys.readouterr()
    assert "An unexpected error occurred during CSV writing: Mapping failed" in captured.err # Generic Exception to stderr

def test_write_steps_to_csv_model_components_none(mocker, tmp_path):
    """
    Tests that the function handles model_components being None gracefully.
    """
    mock_writer = mocker.MagicMock()
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('csv.writer', return_value=mock_writer)
    mock_find_label_mapping = mocker.patch(
        'write_csv.find_label_mapping_with_model',
        return_value="path/to/variable_abc"
    )
    mocker.patch('os.path.exists', return_value=True)

    steps_data = {
        "Scenario 1": [("Given", "I have a step", "10", "50ms")]
    }
    csv_filename = tmp_path / "output.csv"

    write_steps_to_csv(
        steps_data,
        "dummy.xml",
        str(csv_filename),
        [],
        None # model_components is None
    )

    # Verify find_label_mapping_with_model was called with (None, None, None) for model components
    mock_find_label_mapping.assert_called_once_with(
        "dummy.xml", "I have a step", None, None, None
    )
    written_row = mock_writer.writerow.call_args_list[1][0][0]
    assert written_row[5] == "o/variable_abc"
