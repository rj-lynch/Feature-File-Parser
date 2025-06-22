# test_feature_parser.py
import pytest
from feature_parser import select_feature_files
from unittest.mock import MagicMock # For mocking objects
import xml.etree.ElementTree as ET # For ET.ParseError (not directly used here, but kept)
from unittest.mock import mock_open
import re # Not strictly needed for tests, but parse_feature_file uses it.
from typing import Dict, List, Tuple, Optional

# Import the function to be tested.
# This assumes your parse_feature_file function is in a file named feature_parser.py
from feature_parser import parse_feature_file

def test_select_feature_file_success(mocker):
    # When selecting multiple files, the function returns a LIST of paths.
    # So, expected_paths should be a list.
    expected_paths = ["C:/path/to/my_file.feature", "D:/another/path/second.feature"]

    # askopenfilenames returns a TUPLE of paths.
    # So, the mock's return_value must be a tuple.
    mock_ask = mocker.patch('feature_parser.filedialog.askopenfilenames', return_value=tuple(expected_paths))
    
    # PATCHING: feature_parser.tk.Tk
    # This is correct for hiding the Tkinter root window.
    mock_tk = mocker.patch('feature_parser.tk.Tk')

    result = select_feature_files()

    # The result should be the list of paths
    assert result == expected_paths
    mock_ask.assert_called_once_with(
        title="Select .feature files", # Matches the new title in feature_parser.py
        filetypes=[("feature files", "*.feature"), ("All files", "*.*")] # Matches the new filetypes label
    )
    mock_tk.assert_called_once()
    mock_tk.return_value.withdraw.assert_called_once()


def test_select_feature_file_cancelled(mocker):
    # When cancelled, askopenfilenames returns an EMPTY TUPLE ().
    # The function then converts this to an empty list [].
    mock_ask = mocker.patch('feature_parser.filedialog.askopenfilenames', return_value=()) # Changed from "" to ()
    mock_tk = mocker.patch('feature_parser.tk.Tk')

    result = select_feature_files()

    # The result should be an empty list when cancelled
    assert result == []
    mock_ask.assert_called_once_with(
        title="Select .feature files", # Matches the new title in feature_parser.py
        filetypes=[("feature files", "*.feature"), ("All files", "*.*")] # Matches the new filetypes label
    )
    mock_tk.assert_called_once()
    mock_tk.return_value.withdraw.assert_called_once()

# --- Helper for Mocking Files ---
def mock_feature_file_content(mocker, file_content: str):
    """
    Helper function to mock the 'open' built-in for testing file reading.
    It simulates a file containing the given content.
    """
    mock_file = mocker.mock_open(read_data=file_content)
    # Patch 'builtins.open' so that any 'with open(...)' call uses our mock
    mocker.patch('builtins.open', mock_file)
    return mock_file # Return the mock object for assertions on open() calls

# --- Test Cases for parse_feature_file ---

def test_parse_feature_file_basic_scenario(mocker):
    """Tests parsing a simple feature file with a single scenario."""
    file_content = """
    Feature: User Login

    Scenario: Successful Login
      Given I am on the login page
      When I enter username "testuser" and password "password123"
      Then I should be redirected to the dashboard
    """
    mock_file_obj = mock_feature_file_content(mocker, file_content)

    result = parse_feature_file("test_login.feature")

    expected = {
        "Successful Login": [
            ("Given_1", "I am on the login page", None, None),
            ("When_1", 'I enter username "testuser" and password "password123"', None, None),
            ("Then_1", "I should be redirected to the dashboard", None, None)
        ]
    }
    assert result == expected
    mock_file_obj.assert_called_once_with("test_login.feature", 'r', encoding='utf-8')

def test_parse_feature_file_scenario_outline(mocker):
    """Tests parsing a scenario outline (only the title and steps, not examples tables)."""
    file_content = """
    Feature: Calculator

    Scenario Outline: Add two numbers
      Given the first number is <num1>
      And the second number is <num2>
      When I add them
      Then the result should be <result>

      Examples:
        | num1 | num2 | result |
        | 1    | 2    | 3      |
        | 5    | 5    | 10     |
    """
    mock_file_obj = mock_feature_file_content(mocker, file_content)

    result = parse_feature_file("test_outline.feature")

    expected = {
        "Add two numbers": [
            ("Given_1", "the first number is <num1>", None, None),
            ("Given_2", "the second number is <num2>", None, None),
            ("When_1", "I add them", None, None),
            ("Then_1", "the result should be <result>", None, None)
        ]
    }
    assert result == expected
    mock_file_obj.assert_called_once_with("test_outline.feature", 'r', encoding='utf-8')

def test_parse_feature_file_and_steps_context(mocker):
    """Tests 'And' steps correctly associating with their preceding main step type."""
    file_content = """
    Feature: Shopping Cart

    Scenario: Add multiple items
      Given I am logged in
      And I have an empty cart
      When I add "Laptop" to the cart
      And I add "Mouse" to the cart
      Then the cart should contain 2 items
      And the total price should be correct
    """
    mock_file_obj = mock_feature_file_content(mocker, file_content)

    result = parse_feature_file("test_and_steps.feature")

    expected = {
        "Add multiple items": [
            ("Given_1", "I am logged in", None, None),
            ("Given_2", "I have an empty cart", None, None),
            ("When_1", 'I add "Laptop" to the cart', None, None),
            ("When_2", 'I add "Mouse" to the cart', None, None),
            ("Then_1", "the cart should contain 2 items", None, None),
            ("Then_2", "the total price should be correct", None, None)
        ]
    }
    assert result == expected
    mock_file_obj.assert_called_once_with("test_and_steps.feature", 'r', encoding='utf-8')

def test_parse_feature_file_value_latency_extraction(mocker):
    """Tests correct extraction of 'value=' and 'latency=' from step text."""
    file_content = """
    Feature: API Tests

    Scenario: Send request with specific value and latency
      Given an API endpoint is available
      When I send a request with value=123 and latency=50ms
      Then the response should have status 200
      And the response body should contain data with value=abc
    """
    mock_file_obj = mock_feature_file_content(mocker, file_content)

    result = parse_feature_file("test_api.feature")

    expected = {
        "Send request with specific value and latency": [
            ("Given_1", "an API endpoint is available", None, None),
            ("When_1", "I send a request with value=123 and latency=50ms", "123", "50ms"),
            ("Then_1", "the response should have status 200", None, None),
            ("Then_2", "the response body should contain data with value=abc", "abc", None)
        ]
    }
    assert result == expected
    mock_file_obj.assert_called_once_with("test_api.feature", 'r', encoding='utf-8')

def test_parse_feature_file_comments_and_empty_lines(mocker):
    """Tests that comments and empty lines are correctly ignored."""
    file_content = """
    # This is a feature file
    Feature: Cleanup Process

    # Scenario for successful cleanup
    Scenario: Perform system cleanup
      Given the system is in a messy state # inline comment
      
      # Another comment line
      When I trigger the cleanup process
      
      Then the system should be clean
    """
    mock_file_obj = mock_feature_file_content(mocker, file_content)

    result = parse_feature_file("test_cleanup.feature")

    expected = {
        "Perform system cleanup": [
            ("Given_1", "the system is in a messy state", None, None),
            ("When_1", "I trigger the cleanup process", None, None),
            ("Then_1", "the system should be clean", None, None)
        ]
    }
    assert result == expected
    mock_file_obj.assert_called_once_with("test_cleanup.feature", 'r', encoding='utf-8')

def test_parse_feature_file_no_scenarios(mocker):
    """Tests a file that contains no scenarios, only features/comments."""
    file_content = """
    # Just comments and empty lines
    Feature: No Scenarios Here

    """
    mock_file_obj = mock_feature_file_content(mocker, file_content)

    result = parse_feature_file("test_no_scenarios.feature")

    assert result == {} # Expect an empty dictionary if no scenarios are found
    mock_file_obj.assert_called_once_with("test_no_scenarios.feature", 'r', encoding='utf-8')

def test_parse_feature_file_file_not_found(mocker, capfd):
    """Tests handling of FileNotFoundError."""
    # Mock open to raise FileNotFoundError when called
    mocker.patch('builtins.open', side_effect=FileNotFoundError)

    result = parse_feature_file("non_existent.feature")

    assert result == {} # The function should return an empty dict on error
    # Check that the error message was printed to stdout
    outerr = capfd.readouterr()
    assert "Error: Feature file not found at non_existent.feature" in outerr.out

def test_parse_feature_file_empty_file(mocker):
    """Tests parsing an entirely empty file."""
    file_content = ""
    mock_file_obj = mock_feature_file_content(mocker, file_content)

    result = parse_feature_file("empty.feature")

    assert result == {}
    mock_file_obj.assert_called_once_with("empty.feature", 'r', encoding='utf-8')

def test_parse_feature_file_step_outside_scenario_warning(mocker, capfd):
    """Tests that steps found outside a scenario trigger a warning and are ignored."""
    file_content = """
    Given this step is outside a scenario
    Scenario: Real Scenario
      Given a real step
    """
    mock_file_obj = mock_feature_file_content(mocker, file_content)

    result = parse_feature_file("test_step_outside.feature")

    # Expect only the 'Real Scenario' to be parsed
    expected = {
        "Real Scenario": [
            ("Given_1", "a real step", None, None)
        ]
    }
    assert result == expected

    # Check for the warning message in stdout
    outerr = capfd.readouterr()
    assert "Warning: Step 'Given this step is outside a scenario' found outside of a scenario in test_step_outside.feature line 2." in outerr.out

def test_parse_feature_file_and_without_context_warning(mocker, capfd):
    """Tests an 'And' step appearing without a preceding Given/When/Then."""
    file_content = """
    Feature: Independent And

    And this step has no preceding Given/When/Then
    Scenario: Valid Scenario
      Given a setup
      And another setup
    """
    mock_file_obj = mock_feature_file_content(mocker, file_content)

    result = parse_feature_file("test_and_no_context.feature")

    # The 'And' step outside the scenario should not be included in the result.
    # Only steps within 'Valid Scenario' should be parsed.
    expected = {
        "Valid Scenario": [
            ("Given_1", "a setup", None, None),
            ("Given_2", "another setup", None, None)
        ]
    }
    assert result == expected

    # Check for the warning message in stdout
    outerr = capfd.readouterr()
    assert "Warning: 'And' step found without preceding Given/When/Then in test_and_no_context.feature line 4. Treating as first step." in outerr.out

def test_parse_feature_file_general_exception_handling(mocker, capfd):
    """Tests that a general Exception during file processing is caught and reported."""
    # Simulate an arbitrary error during file processing (e.g., in the iteration)
    mock_file = mocker.mock_open(read_data="Scenario: Test\nGiven a step")
    # Make the mock file object raise an exception when read() is called or iterated
    mock_file.side_effect = Exception("Simulated parsing error")
    mocker.patch('builtins.open', mock_file)

    result = parse_feature_file("error.feature")

    assert result == {} # Should return empty dict on any parsing error
    outerr = capfd.readouterr()
    assert "Error parsing file error.feature: Simulated parsing error" in outerr.out

def test_parse_feature_file_success_message(mocker, capfd):
    """Tests that the 'Successfully parsed' message is printed on success."""
    file_content = """
    Feature: Success Test
    Scenario: Simple Success
      Given a step
    """
    mock_feature_file_content(mocker, file_content)

    parse_feature_file("success.feature")

    outerr = capfd.readouterr()
    assert "Successfully parsed success.feature" in outerr.out
