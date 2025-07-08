import pytest
from unittest.mock import patch, MagicMock

# Import the function to be tested
# Assume this function is in a file named 'your_module.py'
# For this example, we'll include it directly for self-containment.
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def map_variables_(matches_dict, model_matches, scenarios):
    """
    Maps Gherkin step text to Testbench IDs using fuzzy matching or a fallback
    to pre-trained model matches if fuzzy matching fails.

    Args:
        matches_dict (dict): A dictionary where keys are framework labels (strings)
                             representing Testbench IDs.
        model_matches (list): A list of tuples/lists, where each inner element is
                              expected to be like `(..., substring_to_match, mapped_id)`.
                              The `substring_to_match` is used for direct containment checks.
        scenarios (list): A list of scenarios, where each scenario is a list of steps.
                          Each step is expected to be a list/tuple like `[alias, step_text]`.

    Returns:
        list: A list of mapped scenarios. Each item is a tuple:
              `(original_scenario_list, alias, step_text, mapped_id_result)`.
              `mapped_id_result` is either `(match_string, score)` from fuzzywuzzy
              or the `mapped_id` from `model_matches`.
    """
    framework_labels_from_xml = list(matches_dict.keys())
    mapped_scenarios = []
    for scenario in scenarios:
        for step in scenario:
            step_text = step[1]
            alias = step[0]
            
            # Attempt fuzzy matching with framework labels
            # process.extractOne returns (match_string, score) or None if no good match
            mapping_from_framework_ids = process.extractOne(step_text, framework_labels_from_xml, scorer=fuzz.token_set_ratio)
            
            if mapping_from_framework_ids is not None:
                # Fuzzy match found, use it
                mapped_scenarios.append((scenario, alias, step_text, mapping_from_framework_ids))
            else:
                # No fuzzy match, try to find a match in model_matches
                found_in_model = False
                for model_match in model_matches:
                    # Check if the step_text contains the substring defined in model_match
                    # Assuming model_match is [..., substring_to_match, mapped_id]
                    if model_match and len(model_match) > 2 and model_match[1] in step_text:
                        mapping_from_model = model_match[2]
                        mapped_scenarios.append((scenario, alias, step_text, mapping_from_model))
                        found_in_model = True
                        break # Found a model match, stop checking other model_matches for this step
                # If neither fuzzy nor model match found, this step is simply skipped from the output.
    return mapped_scenarios

# --- Pytest Tests ---

# Fixture to mock fuzzywuzzy.process.extractOne for controlled testing
@pytest.fixture
def mock_extract_one(mocker):
    """Mocks fuzzywuzzy.process.extractOne to control its return behavior."""
    return mocker.patch('fuzzywuzzy.process.extractOne')

def test_perfect_fuzzy_match(mock_extract_one):
    """
    Tests that the function correctly maps a step when a perfect fuzzy match
    is found in `matches_dict`.
    """
    matches_dict = {"Framework Label A": "id1", "Framework Label B": "id2"}
    model_matches = [] # Model matches should not be used here
    scenarios = [[["alias1", "Framework Label A"]]]

    # Configure mock_extract_one to return a perfect match (score 100)
    mock_extract_one.return_value = ("Framework Label A", 100)

    expected_output = [
        (scenarios[0], "alias1", "Framework Label A", ("Framework Label A", 100))
    ]
    result = map_variables_(matches_dict, model_matches, scenarios)
    assert result == expected_output
    # Verify that extractOne was called correctly
    mock_extract_one.assert_called_once_with("Framework Label A", list(matches_dict.keys()), scorer=fuzz.token_set_ratio)

def test_partial_fuzzy_match(mock_extract_one):
    """
    Tests that the function correctly maps a step when a partial fuzzy match
    is found in `matches_dict`.
    """
    matches_dict = {"This is a test label": "id1"}
    model_matches = []
    scenarios = [[["alias1", "test label this is"]]]

    # Configure mock_extract_one to return a partial match with a good score
    mock_extract_one.return_value = ("This is a test label", 90) # Example score for a partial match

    expected_output = [
        (scenarios[0], "alias1", "test label this is", ("This is a test label", 90))
    ]
    result = map_variables_(matches_dict, model_matches, scenarios)
    assert result == expected_output
    mock_extract_one.assert_called_once_with("test label this is", list(matches_dict.keys()), scorer=fuzz.token_set_ratio)

def test_no_fuzzy_match_fallback_to_model(mock_extract_one):
    """
    Tests that if no fuzzy match is found, the function correctly falls back
    to using `model_matches`.
    """
    matches_dict = {"Framework Label A": "id1"}
    model_matches = [[None, "specific text", "MODEL_ID_123"]] # Example model match structure
    scenarios = [[["alias1", "Some unrelated text containing specific text"]]]

    # Configure mock_extract_one to return None, simulating no fuzzy match
    mock_extract_one.return_value = None

    expected_output = [
        (scenarios[0], "alias1", "Some unrelated text containing specific text", "MODEL_ID_123")
    ]
    result = map_variables_(matches_dict, model_matches, scenarios)
    assert result == expected_output
    mock_extract_one.assert_called_once_with("Some unrelated text containing specific text", list(matches_dict.keys()), scorer=fuzz.token_set_ratio)

def test_no_match_at_all(mock_extract_one):
    """
    Tests that a step is not mapped (and thus not included in the output)
    if neither a fuzzy match nor a model match is found.
    """
    matches_dict = {"Framework Label A": "id1"}
    model_matches = [[None, "specific text", "MODEL_ID_123"]]
    scenarios = [[["alias1", "Completely different text"]]]

    # Configure mock_extract_one to return None (no fuzzy match)
    mock_extract_one.return_value = None

    expected_output = [] # The step should be skipped
    result = map_variables_(matches_dict, model_matches, scenarios)
    assert result == expected_output
    mock_extract_one.assert_called_once_with("Completely different text", list(matches_dict.keys()), scorer=fuzz.token_set_ratio)

def test_multiple_scenarios_and_steps(mock_extract_one, mocker):
    """
    Tests the function with multiple scenarios and steps, including a mix of
    fuzzy matches, model fallbacks, and unmapped steps.
    """
    matches_dict = {
        "Fuzzy Match One": "f1",
        "Another Fuzzy Label": "f2"
    }
    model_matches = [
        [None, "model_text_A", "MODEL_A"],
        [None, "model_text_B", "MODEL_B"]
    ]
    scenarios = [
        [["s1_alias1", "Fuzzy Match One"]], # Should get a perfect fuzzy match
        [["s2_alias1", "Step with model_text_A"], ["s2_alias2", "No match here"]], # Model match, then no match
        [["s3_alias1", "Another Fuzzy Label Is Here"]] # Should get a partial fuzzy match
    ]

    # Use side_effect to provide different return values for sequential calls to extractOne
    mock_extract_one.side_effect = [
        ("Fuzzy Match One", 100), # For "Fuzzy Match One"
        None, # For "Step with model_text_A" (forces model fallback)
        None, # For "No match here" (no fuzzy, no model)
        ("Another Fuzzy Label", 85) # For "Another Fuzzy Label Is Here"
    ]

    expected_output = [
        (scenarios[0], "s1_alias1", "Fuzzy Match One", ("Fuzzy Match One", 100)),
        (scenarios[1], "s2_alias1", "Step with model_text_A", "MODEL_A"),
        (scenarios[2], "s3_alias1", "Another Fuzzy Label Is Here", ("Another Fuzzy Label", 85))
    ]
    result = map_variables_(matches_dict, model_matches, scenarios)
    assert result == expected_output

    # Verify all expected calls to extractOne were made
    assert mock_extract_one.call_count == 4
    mock_extract_one.assert_has_calls([
        mocker.call("Fuzzy Match One", list(matches_dict.keys()), scorer=fuzz.token_set_ratio),
        mocker.call("Step with model_text_A", list(matches_dict.keys()), scorer=fuzz.token_set_ratio),
        mocker.call("No match here", list(matches_dict.keys()), scorer=fuzz.token_set_ratio),
        mocker.call("Another Fuzzy Label Is Here", list(matches_dict.keys()), scorer=fuzz.token_set_ratio),
    ])

def test_empty_scenarios():
    """Tests that an empty `scenarios` list results in an empty output."""
    matches_dict = {"Label": "id"}
    model_matches = [[None, "text", "model_id"]]
    scenarios = []
    
    result = map_variables_(matches_dict, model_matches, scenarios)
    assert result == []

def test_empty_matches_dict(mock_extract_one):
    """
    Tests that if `matches_dict` is empty, fuzzy matching fails (returns None),
    and the function correctly falls back to `model_matches`.
    """
    matches_dict = {} # Empty dictionary
    model_matches = [[None, "some text", "MODEL_A"]]
    scenarios = [[["alias1", "This step has some text"]]]

    # extractOne will be called with an empty list of choices, which should return None
    mock_extract_one.return_value = None

    expected_output = [
        (scenarios[0], "alias1", "This step has some text", "MODEL_A")
    ]
    result = map_variables_(matches_dict, model_matches, scenarios)
    assert result == expected_output
    # Verify extractOne was called with an empty list for framework_labels_from_xml
    mock_extract_one.assert_called_once_with("This step has some text", [], scorer=fuzz.token_set_ratio)

def test_empty_model_matches(mock_extract_one, mocker):
    """
    Tests that if `model_matches` is empty, only fuzzy search is attempted.
    If fuzzy fails, the step should be unmapped.
    """
    matches_dict = {"Fuzzy Label": "id"}
    model_matches = [] # Empty list
    scenarios = [
        [["alias1", "Fuzzy Label"]], # Should fuzzy match
        [["alias2", "No Match Text"]] # Should not match (no fuzzy, no model)
    ]

    # Configure side_effect for two calls
    mock_extract_one.side_effect = [
        ("Fuzzy Label", 100), # For the first step
        None # For the second step
    ]

    expected_output = [
        (scenarios[0], "alias1", "Fuzzy Label", ("Fuzzy Label", 100))
    ]
    result = map_variables_(matches_dict, model_matches, scenarios)
    assert result == expected_output
    # Verify both calls to extractOne were made
    assert mock_extract_one.call_count == 2
    mock_extract_one.assert_has_calls([
        mocker.call("Fuzzy Label", list(matches_dict.keys()), scorer=fuzz.token_set_ratio),
        mocker.call("No Match Text", list(matches_dict.keys()), scorer=fuzz.token_set_ratio)
    ])

def test_multiple_model_matches_for_same_step(mock_extract_one):
    """
    Tests that if multiple `model_matches` could apply to a step, only the
    first one found in the iteration order is used.
    """
    matches_dict = {} # Ensure no fuzzy match
    model_matches = [
        [None, "common text", "MODEL_A_FIRST"],
        [None, "common text", "MODEL_B_SECOND"] # This should NOT be used
    ]
    scenarios = [[["alias1", "This step has common text"]]]

    mock_extract_one.return_value = None # No fuzzy match

    expected_output = [
        (scenarios[0], "alias1", "This step has common text", "MODEL_A_FIRST")
    ]
    result = map_variables_(matches_dict, model_matches, scenarios)
    assert result == expected_output
    # Verify extractOne was called once for the fuzzy check
    mock_extract_one.assert_called_once()