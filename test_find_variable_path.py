from find_variable_path import select_json_file,load_training_data, train_and_save_model, load_model_components, predict_framework_label_from_step # Assuming fuzzy_search_scenario_steps is also imported if used
import pytest
import os
import json
import pickle
from unittest.mock import patch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier # Added for single-class test assertion
from fuzzywuzzy import fuzz, process # Assuming these are used by fuzzy_search_scenario_steps

# --- Helper functions for test setup ---

@pytest.fixture
def temp_dir(tmp_path):
    """Creates a temporary directory for test files and cleans it up."""
    return tmp_path

@pytest.fixture
def dummy_training_data_path(temp_dir):
    """Creates a dummy JSON training data file."""
    data = [
        {"sentence": "Send Vehicle_Speed_Eng = 10", "label": "Vehicle_Speed_Eng_TA_Replace_Value"},
        {"sentence": "Send Vehicle_Speed_Brk = 5", "label": "Vehicle_Speed_Brk_TA_Replace_Value"},
        {"sentence": "Turn vehicle On", "label": "Ignition_TA_Replace_Value"},
        {"sentence": "Open driver door", "label": "Door_Ajar_TA_Replace_Value"},
        {"sentence": "Activate radio", "label": "Radio_TA_Replace_Value"},
    ]
    file_path = temp_dir / "dummy_training_data.json"
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    return str(file_path)

@pytest.fixture
def dummy_training_data_empty_path(temp_dir):
    """Creates an empty JSON training data file."""
    file_path = temp_dir / "dummy_training_data_empty.json"
    with open(file_path, 'w') as f:
        json.dump([], f)
    return str(file_path)

@pytest.fixture
def dummy_training_data_malformed_path(temp_dir):
    """Creates a malformed JSON training data file."""
    file_path = temp_dir / "dummy_training_data_malformed.json"
    with open(file_path, 'w') as f:
        f.write("{invalid json")
    return str(file_path)

@pytest.fixture
def dummy_training_data_missing_cols_path(temp_dir):
    """Creates a JSON training data file missing required columns."""
    data = [
        {"text": "some text", "category": "some_category"},
    ]
    file_path = temp_dir / "dummy_training_data_missing_cols.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return str(file_path)

@pytest.fixture
def trained_ml_components(dummy_training_data_path, temp_dir):
    """Trains and saves ML components, returning their paths and the loaded components."""
    model_file_path = str(temp_dir / "model.pkl")
    vectorizer_file_path = str(temp_dir / "vectorizer.pkl")
    label_encoder_file_path = str(temp_dir / "label_encoder.pkl")

    data = load_training_data(dummy_training_data_path)
    train_and_save_model(data, model_file_path, vectorizer_file_path, label_encoder_file_path)
    
    model, vectorizer, label_encoder = load_model_components(
        model_file_path,
        vectorizer_file_path,
        label_encoder_file_path
    )
    return model, vectorizer, label_encoder, model_file_path, vectorizer_file_path, label_encoder_file_path

@pytest.fixture
def sample_scenario_steps():
    """Provides sample scenario steps for testing."""
    return [
        [
            ("step_1", "Send Vehicle_Speed_Eng = 10"),
            ("step_2", "Send Vehicle_Speed_Brk = 5"),
        ],
        [
            ("step_3", "Turn vehicle On"),
            ("step_4", "Open driver door"),
            ("step_5", "Activate radio"),
            ("step_6", "Completely unrelated phrase"), # This one should not map
        ]
    ]

@pytest.fixture
def sample_fuzzy_matches_dict(): # Renamed for clarity: keys are input sentences
    """Provides a sample matches_dict for fuzzy search validation."""
    return {
        "Send Vehicle_Speed_Eng = 10": "Vehicle_Speed_Eng_TA_Replace_Value",
        "Send Vehicle_Speed_Brk = 5": "Vehicle_Speed_Brk_TA_Replace_Value",
        "Turn vehicle On": "Ignition_TA_Replace_Value",
        "Open driver door": "Door_Ajar_TA_Replace_Value",
        "Activate radio": "Radio_TA_Replace_Value",
    }

@pytest.fixture
def sample_ml_matches_dict(): # NEW FIXTURE: keys are predicted labels
    """Provides a sample matches_dict where keys are ML-predicted labels."""
    return {
        "Vehicle_Speed_Eng_TA_Replace_Value": "Vehicle_Speed_Eng_TA_Replace_Value",
        "Vehicle_Speed_Brk_TA_Replace_Value": "Vehicle_Speed_Brk_TA_Replace_Value",
        "Ignition_TA_Replace_Value": "Ignition_TA_Replace_Value",
        "Door_Ajar_TA_Replace_Value": "Door_Ajar_TA_Replace_Value",
        "Radio_TA_Replace_Value": "Radio_TA_Replace_Value",
    }

# --- Tests for fuzzy_search_scenario_steps ---
# Assuming fuzzy_search_scenario_steps is imported and works as intended from find_variable_path.py
# (i.e., it returns a list of [current_scenario, step, mapped_value] and correctly uses matches_dict[result[0]])

# Placeholder for fuzzy_search_scenario_steps if it's not actually in find_variable_path.py
# If you have fuzzy_search_scenario_steps in find_variable_path.py, remove this placeholder.
def fuzzy_search_scenario_steps(matches_dict, scenario_steps):
    fuzzy_match = []
    for current_scenario in scenario_steps:
        for step_id, step_text in current_scenario:
            lowercase_step = step_text.lower()
            # Use process.extractOne with the values (keys of matches_dict)
            # and a threshold.
            result = process.extractOne(lowercase_step, list(matches_dict.keys()), scorer=fuzz.token_set_ratio)
            if result and result[1] > 80: # Set fuzzy threshold
                fuzzy_match.append([current_scenario, (step_id, step_text), matches_dict[result[0]]])
    return fuzzy_match


def test_fuzzy_search_scenario_steps_basic(sample_fuzzy_matches_dict, sample_scenario_steps):
    results = fuzzy_search_scenario_steps(sample_fuzzy_matches_dict, sample_scenario_steps)

    # Expected matches: all steps in sample_scenario_steps except "Completely unrelated phrase" (5 matches)
    assert len(results) == 5

    # Extract just the mapped values for easier assertion
    actual_mapped_values = sorted([item[2] for item in results])

    expected_mapped_values = sorted([
        "Vehicle_Speed_Eng_TA_Replace_Value",
        "Vehicle_Speed_Brk_TA_Replace_Value",
        "Ignition_TA_Replace_Value",
        "Door_Ajar_TA_Replace_Value",
        "Radio_TA_Replace_Value"
    ])
    assert actual_mapped_values == expected_mapped_values

    # Ensure "Completely unrelated phrase" was not mapped by fuzzy search
    unrelated_phrase_mapped_fuzzy = False
    for scenario, step, mapped_value in results:
        if step[1] == "Completely unrelated phrase":
            unrelated_phrase_mapped_fuzzy = True
            break
    assert not unrelated_phrase_mapped_fuzzy

def test_fuzzy_search_scenario_steps_no_match(sample_fuzzy_matches_dict):
    scenario_steps_no_match = [
        [("step_a", "This is a very unique phrase that won't match")],
        [("step_b", "Another completely irrelevant sentence")],
    ]
    results = fuzzy_search_scenario_steps(sample_fuzzy_matches_dict, scenario_steps_no_match)
    assert len(results) == 0

def test_fuzzy_search_scenario_steps_empty_inputs():
    assert fuzzy_search_scenario_steps({}, []) == []
    assert fuzzy_search_scenario_steps({"key": "value"}, []) == []
    assert fuzzy_search_scenario_steps({}, [[("id", "text")]]) == []

# --- Tests for select_json_file ---

@patch('tkinter.filedialog.askopenfilename', return_value="/path/to/test.json")
@patch('tkinter.Tk')
def test_select_json_file_returns_path(mock_tk, mock_askopenfilename):
    path = select_json_file()
    mock_tk.assert_called_once()
    mock_tk.return_value.withdraw.assert_called_once()
    mock_askopenfilename.assert_called_once_with(
        title="Select .json file",
        filetypes=[("json files", "*.json"), ("All files", "*.*")]
    )
    assert path == "/path/to/test.json"

@patch('tkinter.filedialog.askopenfilename', return_value="")
@patch('tkinter.Tk')
def test_select_json_file_no_selection(mock_tk, mock_askopenfilename):
    path = select_json_file()
    assert path == ""

# --- Tests for load_training_data ---

def test_load_training_data_success(dummy_training_data_path):
    data = load_training_data(dummy_training_data_path)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "sentence" in data[0]
    assert "label" in data[0]

def test_load_training_data_file_not_found():
    data = load_training_data("non_existent_file.json")
    assert data == []

def test_load_training_data_malformed_json(dummy_training_data_malformed_path):
    data = load_training_data(dummy_training_data_malformed_path)
    assert data == []

def test_load_training_data_empty_file(dummy_training_data_empty_path):
    data = load_training_data(dummy_training_data_empty_path)
    assert data == []

# --- Tests for train_and_save_model ---

def test_train_and_save_model_success(dummy_training_data_path, temp_dir):
    model_file_path = str(temp_dir / "model.pkl")
    vectorizer_file_path = str(temp_dir / "vectorizer.pkl")
    label_encoder_file_path = str(temp_dir / "label_encoder.pkl")
    
    data = load_training_data(dummy_training_data_path)
    train_and_save_model(data, model_file_path, vectorizer_file_path, label_encoder_file_path)

    assert os.path.exists(model_file_path)
    assert os.path.exists(vectorizer_file_path)
    assert os.path.exists(label_encoder_file_path)

def test_train_and_save_model_no_data(temp_dir): # Added temp_dir fixture to ensure clean paths
    model_file_path = str(temp_dir / "model_no_data.pkl")
    vectorizer_file_path = str(temp_dir / "vec_no_data.pkl")
    label_encoder_file_path = str(temp_dir / "le_no_data.pkl")
    train_and_save_model([], model_file_path, vectorizer_file_path, label_encoder_file_path)
    assert not os.path.exists(model_file_path)
    assert not os.path.exists(vectorizer_file_path)
    assert not os.path.exists(label_encoder_file_path)

def test_train_and_save_model_missing_cols(dummy_training_data_missing_cols_path, temp_dir):
    model_file_path = str(temp_dir / "model_missing_cols.pkl")
    vectorizer_file_path = str(temp_dir / "vec_missing_cols.pkl")
    label_encoder_file_path = str(temp_dir / "le_missing_cols.pkl")
    train_and_save_model(load_training_data(dummy_training_data_missing_cols_path), model_file_path, vectorizer_file_path, label_encoder_file_path)
    assert not os.path.exists(model_file_path)
    assert not os.path.exists(vectorizer_file_path)
    assert not os.path.exists(label_encoder_file_path)

# --- Tests for load_model_components ---

def test_load_model_components_success(trained_ml_components):
    # Unpack only the component objects
    model, vectorizer, label_encoder = trained_ml_components[0:3] 
    assert isinstance(model, SGDClassifier)
    assert isinstance(vectorizer, TfidfVectorizer)
    assert isinstance(label_encoder, LabelEncoder)

def test_load_model_components_file_not_found(temp_dir):
    model, vectorizer, label_encoder = load_model_components(
        str(temp_dir / "non_existent_model.pkl"),
        str(temp_dir / "non_existent_vec.pkl"),
        str(temp_dir / "non_existent_le.pkl")
    )
    assert model is None
    assert vectorizer is None
    assert label_encoder is None

def test_load_model_components_corrupted_file(temp_dir):
    corrupted_file = temp_dir / "corrupted.pkl"
    with open(corrupted_file, 'wb') as f:
        f.write(b'this is not a pickle')

    model, vectorizer, label_encoder = load_model_components(
        str(corrupted_file),
        str(corrupted_file),
        str(corrupted_file)
    )
    assert model is None
    assert vectorizer is None
    assert label_encoder is None

# --- Tests for predict_framework_label_from_scenario_steps ---

# Make sure `predict_framework_label_from_step` from `find_variable_path.py`
# has been updated as per previous suggestions:
# 1. Returns `[]` (empty list) instead of `None` when components are missing or on error.
# 2. Correctly appends `(current_scenario, step, matches_dict[predicted_label])` to `model_match`.
# 3. Type hint is `-> List[Tuple[List[Tuple[str, str]], Tuple[str, str], str]]`

def test_predict_framework_label_from_scenario_steps_no_ml_components(
    sample_ml_matches_dict, sample_scenario_steps): # Use sample_ml_matches_dict
    
    results = predict_framework_label_from_step(
        model=None,
        vectorizer=None,
        label_encoder=None,
        matches_dict=sample_ml_matches_dict,
        scenario_steps=sample_scenario_steps
    )
    assert results == [] # Should return an empty list now, not None

def test_predict_framework_label_from_scenario_steps_empty_scenario_steps(
    trained_ml_components, sample_ml_matches_dict): # Use sample_ml_matches_dict
    
    model, vectorizer, label_encoder = trained_ml_components[0:3]
    results = predict_framework_label_from_step(
        model=model,
        vectorizer=vectorizer,
        label_encoder=label_encoder,
        matches_dict=sample_ml_matches_dict,
        scenario_steps=[]
    )
    assert results == []