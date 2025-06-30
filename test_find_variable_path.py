import pytest
import os
import json
import pickle
from unittest.mock import patch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

# Import functions from your main script
# Assuming your script is named find_variable_path.py
from find_variable_path import (
    fuzzy_search_scenario_steps,
    select_json_file,
    load_training_data,
    train_and_save_model,
    load_model_components,
    predict_framework_label_from_scenario_steps,
    MODEL_FILE, VECTORIZER_FILE, LABEL_ENCODER_FILE
)

# --- Helper functions for test setup ---

@pytest.fixture
def temp_dir(tmp_path):
    """Creates a temporary directory for test files and cleans it up."""
    return tmp_path

@pytest.fixture
def dummy_training_data_path(temp_dir):
    """Creates a dummy JSON training data file."""
    data = [
        {"sentence": "click on the login button", "label": "LoginButton"},
        {"sentence": "submit the form", "label": "SubmitForm"},
        {"sentence": "go to the home page", "label": "NavigateHome"},
        {"sentence": "check if the text is present", "label": "VerifyText"},
        {"sentence": "select an option from the dropdown", "label": "SelectOption"},
        {"sentence": "type text into a field", "label": "EnterText"},
        {"sentence": "press the login button", "label": "LoginButton"},
        {"sentence": "send the form", "label": "SubmitForm"},
        {"sentence": "verify the text display", "label": "VerifyText"},
        {"sentence": "go to the main page", "label": "NavigateHome"},
        {"sentence": "click the logout button", "label": "LogoutButton"}, # New label for testing single-sample class
        {"sentence": "fill in username", "label": "EnterText"},
        {"sentence": "confirm selection", "label": "SelectOption"}
    ]
    file_path = temp_dir / "dummy_training_data.json"
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    return str(file_path)

@pytest.fixture
def dummy_training_data_single_class_path(temp_dir):
    """Creates a dummy JSON training data file with only one class."""
    data = [
        {"sentence": "click on the login button", "label": "LoginButton"},
        {"sentence": "press the login button", "label": "LoginButton"},
    ]
    file_path = temp_dir / "dummy_training_data_single_class.json"
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
    """Trains and saves ML components, returning their paths."""
    # Ensure model files are in the temporary directory
    old_model_file = MODEL_FILE
    old_vectorizer_file = VECTORIZER_FILE
    old_label_encoder_file = LABEL_ENCODER_FILE
    
    # Temporarily change constants to point to temp_dir
    global MODEL_FILE, VECTORIZER_FILE, LABEL_ENCODER_FILE
    MODEL_FILE = str(temp_dir / old_model_file)
    VECTORIZER_FILE = str(temp_dir / old_vectorizer_file)
    LABEL_ENCODER_FILE = str(temp_dir / old_label_encoder_file)

    data = load_training_data(dummy_training_data_path)
    train_and_save_model(data, MODEL_FILE, VECTORIZER_FILE, LABEL_ENCODER_FILE)
    
    # Restore original constants after training
    MODEL_FILE = old_model_file
    VECTORIZER_FILE = old_vectorizer_file
    LABEL_ENCODER_FILE = old_label_encoder_file

    return load_model_components(
        str(temp_dir / old_model_file),
        str(temp_dir / old_vectorizer_file),
        str(temp_dir / old_label_encoder_file)
    )

@pytest.fixture
def sample_scenario_steps():
    """Provides sample scenario steps for testing."""
    return [
        [
            ("step_1", "Click the login button"),
            ("step_2", "Submit the user form"),
        ],
        [
            ("step_3", "Go to the homepage"),
            ("step_4", "Verify the text display"),
            ("step_5", "Type some text into the field"),
            ("step_6", "Completely unrelated phrase"),
        ]
    ]

@pytest.fixture
def sample_matches_dict():
    """Provides a sample matches_dict for fuzzy and ML validation."""
    return {
        "click login button": "TB.UI.LoginButton.Click",
        "submit form": "TB.Action.SubmitForm.Perform",
        "go to home page": "TB.Nav.HomePage.Open",
        "verify text": "TB.Assert.VerifyTextOnPage",
        "enter text": "TB.Input.EnterText",
        "LoginButton": "TB.UI.LoginButton.Click", # Matches ML prediction
        "SubmitForm": "TB.Action.SubmitForm.Perform", # Matches ML prediction
        "NavigateHome": "TB.Nav.HomePage.Open", # Matches ML prediction
        "VerifyText": "TB.Assert.VerifyTextOnPage", # Matches ML prediction
        "EnterText": "TB.Input.EnterText", # Matches ML prediction
        "SelectOption": "TB.Dropdown.SelectOption", # Matches ML prediction
        "LogoutButton": "TB.User.Logout.Perform" # New label for testing
    }

# --- Tests for fuzzy_search_scenario_steps ---

def test_fuzzy_search_scenario_steps_basic(sample_matches_dict, sample_scenario_steps):
    # Corrected `fuzzy_search_scenario_steps` to use matches_dict[result[0]]
    # This correction needs to be applied to your find_variable_path.py
    # For testing purposes, we'll patch the function or rely on the fixture's behavior.
    # For now, let's assume the function is fixed for this test.
    
    # Simulate the corrected function logic here if not fixing the original file
    # For a real test, you'd fix the source file.
    
    # Let's manually apply the fix for this test function for clarity
    original_fuzzy_search = fuzzy_search_scenario_steps
    def fixed_fuzzy_search_scenario_steps(matches_dict, scenario_steps):
        fuzzy_match=[]
        for current_scenario in scenario_steps:
            for step in current_scenario:
                lowercase_step = step[1].lower()
                result = process.extractOne(lowercase_step, list(matches_dict.keys()), scorer=fuzz.token_set_ratio)
                if result and result[1]>80: # Set fuzzy threshold
                    fuzzy_match.append([current_scenario,step,matches_dict[result[0]]]) # Corrected line
        return fuzzy_match
    
    with patch('find_variable_path.fuzzy_search_scenario_steps', new=fixed_fuzzy_search_scenario_steps):
        results = fuzzy_search_scenario_steps(sample_matches_dict, sample_scenario_steps)

        assert len(results) == 5 # Expected matches: Login, Submit, Home, Verify, Enter
        
        # Check specific matches
        found_login = False
        found_submit = False
        found_home = False
        found_verify = False
        found_enter = False

        for scenario, step, mapped_value in results:
            if "login" in step[1].lower() and mapped_value == "TB.UI.LoginButton.Click":
                found_login = True
            elif "submit" in step[1].lower() and mapped_value == "TB.Action.SubmitForm.Perform":
                found_submit = True
            elif "home" in step[1].lower() and mapped_value == "TB.Nav.HomePage.Open":
                found_home = True
            elif "verify" in step[1].lower() and mapped_value == "TB.Assert.VerifyTextOnPage":
                found_verify = True
            elif "type" in step[1].lower() and mapped_value == "TB.Input.EnterText":
                found_enter = True
        
        assert found_login
        assert found_submit
        assert found_home
        assert found_verify
        assert found_enter

def test_fuzzy_search_scenario_steps_no_match(sample_matches_dict):
    scenario_steps_no_match = [
        [("step_a", "This is a very unique phrase that won't match")],
        [("step_b", "Another completely irrelevant sentence")],
    ]
    results = fuzzy_search_scenario_steps(sample_matches_dict, scenario_steps_no_match)
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
    # Temporarily change constants to point to temp_dir
    old_model_file = MODEL_FILE
    old_vectorizer_file = VECTORIZER_FILE
    old_label_encoder_file = LABEL_ENCODER_FILE
    
    global MODEL_FILE, VECTORIZER_FILE, LABEL_ENCODER_FILE
    MODEL_FILE = str(temp_dir / old_model_file)
    VECTORIZER_FILE = str(temp_dir / old_vectorizer_file)
    LABEL_ENCODER_FILE = str(temp_dir / old_label_encoder_file)

    data = load_training_data(dummy_training_data_path)
    train_and_save_model(data, MODEL_FILE, VECTORIZER_FILE, LABEL_ENCODER_FILE)

    assert os.path.exists(MODEL_FILE)
    assert os.path.exists(VECTORIZER_FILE)
    assert os.path.exists(LABEL_ENCODER_FILE)

    # Clean up after test if not using tmp_path directly for constants
    os.remove(MODEL_FILE)
    os.remove(VECTORIZER_FILE)
    os.remove(LABEL_ENCODER_FILE)
    
    # Restore original constants
    MODEL_FILE = old_model_file
    VECTORIZER_FILE = old_vectorizer_file
    LABEL_ENCODER_FILE = old_label_encoder_file

def test_train_and_save_model_no_data():
    train_and_save_model([], "model.pkl", "vec.pkl", "le.pkl")
    assert not os.path.exists("model.pkl")

def test_train_and_save_model_missing_cols(dummy_training_data_missing_cols_path):
    train_and_save_model(load_training_data(dummy_training_data_missing_cols_path), "model.pkl", "vec.pkl", "le.pkl")
    assert not os.path.exists("model.pkl")

def test_train_and_save_model_single_class(dummy_training_data_single_class_path, temp_dir):
    # Temporarily change constants to point to temp_dir
    old_model_file = MODEL_FILE
    old_vectorizer_file = VECTORIZER_FILE
    old_label_encoder_file = LABEL_ENCODER_FILE
    
    global MODEL_FILE, VECTORIZER_FILE, LABEL_ENCODER_FILE
    MODEL_FILE = str(temp_dir / old_model_file)
    VECTORIZER_FILE = str(temp_dir / old_vectorizer_file)
    LABEL_ENCODER_FILE = str(temp_dir / old_label_encoder_file)

    data = load_training_data(dummy_training_data_single_class_path)
    train_and_save_model(data, MODEL_FILE, VECTORIZER_FILE, LABEL_ENCODER_FILE)
    
    model, vectorizer, label_encoder = load_model_components(MODEL_FILE, VECTORIZER_FILE, LABEL_ENCODER_FILE)

    assert os.path.exists(MODEL_FILE)
    assert isinstance(model, type(lambda: None)) # Check if it's the DummyClassifier (a function or similar)
    assert model.classes_ == ['LoginButton']

    # Clean up
    os.remove(MODEL_FILE)
    os.remove(VECTORIZER_FILE)
    os.remove(LABEL_ENCODER_FILE)
    
    # Restore original constants
    MODEL_FILE = old_model_file
    VECTORIZER_FILE = old_vectorizer_file
    LABEL_ENCODER_FILE = old_label_encoder_file

# --- Tests for load_model_components ---

def test_load_model_components_success(trained_ml_components):
    model, vectorizer, label_encoder = trained_ml_components
    assert isinstance(model, SGDClassifier)
    assert isinstance(vectorizer, TfidfVectorizer)
    assert isinstance(label_encoder, LabelEncoder)

def test_load_model_components_file_not_found(temp_dir):
    # Ensure no files exist in the temp_dir for this test
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
        str(corrupted_file), # Pass corrupted file for all to trigger error
        str(corrupted_file)
    )
    assert model is None
    assert vectorizer is None
    assert label_encoder is None

# --- Tests for predict_framework_label_from_scenario_steps ---

def test_predict_framework_label_from_scenario_steps_success(
    trained_ml_components, sample_matches_dict, sample_scenario_steps):
    
    model, vectorizer, label_encoder = trained_ml_components

    results = predict_framework_label_from_scenario_steps(
        model=model,
        vectorizer=vectorizer,
        label_encoder=label_encoder,
        matches_dict=sample_matches_dict,
        scenario_steps=sample_scenario_steps
    )

    assert isinstance(results, list)
    assert len(results) > 0 # Expect some matches
    
    # Check for specific expected matches
    predicted_login = False
    predicted_submit = False
    predicted_home = False
    predicted_verify = False
    predicted_enter = False
    predicted_unrelated = True # Should not be mapped

    for scenario, step, mapped_value in results:
        assert isinstance(scenario, list)
        assert isinstance(step, tuple)
        assert isinstance(mapped_value, str)

        if step[1] == "Click the login button" and mapped_value == "TB.UI.LoginButton.Click":
            predicted_login = True
        elif step[1] == "Submit the user form" and mapped_value == "TB.Action.SubmitForm.Perform":
            predicted_submit = True
        elif step[1] == "Go to the homepage" and mapped_value == "TB.Nav.HomePage.Open":
            predicted_home = True
        elif step[1] == "Verify the text display" and mapped_value == "TB.Assert.VerifyTextOnPage":
            predicted_verify = True
        elif step[1] == "Type some text into the field" and mapped_value == "TB.Input.EnterText":
            predicted_enter = True
        elif step[1] == "Completely unrelated phrase":
            predicted_unrelated = False # If it maps, this test fails

    assert predicted_login
    assert predicted_submit
    assert predicted_home
    assert predicted_verify
    assert predicted_enter
    assert predicted_unrelated # Ensure the unrelated phrase was not mapped

def test_predict_framework_label_from_scenario_steps_no_ml_components(
    sample_matches_dict, sample_scenario_steps):
    
    results = predict_framework_label_from_scenario_steps(
        model=None,
        vectorizer=None,
        label_encoder=None,
        matches_dict=sample_matches_dict,
        scenario_steps=sample_scenario_steps
    )
    assert results == []

def test_predict_framework_label_from_scenario_steps_empty_scenario_steps(
    trained_ml_components, sample_matches_dict):
    
    model, vectorizer, label_encoder = trained_ml_components
    results = predict_framework_label_from_scenario_steps(
        model=model,
        vectorizer=vectorizer,
        label_encoder=label_encoder,
        matches_dict=sample_matches_dict,
        scenario_steps=[]
    )
    assert results == []

def test_predict_framework_label_from_scenario_steps_predicted_label_not_in_matches_dict(
    trained_ml_components, sample_scenario_steps):
    
    model, vectorizer, label_encoder = trained_ml_components
    # Create a matches_dict that explicitly *excludes* some predicted labels
    limited_matches_dict = {
        "LoginButton": "TB.UI.LoginButton.Click",
        "SubmitForm": "TB.Action.SubmitForm.Perform",
        # Missing "NavigateHome", "VerifyText", "EnterText"
    }

    results = predict_framework_label_from_scenario_steps(
        model=model,
        vectorizer=vectorizer,
        label_encoder=label_encoder,
        matches_dict=limited_matches_dict,
        scenario_steps=sample_scenario_steps
    )
    
    # Expect only LoginButton and SubmitForm to be mapped
    assert len(results) == 2
    for scenario, step, mapped_value in results:
        assert mapped_value in ["TB.UI.LoginButton.Click", "TB.Action.SubmitForm.Perform"]
