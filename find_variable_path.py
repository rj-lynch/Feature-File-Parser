from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import json
from typing import List, Dict, Tuple, Any, Optional, Union
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# --- Constants ---
MODEL_FILE = 'signal_prediction_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
LABEL_ENCODER_FILE = 'label_encoder.pkl'

def select_json_file():
    """Opens a file dialog to select XML file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    json_path = filedialog.askopenfilename(
        title="Select .json file",
        filetypes=[("json files", "*.json"), ("All files", "*.*")]
    )
    return json_path

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
        # The model could be a DummyClassifier, so return type should allow for Any
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

def predict_framework_label_from_step(
    step_text: str,
    model: Optional[Any],
    vectorizer: Optional[TfidfVectorizer],
    label_encoder: Optional[LabelEncoder],
    matches_dict, 
    scenario_steps
) -> Optional[str]:
    """
    Predicts the most likely FrameworkLabel for a given scenario step text
    using the pre-trained machine learning model.

    Args:
        step_text (str): The text of the scenario step (e.g., "click the login button").
        model (Optional[Any]): The loaded ML model (SGDClassifier or DummyClassifier).
        vectorizer (Optional[TfidfVectorizer]): The loaded TF-IDF vectorizer.
        label_encoder (Optional[LabelEncoder]): The loaded label encoder.
        valid_framework_labels (List[str]): A list of all valid FrameworkLabel IDs
                                             expected from your XML configuration.
                                             The prediction will be validated against this list.

    Returns:
        Optional[str]: The predicted FrameworkLabel ID if a valid prediction is made,
                       otherwise None.
    """
    model_match = []

    for current_scenario in scenario_steps:
        for step in current_scenario:
            lowercase_step = step[1].lower()
            print(f"\n[ML Predict] Processing step: '{step[1]}'")

            if not model or not vectorizer or not label_encoder:
                print("[ML Predict] ML components are missing or not loaded. Cannot predict.")
                return None

            # Check if the model/encoder are properly initialized for prediction
            if not (hasattr(model, 'predict') and hasattr(label_encoder, 'inverse_transform') and len(label_encoder.classes_) > 0):
                print("[ML Predict] ML model/encoder not fully initialized or not suitable for prediction.")
                return None

            try:
                # Transform the input text into numerical features
                X = vectorizer.transform([lowercase_step])

                # Get the numerical prediction from the model
                y_pred = model.predict(X)

                # Convert the numerical prediction back to a human-readable label
                predicted_label = label_encoder.inverse_transform(y_pred)[0]

                # Validate the predicted label against the list of known FrameworkLabels
                if predicted_label in matches_dict:
                    print(f"[ML Predict] Predicted FrameworkLabel: {predicted_label}")
                    model_match.append(current_scenario,step,matches_dict[predicted_label])
                    
                else:
                    print(f"[ML Predict] Predicted label '{predicted_label}' not found in valid FrameworkLabels. Returning None.")
                    
            except Exception as e:
                print(f"[ML Predict] Error during ML prediction: {e}")
                return None
    return model_match