from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# --- Fuzzy search framework labelIDs to find correct Testbench ID.
#  --- If no matches are found use ML to select ID instead.
def map_variables_(matches_dict, model_prediction, scenarios):
    framework_labels_from_xml = list(matches_dict.keys())
    for scenario in scenarios:
        for step in scenario:
            step_text = step[1]
            mapping_from_framework_ids = process.extractOne(step_text, framework_labels_from_xml, scorer=fuzz.token_set_ratio)
            if mapping_from_framework_ids != None:
                updated_step = step.append(mapping_from_framework_ids)
            else:
                step_text_lower=step_text.lower()
                model: Optional[SGDClassifier],  # Model can be None
                vectorizer: Optional[TfidfVectorizer],  # Vectorizer can be None
                label_encoder: Optional[LabelEncoder],  # LabelEncoder can be None
                if framework_labels_from_xml:
                    # Check if the model has been fitted and has classes (prevents errors with dummy models for single-class data)
                    if hasattr(model, 'predict') and hasattr(label_encoder, 'inverse_transform') and len(label_encoder.classes_) > 0:
                        try:
                            X = vectorizer.transform([step_text_lower])
                            y_pred = model.predict(X)
                            predicted_label = label_encoder.inverse_transform(y_pred)[0]
                            if predicted_label in framework_labels_from_xml: # Ensure ML predicted label is actually in XML
                                identified_framework_label = predicted_label
                                updated_step = step.append(matches_dict[identified_framework_label])
                        except Exception as e:
                            print(f"Error during ML prediction for FrameworkLabel: {e}")
    
            