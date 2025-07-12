# main.py
from feature_parser import select_feature_files, parse_feature_file
from scan_xml import select_xml_file, extract_framework_labelids, scrub_labelids, match_testbench_to_framework_labels
from find_variable_path import select_json_file, load_training_data, train_and_save_model, load_model_components, predict_framework_label_from_step
from map_variables import map_variables
from write_csv import write_steps_to_csv

# Profiling imports
import cProfile
import pstats
import io

def main():
    """
    Main function to parse feature files, scan xml file, predict variable path,map variables, and write results to CSV.
    """
    # 1). Scenario steps collected
    # -- User selects .feature files
    feature_files = select_feature_files()
    # -- Scenario_steps are returned in dictionary of scenario and steps strings
    scenario_steps = {}
    for feature_file in feature_files:
        scenario_steps.update(parse_feature_file(feature_file))

    # 2). ID dictionary created with scrubbed values to aid fuzzy matching
    # -- User selects .xml file
    xml = select_xml_file()
    # -- Scrubbed labelids list is returned
    labelids = scrub_labelids(extract_framework_labelids(xml))
    # -- Scrubbed labelids used to construct dictionary with testbench and framework labels
    id_dict = match_testbench_to_framework_labels(xml, labelids) # Inefficient needs to be optimised

    # 3). Predicted variable path
    # -- Attempted loading of existing model components
    model, vectorizer, label_encoder = load_model_components('signal_prediction_model.pkl', 'tfidf_vectorizer.pkl', 'label_encoder.pkl')
    # -- If model components are present use model to predict framework labels
    if all([model, vectorizer, label_encoder]):
        model_matches = predict_framework_label_from_step(model, vectorizer, label_encoder, id_dict, scenario_steps)
    else:
        print("Warning: Not all model components loaded. Training new model...")
        json = select_json_file()
        data = load_training_data(json)
        train_and_save_model(data, 'signal_prediction_model.pkl', 'tfidf_vectorizer.pkl', 'label_encoder.pkl')
        # Reload model components after training
        model, vectorizer, label_encoder = load_model_components('signal_prediction_model.pkl', 'tfidf_vectorizer.pkl', 'label_encoder.pkl')
        model_matches = predict_framework_label_from_step(model, vectorizer, label_encoder, id_dict, scenario_steps)

    # 4). Variables mapped to test steps
    # -- Map fuzzy match variable or ML predicted variable as backup
    mapped_scenarios = map_variables(id_dict, model_matches, scenario_steps)

    # 5). Write Scenario, step type, step text, value, latency and variable path to .csv
    # -- Write to .csv file hardcoded as gherkin_steps_with_paths.csv
    write_steps_to_csv("gherkin_steps_with_paths.csv", mapped_scenarios, scenario_steps)

# Execute script directly
if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(30)  # Show top 30 lines
    print("\n--- Profiling Results (Top 30 by cumulative time) ---\n")
    print(s.getvalue())