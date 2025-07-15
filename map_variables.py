from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# --- Fuzzy search framework labelIDs to find correct Testbench ID.
#  --- If no matches are found use ML to select ID instead.
def map_variables(matches_dict, model_matches, scenarios):
    framework_labels_from_xml = list(matches_dict.keys())
    mapped_scenarios = []
    for scenario in scenarios:
            steps = scenarios[scenario]
            for step in steps:
                step_type = step[0]
                step_text = step[1] 
                mapping_from_framework_ids = process.extractOne(step_text, framework_labels_from_xml, scorer=fuzz.token_set_ratio)
                if mapping_from_framework_ids is not None:
                    framework_label = mapping_from_framework_ids[0]
                    # Fuzzy search for the best match in matches_dict keys
                    best_match = process.extractOne(framework_label, matches_dict.keys(), scorer=fuzz.token_set_ratio)
                    if best_match:
                        testbench_id = matches_dict[best_match[0]]
                    else:
                        testbench_id = None
                    mapped_scenarios.append((scenario, step_type, step_text, testbench_id))
                else:
                    for model_match in model_matches:
                        if step_text in model_match[1]:
                            mapping_from_model = model_match[2]
                            mapped_scenarios.append((scenario, step_type, step_text, mapping_from_model))
    return mapped_scenarios