from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# --- Fuzzy search framework labelIDs to find correct Testbench ID.
#  --- If no matches are found use ML to select ID instead.
def map_variables_(matches_dict, model_matches, scenarios):
    framework_labels_from_xml = list(matches_dict.keys())
    mapped_scenarios = []
    for scenario in scenarios:
        for step in scenario:
            step_text = step[1]
            alias = step[0]
            mapping_from_framework_ids = process.extractOne(step_text, framework_labels_from_xml, scorer=fuzz.token_set_ratio)
            if mapping_from_framework_ids != None:
                mapped_scenarios.append((scenario, alias, step_text, mapping_from_framework_ids))
            else:
                for model_match in model_matches:
                    if step_text in model_match[1]:
                        mapping_from_model = model_match[2]
                        mapped_scenarios.append((scenario, alias, step_text, mapping_from_model))
    return mapped_scenarios