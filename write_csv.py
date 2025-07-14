import csv

def write_steps_to_csv(
    csv_filename,
    mapped_scenarios,
    scenario_steps
):
    """
    Writes parsed Gherkin steps, extracted data, and mapped variable paths to a CSV file.
    Output is .csv file with collumns Scenario, Step Type, step_text, Value, Latency, Variable Path
    """
    try:
        # Open the CSV file for writing
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header row
            writer.writerow(['Scenario', 'Step Type', 'Step Text', 'Value', 'Latency', 'Variable Path'])

            # Write each mapped_scenario to the CSV, sourcing value and latency from scenario_steps
            for mapped_scenario in mapped_scenarios:
                scenario = mapped_scenario[0]
                step_type = mapped_scenario[1]
                step_text = mapped_scenario[2]
                variable_path = mapped_scenario[-1] if len(mapped_scenario) > 3 else None

                # Find the matching step in scenario_steps to get value and latency
                value, latency = None, None
                if scenario in scenario_steps:
                    for step in scenario_steps[scenario]:
                        if step[0] == step_type and step[1] == step_text:
                            value = step[2] if len(step) > 2 else None
                            latency = step[3] if len(step) > 3 else None
                            break
                writer.writerow([scenario, step_type, step_text, value, latency, variable_path])

            # Iterate through each scenario and its steps
            

        # If the loop completes without errors, print success message
        print(f"Gherkin steps, value (converted), latency, and variable paths written to {csv_filename}")

    except IOError as e:
        print(f"Error writing to CSV file {csv_filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")