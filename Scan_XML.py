import tkinter as tk
from tkinter import filedialog
import xml.etree.ElementTree as ET
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from typing import List

# XML Namespace for parsing
XIL_NAMESPACE = "http://www.asam.net/XIL/Mapping/2.2.0"
# IMPORTANT: Ensure this matches the actual namespace URI from your XML.
# If your XML elements (FrameworkLabel, TestbenchLabel) do NOT have a namespace prefix (e.g., <FrameworkLabel>),
# then set NAMESPACES = {} and remove all 'ns0:' prefixes from the XPath queries below.
NAMESPACES = {'ns0': XIL_NAMESPACE} 

# -- Selects an XML file to parse. The file should follow XIL format specified above--
def select_xml_file():
     """Opens a file dialog to select XML file."""
     root = tk.Tk()
     root.withdraw()  # Hide the main window
     xml_path = filedialog.askopenfilename(
         title="Select .xml file",
         filetypes=[("xml files", "*.xml"), ("All files", "*.*")]
     )
     return xml_path

# -- Extracts all framework labelID values from XML. These values are similar to the signal value as specified in a .dbc or .arxml --
def extract_framework_labelids(xml_file: str) -> List[str]:
     """Extracts all LabelId values from FrameworkLabel elements in the XML file."""
     try:
         tree = ET.parse(xml_file)
         root = tree.getroot()
         # Ensure the XPath matches your namespace prefix and element name
         labelids = [elem.attrib['Id'] for elem in root.findall('.//ns0:FrameworkLabel', NAMESPACES)]
         print(labelids)
         return labelids
     except FileNotFoundError:
         print(f"Error: XML file not found at '{xml_file}'")
         return []
     except ET.ParseError as e:
         print(f"Error parsing XML file '{xml_file}': {e}")
         return []
     except Exception as e:
         print(f"An unexpected error occurred in get_xml_labels: {e}")
         return []

# -- Scrub LabelIds by removing suffix and making them lowercase --
def scrub_labelids(labelids):
    # set to lowercase before fuzzy search
    lowercase_labelids = [str(labelid).lower() for labelid in labelids]
    # Removing unnecessary prefixes before fuzzy search
    scrubbed_labelids=[]
    for lowercase_labelid in lowercase_labelids:
        if "io_signal" in lowercase_labelid:
            scrubbed_labelid = lowercase_labelid.replace("_io_signal","")
            scrubbed_labelids.append(scrubbed_labelid)
        elif "ta_replacevalue" in lowercase_labelid:
            scrubbed_labelid = lowercase_labelid.replace("_ta_replacevalue","")
            scrubbed_labelids.append(scrubbed_labelid)
        elif "value_io_signal" in lowercase_labelid:
            scrubbed_labelid = lowercase_labelid.replace("_value_io_signal","")
            scrubbed_labelids.append(scrubbed_labelid)
        elif "value_ta_replacevalue" in lowercase_labelid:
            scrubbed_labelid = lowercase_labelid.replace("_value_ta_replacevalue","")
            scrubbed_labelids.append(scrubbed_labelid)
        else:
            scrubbed_labelids.append(lowercase_labelid)

    return scrubbed_labelids

# -- Get all TestbenchLabel IDs, remove uneccessary characters
# -- Set to lowercase 
# -- Assign to dictionary (Framework LabelIds = Key and TestbenchLabel Ids = Value)
import tkinter as tk
from tkinter import filedialog
import xml.etree.ElementTree as ET
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from typing import List, Dict, Union # Added Union for clarity on return types

# XML Namespace for parsing
XIL_NAMESPACE = "http://www.asam.net/XIL/Mapping/2.2.0"
NAMESPACES = {'ns0': XIL_NAMESPACE}

# -- Selects an XML file to parse. The file should follow XIL format specified above--
def select_xml_file() -> str:
     """Opens a file dialog to select XML file."""
     root = tk.Tk()
     root.withdraw()  # Hide the main window
     xml_path = filedialog.askopenfilename(
         title="Select .xml file",
         filetypes=[("xml files", "*.xml"), ("All files", "*.*")]
     )
     return xml_path

# -- Extracts all framework labelID values from XML. These values are similar to the signal value as specified in a .dbc or .arxml --
def extract_framework_labelids(xml_file: str) -> List[str]:
     """Extracts all LabelId values from FrameworkLabel elements in the XML file."""
     try:
         tree = ET.parse(xml_file)
         root = tree.getroot()
         labelids = [elem.attrib['Id'] for elem in root.findall('.//ns0:FrameworkLabel', NAMESPACES)]
         # print(labelids) # Optional: remove this print in production code
         return labelids
     except FileNotFoundError:
         print(f"Error: XML file not found at '{xml_file}'")
         return []
     except ET.ParseError as e:
         print(f"Error parsing XML file '{xml_file}': {e}")
         return []
     except Exception as e:
         print(f"An unexpected error occurred in extract_framework_labelids: {e}") # Renamed function for clarity
         return []

# -- Scrub LabelIds by removing suffix and making them lowercase --
def scrub_labelids(labelids: List[str]) -> List[str]:
    scrubbed_list = []
    for labelid in labelids:
        lowercase_labelid = str(labelid).lower() # Ensure it's a string before operations

        # Use a more robust check: remove suffix if it exists
        if lowercase_labelid.endswith("_io_signal"):
            scrubbed_id = lowercase_labelid.replace("_io_signal","")
        elif lowercase_labelid.endswith("_ta_replacevalue"):
            scrubbed_id = lowercase_labelid.replace("_ta_replacevalue","")
        elif lowercase_labelid.endswith("_value_io_signal"): # Make sure order of checks is from longest to shortest if they can overlap
            scrubbed_id = lowercase_labelid.replace("_value_io_signal","")
        elif lowercase_labelid.endswith("_value_ta_replacevalue"):
            scrubbed_id = lowercase_labelid.replace("_value_ta_replacevalue","")
        else:
            scrubbed_id = lowercase_labelid # No suffix to remove

        scrubbed_list.append(scrubbed_id)
    return scrubbed_list

# -- Get all TestbenchLabel IDs, remove unnecessary characters
# -- Set to lowercase
# -- Assign to dictionary (Framework LabelIds = Key and TestbenchLabel Ids = Value)
def match_testbench_to_framework_labels(xml_file: str, framework_labels: List[str]) -> Dict[str, str]:
    """
    Get all TestbenchLabel IDs, remove unnecessary characters,
    and assigns to dictionary (Framework LabelIds = Key and TestbenchLabel Ids = Value).

    Args:
        xml_file (str): Path to the XML file containing TestbenchLabel elements.
        framework_labels (List[str]): A list of scrubbed (lowercased, suffix-removed)
                                     framework labels to match against.

    Returns:
        Dict[str, str]: A dictionary where keys are matched framework labels
                        (from framework_labels list) and values are the scrubbed
                        TestbenchLabel IDs (after removing "()://").
    """
    # Defensive check: Ensure framework_labels is an iterable
    if not isinstance(framework_labels, (list, tuple)):
        print(f"Error: 'framework_labels' must be a list or tuple of strings, but got {type(framework_labels)}.")
        return {}

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Find all TestbenchLabel elements
        raw_testbench_ids = [elem.attrib['Id'] for elem in root.findall('.//ns0:TestbenchLabel', NAMESPACES)]

        matches_dict = {}
        # Only proceed with fuzzy matching if there are framework labels to compare against
        if not framework_labels:
            print("Warning: 'framework_labels' list is empty. Cannot perform fuzzy matching.")
            return {}

        for testbench_id_raw in raw_testbench_ids:
            # 1. This is the ID that will be the VALUE in the final dict (after "()://" removal)
            testbench_id_for_dict_value = testbench_id_raw.replace("()://", "")

            # 2. This is the ID used for fuzzy comparison (after "()://" removal AND lowercasing)
            # It's crucial that this matches the case and scrubbing of `framework_labels`
            testbench_id_for_fuzzy_matching = testbench_id_for_dict_value.lower()

            result = process.extractOne(testbench_id_for_fuzzy_matching, framework_labels, scorer=fuzz.token_set_ratio)

            # Crucial check: process.extractOne returns None if no choices are provided
            # or if no sufficiently good match is found (e.g., if you set a very high cutoff, not applicable here)
            if result is not None:
                # result[0] is the matched string from framework_labels (which are already scrubbed and lowercased)
                matches_dict[result[0]] = testbench_id_for_dict_value
            else:
                print(f"Warning: No suitable framework label found for '{testbench_id_for_fuzzy_matching}'. Skipping.")

        return matches_dict

    except FileNotFoundError:
         print(f"Error: XML file not found at '{xml_file}'")
         return {} # <--- CORRECTED: Changed from [] to {}
    except ET.ParseError as e:
         print(f"Error parsing XML file '{xml_file}': {e}")
         return {} # <--- CORRECTED: Changed from [] to {}
    except KeyError as e: # Catch if 'Id' attribute is missing from a TestbenchLabel
        print(f"Error: Missing expected 'Id' attribute in a TestbenchLabel element: {e}. Check XML structure.")
        return {}
    except Exception as e:
         # Corrected the error message string here
         print(f"An unexpected error occurred in match_testbench_to_framework_labels: {e}")
         return {} # <--- CORRECTED: Changed from [] to {}

# --- Corrected Function Calls ---
xml_file_path = select_xml_file() # This is 'x' from your original code (the XML file path)

if xml_file_path: # Only proceed if a file was selected
    # Extract and scrub framework labels
    raw_framework_labels = extract_framework_labelids(xml_file_path)
    scrubbed_framework_labels = scrub_labelids(raw_framework_labels) # This is 'y' from your original code (the list of scrubbed labels)

    # Now, call the function with the correct arguments
    # xml_file (path) is the first argument, framework_labels (list) is the second
    matched_labels = match_testbench_to_framework_labels(xml_file_path, scrubbed_framework_labels)

    print("\n--- Matching Results ---")
    if matched_labels: # Now `matched_labels` will always be a dict, so `if matched_labels` works for non-empty dicts
        for framework_label, testbench_id in matched_labels.items():
            print(f"Framework Label: '{framework_label}' matched to Testbench ID: '{testbench_id}'")
    else:
        print("No matches found or an error occurred during matching.")

x=select_xml_file()
y=scrub_labelids(extract_framework_labelids(x))
z=match_testbench_to_framework_labels(x,y)
print(z)