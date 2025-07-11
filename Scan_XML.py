import tkinter as tk
from tkinter import filedialog
import xml.etree.ElementTree as ET
from rapidfuzz import process, fuzz
from typing import List, Dict
import sys

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
         return labelids
     except FileNotFoundError:
         print(f"Error: XML file not found at '{xml_file}'", file=sys.stderr)
         return []
     except ET.ParseError as e:
         print(f"Error parsing XML file '{xml_file}': {e}")
         return []
     except Exception as e:
         print(f"An unexpected error occurred in extract_framework_labelids: {e}", file=sys.stderr)
         return []

# -- Scrub LabelIds by removing suffix and making them lowercase --
def scrub_labelids(labelids):
    # set to lowercase before fuzzy search
    lowercase_labelids = [str(labelid).lower() for labelid in labelids]
    # Removing unnecessary prefixes before fuzzy search
    scrubbed_labelids=[]
    for lowercase_labelid in lowercase_labelids:
        if "_value_io_signal" in lowercase_labelid:
            scrubbed_labelid = lowercase_labelid.replace("_value_io_signal","")
            scrubbed_labelids.append(scrubbed_labelid)
        elif "_value_ta_replacevalue" in lowercase_labelid:
            scrubbed_labelid = lowercase_labelid.replace("_value_ta_replacevalue","")
            scrubbed_labelids.append(scrubbed_labelid)
        elif "_io_signal" in lowercase_labelid:
            scrubbed_labelid = lowercase_labelid.replace("_io_signal","")
            scrubbed_labelids.append(scrubbed_labelid)
        elif "_ta_replacevalue" in lowercase_labelid:
            scrubbed_labelid = lowercase_labelid.replace("_ta_replacevalue","")
            scrubbed_labelids.append(scrubbed_labelid)
        else:
            scrubbed_labelids.append(lowercase_labelid)

    return scrubbed_labelids

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

        # Preprocess testbench IDs: remove '()://', lowercase
        processed_testbench_ids = [(tb_id.replace("()://", ""), tb_id.replace("()://", "").lower()) for tb_id in raw_testbench_ids]

        matches_dict = {}
        if not framework_labels:
            print("Warning: 'framework_labels' list is empty. Cannot perform matching.")
            return {}

        # First, try exact matches
        framework_set = set(framework_labels)
        used_testbench = set()
        for fw_label in framework_labels:
            # Find exact match in processed testbench IDs
            for tb_value, tb_lower in processed_testbench_ids:
                if fw_label == tb_lower:
                    matches_dict[fw_label] = tb_value
                    used_testbench.add(tb_value)
                    break

        # For unmatched framework labels, use fuzzy matching
        unmatched_fw_labels = [fw for fw in framework_labels if fw not in matches_dict]
        available_tb = [tb_lower for tb_value, tb_lower in processed_testbench_ids if tb_value not in used_testbench]
        tb_map = {tb_lower: tb_value for tb_value, tb_lower in processed_testbench_ids}

        for fw_label in unmatched_fw_labels:
            # Only fuzzy match against unused testbench labels
            result = process.extractOne(fw_label, available_tb, scorer=fuzz.token_set_ratio)
            if result is not None:
                tb_lower = result[0]
                matches_dict[fw_label] = tb_map[tb_lower]
                used_testbench.add(tb_map[tb_lower])
            else:
                print(f"Warning: No suitable testbench label found for '{fw_label}'. Skipping.")

        return matches_dict

    except FileNotFoundError:
         print(f"Error: XML file not found at '{xml_file}'")
         return {} 
    except ET.ParseError as e:
         print(f"Error parsing XML file '{xml_file}': {e}")
         return {}
    except KeyError as e: # Catch if 'Id' attribute is missing from a TestbenchLabel
        print(f"Error: Missing expected 'Id' attribute in a TestbenchLabel element: {e}. Check XML structure.")
        return {}
    except Exception as e:
         print(f"An unexpected error occurred in match_testbench_to_framework_labels: {e}")
         return {}