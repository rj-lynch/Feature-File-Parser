import re
import csv
import tkinter as tk
from tkinter import filedialog
import os
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import json
from typing import List, Dict, Tuple, Any, Optional, Union

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

answer=scrub_labelids(extract_framework_labelids(r"C:\Users\RLYNCH39\Downloads\Mapping.BE672F86-DAE9-4A56-8C8D-5A34762EC14F.xml"))
print(answer)
# NEW FUNCTION: To get all TestbenchLabel IDs
# def get_all_testbench_label_ids(xml_file: str) -> List[str]:
#     """Extracts all Id values from TestbenchLabel elements in the XML file."""
#     try:
#         tree = ET.parse(xml_file)
#         root = tree.getroot()
#         # Find all TestbenchLabel elements, assuming they are also namespaced
#         testbench_ids = [elem.attrib['Id'] for elem in root.findall('.//ns0:TestbenchLabel', NAMESPACES)]
#         return testbench_ids
#     except FileNotFoundError:
#         print(f"Error: XML file not found at '{xml_file}'")
#         return []
#     except ET.ParseError as e:
#         print(f"Error parsing XML file '{xml_file}': {e}")
#         return []
#     except Exception as e:
#         print(f"An unexpected error occurred in get_all_testbench_label_ids: {e}")
#         return []

# def fuzzy_search_xml_labels(search_text: str, target_labels: List[str], fuzzy_threshold: int = 1) -> Optional[str]:
#     """
#     Fuzzy matches search_text to a list of target_labels, using lowercased comparison
#     but returning the original label.
#     """
#     if not target_labels:
#         return None

#     target_labels_lower = [str(label).lower() for label in target_labels]
#     search_text_lower = search_text.lower()

#     # Using fuzz.token_set_ratio is often better for phrases as it handles word order and extra words
#     result = process.extractOne(search_text_lower, target_labels_lower, scorer=fuzz.token_set_ratio)
    
#     if result:
#         match_lower, score = result
#         if score >= fuzzy_threshold:
#             # Find the original label from the target_labels list
#             try:
#                 idx = target_labels_lower.index(match_lower)
#                 print(f"Fuzzy match {Score} {target_label[idx]}")
#                 return target_labels[idx]  # Return the original, case-preserved label
#             except ValueError:
#                 # This should ideally not happen if match_lower came from target_labels_lower
#                 print(f"Warning: Fuzzy match '{match_lower}' not found in original lowercased list.")
#                 return None
#     return None


#select_xml_file()