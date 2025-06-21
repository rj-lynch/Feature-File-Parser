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
def match_testbench_to_framework_labels(xml_file, framework_labels):
    """Get all TestbenchLabel IDs, remove uneccessary characters,
    and assigns to dictionary (Framework LabelIds = Key and TestbenchLabel Ids = Value)"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # Find all TestbenchLabel elements, assuming they are also namespaced
        testbench_ids = [elem.attrib['Id'] for elem in root.findall('.//ns0:TestbenchLabel', NAMESPACES)]
        # Scrub Testbench Ids to remove excess characters
        scrubbed_testbench_ids=[]
        for testbench_id in testbench_ids:
            scrubbed_testbench_id = testbench_id.replace("()://","")
            scrubbed_testbench_ids.append(scrubbed_testbench_id)
        # Fuzzy search Framework Label Id against Testbench label Id and assign closest match to dictionary
        matches_dict = {}
        for scrubbed_testbench_id in scrubbed_testbench_ids:
            result = process.extractOne(scrubbed_testbench_id, framework_labels, scorer=fuzz.token_set_ratio)
            matches_dict[result[0]] = scrubbed_testbench_id
        return matches_dict
    
    except FileNotFoundError:
         print(f"Error: XML file not found at '{xml_file}'")
         return []
    except ET.ParseError as e:
         print(f"Error parsing XML file '{xml_file}': {e}")
         return []
    except Exception as e:
         print(f"An unexpected error occurred in get_all_testbench_label_ids: {e}")
         return []