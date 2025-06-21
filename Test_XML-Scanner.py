# test_Scan_XML.py
import pytest
from Scan_XML import select_xml_file
from Scan_XML import extract_framework_labelids, NAMESPACES # Import the function and NAMESPACES
from unittest.mock import MagicMock # For mocking objects
import xml.etree.ElementTree as ET # For ET.ParseError

def test_select_xml_file_success(mocker):
    expected_path = "C:/path/to/my_file.xml"
    # PATCHING: Scan_XML.filedialog.askopenfilename
    # This is correct because 'filedialog' is imported directly into Scan_XML.'s namespace
    mock_ask = mocker.patch('Scan_XML.filedialog.askopenfilename', return_value=expected_path)
    
    # PATCHING: Scan_XML.tk.Tk
    # This is correct because 'tk' is imported as 'tk' into Scan_XML.'s namespace
    mock_tk = mocker.patch('Scan_XML.tk.Tk')

    result = select_xml_file()

    assert result == expected_path
    mock_ask.assert_called_once_with(
        title="Select .xml file",
        filetypes=[("xml files", "*.xml"), ("All files", "*.*")]
    )
    mock_tk.assert_called_once()
    mock_tk.return_value.withdraw.assert_called_once()


def test_select_xml_file_cancelled(mocker):
    mock_ask = mocker.patch('Scan_XML.filedialog.askopenfilename', return_value="")
    mock_tk = mocker.patch('Scan_XML.tk.Tk')

    result = select_xml_file()

    assert result == ""
    mock_ask.assert_called_once_with(
        title="Select .xml file",
        filetypes=[("xml files", "*.xml"), ("All files", "*.*")]
    )
    mock_tk.assert_called_once()
    mock_tk.return_value.withdraw.assert_called_once()

# --- Test Cases for extract_framework_labelids ---

def test_extract_framework_labelids_success(tmp_path, capsys):
    """
    Tests successful extraction of LabelIds from a valid XML file.
    """
    xml_content = f"""<?xml version="1.0"?>
    <root xmlns:ns0="{NAMESPACES['ns0']}">
        <Section>
            <ns0:FrameworkLabel Id="LabelId1"/>
            <OtherElement/>
            <ns0:FrameworkLabel Id="LabelId2"/>
        </Section>
        <AnotherSection>
            <ns0:FrameworkLabel Id="LabelId3"/>
        </AnotherSection>
    </root>
    """
    xml_file = tmp_path / "valid.xml"
    xml_file.write_text(xml_content)

    expected_ids = ["LabelId1", "LabelId2", "LabelId3"]
    result = extract_framework_labelids(str(xml_file))

    assert result == expected_ids
    captured = capsys.readouterr()
    assert captured.out.strip() == str(expected_ids) # Check the print output

def test_extract_framework_labelids_no_matching_elements(tmp_path, capsys):
    """
    Tests extraction when the XML file contains no matching FrameworkLabel elements.
    """
    xml_content = f"""<?xml version="1.0"?>
    <root xmlns:ns0="{NAMESPACES['ns0']}">
        <Section>
            <OtherLabel Id="NotThis"/>
            <AnotherElement/>
        </Section>
    </root>
    """
    xml_file = tmp_path / "no_labels.xml"
    xml_file.write_text(xml_content)

    expected_ids = []
    result = extract_framework_labelids(str(xml_file))

    assert result == expected_ids
    captured = capsys.readouterr()
    assert captured.out.strip() == str(expected_ids) # Check the print output (empty list)

def test_extract_framework_labelids_file_not_found(capsys):
    """
    Tests handling of FileNotFoundError when the XML file does not exist.
    """
    non_existent_file = "non_existent.xml"
    result = extract_framework_labelids(non_existent_file)

    assert result == []
    captured = capsys.readouterr()
    assert f"Error: XML file not found at '{non_existent_file}'" in captured.out

def test_extract_framework_labelids_malformed_xml(tmp_path, capsys):
    """
    Tests handling of ET.ParseError for malformed XML content.
    """
    malformed_content = "<root><unclosed_tag>"
    xml_file = tmp_path / "malformed.xml"
    xml_file.write_text(malformed_content)

    result = extract_framework_labelids(str(xml_file))

    assert result == []
    captured = capsys.readouterr()
    assert "Error parsing XML file" in captured.out
    assert "unclosed_tag" in captured.out # Check for specific error detail

def test_extract_framework_labelids_missing_id_attribute(tmp_path, capsys):
    """
    Tests handling of FrameworkLabel elements missing the 'Id' attribute,
    which should trigger the generic Exception handler (KeyError).
    """
    xml_content = f"""<?xml version="1.0"?>
    <root xmlns:ns0="{NAMESPACES['ns0']}">
        <ns0:FrameworkLabel Id="LabelId1"/>
        <ns0:FrameworkLabel NoIdAttribute="Value"/>
        <ns0:FrameworkLabel Id="LabelId2"/>
    </root>
    """
    xml_file = tmp_path / "missing_id.xml"
    xml_file.write_text(xml_content)

    result = extract_framework_labelids(str(xml_file))

    assert result == [] # Should return empty list due to the error
    captured = capsys.readouterr()
    # The error message will contain "KeyError: 'Id'"
    assert "An unexpected error occurred in get_xml_labels: KeyError: 'Id'" in captured.out

def test_extract_framework_labelids_empty_file(tmp_path, capsys):
    """
    Tests handling of an empty XML file, which should also cause a ParseError.
    """
    xml_file = tmp_path / "empty.xml"
    xml_file.write_text("")

    result = extract_framework_labelids(str(xml_file))

    assert result == []
    captured = capsys.readouterr()
    assert "Error parsing XML file" in captured.out
    assert "no element found" in captured.out # Specific error for empty XML



# test_label_scrubber.py
import pytest
from Scan_XML import scrub_labelids # Import the corrected function

# --- Test Cases for scrub_labelids ---

def test_scrub_labelids_basic_functionality():
    """
    Tests general cases with various suffixes and mixed casing.
    """
    input_ids = [
        "MyLabel_Value_IO_Signal",
        "AnotherLabel_Value_TA_Replacevalue",
        "NoSuffixHere",
        "MixedCase_Value_IO_Signal1",
        "AlreadyLower",
        "ALLUPPER_VALUE_IO_SIGNAL" # Suffix case doesn't match, so it should remain
    ]
    expected_output = [
        "mylabel",
        "anotherlabel",
        "nosuffixhere",
        "mixedcase",
        "alreadylower",
        "allupper_value_io_signal" # Still has suffix because case didn't match
    ]
    assert scrub_labelids(input_ids) == expected_output

def test_scrub_labelids_no_suffixes():
    """
    Tests a list where no label IDs have the specified suffixes.
    Should only perform lowercasing.
    """
    input_ids = [
        "SimpleLabel",
        "AnotherOne",
        "CamelCaseLabel"
    ]
    expected_output = [
        "simplelabel",
        "anotherone",
        "camelcaselabel"
    ]
    assert scrub_labelids(input_ids) == expected_output

def test_scrub_labelids_empty_list():
    """
    Tests handling of an empty input list.
    """
    assert scrub_labelids([]) == []

def test_scrub_labelids_suffixes_with_and_without_1():
    """
    Ensures that longer suffixes (ending in '1') are correctly removed
    before shorter ones.
    """
    input_ids = [
        "SpecificLabel_Value_IO_Signal1",
        "GeneralLabel_Value_IO_Signal",
        "TestReplace_Value_TA_Replacevalue1",
        "AnotherReplace_Value_TA_Replacevalue"
    ]
    expected_output = [
        "specificlabel",
        "generallabel",
        "testreplace",
        "anotherreplace"
    ]
    assert scrub_labelids(input_ids) == expected_output

def test_scrub_labelids_suffix_not_at_end():
    """
    Tests that suffixes are only removed if they are at the end of the string.
    """
    input_ids = [
        "Middle_Value_IO_Signal_End",
        "Start_Value_TA_Replacevalue_Middle"
    ]
    expected_output = [
        "middle_value_io_signal_end",
        "start_value_ta_replacevalue_middle"
    ]
    assert scrub_labelids(input_ids) == expected_output

def test_scrub_labelids_label_is_just_suffix():
    """
    Tests a label ID that consists only of a suffix.
    """
    input_ids = [
        "_Value_IO_Signal",
        "_Value_TA_Replacevalue1"
    ]
    expected_output = [
        "", # Should become empty string after removal
        ""
    ]
    assert scrub_labelids(input_ids) == expected_output

def test_scrub_labelids_contains_multiple_suffixes_but_only_one_removed():
    """
    Tests that if a label ID contains multiple potential suffixes, only the
    first (longest) matching one at the end is removed.
    """
    input_ids = [
        "Label_Value_IO_Signal_Value_IO_Signal1" # Should remove '_Value_IO_Signal1' only
    ]
    expected_output = [
        "label_value_io_signal"
    ]
    assert scrub_labelids(input_ids) == expected_output

def test_scrub_labelids_non_string_inputs():
    """
    Tests handling of non-string inputs, which should be converted to string
    before lowercasing.
    """
    input_ids = [
        123,
        True,
        None,
        ["list_as_id"], # Will become "['list_as_id']"
        {"dict_as_id": 1} # Will become "{'dict_as_id': 1}"
    ]
    expected_output = [
        "123",
        "true",
        "none",
        "['list_as_id']",
        "{'dict_as_id': 1}"
    ]
    assert scrub_labelids(input_ids) == expected_output


# test_label_matcher.py
# Import the function and NAMESPACES from your module
from Scan_XML import match_testbench_to_framework_labels, NAMESPACES

# --- Helper to create a mock Element for ET.findall ---
def create_mock_element(element_id):
    """Creates a MagicMock object that behaves like an ElementTree Element
    with an 'Id' attribute."""
    mock_elem = MagicMock()
    mock_elem.attrib = {'Id': element_id}
    return mock_elem

# --- Test Cases for match_testbench_to_framework_labels ---

def test_match_testbench_to_framework_labels_success(mocker):
    """
    Tests successful extraction, scrubbing, and fuzzy matching.
    """
    # 1. Mock xml.etree.ElementTree.parse and its subsequent calls
    mock_tree = MagicMock()
    mock_root = MagicMock()
    mock_tree.getroot.return_value = mock_root
    
    # Simulate findall returning elements with IDs, some needing scrubbing
    mock_root.findall.return_value = [
        create_mock_element("TB_Label_A"),
        create_mock_element("TB_Label_B()://"), # This one needs scrubbing
        create_mock_element("TB_Label_C_Long")
    ]
    mocker.patch('xml.etree.ElementTree.parse', return_value=mock_tree)

    # 2. Mock fuzzywuzzy.process.extractOne
    # We use side_effect to define what each call to extractOne returns
    # The return format is (matched_string, score, index)
    mock_extract_one = mocker.patch('fuzzywuzzy.process.extractOne', side_effect=[
        ("Framework_Label_A", 90, 0),       # For "TB_Label_A"
        ("Framework_Label_B_Match", 95, 1), # For "TB_Label_B" (after scrubbing)
        ("Framework_Label_C_Long", 85, 2)   # For "TB_Label_C_Long"
    ])

    framework_labels = ["Framework_Label_A", "Framework_Label_B_Match", "Framework_Label_C_Long"]
    
    # Expected output: key is the framework label, value is the scrubbed testbench label
    expected_matches = {
        "Framework_Label_A": "TB_Label_A",
        "Framework_Label_B_Match": "TB_Label_B", # Note: this is the scrubbed value
        "Framework_Label_C_Long": "TB_Label_C_Long"
    }

    result = match_testbench_to_framework_labels("dummy.xml", framework_labels)

    assert result == expected_matches
    
    # Verify that xml.etree.ElementTree.parse was called
    mocker.patch('xml.etree.ElementTree.parse').assert_called_once_with("dummy.xml")
    
    # Verify that findall was called with the correct arguments
    mock_root.findall.assert_called_once_with('.//ns0:TestbenchLabel', NAMESPACES)
    
    # Verify extractOne was called for each scrubbed testbench ID with correct arguments
    mock_extract_one.assert_any_call("TB_Label_A", framework_labels, scorer=mocker.ANY) # mocker.ANY for fuzz.token_set_ratio
    mock_extract_one.assert_any_call("TB_Label_B", framework_labels, scorer=mocker.ANY) # Assert scrubbed value
    mock_extract_one.assert_any_call("TB_Label_C_Long", framework_labels, scorer=mocker.ANY)
    assert mock_extract_one.call_count == len(mock_root.findall.return_value)


def test_match_testbench_to_framework_labels_no_testbench_elements(mocker):
    """
    Tests scenario where no TestbenchLabel elements are found in XML.
    """
    mock_tree = MagicMock()
    mock_root = MagicMock()
    mock_tree.getroot.return_value = mock_root
    mock_root.findall.return_value = [] # Simulate no elements found
    mocker.patch('xml.etree.ElementTree.parse', return_value=mock_tree)
    
    # fuzzywuzzy.process.extractOne should not be called at all
    mock_extract_one = mocker.patch('fuzzywuzzy.process.extractOne')

    framework_labels = ["F_Label_1", "F_Label_2"]
    result = match_testbench_to_framework_labels("dummy.xml", framework_labels)

    assert result == {}
    mock_root.findall.assert_called_once_with('.//ns0:TestbenchLabel', NAMESPACES)
    mock_extract_one.assert_not_called()

def test_match_testbench_to_framework_labels_file_not_found(mocker, capsys):
    """
    Tests handling of FileNotFoundError.
    """
    mocker.patch('xml.etree.ElementTree.parse', side_effect=FileNotFoundError)

    result = match_testbench_to_framework_labels("non_existent.xml", ["F_Label"])

    assert result == {}
    captured = capsys.readouterr()
    assert "Error: XML file not found at 'non_existent.xml'" in captured.out

def test_match_testbench_to_framework_labels_parse_error(mocker, capsys):
    """
    Tests handling of ET.ParseError for malformed XML.
    """
    # Simulate a ParseError being raised by ET.parse
    mocker.patch('xml.etree.ElementTree.parse', side_effect=ET.ParseError("syntax error detail", 0, 0))

    result = match_testbench_to_framework_labels("malformed.xml", ["F_Label"])

    assert result == {}
    captured = capsys.readouterr()
    assert "Error parsing XML file 'malformed.xml': syntax error detail" in captured.out

def test_match_testbench_to_framework_labels_missing_id_attribute(mocker, capsys):
    """
    Tests handling of KeyError when an element is missing the 'Id' attribute.
    This should be caught by the generic Exception handler.
    """
    mock_tree = MagicMock()
    mock_root = MagicMock()
    mock_tree.getroot.return_value = mock_root
    
    # Simulate an element without 'Id' attribute
    mock_elem_no_id = MagicMock()
    mock_elem_no_id.attrib = {'Name': 'Test'} # No 'Id'
    mock_root.findall.return_value = [
        create_mock_element("ValidId"),
        mock_elem_no_id # This will cause a KeyError when .attrib['Id'] is accessed
    ]
    mocker.patch('xml.etree.ElementTree.parse', return_value=mock_tree)
    
    # fuzzywuzzy.process.extractOne should not be called if an error occurs earlier
    mock_extract_one = mocker.patch('fuzzywuzzy.process.extractOne')

    framework_labels = ["F_Label"]
    result = match_testbench_to_framework_labels("dummy.xml", framework_labels)

    assert result == {}
    captured = capsys.readouterr()
    assert "An unexpected error occurred in get_all_testbench_label_ids: KeyError: 'Id'" in captured.out
    mock_extract_one.assert_not_called()

def test_match_testbench_to_framework_labels_empty_framework_labels(mocker):
    """
    Tests scenario where framework_labels list is empty.
    Fuzzy matching should be skipped, resulting in an empty dict.
    """
    mock_tree = MagicMock()
    mock_root = MagicMock()
    mock_tree.getroot.return_value = mock_root
    mock_root.findall.return_value = [
        create_mock_element("TB_Label_A") # Even if testbench labels exist
    ]
    mocker.patch('xml.etree.ElementTree.parse', return_value=mock_tree)
    
    # extractOne will raise ValueError if choices is empty, so it should not be called
    mock_extract_one = mocker.patch('fuzzywuzzy.process.extractOne')

    framework_labels = [] # Empty list
    result = match_testbench_to_framework_labels("dummy.xml", framework_labels)

    assert result == {}
    mock_extract_one.assert_not_called()
    
def test_match_testbench_to_framework_labels_no_scrubbing_needed(mocker):
    """
    Tests that scrubbing doesn't alter IDs if '()://' is not present.
    """
    mock_tree = MagicMock()
    mock_root = MagicMock()
    mock_tree.getroot.return_value = mock_root
    mock_root.findall.return_value = [
        create_mock_element("TB_Label_A_Clean"),
        create_mock_element("TB_Label_B_NoSpecialChars")
    ]
    mocker.patch('xml.etree.ElementTree.parse', return_value=mock_tree)

    mock_extract_one = mocker.patch('fuzzywuzzy.process.extractOne', side_effect=[
        ("F_Label_A_Clean", 90, 0),
        ("F_Label_B_NoSpecialChars", 95, 1)
    ])

    framework_labels = ["F_Label_A_Clean", "F_Label_B_NoSpecialChars"]
    expected_matches = {
        "F_Label_A_Clean": "TB_Label_A_Clean",
        "F_Label_B_NoSpecialChars": "TB_Label_B_NoSpecialChars"
    }

    result = match_testbench_to_framework_labels("dummy.xml", framework_labels)
    assert result == expected_matches
    # Verify extractOne was called with the original (unscrubbed, but clean) values
    mock_extract_one.assert_any_call("TB_Label_A_Clean", framework_labels, scorer=mocker.ANY)
    mock_extract_one.assert_any_call("TB_Label_B_NoSpecialChars", framework_labels, scorer=mocker.ANY)

def test_match_testbench_to_framework_labels_scrubbing_performed(mocker):
    """
    Tests that scrubbing correctly removes '()://'.
    """
    mock_tree = MagicMock()
    mock_root = MagicMock()
    mock_tree.getroot.return_value = mock_root
    mock_root.findall.return_value = [
        create_mock_element("TB_Label_With_Scrub()://"),
        create_mock_element("Another_TB_Label_With_Scrub()://")
    ]
    mocker.patch('xml.etree.ElementTree.parse', return_value=mock_tree)

    mock_extract_one = mocker.patch('fuzzywuzzy.process.extractOne', side_effect=[
        ("F_Label_With_Scrub", 90, 0),
        ("Another_F_Label_With_Scrub", 95, 1)
    ])

    framework_labels = ["F_Label_With_Scrub", "Another_F_Label_With_Scrub"]
    expected_matches = {
        "F_Label_With_Scrub": "TB_Label_With_Scrub",
        "Another_F_Label_With_Scrub": "Another_TB_Label_With_Scrub"
    }

    result = match_testbench_to_framework_labels("dummy.xml", framework_labels)
    assert result == expected_matches
    # Crucially, check that extractOne was called with the *scrubbed* values
    mock_extract_one.assert_any_call("TB_Label_With_Scrub", framework_labels, scorer=mocker.ANY)
    mock_extract_one.assert_any_call("Another_TB_Label_With_Scrub", framework_labels, scorer=mocker.ANY)