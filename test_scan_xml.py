# test_scan_xml.py
import pytest
from scan_xml import select_xml_file
from scan_xml import extract_framework_labelids, NAMESPACES # Import the function and NAMESPACES
from unittest.mock import MagicMock # For mocking objects
import xml.etree.ElementTree as ET # For ET.ParseError
from scan_xml import scrub_labelids
from scan_xml import match_testbench_to_framework_labels, NAMESPACES

# --- Test Cases for select_xml_file ---

def test_select_xml_file_success(mocker):
    expected_path = "C:/path/to/my_file.xml"
    mock_ask = mocker.patch('scan_xml.filedialog.askopenfilename', return_value=expected_path)
    mock_tk = mocker.patch('scan_xml.tk.Tk')
    result = select_xml_file()
    assert result == expected_path
    mock_ask.assert_called_once_with(
        title="Select .xml file",
        filetypes=[("xml files", "*.xml"), ("All files", "*.*")]
    )
    mock_tk.assert_called_once()
    mock_tk.return_value.withdraw.assert_called_once()

def test_select_xml_file_cancelled(mocker):
    mock_ask = mocker.patch('scan_xml.filedialog.askopenfilename', return_value="")
    mock_tk = mocker.patch('scan_xml.tk.Tk')

    result = select_xml_file()

    assert result == ""
    mock_ask.assert_called_once_with(
        title="Select .xml file",
        filetypes=[("xml files", "*.xml"), ("All files", "*.*")]
    )
    mock_tk.assert_called_once()
    mock_tk.return_value.withdraw.assert_called_once()

# --- Test Cases for extract_framework_labelids ---

def test_extract_framework_labelids_success(tmp_path):
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

    expected_ids = ['LabelId1', 'LabelId2', 'LabelId3']

    result = extract_framework_labelids(str(xml_file))
    assert result == expected_ids

def test_extract_framework_labelids_no_matching_elements(tmp_path):
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

def test_extract_framework_labelids_file_not_found(capsys):
    """
    Tests handling of FileNotFoundError when the XML file does not exist.
    """
    non_existent_file = "non_existent.xml"
    result = extract_framework_labelids(non_existent_file)

    assert result == []
    captured = capsys.readouterr()
    assert f"Error: XML file not found at '{non_existent_file}'" in captured.err

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
    assert "no element found" in captured.out

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
    assert "An unexpected error occurred in extract_framework_labelids: 'Id'" in captured.err

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

# --- Test Cases for scrub_labelids ---

def test_scrub_labelids_basic_functionality():
    """
    Tests general cases with various suffixes and mixed casing.
    """
    input_ids = [
        "MyLabel_Value_IO_Signal",
        "AnotherLabel_Value_TA_Replacevalue",
        "NoSuffixHere",
        "MixedCase_Value_IO_Signal",
        "AlreadyLower",
        "ALLUPPER_VALUE_IO_SIGNAL"
    ]
    expected_output = [
        "mylabel",
        "anotherlabel",
        "nosuffixhere",
        "mixedcase",
        "alreadylower",
        "allupper"
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

def test_scrub_labelids_label_is_just_suffix():
    """
    Tests a label ID that consists only of a suffix.
    """
    input_ids = [
        "_Value_IO_Signal",
        "_Value_TA_Replacevalue"
    ]
    expected_output = [
        "", # Should become empty string after removal
        ""
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
        create_mock_element("TB_Label_acceleration"),
        create_mock_element("TB_Label_speed()://"), # This one needs scrubbing
        create_mock_element("TB_Label_pitch")
    ]
    mock_parse = mocker.patch('xml.etree.ElementTree.parse', return_value=mock_tree)

    # 2. Mock fuzzywuzzy.process.extractOne
    # We use side_effect to define what each call to extractOne returns
    # The return format is (matched_string, score, index)
    mock_extract_one = mocker.patch('rapidfuzz.process.extractOne', side_effect=[
        ("tb_label_acceleration", 95, 0),
        ("tb_label_speed", 95, 1),
        ("tb_label_pitch", 95, 2)   
    ])

    framework_labels = ["Framework_Label_acceleration", "Framework_Label_speed", "Framework_Label_pitch"]
    # Scrub and lowercase framework labels to match production expectations
    from scan_xml import scrub_labelids
    framework_labels_scrubbed = scrub_labelids(framework_labels)

    # Expected output: key is the scrubbed framework label, value is the scrubbed testbench label
    expected_matches = {
        "framework_label_acceleration": "TB_Label_acceleration",
        "framework_label_speed": "TB_Label_speed",
        "framework_label_pitch": "TB_Label_pitch"
    }

    result = match_testbench_to_framework_labels("dummy.xml", framework_labels_scrubbed)

    assert result == expected_matches
    
    # Verify that xml.etree.ElementTree.parse was called
    mock_parse.assert_called_once_with("dummy.xml")
    
    # Verify that findall was called with the correct arguments
    mock_root.findall.assert_called_once_with('.//ns0:TestbenchLabel', NAMESPACES)
    
    # The list of available_tb in production is the lowercased, scrubbed testbench labels
    available_tb = [elem.attrib['Id'].replace("()://", "").lower() for elem in mock_root.findall.return_value]
    for scrubbed_fw_label in framework_labels_scrubbed:
        mock_extract_one.assert_any_call(scrubbed_fw_label, available_tb, scorer=mocker.ANY)
    assert mock_extract_one.call_count == len(framework_labels_scrubbed)

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