# test_Scan_XML.py
import pytest
from Scan_XML import select_xml_file
from Scan_XML import extract_framework_labelids, NAMESPACES # Import the function and NAMESPACES

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


