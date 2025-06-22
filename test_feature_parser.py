# test_feature_parser.py
import pytest
from feature_parser import select_feature_files
from unittest.mock import MagicMock # For mocking objects
import xml.etree.ElementTree as ET # For ET.ParseError (not directly used here, but kept)

def test_select_feature_file_success(mocker):
    # When selecting multiple files, the function returns a LIST of paths.
    # So, expected_paths should be a list.
    expected_paths = ["C:/path/to/my_file.feature", "D:/another/path/second.feature"]

    # askopenfilenames returns a TUPLE of paths.
    # So, the mock's return_value must be a tuple.
    mock_ask = mocker.patch('feature_parser.filedialog.askopenfilenames', return_value=tuple(expected_paths))
    
    # PATCHING: feature_parser.tk.Tk
    # This is correct for hiding the Tkinter root window.
    mock_tk = mocker.patch('feature_parser.tk.Tk')

    result = select_feature_files()

    # The result should be the list of paths
    assert result == expected_paths
    mock_ask.assert_called_once_with(
        title="Select .feature files", # Matches the new title in feature_parser.py
        filetypes=[("feature files", "*.feature"), ("All files", "*.*")] # Matches the new filetypes label
    )
    mock_tk.assert_called_once()
    mock_tk.return_value.withdraw.assert_called_once()


def test_select_feature_file_cancelled(mocker):
    # When cancelled, askopenfilenames returns an EMPTY TUPLE ().
    # The function then converts this to an empty list [].
    mock_ask = mocker.patch('feature_parser.filedialog.askopenfilenames', return_value=()) # Changed from "" to ()
    mock_tk = mocker.patch('feature_parser.tk.Tk')

    result = select_feature_files()

    # The result should be an empty list when cancelled
    assert result == []
    mock_ask.assert_called_once_with(
        title="Select .feature files", # Matches the new title in feature_parser.py
        filetypes=[("feature files", "*.feature"), ("All files", "*.*")] # Matches the new filetypes label
    )
    mock_tk.assert_called_once()
    mock_tk.return_value.withdraw.assert_called_once()