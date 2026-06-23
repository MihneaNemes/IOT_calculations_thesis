"""Data persistence for saved names and user data."""

import os
import json


class DataManager:
    """Handles loading and saving user data to JSON files."""
    
    def __init__(self, names_file="/home/mihnea/Desktop/bodyfat_preictor/saved_names.json"):
        self.names_file = names_file
    
    def load_names(self):
        """Load saved names from JSON file.
        
        Returns:
            List of names, or ["Mihnea"] if file doesn't exist
        """
        if os.path.exists(self.names_file):
            try:
                with open(self.names_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading names: {e}")
        return ["Mihnea"]
    
    def save_name(self, new_name, current_names):
        """Add and save a new name to the list.
        
        Args:
            new_name: Name to add
            current_names: Current list of names
            
        Returns:
            Updated list of names
        """
        if new_name and new_name != "Select a name" and new_name not in current_names:
            current_names.append(new_name)
            current_names.sort()
            
            try:
                with open(self.names_file, "w") as f:
                    json.dump(current_names, f)
            except Exception as e:
                print(f"Error saving names: {e}")
        
        return current_names
