import json
import re
from typing import Dict, Any, Optional


class DictionaryAgent:
    """
    A simple agent that extracts dictionary structures and levels from responses.
    """
    
    def __init__(self):
        self.storage = {}  # Storage: {level: dict_content}
    
    def process_prompt(self, prompt: str) -> Dict[int, Dict[str, Any]]:
        """
        Step 1: Process a prompt/response and extract dictionary structures with levels.
        
        Args:
            prompt: The input text/response to process
            
        Returns:
            Dictionary with key: level (int), value: extracted dict
        """
        # Extract dictionaries from the text
        extracted_dicts = self._extract_dictionaries(prompt)
        
        # Extract level information and associate with dicts
        for idx, dict_content in enumerate(extracted_dicts):
            level = self._extract_level(prompt, dict_content, idx)
            self.storage[level] = dict_content
        
        return self.storage
    
    def _extract_dictionaries(self, text: str) -> list:
        """Extract all dictionary structures from text."""
        dicts = []
        
        # Try to find JSON objects in the text
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                dict_obj = json.loads(match.group())
                dicts.append(dict_obj)
            except json.JSONDecodeError:
                # Try to evaluate as Python dict
                try:
                    dict_obj = eval(match.group())
                    if isinstance(dict_obj, dict):
                        dicts.append(dict_obj)
                except:
                    continue
        
        return dicts
    
    def _extract_level(self, text: str, dict_content: Dict, idx: int) -> int:
        """
        Extract level information from the text or dictionary.
        
        Priority:
        1. Look for 'level' key in the dictionary itself
        2. Search for level mentions near the dictionary in text
        3. Use index as fallback
        """
        # Check if dict has a 'level' key
        if 'level' in dict_content:
            return int(dict_content['level'])
        
        # Search for level patterns in text
        level_patterns = [
            r'level[:\s]+(\d+)',
            r'Level[:\s]+(\d+)',
            r'LEVEL[:\s]+(\d+)',
            r'lvl[:\s]+(\d+)',
        ]
        
        for pattern in level_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Use the first level found, or match by index
                if idx < len(matches):
                    return int(matches[idx])
                return int(matches[0])
        
        # Fallback to index
        return idx
    
    def get_storage(self) -> Dict[int, Dict[str, Any]]:
        """Return the current storage."""
        return self.storage
    
    def get_by_level(self, level: int) -> Optional[Dict[str, Any]]:
        """Retrieve dictionary by level."""
        return self.storage.get(level)
    
    def clear_storage(self):
        """Clear all stored data."""
        self.storage = {}
    
    def save_to_json(self, filename: str = "agent_storage.json"):
        """
        Save the storage dictionary to a JSON file.
        
        Args:
            filename: Name of the JSON file to save to
        """
        # Convert integer keys to strings for JSON compatibility
        json_data = {str(k): v for k, v in self.storage.items()}
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Data saved to {filename}")
    
    def load_from_json(self, filename: str = "agent_storage.json"):
        """
        Load storage dictionary from a JSON file.
        
        Args:
            filename: Name of the JSON file to load from
        """
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Convert string keys back to integers
        self.storage = {int(k): v for k, v in json_data.items()}
        
        print(f"✓ Data loaded from {filename}")
        return self.storage

agent = DictionaryAgent()
with open("claude_response", "r") as f:
    sys_prompt = f.read()
result = agent.process_prompt(sys_prompt)
agent.save_to_json("phenotype.json")