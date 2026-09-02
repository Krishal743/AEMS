#!/usr/bin/env python3
"""
Debug script to understand metadata structure and fix evaluation
"""

import json
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.metadata import load_metadata, filter_by_split

def debug_metadata_structure():
    print("=== DEBUG METADATA STRUCTURE ===")
    
    try:
        metadata = load_metadata("data/processed/aems/metadata/aems_manifest_v1.json")
        print(f"Total metadata items: {len(metadata)}")
        
        # Sample first few items
        for i, item in enumerate(metadata[:3]):
            print(f"\nItem {i}:")
            for key, value in item.items():
                if key == 'qa_questions':
                    print(f"  {key}: {len(value)} questions")
                    for j, q in enumerate(value[:2]):
                        print(f"    Q{j}: {q}")
                else:
                    print(f"  {key}: {type(value)} - {value if len(str(value)) < 100 else '...'}")
        
        # Test queries structure
        test_items = filter_by_split(metadata, "test")
        print(f"\nTest items: {len(test_items)}")
        
        if test_items:
            sample_test = test_items[0]
            print(f"\nSample test item:")
            for key, value in sample_test.items():
                if key == 'qa_questions':
                    print(f"  {key}: {len(value)} questions")
                    for j, q in enumerate(value[:3]):
                        print(f"    Q{j}: '{q}'")
                else:
                    print(f"  {key}: {type(value)} - {value}")
        
        return metadata, test_items
        
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None, None

if __name__ == "__main__":
    debug_metadata_structure()