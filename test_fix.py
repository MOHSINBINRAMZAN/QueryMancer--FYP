import json
import re
import sys
import os

# Add current directory to path to import modules
sys.path.append('.')

# Test the schema manager fix
with open('schema.json', 'r') as f:
    schema_data = json.load(f)

def test_table_extraction(query):
    print(f"\n=== Testing Query: '{query}' ===")
    
    # Original broken method (substring matching)
    query_upper = query.upper()
    old_method_tables = []
    for table_name in schema_data.keys():
        if table_name in query_upper:  # This is the broken substring matching
            old_method_tables.append(table_name)
    
    # Fixed method (word boundary matching)
    potential_table_names = set(re.findall(r'[A-Z_][A-Z0-9_]*', query))
    new_method_tables = []
    for table_name in schema_data.keys():
        if table_name in potential_table_names:
            new_method_tables.append(table_name)
    
    print(f"OLD METHOD (broken): {len(old_method_tables)} tables - {old_method_tables}")
    print(f"NEW METHOD (fixed):  {len(new_method_tables)} tables - {new_method_tables}")
    print(f"Potential table names extracted: {potential_table_names}")

# Test cases
test_table_extraction("tell me about FAX_CAMPAIGN_ACCOUNT and FACULTY")
test_table_extraction("show me USER and ACCOUNT data")
test_table_extraction("VAT_RATE information")
