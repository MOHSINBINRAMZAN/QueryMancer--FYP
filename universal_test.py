import json
import re

# Load schema to test with real table names
with open('schema.json', 'r') as f:
    schema_data = json.load(f)

def test_universal_table_detection(query):
    print(f"\n=== Testing: '{query}' ===")
    
    # Fixed method (what's now in the app)
    potential_table_names = set(re.findall(r'[A-Z_][A-Z0-9_]*', query))
    found_tables = []
    for table_name in schema_data.keys():
        if table_name in potential_table_names:
            found_tables.append(table_name)
    
    print(f"Extracted words: {potential_table_names}")
    print(f"Found tables: {found_tables}")
    print(f"Count: {len(found_tables)}")

# Test various table combinations from your schema
test_universal_table_detection("tell me about FAX_CAMPAIGN_ACCOUNT and FACULTY")
test_universal_table_detection("show USER and ACCOUNT data")
test_universal_table_detection("EMAIL_CAMPAIGN_CONTACT information")
test_universal_table_detection("ACCOUNT_STATUS and USER_ACTIVITY")
test_universal_table_detection("compare ABSTRACT and ABSTRACT_REVIEW")
test_universal_table_detection("VAT_RATE details")
test_universal_table_detection("show me CONTACT and MEMBER data")
test_universal_table_detection("ACTIVITY_TASK and TASK_TYPE relationship")
