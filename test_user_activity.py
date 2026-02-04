#!/usr/bin/env python3
"""
Test script to verify USER and ACTIVITY table handling
"""

import json
import logging
from app import SchemaManager, AIQueryProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_schema_loading():
    """Test if USER and ACTIVITY tables are loaded correctly"""
    print("=" * 50)
    print("TESTING SCHEMA LOADING")
    print("=" * 50)
    
    schema_manager = SchemaManager()
    
    # Check if USER table exists
    if 'USER' in schema_manager.schema_data:
        print("✅ USER table found in schema")
        user_info = schema_manager.schema_data['USER']
        print(f"   Columns: {len(user_info.get('columns', []))}")
        print(f"   Primary Keys: {user_info.get('primary_keys', [])}")
        print(f"   Foreign Keys: {user_info.get('foreign_keys', {})}")
        print(f"   Variations: {user_info.get('variations', [])}")
    else:
        print("❌ USER table NOT found in schema")
    
    # Check if ACTIVITY table exists  
    if 'ACTIVITY' in schema_manager.schema_data:
        print("✅ ACTIVITY table found in schema")
        activity_info = schema_manager.schema_data['ACTIVITY']
        print(f"   Columns: {len(activity_info.get('columns', []))}")
        print(f"   Primary Keys: {activity_info.get('primary_keys', [])}")
        print(f"   Foreign Keys: {activity_info.get('foreign_keys', {})}")
        print(f"   Variations: {activity_info.get('variations', [])}")
    else:
        print("❌ ACTIVITY table NOT found in schema")
    
    print(f"Total tables in schema: {len(schema_manager.schema_data)}")
    return schema_manager

def test_table_detection(schema_manager):
    """Test if tables are detected correctly in queries"""
    print("\n" + "=" * 50)
    print("TESTING TABLE DETECTION")
    print("=" * 50)
    
    test_queries = [
        "tell me about user and activity tables",
        "show all users",
        "list all activities", 
        "show user information",
        "get activity data",
        "join user and activity tables"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        relevant_tables = schema_manager.get_relevant_tables(query)
        table_names = [t['name'] for t in relevant_tables]
        print(f"   Detected tables: {table_names}")
        
        if not table_names:
            print("   ❌ No tables detected!")
        elif any(name in ['USER', 'ACTIVITY'] for name in table_names):
            print("   ✅ USER/ACTIVITY tables detected correctly")
        else:
            print("   ⚠️  Wrong tables detected")

def test_schema_context(schema_manager):
    """Test schema context generation"""
    print("\n" + "=" * 50) 
    print("TESTING SCHEMA CONTEXT GENERATION")
    print("=" * 50)
    
    test_query = "tell me about user and activity tables"
    context = schema_manager.get_schema_context(test_query)
    
    print(f"Query: '{test_query}'")
    print(f"Context length: {len(context)} characters")
    print("Context preview:")
    print(context[:500] + "..." if len(context) > 500 else context)
    
    # Check if context contains expected tables
    if 'USER' in context:
        print("✅ USER table included in context")
    else:
        print("❌ USER table missing from context")
        
    if 'ACTIVITY' in context:
        print("✅ ACTIVITY table included in context")  
    else:
        print("❌ ACTIVITY table missing from context")

def main():
    """Run all tests"""
    print("QUERYMANCER USER/ACTIVITY TABLE FIX VERIFICATION")
    print("=" * 60)
    
    try:
        # Test 1: Schema Loading
        schema_manager = test_schema_loading()
        
        # Test 2: Table Detection
        test_table_detection(schema_manager)
        
        # Test 3: Schema Context
        test_schema_context(schema_manager)
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR DURING TESTING: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
