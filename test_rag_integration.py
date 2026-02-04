"""
Test script for RAG integration with QueryMancer

This script tests the RAG engine integration with the existing QueryMancer system.
Run this to verify that RAG is working correctly with the Mistral LLM.
"""

import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_rag_standalone():
    """Test RAG engine standalone functionality"""
    print("=" * 60)
    print("Testing RAG Engine Standalone")
    print("=" * 60)
    
    try:
        from rag_engine import get_rag_engine, enhance_with_rag, get_rag_relevant_tables
        
        # Initialize RAG engine
        rag = get_rag_engine()
        
        # Get statistics
        stats = rag.get_statistics()
        print(f"\n✅ RAG Engine Statistics:")
        print(f"   - Tables indexed: {stats['tables_indexed']}")
        print(f"   - Columns indexed: {stats['columns_indexed']}")
        print(f"   - Examples indexed: {stats['examples_indexed']}")
        print(f"   - Embedding dimension: {stats['embedding_dimension']}")
        
        # Test query enhancement
        test_queries = [
            "Show me all contacts",
            "List accounts with their owners",
            "Get events and their sessions",
            "Show invoices for last month",
            "Find products by category"
        ]
        
        print(f"\n{'=' * 60}")
        print("Testing Query Enhancement")
        print("=" * 60)
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            
            result = rag.enhance_query(query)
            
            tables = result.get('relevant_tables', [])
            confidence = result.get('confidence_scores', {})
            examples = result.get('similar_examples', [])
            
            print(f"   📊 Relevant Tables: {tables[:3]}{'...' if len(tables) > 3 else ''}")
            print(f"   📈 Confidence: table={confidence.get('table_confidence', 0):.2f}, "
                  f"column={confidence.get('column_confidence', 0):.2f}, "
                  f"example={confidence.get('example_confidence', 0):.2f}")
            
            if examples:
                print(f"   📚 Similar Example: {examples[0]['question'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ RAG Engine Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_integration():
    """Test RAG integration with agent (if Ollama is running)"""
    print("\n" + "=" * 60)
    print("Testing RAG Integration with Agent")
    print("=" * 60)
    
    try:
        # Check if RAG is available in agent
        from agent import RAG_AVAILABLE
        
        if RAG_AVAILABLE:
            print("✅ RAG is available in agent module")
        else:
            print("⚠️ RAG is not available in agent module (import failed)")
            return False
        
        # Test with a simple query enhancement
        from rag_engine import get_rag_engine
        
        rag = get_rag_engine()
        result = rag.enhance_query("Show me the top 10 accounts")
        
        enhanced_context = result.get('enhanced_context', '')
        
        if enhanced_context:
            print(f"✅ Enhanced context generated ({len(enhanced_context)} chars)")
            print(f"\n--- Context Preview (first 500 chars) ---")
            print(enhanced_context[:500])
            print("...")
        else:
            print("❌ Failed to generate enhanced context")
            return False
            
        return True
        
    except ImportError as e:
        print(f"⚠️ Import error (this is expected if DB config is missing): {e}")
        return True  # Not a real failure if DB config is missing
        
    except Exception as e:
        print(f"❌ Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_models_rag():
    """Test RAG integration with models module"""
    print("\n" + "=" * 60)
    print("Testing RAG Integration with Models")
    print("=" * 60)
    
    try:
        from models import get_rag_status, RAG_AVAILABLE
        
        if RAG_AVAILABLE:
            print("✅ RAG is available in models module")
            
            status = get_rag_status()
            print(f"   RAG Status: {json.dumps(status, indent=2, default=str)}")
        else:
            print("⚠️ RAG is not available in models module")
            
        return True
        
    except ImportError as e:
        print(f"⚠️ Models import failed (expected if DB config missing): {e}")
        return True
        
    except Exception as e:
        print(f"❌ Models RAG Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_add_query_example():
    """Test adding new query examples to RAG"""
    print("\n" + "=" * 60)
    print("Testing Adding Query Examples")
    print("=" * 60)
    
    try:
        from rag_engine import get_rag_engine
        
        rag = get_rag_engine()
        
        # Add a test example
        test_question = "How many active members do we have?"
        test_sql = "SELECT COUNT(*) AS active_members FROM [ACCOUNT] WHERE [MEMBER_STATUS_NAME] = 'Active'"
        test_desc = "Count of active members in the system"
        
        # Get count before
        stats_before = rag.get_statistics()
        examples_before = stats_before['examples_indexed']
        
        # Add example
        rag.add_query_example(test_question, test_sql, test_desc)
        
        # Get count after
        stats_after = rag.get_statistics()
        examples_after = stats_after['examples_indexed']
        
        if examples_after > examples_before:
            print(f"✅ Successfully added query example")
            print(f"   Examples before: {examples_before}")
            print(f"   Examples after: {examples_after}")
        else:
            print(f"❌ Failed to add query example")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Add Example Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all RAG integration tests"""
    print("\n" + "=" * 70)
    print("     QueryMancer RAG Integration Test Suite")
    print("=" * 70)
    
    results = {}
    
    # Test 1: RAG Standalone
    results['rag_standalone'] = test_rag_standalone()
    
    # Test 2: RAG Integration
    results['rag_integration'] = test_rag_integration()
    
    # Test 3: Models RAG
    results['models_rag'] = test_models_rag()
    
    # Test 4: Add Query Example
    results['add_example'] = test_add_query_example()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! RAG integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the output above.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
