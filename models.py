"""
models.py - Simplified Model Management for Local Ollama + Mistral Setup

This module provides streamlined model management for natural language to SQL translation
using local Ollama + Mistral, schema.json integration, RAG enhancement with FAISS, 
and AWS SQL Server connection.
"""

import os
import json
import time
import logging
import re
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from pathlib import Path

# Core LangChain imports
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Ollama integration
try:
    from langchain_ollama import ChatOllama
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        ChatOllama = None
        logging.error("ChatOllama not available. Install: pip install langchain-ollama or langchain-community")

# RAG Engine integration
try:
    from rag_engine import get_rag_engine, enhance_with_rag, QueryMancerRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logging.warning("RAG engine not available. Using standard schema context.")

# Database connectivity
import pyodbc
import sqlalchemy
from sqlalchemy import create_engine, text

# Import your configuration
try:
    from config import Config
except ImportError:
    # Fallback if config is in different location
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Simplified model manager for local Ollama + Mistral setup"""
    
    def __init__(self):
        self.config = Config()
        self.model = None
        self.schema_data = None
        self.db_engine = None
        
        # Initialize components
        self._load_schema()
        self._initialize_model()
        self._initialize_database()
    
    def _load_schema(self):
        """Load schema from JSON file"""
        try:
            schema_path = Path("schema.json")
            if not schema_path.exists():
                # Try different possible locations
                possible_paths = ["schemas/schema.json", "data/schema.json", "../schema.json"]
                for path in possible_paths:
                    if Path(path).exists():
                        schema_path = Path(path)
                        break
                else:
                    raise FileNotFoundError("schema.json not found")
            
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.schema_data = json.load(f)
            
            logger.info(f"✅ Schema loaded successfully from {schema_path}")
            logger.info(f"📊 Found {len(self.schema_data.get('tables', {}))} tables")
            
        except Exception as e:
            logger.error(f"❌ Error loading schema: {e}")
            self.schema_data = {"tables": {}, "relationships": []}
    
    def _initialize_model(self):
        """Initialize Ollama + Mistral model"""
        try:
            if ChatOllama is None:
                raise ImportError("ChatOllama not available")
            
            # Configure Ollama with optimized settings for SQL generation
            self.model = ChatOllama(
                model=self.config.OLLAMA_MODEL,
                base_url=self.config.OLLAMA_BASE_URL,
                temperature=0.1,  # Low temperature for more deterministic SQL output
                num_ctx=4096,     # Context window
                num_predict=1000, # Max tokens for response
                repeat_penalty=1.05,
                top_k=20,
                top_p=0.8,
                keep_alive="15m"  # Keep model loaded
            )
            
            # Test the model connection
            test_response = self.model.invoke("SELECT 1")
            logger.info(f"✅ Ollama model '{self.config.OLLAMA_MODEL}' initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Ollama model: {e}")
            raise
    
    def _initialize_database(self):
        """Initialize AWS SQL Server connection"""
        try:
            # Build connection string for SQL Server
            connection_string = (
                f"mssql+pyodbc://{self.config.DB_USER}:{self.config.DB_PASSWORD}@"
                f"{self.config.DB_HOST}:{self.config.DB_PORT}/{self.config.DB_NAME}"
                f"?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
            )
            
            self.db_engine = create_engine(
                connection_string,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600,
                echo=False  # Set to True for SQL debugging
            )
            
            # Test connection
            with self.db_engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test"))
                test_result = result.fetchone()
            
            logger.info("✅ Database connection established successfully")
            
        except Exception as e:
            logger.error(f"❌ Error connecting to database: {e}")
            raise
    
    def get_relevant_schema_context(self, question: str, max_tables: int = 5) -> str:
        """Get relevant schema context based on the question"""
        if not self.schema_data or not self.schema_data.get('tables'):
            return "No schema information available."
        
        question_lower = question.lower()
        
        relevant_tables = []
        
        # Score tables based on relevance to the question
        table_scores = {}
        
        for table_name, table_info in self.schema_data['tables'].items():
            score = 0
            table_name_lower = table_name.lower()
            
            # Direct table name match
            if table_name_lower in question_lower:
                score += 10
            
            # Partial table name match
            for part in table_name_lower.split('_'):
                if part in question_lower:
                    score += 3
            
            # Check natural language variations
            natural_names = table_info.get('natural_language', [])
            for nat_name in natural_names:
                if nat_name.lower() in question_lower:
                    score += 8
            
            # Check column names
            columns = table_info.get('columns', {})
            for col_name, col_info in columns.items():
                if col_name.lower() in question_lower:
                    score += 5
                
                # Check column description
                if isinstance(col_info, dict):
                    description = col_info.get('description', '').lower()
                    if description and any(word in description for word in question_lower.split()):
                        score += 2
            
            # Check table description
            description = table_info.get('description', '').lower()
            if description and any(word in description for word in question_lower.split()):
                score += 3
            
            if score > 0:
                table_scores[table_name] = score
        
        # Get top scoring tables
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        relevant_table_names = [table for table, score in sorted_tables[:max_tables]]
        
        # If no relevant tables found, include first few tables
        if not relevant_table_names:
            relevant_table_names = list(self.schema_data['tables'].keys())[:max_tables]
        
        # Build schema context
        context_parts = []
        context_parts.append("=== DATABASE SCHEMA ===")
        
        for table_name in relevant_table_names:
            table_info = self.schema_data['tables'][table_name]
            context_parts.append(f"\nTable: {table_name}")
            
            # Add description
            if table_info.get('description'):
                context_parts.append(f"Description: {table_info['description']}")
            
            # Add columns
            columns = table_info.get('columns', {})
            if columns:
                context_parts.append("Columns:")
                for col_name, col_info in columns.items():
                    if isinstance(col_info, dict):
                        col_type = col_info.get('type', col_info.get('data_type', 'Unknown'))
                        col_desc = col_info.get('description', '')
                        
                        # Build column flags
                        flags = []
                        if col_info.get('primary_key', False):
                            flags.append('PK')
                        if col_info.get('foreign_key', False):
                            flags.append('FK')
                        if not col_info.get('nullable', True):
                            flags.append('NOT NULL')
                        
                        flag_str = f" ({', '.join(flags)})" if flags else ""
                        context_parts.append(f"  - {col_name}: {col_type}{flag_str} - {col_desc}")
                    else:
                        context_parts.append(f"  - {col_name}: {col_info}")
            
            # Add natural language variations
            if table_info.get('natural_language'):
                context_parts.append(f"Also known as: {', '.join(table_info['natural_language'])}")
        
        # Add relationships if available
        if self.schema_data.get('relationships'):
            context_parts.append("\n=== RELATIONSHIPS ===")
            for rel in self.schema_data['relationships']:
                if rel.get('source_table') in relevant_table_names or rel.get('target_table') in relevant_table_names:
                    source = f"{rel.get('source_table', '')}.{rel.get('source_column', '')}"
                    target = f"{rel.get('target_table', '')}.{rel.get('target_column', '')}"
                    rel_type = rel.get('relationship_type', 'REFERENCES')
                    context_parts.append(f"- {source} {rel_type} {target}")
        
        return "\n".join(context_parts)
    
    def generate_sql(self, question: str, use_rag: bool = True) -> Dict[str, Any]:
        """Generate SQL query from natural language question with optional RAG enhancement
        
        Args:
            question: Natural language question to convert to SQL
            use_rag: Whether to use RAG enhancement (default: True)
            
        Returns:
            Dict containing generated SQL and metadata
        """
        try:
            start_time = time.time()
            rag_metadata = {}
            
            # Try RAG-enhanced context if available and enabled
            if use_rag and RAG_AVAILABLE:
                try:
                    rag_engine = get_rag_engine()
                    rag_result = rag_engine.enhance_query(question)
                    
                    # Use RAG-enhanced context
                    schema_context = rag_result.get('enhanced_context', '')
                    
                    # Add similar examples to context if available
                    similar_examples = rag_result.get('similar_examples', [])
                    if similar_examples:
                        example_context = "\n\n=== SIMILAR QUERY EXAMPLES ===\n"
                        for ex in similar_examples[:2]:  # Top 2 examples
                            example_context += f"Question: {ex['question']}\n"
                            example_context += f"SQL: {ex['sql']}\n\n"
                        schema_context += example_context
                    
                    rag_metadata = {
                        'rag_enabled': True,
                        'confidence_scores': rag_result.get('confidence_scores', {}),
                        'tables_found': rag_result.get('relevant_tables', []),
                        'examples_used': len(similar_examples)
                    }
                    
                    logger.info(f"RAG enhancement applied - Confidence: {rag_result.get('confidence_scores', {})}")
                    
                except Exception as rag_error:
                    logger.warning(f"RAG enhancement failed: {rag_error}. Using standard context.")
                    schema_context = self.get_relevant_schema_context(question)
                    rag_metadata = {'rag_enabled': False, 'fallback_reason': str(rag_error)}
            else:
                # Get standard schema context
                schema_context = self.get_relevant_schema_context(question)
                rag_metadata = {'rag_enabled': False}
            
            # Create enhanced prompt for SQL generation
            sql_prompt = ChatPromptTemplate.from_template("""
You are an expert SQL Server database analyst. Convert natural language questions to precise SQL Server queries.

{schema_context}

=== CRITICAL REQUIREMENTS ===
1. Generate ONLY SQL Server compatible queries
2. Use EXACT table and column names from the schema above
3. Use TOP instead of LIMIT for row limiting
4. Use square brackets [Table Name] for names with spaces or special characters
5. Use SQL Server functions like GETDATE(), DATEDIFF(), YEAR(), MONTH(), etc.
6. Include proper WHERE clauses for filtering
7. Use appropriate JOINs when multiple tables are needed
8. Handle NULL values correctly with IS NULL or IS NOT NULL
9. Use proper SQL Server date formats
10. Return ONLY the SQL query without explanations or formatting

=== QUESTION ===
{question}

=== SQL QUERY ===
""")
            
            # Generate SQL using the model
            chain = sql_prompt | self.model | StrOutputParser()
            
            response = chain.invoke({
                "schema_context": schema_context,
                "question": question
            })
            
            # Clean and extract SQL
            sql_query = self._clean_sql_response(response)
            
            # Validate SQL structure
            is_valid, validation_message = self._validate_sql(sql_query)
            
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "sql": sql_query,
                "question": question,
                "schema_used": len(schema_context.split('\n')),
                "execution_time": execution_time,
                "is_valid": is_valid,
                "validation_message": validation_message,
                "timestamp": datetime.now().isoformat(),
                "rag_metadata": rag_metadata  # Include RAG enhancement metadata
            }
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return {
                "success": False,
                "error": str(e),
                "question": question,
                "timestamp": datetime.now().isoformat()
            }
    
    def _clean_sql_response(self, response: str) -> str:
        """Clean and extract SQL from model response"""
        # Remove markdown code blocks
        sql_match = re.search(r'```(?:sql)?\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            sql = sql_match.group(1)
        else:
            sql = response
        
        # Clean the SQL
        sql = sql.strip()
        
        # Remove common prefixes that models sometimes add
        prefixes_to_remove = [
            'sql query:', 'query:', 'select query:', 'here is the sql:', 'sql:'
        ]
        for prefix in prefixes_to_remove:
            if sql.lower().startswith(prefix):
                sql = sql[len(prefix):].strip()
        
        # Remove trailing semicolons and clean up
        sql = sql.rstrip(';').strip()
        
        # Ensure it starts with SELECT, WITH, or other valid SQL commands
        if not re.match(r'^\s*(SELECT|WITH|EXEC|EXECUTE)', sql, re.IGNORECASE):
            # Try to find SELECT in the response
            select_match = re.search(r'(SELECT.*?)(?:\n\n|$|```)', sql, re.DOTALL | re.IGNORECASE)
            if select_match:
                sql = select_match.group(1).strip()
        
        return sql
    
    def _validate_sql(self, sql: str) -> Tuple[bool, str]:
        """Validate basic SQL structure"""
        if not sql or not sql.strip():
            return False, "Empty SQL query"
        
        sql_upper = sql.upper().strip()
        
        # Must start with valid SQL command
        valid_starts = ['SELECT', 'WITH', 'EXEC', 'EXECUTE']
        if not any(sql_upper.startswith(start) for start in valid_starts):
            return False, f"SQL must start with one of: {', '.join(valid_starts)}"
        
        # For SELECT queries, must contain FROM (unless it's a simple calculation)
        if sql_upper.startswith('SELECT'):
            if 'FROM' not in sql_upper and not re.search(r'SELECT\s+\d+', sql_upper):
                return False, "SELECT query missing FROM clause"
        
        # Check for dangerous keywords
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False, f"Potentially dangerous keyword '{keyword}' detected"
        
        # Basic syntax checks
        if sql.count('(') != sql.count(')'):
            return False, "Unmatched parentheses"
        
        if sql.count("'") % 2 != 0:
            return False, "Unmatched single quotes"
        
        return True, "SQL structure is valid"
    
    def execute_query(self, sql: str, limit_results: int = 1000) -> Dict[str, Any]:
        """Execute SQL query safely against the database"""
        try:
            start_time = time.time()
            
            # Validate SQL first
            is_valid, validation_message = self._validate_sql(sql)
            if not is_valid:
                return {
                    "success": False,
                    "error": f"Invalid SQL: {validation_message}",
                    "sql": sql
                }
            
            # Add result limiting if not already present
            sql_upper = sql.upper()
            if 'TOP' not in sql_upper and 'SELECT' in sql_upper and limit_results:
                # Simple injection of TOP clause
                sql = re.sub(r'(SELECT\s+)', f'SELECT TOP {limit_results} ', sql, flags=re.IGNORECASE)
            
            # Execute query
            with self.db_engine.connect() as conn:
                result = conn.execute(text(sql))
                
                # Fetch results
                rows = result.fetchall()
                columns = list(result.keys())
                
                # Convert to list of dictionaries
                data = []
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        # Handle datetime objects
                        if hasattr(value, 'isoformat'):
                            value = value.isoformat()
                        row_dict[col] = value
                    data.append(row_dict)
            
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "data": data,
                "columns": columns,
                "row_count": len(data),
                "sql": sql,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "sql": sql,
                "timestamp": datetime.now().isoformat()
            }
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """Complete pipeline: Question -> SQL -> Results"""
        try:
            logger.info(f"Processing question: {question}")
            
            # Generate SQL
            sql_result = self.generate_sql(question)
            
            if not sql_result["success"]:
                return sql_result
            
            # Execute SQL if valid
            if sql_result["is_valid"]:
                execution_result = self.execute_query(sql_result["sql"])
                
                # Combine results
                final_result = {
                    **sql_result,
                    "execution": execution_result,
                    "has_results": execution_result.get("success", False)
                }
                
                if execution_result.get("success"):
                    final_result.update({
                        "data": execution_result["data"],
                        "columns": execution_result["columns"],
                        "row_count": execution_result["row_count"]
                    })
                
                return final_result
            
            else:
                return {
                    **sql_result,
                    "execution": {
                        "success": False,
                        "error": "SQL validation failed"
                    }
                }
        
        except Exception as e:
            logger.error(f"Error in process_question: {e}")
            return {
                "success": False,
                "error": str(e),
                "question": question,
                "timestamp": datetime.now().isoformat()
            }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test all connections (model and database)"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # Test Ollama model
        try:
            test_response = self.model.invoke("Generate a simple SELECT statement")
            results["tests"]["ollama"] = {
                "success": True,
                "message": "Ollama model responding",
                "response_length": len(str(test_response))
            }
        except Exception as e:
            results["tests"]["ollama"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test database connection
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text("SELECT GETDATE() as current_time"))
                db_time = result.fetchone()[0]
            
            results["tests"]["database"] = {
                "success": True,
                "message": "Database connection successful",
                "server_time": str(db_time)
            }
        except Exception as e:
            results["tests"]["database"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test schema loading
        results["tests"]["schema"] = {
            "success": bool(self.schema_data and self.schema_data.get('tables')),
            "tables_count": len(self.schema_data.get('tables', {})) if self.schema_data else 0
        }
        
        # Overall status
        all_tests_passed = all(test.get("success", False) for test in results["tests"].values())
        results["overall_status"] = "ready" if all_tests_passed else "issues_detected"
        
        return results
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about loaded schema"""
        if not self.schema_data:
            return {"error": "No schema loaded"}
        
        tables = self.schema_data.get('tables', {})
        relationships = self.schema_data.get('relationships', [])
        
        table_info = []
        for table_name, table_data in tables.items():
            columns = table_data.get('columns', {})
            table_info.append({
                "name": table_name,
                "description": table_data.get('description', ''),
                "column_count": len(columns),
                "has_primary_key": any(
                    isinstance(col, dict) and col.get('primary_key', False) 
                    for col in columns.values()
                ),
                "natural_language": table_data.get('natural_language', [])
            })
        
        return {
            "total_tables": len(tables),
            "total_relationships": len(relationships),
            "tables": table_info,
            "timestamp": datetime.now().isoformat()
        }

# Global model manager instance
model_manager = None

def get_model_manager() -> ModelManager:
    """Get or create the global model manager instance"""
    global model_manager
    if model_manager is None:
        model_manager = ModelManager()
    return model_manager

def initialize_models():
    """Initialize the model manager"""
    global model_manager
    try:
        model_manager = ModelManager()
        logger.info("✅ Model manager initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize model manager: {e}")
        return False

# Convenience functions for backward compatibility
def nl_to_sql(question: str) -> Dict[str, Any]:
    """Convert natural language to SQL"""
    manager = get_model_manager()
    sql_result = manager.generate_sql(question)
    return sql_result

def process_query(question: str) -> Dict[str, Any]:
    """Process complete question pipeline"""
    manager = get_model_manager()
    return manager.process_question(question)

def execute_sql(sql: str) -> Dict[str, Any]:
    """Execute SQL query"""
    manager = get_model_manager()
    return manager.execute_query(sql)

def test_system() -> Dict[str, Any]:
    """Test system components"""
    manager = get_model_manager()
    return manager.test_connection()


def get_rag_status() -> Dict[str, Any]:
    """Get RAG engine status and statistics"""
    if not RAG_AVAILABLE:
        return {
            "available": False,
            "reason": "RAG engine module not imported"
        }
    
    try:
        rag_engine = get_rag_engine()
        stats = rag_engine.get_statistics()
        return {
            "available": True,
            "initialized": True,
            "statistics": stats
        }
    except Exception as e:
        return {
            "available": True,
            "initialized": False,
            "error": str(e)
        }


def reindex_rag_schema() -> Dict[str, Any]:
    """Re-index the schema in RAG vector store"""
    if not RAG_AVAILABLE:
        return {
            "success": False,
            "reason": "RAG engine not available"
        }
    
    try:
        rag_engine = get_rag_engine()
        rag_engine.reindex_schema()
        return {
            "success": True,
            "statistics": rag_engine.get_statistics()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def get_schema_info() -> Dict[str, Any]:
    """Get schema information"""
    manager = get_model_manager()
    return manager.get_schema_info()

# Example usage and testing
if __name__ == "__main__":
    print("🚀 QueryMancer Model Manager - Local Setup")
    print("=" * 50)
    
    # Initialize
    try:
        initialize_models()
        manager = get_model_manager()
        
        # Test connections
        print("\n🔧 Testing System Components...")
        test_results = manager.test_connection()
        
        for component, result in test_results["tests"].items():
            status = "✅" if result["success"] else "❌"
            print(f"{status} {component.capitalize()}: {result.get('message', result.get('error', 'Unknown'))}")
        
        print(f"\n📊 Overall Status: {test_results['overall_status'].upper()}")
        
        # Show schema info
        schema_info = manager.get_schema_info()
        if "error" not in schema_info:
            print(f"\n📋 Schema Info:")
            print(f"   Tables: {schema_info['total_tables']}")
            print(f"   Relationships: {schema_info['total_relationships']}")
        
        # Test example query
        if test_results["overall_status"] == "ready":
            print("\n🧪 Testing Example Query...")
            example_question = "Show me the first 5 rows from any table"
            result = manager.process_question(example_question)
            
            if result["success"]:
                print(f"✅ SQL Generated: {result['sql'][:100]}...")
                if result.get("has_results"):
                    print(f"📊 Results: {result.get('row_count', 0)} rows returned")
                else:
                    print("⚠️ SQL generated but execution failed")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        print("\n🎯 QueryMancer ready for natural language to SQL conversion!")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        print("Please check your configuration and dependencies.")