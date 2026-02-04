"""
Enhanced Querymancer Tools - Local AI SQL Chatbot with Ollama + Mistral

This module provides comprehensive database tools for natural language to SQL translation
using local Ollama + Mistral integration with schema.json file and AWS SQL Server.

Dependencies: LangChain, Ollama, pyodbc, sqlalchemy
Database: AWS SQL Server
LLM: Mistral via Ollama (local)
Schema: schema.json file
"""

import re
import logging
import pyodbc
import time
import traceback
import json
import hashlib
from contextlib import contextmanager
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import threading
import os
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import pandas as pd

# Import configuration and model modules
try:
    from . import app_config, logger
    from .models import process_query, get_model_status
    from .agent import ConversationMemory
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from config import app_config, logger
        from models import process_query, get_model_status
        from agent import ConversationMemory
    except ImportError:
        # Create basic fallback logger if imports fail
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

# Enhanced Constants
CURRENT_USER = os.getenv("DB_USER", "usr_mohsin")
CURRENT_DATETIME = datetime.now()
SCHEMA_FILE_PATH = os.path.join(os.path.dirname(__file__), "schema.json")

# Thread-safe connection pool
_connection_pool = {}
_pool_lock = threading.Lock()

class QueryMancerError(Exception):
    """Base exception for QueryMancer tools"""
    pass

class TableResolutionError(QueryMancerError):
    """Exception for table resolution issues"""
    pass

class SQLExecutionError(QueryMancerError):
    """Exception for SQL execution issues"""
    pass

class SchemaError(QueryMancerError):
    """Exception for schema-related issues"""
    pass

class LocalSchemaManager:
    """
    Enhanced schema manager that loads and processes schema.json file
    for local AI SQL chatbot integration
    """
    
    def __init__(self, schema_file_path: str = SCHEMA_FILE_PATH):
        self.schema_file_path = schema_file_path
        self.schema_data = None
        self.tables_cache = {}
        self.columns_cache = {}
        self.relationships_cache = {}
        self.variations_cache = {}
        self.last_loaded = None
        self._load_schema()
    
    def _load_schema(self):
        """Load schema from JSON file"""
        try:
            if not os.path.exists(self.schema_file_path):
                logger.warning(f"Schema file not found: {self.schema_file_path}")
                self.schema_data = {"tables": []}
                return
            
            with open(self.schema_file_path, 'r', encoding='utf-8') as f:
                self.schema_data = json.load(f)
            
            self.last_loaded = datetime.now()
            self._build_caches()
            logger.info(f"Schema loaded successfully: {len(self.get_all_tables())} tables")
            
        except Exception as e:
            logger.error(f"Error loading schema: {e}")
            self.schema_data = {"tables": []}
            raise SchemaError(f"Failed to load schema: {e}")
    
    def _build_caches(self):
        """Build internal caches for fast lookup"""
        self.tables_cache = {}
        self.columns_cache = {}
        self.relationships_cache = {}
        self.variations_cache = {}
        
        if not self.schema_data or "tables" not in self.schema_data:
            return
        
        for table_info in self.schema_data["tables"]:
            table_name = table_info.get("table_name", "")
            if not table_name:
                continue
            
            # Cache table info
            self.tables_cache[table_name.upper()] = table_info
            
            # Cache columns
            columns = table_info.get("columns", [])
            self.columns_cache[table_name.upper()] = columns
            
            # Cache relationships
            foreign_keys = table_info.get("foreign_keys", [])
            self.relationships_cache[table_name.upper()] = foreign_keys
            
            # Cache variations
            variations = table_info.get("variations", [])
            variations.extend(self._generate_table_variations(table_name))
            self.variations_cache[table_name.upper()] = list(set(variations))
    
    def _generate_table_variations(self, table_name: str) -> List[str]:
        """Generate common variations for a table name"""
        variations = []
        
        # Basic variations
        variations.extend([
            table_name,
            table_name.lower(),
            table_name.upper(),
            table_name.replace('_', ' '),
            table_name.replace('_', '').lower()
        ])
        
        # Handle specific patterns
        table_upper = table_name.upper()
        
        # Type tables
        if table_upper.endswith('_TYPE'):
            base = table_upper.replace('_TYPE', '').lower()
            variations.extend([
                f"{base} type", f"{base} types", f"type of {base}",
                f"{base} type data", f"show {base} types"
            ])
        
        # Status tables
        elif table_upper.endswith('_STATUS'):
            base = table_upper.replace('_STATUS', '').lower()
            variations.extend([
                f"{base} status", f"status of {base}",
                f"{base} statuses", f"show {base} status"
            ])
        
        # Category tables
        elif table_upper.endswith('_CATEGORY'):
            base = table_upper.replace('_CATEGORY', '').lower()
            variations.extend([
                f"{base} category", f"{base} categories",
                f"category of {base}", f"show {base} categories"
            ])
        
        # Common action variations
        table_lower = table_name.lower()
        variations.extend([
            f"show {table_lower}", f"list {table_lower}", f"get {table_lower}",
            f"display {table_lower}", f"view {table_lower}", f"all {table_lower}",
            f"{table_lower} data", f"{table_lower} table"
        ])
        
        return list(set(variations))
    
    def get_all_tables(self) -> List[str]:
        """Get all table names"""
        if not self.schema_data:
            return []
        
        tables = []
        for table_info in self.schema_data.get("tables", []):
            table_name = table_info.get("table_name")
            if table_name:
                tables.append(table_name)
        
        return tables
    
    def get_table_info(self, table_name: str) -> Dict:
        """Get complete table information"""
        table_key = table_name.upper()
        return self.tables_cache.get(table_key, {})
    
    def get_table_columns(self, table_name: str) -> List[Dict]:
        """Get table columns information"""
        table_key = table_name.upper()
        return self.columns_cache.get(table_key, [])
    
    def get_table_relationships(self, table_name: str) -> List[Dict]:
        """Get table foreign key relationships"""
        table_key = table_name.upper()
        return self.relationships_cache.get(table_key, [])
    
    def resolve_table_name(self, user_input: str) -> Tuple[str, float]:
        """Resolve table name from user input with confidence scoring"""
        user_input_clean = user_input.lower().strip()
        best_match = None
        best_confidence = 0.0
        
        # Direct match
        for table_name in self.get_all_tables():
            if user_input_clean == table_name.lower():
                return table_name, 1.0
        
        # Variation matching
        for table_name in self.get_all_tables():
            variations = self.variations_cache.get(table_name.upper(), [])
            
            for variation in variations:
                if user_input_clean == variation.lower():
                    confidence = 0.95 if variation == table_name else 0.9
                    if confidence > best_confidence:
                        best_match = table_name
                        best_confidence = confidence
        
        # Fuzzy matching
        if not best_match:
            for table_name in self.get_all_tables():
                # Check if user input is contained in table name or vice versa
                table_lower = table_name.lower()
                if user_input_clean in table_lower or table_lower in user_input_clean:
                    confidence = min(len(user_input_clean), len(table_lower)) / max(len(user_input_clean), len(table_lower))
                    if confidence > 0.7 and confidence > best_confidence:
                        best_match = table_name
                        best_confidence = confidence
        
        return best_match, best_confidence
    
    def get_relevant_schema_context(self, user_query: str, max_tables: int = 5) -> str:
        """Get relevant schema context for the user query"""
        context_parts = []
        
        # Try to identify relevant tables
        relevant_tables = []
        query_lower = user_query.lower()
        
        # Direct table mentions
        for table_name in self.get_all_tables():
            if table_name.lower() in query_lower:
                relevant_tables.append(table_name)
        
        # Variation matching
        if not relevant_tables:
            for table_name in self.get_all_tables():
                variations = self.variations_cache.get(table_name.upper(), [])
                for variation in variations:
                    if variation.lower() in query_lower:
                        relevant_tables.append(table_name)
                        break
        
        # If no specific tables found, include the most common ones
        if not relevant_tables:
            all_tables = self.get_all_tables()
            relevant_tables = all_tables[:max_tables]
        
        # Limit tables
        relevant_tables = relevant_tables[:max_tables]
        
        context_parts.append("=== DATABASE SCHEMA CONTEXT ===")
        context_parts.append(f"Database: 146_36156520-AC21-435A-9C9B-1EC9145A9090")
        context_parts.append(f"Available Tables: {len(self.get_all_tables())}")
        context_parts.append("")
        
        # Add detailed info for relevant tables
        for table_name in relevant_tables:
            table_info = self.get_table_info(table_name)
            columns = self.get_table_columns(table_name)
            relationships = self.get_table_relationships(table_name)
            
            context_parts.append(f"TABLE: {table_name}")
            context_parts.append(f"Description: {table_info.get('description', 'No description available')}")
            
            # Add columns
            if columns:
                context_parts.append("COLUMNS:")
                for col in columns:
                    col_name = col.get('column_name', 'Unknown')
                    col_type = col.get('data_type', 'Unknown')
                    is_nullable = col.get('is_nullable', 'Unknown')
                    is_pk = col.get('is_primary_key', False)
                    
                    pk_indicator = " (PRIMARY KEY)" if is_pk else ""
                    context_parts.append(f"  - {col_name}: {col_type} (Nullable: {is_nullable}){pk_indicator}")
            
            # Add relationships
            if relationships:
                context_parts.append("FOREIGN KEYS:")
                for fk in relationships:
                    fk_column = fk.get('column_name', 'Unknown')
                    ref_table = fk.get('referenced_table', 'Unknown')
                    ref_column = fk.get('referenced_column', 'Unknown')
                    context_parts.append(f"  - {fk_column} -> {ref_table}.{ref_column}")
            
            context_parts.append("")
        
        # Add sample queries for context
        context_parts.append("=== EXAMPLE SQL PATTERNS ===")
        context_parts.append("SELECT TOP 10 * FROM [TableName]")
        context_parts.append("SELECT column1, column2 FROM [TableName] WHERE condition")
        context_parts.append("SELECT COUNT(*) FROM [TableName]")
        context_parts.append("")
        
        return "\n".join(context_parts)

# Global schema manager instance
schema_manager = LocalSchemaManager()

@contextmanager
def get_db_connection():
    """
    Enhanced database connection with comprehensive error handling
    Uses environment variables for AWS SQL Server connection
    """
    connection = None
    max_retries = 3
    retry_delay = 1
    
    # Get connection parameters from environment
    server = os.getenv('DB_SERVER', '10.0.0.45')
    database = os.getenv('DB_DATABASE', '146_36156520-AC21-435A-9C9B-1EC9145A9090')
    username = os.getenv('DB_USER', 'usr_mohsin')
    password = os.getenv('DB_PASSWORD', 'blY|5K:3pe10')
    driver = os.getenv('DB_DRIVER', 'ODBC Driver 17 for SQL Server')
    port = os.getenv('DB_PORT', '1433')
    
    # Build connection string
    connection_string = f"""
    DRIVER={{{driver}}};
    SERVER={server},{port};
    DATABASE={database};
    UID={username};
    PWD={password};
    TrustServerCertificate=yes;
    """
    
    for attempt in range(max_retries):
        try:
            connection = pyodbc.connect(
                connection_string.replace('\n', '').replace(' ', ''),
                timeout=30,
                autocommit=True
            )
            
            # Test connection
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            
            logger.debug(f"Database connection established (attempt {attempt + 1})")
            yield connection
            return
            
        except Exception as e:
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            if connection:
                try:
                    connection.close()
                except:
                    pass
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"All connection attempts failed: {e}")
                raise SQLExecutionError(f"Database connection failed after {max_retries} attempts: {e}")

def validate_sql_query(query: str) -> Tuple[bool, str]:
    """
    Enhanced SQL query validation for security
    """
    if not query or not query.strip():
        return False, "Empty query"
    
    query_upper = query.upper().strip()
    
    # Check for allowed statement types
    allowed_starts = ['SELECT', 'WITH']
    if not any(query_upper.startswith(start) for start in allowed_starts):
        return False, "Only SELECT and WITH statements are allowed"
    
    # Check for blocked keywords (SQL injection prevention)
    blocked_keywords = [
        "DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT", "ALTER", 
        "EXECUTE", "EXEC", "MODIFY", "GRANT", "REVOKE", "CREATE", 
        "MERGE", "OPENROWSET", "BULK", "RECONFIGURE", "SYSTEM", 
        "XP_", "SP_", "CMDSHELL", "SHUTDOWN", "DBCC"
    ]
    
    for keyword in blocked_keywords:
        if re.search(r'\b' + keyword + r'\b', query_upper):
            return False, f"Blocked keyword detected: {keyword}"
    
    # Check for suspicious patterns
    suspicious_patterns = [
        (r';\s*\w', 'Multiple statements detected'),
        (r'--', 'SQL comments detected'),
        (r'/\*.*?\*/', 'Block comments detected'),
        (r'UNION\s+(?:ALL\s+)?SELECT', 'UNION queries detected'),
        (r'WAITFOR\s+DELAY', 'Time-based attacks detected'),
    ]
    
    for pattern, description in suspicious_patterns:
        if re.search(pattern, query_upper, re.IGNORECASE | re.DOTALL):
            return False, f"Potentially dangerous SQL pattern: {description}"
    
    return True, "Query is valid"

def format_query_results(columns: List[str], rows: List[Any], 
                        execution_time: float = None, 
                        query: str = None) -> str:
    """
    Enhanced result formatting for the chatbot interface
    """
    if not rows:
        result = "✅ **Query executed successfully**\n\n"
        result += "📊 **Results:** No data returned\n"
        return result
    
    try:
        result = "✅ **Query executed successfully**\n\n"
        result += "📊 **Results:**\n\n"
        
        # Limit display rows for performance
        max_display_rows = 100
        display_rows = rows[:max_display_rows]
        
        if columns:
            # Generate markdown table
            result += "| " + " | ".join(columns) + " |\n"
            result += "| " + " | ".join(["---"] * len(columns)) + " |\n"
            
            for row in display_rows:
                formatted_row = []
                for item in row:
                    if item is None:
                        formatted_row.append("*NULL*")
                    elif isinstance(item, str):
                        # Truncate long strings
                        truncated = item[:100] + "..." if len(item) > 100 else item
                        formatted_row.append(truncated.replace("|", "\\|").replace("\n", "<br>"))
                    elif isinstance(item, datetime):
                        formatted_row.append(item.strftime("%Y-%m-%d %H:%M:%S"))
                    elif isinstance(item, (int, float)):
                        if isinstance(item, float):
                            formatted_row.append(f"{item:,.2f}")
                        else:
                            formatted_row.append(f"{item:,}")
                    else:
                        formatted_row.append(str(item).replace("|", "\\|"))
                
                result += "| " + " | ".join(formatted_row) + " |\n"
        
        # Add metadata
        result += f"\n📈 **Query Metadata:**\n"
        result += f"- **Rows Returned:** {len(rows):,}\n"
        result += f"- **Columns:** {len(columns)}\n"
        
        if len(rows) > max_display_rows:
            result += f"- **Display:** Showing first {max_display_rows:,} of {len(rows):,} rows\n"
        
        if execution_time:
            result += f"- **Execution Time:** {execution_time:.3f}s\n"
        
        result += f"- **Executed At:** {CURRENT_DATETIME.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error formatting results: {e}")
        return f"❌ **Error formatting results:** {str(e)}"

@tool
def natural_language_to_sql_tool(user_query: str) -> str:
    """
    Convert natural language to SQL using local Ollama + Mistral integration
    with schema.json context for accurate SQL generation.
    
    Args:
        user_query: Natural language question about the database
        
    Returns:
        Formatted query results or error message
    """
    try:
        if not user_query or not user_query.strip():
            return "❌ **Error:** Please provide a valid query"
        
        start_time = time.time()
        logger.info(f"Processing natural language query: {user_query}")
        
        # Get relevant schema context
        schema_context = schema_manager.get_relevant_schema_context(user_query)
        
        # Try to identify the most likely table
        resolved_table, confidence = schema_manager.resolve_table_name(user_query)
        
        if resolved_table and confidence > 0.8:
            logger.info(f"High confidence table match: {resolved_table} (confidence: {confidence:.1%})")
            
            # For high confidence simple queries, generate basic SQL
            simple_patterns = [
                'show', 'list', 'display', 'get', 'view', 'all', 
                'what is', 'what are', 'tell me'
            ]
            
            if any(pattern in user_query.lower() for pattern in simple_patterns):
                sql_query = f"SELECT TOP 50 * FROM [{resolved_table}]"
                
                # Add WHERE clause if specific conditions are mentioned
                if 'where' in user_query.lower() or 'with' in user_query.lower():
                    # Let the model handle complex conditions
                    pass
                else:
                    # Execute simple query directly
                    is_valid, validation_msg = validate_sql_query(sql_query)
                    if not is_valid:
                        return f"❌ **Security Error:** {validation_msg}"
                    
                    with get_db_connection() as connection:
                        cursor = connection.cursor()
                        cursor.execute(sql_query)
                        rows = cursor.fetchall()
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                        cursor.close()
                        
                        execution_time = time.time() - start_time
                        
                        result = format_query_results(columns, rows, execution_time, sql_query)
                        result += f"\n**🔍 Generated SQL:**\n```sql\n{sql_query}\n```"
                        
                        return result
        
        # For complex queries or low confidence matches, use the model
        try:
            # Prepare context for the model
            context_data = {
                'user_query': user_query,
                'schema_context': schema_context,
                'resolved_table': resolved_table,
                'confidence': confidence
            }
            
            # Call the model to generate SQL
            response = process_query(user_query, context_data)
            
            if response.get('success') and 'sql' in response:
                sql_query = response['sql'].strip()
                
                # Validate the generated SQL
                is_valid, validation_msg = validate_sql_query(sql_query)
                if not is_valid:
                    return f"❌ **Security Error:** {validation_msg}"
                
                # Execute the SQL
                with get_db_connection() as connection:
                    cursor = connection.cursor()
                    cursor.execute(sql_query)
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    cursor.close()
                    
                    execution_time = time.time() - start_time
                    
                    result = format_query_results(columns, rows, execution_time, sql_query)
                    result += f"\n**🔍 Generated SQL:**\n```sql\n{sql_query}\n```"
                    
                    # Add explanation if provided by model
                    if 'explanation' in response:
                        result += f"\n**💡 Explanation:** {response['explanation']}"
                    
                    return result
            
            elif response.get('success') and 'response' in response:
                # Model provided direct answer without SQL
                return response['response']
            
            else:
                error_msg = response.get('error', 'Unknown error occurred')
                return f"❌ **Query Processing Error:** {error_msg}"
                
        except Exception as e:
            logger.error(f"Error in model processing: {e}")
            return f"❌ **Model Error:** {str(e)}"
                
    except Exception as e:
        logger.error(f"Error in natural_language_to_sql_tool: {e}")
        return f"❌ **Error:** {str(e)}"

@tool
def execute_sql_query_tool(sql_query: str) -> str:
    """
    Execute SQL query with validation and security checks.
    
    Args:
        sql_query: SQL query to execute
        
    Returns:
        Formatted query results
    """
    try:
        if not sql_query or not sql_query.strip():
            return "❌ **Error:** Please provide a valid SQL query"
        
        sql_query = sql_query.strip().rstrip(';')
        
        # Validate the query
        is_valid, validation_msg = validate_sql_query(sql_query)
        if not is_valid:
            return f"❌ **Security Error:** {validation_msg}"
        
        start_time = time.time()
        
        with get_db_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(sql_query)
            
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                cursor.close()
                
                execution_time = time.time() - start_time
                
                result = format_query_results(columns, rows, execution_time, sql_query)
                result += f"\n**🔍 Executed SQL:**\n```sql\n{sql_query}\n```"
                
                return result
            else:
                affected_rows = cursor.rowcount if cursor.rowcount >= 0 else 0
                cursor.close()
                
                execution_time = time.time() - start_time
                
                result = f"✅ **Query executed successfully**\n\n"
                result += f"📊 **Affected Rows:** {affected_rows:,}\n"
                result += f"⏱️ **Execution Time:** {execution_time:.3f}s\n"
                
                return result
                
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        return f"❌ **Execution Error:** {str(e)}"

@tool
def get_database_schema_tool() -> str:
    """
    Get comprehensive database schema information from schema.json file.
    
    Returns:
        Formatted schema information
    """
    try:
        all_tables = schema_manager.get_all_tables()
        
        if not all_tables:
            return "📋 **Database Schema:** No tables found in schema"
        
        result = f"📋 **Database Schema Information**\n\n"
        result += f"**Database:** 146_36156520-AC21-435A-9C9B-1EC9145A9090\n"
        result += f"**Total Tables:** {len(all_tables)}\n\n"
        
        # Categorize tables
        type_tables = [t for t in all_tables if t.upper().endswith('_TYPE')]
        status_tables = [t for t in all_tables if t.upper().endswith('_STATUS')]
        category_tables = [t for t in all_tables if t.upper().endswith('_CATEGORY')]
        single_word_tables = [t for t in all_tables if '_' not in t]
        other_tables = [t for t in all_tables if t not in type_tables + status_tables + category_tables + single_word_tables]
        
        # Summary by category
        result += "**📊 Table Categories:**\n"
        if type_tables:
            result += f"- TYPE Tables: {len(type_tables)} ({', '.join(type_tables[:5])}{'...' if len(type_tables) > 5 else ''})\n"
        if status_tables:
            result += f"- STATUS Tables: {len(status_tables)} ({', '.join(status_tables[:5])}{'...' if len(status_tables) > 5 else ''})\n"
        if category_tables:
            result += f"- CATEGORY Tables: {len(category_tables)} ({', '.join(category_tables[:5])}{'...' if len(category_tables) > 5 else ''})\n"
        if single_word_tables:
            result += f"- Single-Word Tables: {len(single_word_tables)} ({', '.join(single_word_tables[:5])}{'...' if len(single_word_tables) > 5 else ''})\n"
        if other_tables:
            result += f"- Other Tables: {len(other_tables)} ({', '.join(other_tables[:5])}{'...' if len(other_tables) > 5 else ''})\n"
        
        result += "\n**🔍 All Tables:**\n"
        for i, table in enumerate(all_tables, 1):
            result += f"{i:2d}. {table}\n"
        
        result += f"\n**💡 Usage Examples:**\n"
        result += f"- \"Show me data from {all_tables[0] if all_tables else 'TABLE_NAME'}\"\n"
        result += f"- \"List all records in {all_tables[1] if len(all_tables) > 1 else 'TABLE_NAME'}\"\n"
        result += f"- \"Get count of {all_tables[2] if len(all_tables) > 2 else 'TABLE_NAME'}\"\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error in get_database_schema_tool: {e}")
        return f"❌ **Error:** {str(e)}"

@tool
def analyze_table_structure_tool(table_name: str) -> str:
    """
    Analyze specific table structure with detailed column and relationship information.
    
    Args:
        table_name: Name of the table to analyze
        
    Returns:
        Detailed table structure information
    """
    try:
        if not table_name or not table_name.strip():
            return "❌ **Error:** Please provide a valid table name"
        
        # Resolve table name
        actual_table, confidence = schema_manager.resolve_table_name(table_name)
        
        if not actual_table:
            available_tables = schema_manager.get_all_tables()
            return f"❌ **Error:** Table '{table_name}' not found. Available tables: {', '.join(available_tables[:10])}"
        
        # Get table information
        table_info = schema_manager.get_table_info(actual_table)
        columns = schema_manager.get_table_columns(actual_table)
        relationships = schema_manager.get_table_relationships(actual_table)
        
        result = f"🔍 **Table Structure Analysis: {actual_table}**\n\n"
        
        if table_name.lower() != actual_table.lower():
            result += f"**Resolved:** '{table_name}' → '{actual_table}' (confidence: {confidence:.1%})\n\n"
        
        # Basic table information
        description = table_info.get('description', 'No description available')
        result += f"**📊 Table Information:**\n"
        result += f"- **Description:** {description}\n"
        result += f"- **Columns:** {len(columns)}\n"
        result += f"- **Foreign Keys:** {len(relationships)}\n\n"
        
        # Column details
        if columns:
            result += "**📋 Column Details:**\n\n"
            result += "| Column | Type | Nullable | Primary Key | Description |\n"
            result += "| --- | --- | --- | --- | --- |\n"
            
            for col in columns:
                col_name = col.get('column_name', 'Unknown')
                data_type = col.get('data_type', 'Unknown')
                is_nullable = "Yes" if col.get('is_nullable', True) else "No"
                is_pk = "✅" if col.get('is_primary_key', False) else "❌"
                col_desc = col.get('description', 'No description')
                
                result += f"| {col_name} | {data_type} | {is_nullable} | {is_pk} | {col_desc} |\n"
        
        # Relationship details
        if relationships:
            result += "\n**🔗 Foreign Key Relationships:**\n\n"
            result += "| Column | References | Description |\n"
            result += "| --- | --- | --- |\n"
            
            for rel in relationships:
                col_name = rel.get('column_name', 'Unknown')
                ref_table = rel.get('referenced_table', 'Unknown')
                ref_column = rel.get('referenced_column', 'Unknown')
                rel_desc = rel.get('description', 'Foreign key relationship')
                
                result += f"| {col_name} | {ref_table}.{ref_column} | {rel_desc} |\n"
        
        # Sample queries
        result += f"\n**💡 Sample Queries:**\n"
        result += f"```sql\n"
        result += f"-- Get all records\n"
        result += f"SELECT TOP 10 * FROM [{actual_table}]\n\n"
        result += f"-- Count records\n"
        result += f"SELECT COUNT(*) as Total FROM [{actual_table}]\n\n"
        
        if columns:
            primary_cols = [col['column_name'] for col in columns if col.get('is_primary_key')]
            if primary_cols:
                result += f"-- Get specific record\n"
                result += f"SELECT * FROM [{actual_table}] WHERE {primary_cols[0]} = 'value'\n"
        
        result += f"```\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_table_structure_tool: {e}")
        return f"❌ **Error:** {str(e)}"

@tool
def test_database_connection_tool() -> str:
    """
    Test database connection and return system health information.
    
    Returns:
        Database connection and system health status
    """
    try:
        result = f"🏥 **System Health Check**\n\n"
        result += f"**Timestamp:** {CURRENT_DATETIME.strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += f"**User:** {CURRENT_USER}\n\n"
        
        # Test database connection
        result += f"**🗄️ Database Health:**\n"
        try:
            with get_db_connection() as connection:
                cursor = connection.cursor()
                
                # Get basic database info
                cursor.execute("SELECT @@VERSION as Version")
                version_info = cursor.fetchone()[0]
                
                cursor.execute("SELECT DB_NAME() as DatabaseName")
                db_name = cursor.fetchone()[0]
                
                cursor.execute("SELECT GETDATE() as ServerTime")
                server_time = cursor.fetchone()[0]
                
                cursor.close()
                
                result += f"- Status: ✅ Connected\n"
                result += f"- Database: {db_name}\n"
                result += f"- Server Time: {server_time}\n"
                result += f"- Version: {version_info[:100]}...\n"
                
        except Exception as db_error:
            result += f"- Status: ❌ Connection Failed\n"
            result += f"- Error: {str(db_error)}\n"
        
        # Test schema loading
        result += f"\n**📊 Schema Health:**\n"
        try:
            all_tables = schema_manager.get_all_tables()
            result += f"- Status: ✅ Schema Loaded\n"
            result += f"- Tables Available: {len(all_tables)}\n"
            result += f"- Schema File: {schema_manager.schema_file_path}\n"
            result += f"- Last Loaded: {schema_manager.last_loaded}\n"
            
        except Exception as schema_error:
            result += f"- Status: ❌ Schema Error\n"
            result += f"- Error: {str(schema_error)}\n"
        
        # Test model availability
        result += f"\n**🤖 Model Health:**\n"
        try:
            model_status = get_model_status()
            result += f"- Status: {model_status}\n"
        except Exception as model_error:
            result += f"- Status: ❌ Model Unavailable\n"
            result += f"- Error: {str(model_error)}\n"
        
        # Environment check
        result += f"\n**🔧 Environment Configuration:**\n"
        env_vars = ['DB_SERVER', 'DB_DATABASE', 'DB_USER', 'OLLAMA_BASE_URL']
        for var in env_vars:
            value = os.getenv(var, 'Not Set')
            # Mask sensitive information
            if var in ['DB_PASSWORD']:
                value = '***' if value != 'Not Set' else 'Not Set'
            elif var == 'DB_USER' and value != 'Not Set':
                value = f"{value[:3]}***"
            result += f"- {var}: {value}\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error in test_database_connection_tool: {e}")
        return f"❌ **Health Check Error:** {str(e)}"

@tool 
def query_suggestions_tool(context: str = "") -> str:
    """
    Generate intelligent query suggestions based on available schema and context.
    
    Args:
        context: Optional context to focus suggestions
        
    Returns:
        List of suggested queries
    """
    try:
        all_tables = schema_manager.get_all_tables()
        
        if not all_tables:
            return "❌ **Error:** No tables available for suggestions"
        
        result = f"💡 **Query Suggestions**\n\n"
        
        # Basic exploration queries
        result += f"**🔍 Basic Exploration:**\n"
        for i, table in enumerate(all_tables[:5], 1):
            result += f"{i}. Show me data from {table}\n"
            result += f"{i+5}. How many records are in {table}?\n"
        
        # Type tables suggestions
        type_tables = [t for t in all_tables if t.upper().endswith('_TYPE')]
        if type_tables:
            result += f"\n**📂 Reference Data:**\n"
            for i, table in enumerate(type_tables[:3], 1):
                base_name = table.replace('_TYPE', '').replace('_', ' ').lower()
                result += f"{i}. Show me all {base_name} types\n"
                result += f"{i+3}. List {base_name} categories\n"
        
        # Status tables suggestions
        status_tables = [t for t in all_tables if t.upper().endswith('_STATUS')]
        if status_tables:
            result += f"\n**📊 Status Information:**\n"
            for i, table in enumerate(status_tables[:3], 1):
                base_name = table.replace('_STATUS', '').replace('_', ' ').lower()
                result += f"{i}. What are the different {base_name} statuses?\n"
        
        # Common analytical queries
        result += f"\n**📈 Analytical Queries:**\n"
        if all_tables:
            sample_table = all_tables[0]
            result += f"1. Count records by category in {sample_table}\n"
            result += f"2. Show recent records from {sample_table}\n"
            result += f"3. Find duplicates in {sample_table}\n"
        
        # Context-specific suggestions
        if context and context.strip():
            context_lower = context.lower()
            relevant_tables = []
            
            for table in all_tables:
                if any(word in table.lower() for word in context_lower.split()):
                    relevant_tables.append(table)
            
            if relevant_tables:
                result += f"\n**🎯 Context-Specific ('{context}'):**\n"
                for i, table in enumerate(relevant_tables[:3], 1):
                    result += f"{i}. Show {context} data from {table}\n"
                    result += f"{i+3}. Analyze {context} trends in {table}\n"
        
        # Advanced query patterns
        result += f"\n**🔧 Advanced Patterns:**\n"
        result += f"1. Show me the top 10 records from [TABLE] ordered by [COLUMN]\n"
        result += f"2. Find all records in [TABLE] where [COLUMN] contains '[VALUE]'\n"
        result += f"3. Group [TABLE] by [COLUMN] and count occurrences\n"
        result += f"4. Join data between related tables\n"
        
        result += f"\n**💭 Tips:**\n"
        result += f"- Be specific about what data you want\n"
        result += f"- Mention table names when possible\n"
        result += f"- Use natural language - I'll convert it to SQL\n"
        result += f"- Ask for counts, totals, or specific conditions\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error in query_suggestions_tool: {e}")
        return f"❌ **Error:** {str(e)}"

@tool
def optimize_query_tool(user_query: str) -> str:
    """
    Analyze and optimize a user query for better performance and accuracy.
    
    Args:
        user_query: User's natural language query to optimize
        
    Returns:
        Optimized query suggestions and analysis
    """
    try:
        if not user_query or not user_query.strip():
            return "❌ **Error:** Please provide a query to optimize"
        
        result = f"⚡ **Query Optimization Analysis**\n\n"
        result += f"**Original Query:** {user_query}\n\n"
        
        # Table resolution analysis
        resolved_table, confidence = schema_manager.resolve_table_name(user_query)
        
        result += f"**🎯 Table Resolution:**\n"
        if resolved_table:
            result += f"- Identified Table: {resolved_table} (confidence: {confidence:.1%})\n"
            
            if confidence < 0.8:
                result += f"- ⚠️ Low confidence - consider being more specific\n"
                # Suggest similar tables
                all_tables = schema_manager.get_all_tables()
                similar_tables = [t for t in all_tables if any(word in t.lower() for word in user_query.lower().split())]
                if similar_tables:
                    result += f"- 💡 Similar tables: {', '.join(similar_tables[:5])}\n"
        else:
            result += f"- ❌ No table identified\n"
            result += f"- 💡 Try mentioning a specific table name\n"
        
        # Query complexity analysis
        result += f"\n**🔍 Query Analysis:**\n"
        
        query_lower = user_query.lower()
        complexity_factors = []
        
        # Check for aggregation keywords
        aggregation_words = ['count', 'sum', 'average', 'max', 'min', 'total']
        if any(word in query_lower for word in aggregation_words):
            complexity_factors.append("Aggregation detected")
        
        # Check for filtering keywords
        filter_words = ['where', 'with', 'having', 'filter', 'only', 'specific']
        if any(word in query_lower for word in filter_words):
            complexity_factors.append("Filtering required")
        
        # Check for sorting keywords
        sort_words = ['order', 'sort', 'arrange', 'top', 'bottom', 'first', 'last']
        if any(word in query_lower for word in sort_words):
            complexity_factors.append("Sorting detected")
        
        # Check for join indicators
        join_words = ['and', 'with', 'related', 'connected', 'linked']
        if any(word in query_lower for word in join_words):
            complexity_factors.append("Potential joins needed")
        
        if complexity_factors:
            result += f"- Complexity factors: {', '.join(complexity_factors)}\n"
        else:
            result += f"- Simple query detected\n"
        
        # Optimization suggestions
        result += f"\n**⚡ Optimization Suggestions:**\n"
        
        if not resolved_table:
            result += f"1. **Specify table name explicitly** - mention the exact table you want to query\n"
            available_tables = schema_manager.get_all_tables()
            result += f"   Available tables: {', '.join(available_tables[:10])}\n"
        
        if confidence and confidence < 0.8:
            result += f"2. **Improve table reference** - use more specific table names\n"
            if resolved_table:
                variations = schema_manager.variations_cache.get(resolved_table.upper(), [])[:5]
                result += f"   Try: {', '.join(variations)}\n"
        
        if 'all' in query_lower and 'count' not in query_lower:
            result += f"3. **Consider limiting results** - add 'top 10' or 'limit 100' for large tables\n"
        
        if any(word in query_lower for word in ['recent', 'latest', 'new']):
            result += f"4. **Specify date range** - mention specific dates or time periods\n"
        
        # Generate optimized query suggestions
        result += f"\n**💡 Optimized Query Suggestions:**\n"
        
        if resolved_table:
            table_info = schema_manager.get_table_info(resolved_table)
            columns = schema_manager.get_table_columns(resolved_table)
            
            # Simple data retrieval
            result += f"1. **Basic Data:** \"Show me the first 10 records from {resolved_table}\"\n"
            
            # Count query
            result += f"2. **Record Count:** \"How many records are in {resolved_table}?\"\n"
            
            # Specific columns
            if columns:
                key_columns = [col['column_name'] for col in columns[:3]]
                result += f"3. **Specific Columns:** \"Show me {', '.join(key_columns)} from {resolved_table}\"\n"
            
            # Filtering
            if columns:
                first_col = columns[0]['column_name']
                result += f"4. **With Filtering:** \"Show me {resolved_table} where {first_col} is not null\"\n"
        
        else:
            result += f"1. First, specify which table you want to query\n"
            result += f"2. Then add your specific requirements\n"
        
        # Performance tips
        result += f"\n**🚀 Performance Tips:**\n"
        result += f"- Use specific column names instead of SELECT *\n"
        result += f"- Add WHERE clauses to filter data\n"
        result += f"- Use TOP/LIMIT for large result sets\n"
        result += f"- Be specific about date ranges\n"
        result += f"- Mention primary key values when looking for specific records\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error in optimize_query_tool: {e}")
        return f"❌ **Error:** {str(e)}"

# Utility functions for integration
def reload_schema() -> bool:
    """Reload schema from file"""
    try:
        global schema_manager
        schema_manager = LocalSchemaManager()
        return True
    except Exception as e:
        logger.error(f"Error reloading schema: {e}")
        return False

def get_available_tools() -> List[str]:
    """Get list of available tool names"""
    return [
        "natural_language_to_sql_tool",
        "execute_sql_query_tool", 
        "get_database_schema_tool",
        "analyze_table_structure_tool",
        "test_database_connection_tool",
        "query_suggestions_tool",
        "optimize_query_tool"
    ]

def get_tool_descriptions() -> Dict[str, str]:
    """Get descriptions of all tools"""
    return {
        "natural_language_to_sql_tool": "Convert natural language questions to SQL and execute them",
        "execute_sql_query_tool": "Execute SQL queries with security validation", 
        "get_database_schema_tool": "Get comprehensive database schema information",
        "analyze_table_structure_tool": "Analyze specific table structure and relationships",
        "test_database_connection_tool": "Test database connection and system health",
        "query_suggestions_tool": "Generate intelligent query suggestions",
        "optimize_query_tool": "Analyze and optimize queries for better performance"
    }

# Legacy compatibility for existing integrations
@tool
def instant_text_to_sql_tool(text_query: str) -> str:
    """Legacy compatibility wrapper"""
    return natural_language_to_sql_tool(text_query)

@tool
def run_sql_query_tool(sql_query: str) -> str:
    """Legacy compatibility wrapper"""
    return execute_sql_query_tool(sql_query)

@tool
def list_tables_instant() -> str:
    """Legacy compatibility wrapper"""
    return get_database_schema_tool()

# Main exports
__all__ = [
    # Primary tools
    "natural_language_to_sql_tool",
    "execute_sql_query_tool",
    "get_database_schema_tool", 
    "analyze_table_structure_tool",
    "test_database_connection_tool",
    "query_suggestions_tool",
    "optimize_query_tool",
    
    # Legacy compatibility
    "instant_text_to_sql_tool",
    "run_sql_query_tool", 
    "list_tables_instant",
    
    # Utility functions
    "get_available_tools",
    "get_tool_descriptions",
    "reload_schema",
    "get_db_connection",
    "validate_sql_query",
    "format_query_results",
    
    # Classes
    "LocalSchemaManager",
    "QueryMancerError",
    "TableResolutionError", 
    "SQLExecutionError",
    "SchemaError"
]

# Initialize on import
if __name__ == "__main__":
    print("🚀 Enhanced Tools Module Loaded")
    print(f"📅 Current Time: {CURRENT_DATETIME.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👤 Current User: {CURRENT_USER}")
    print(f"📁 Schema File: {SCHEMA_FILE_PATH}")
    
    print(f"\n📋 Available Tools:")
    for tool_name in get_available_tools():
        description = get_tool_descriptions().get(tool_name, "No description")
        print(f"  - {tool_name}: {description}")
    
    print(f"\n🎯 Optimized for local Ollama + Mistral integration!")
    print(f"✨ Secure AWS SQL Server connection with schema.json context")
    
    # Quick health check
    try:
        health_status = test_database_connection_tool()
        if "✅ Connected" in health_status:
            print(f"🏥 System Status: ✅ READY")
        else:
            print(f"🏥 System Status: ⚠️ CHECK REQUIRED") 
    except:
        print(f"🏥 System Status: ❓ UNKNOWN")