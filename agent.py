"""
Querymancer Agent - Local AI-Powered SQL Generation with Ollama + Mistral

This module provides the core agent functionality for natural language to SQL translation
using local Ollama + Mistral model with manual schema injection from schema.json.
Enhanced with RAG (Retrieval-Augmented Generation) using FAISS vector database for 
improved contextual accuracy and query performance.
"""

import logging
import time
import re
import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# Core LangChain imports for local Ollama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import json

load_dotenv()


# RAG Engine imports for enhanced context retrieval
try:
    from rag_engine import get_rag_engine, enhance_with_rag, get_rag_relevant_tables, QueryMancerRAG
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    logging.warning(f"RAG engine not available: {e}. Using standard schema context.")
# Database imports
from sqlalchemy import create_engine, text as sql_text, inspect
import pyodbc
from sqlalchemy.exc import SQLAlchemyError

# Import configuration
try:
    from . import app_config, logger
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import app_config, logger

@dataclass
class AgentConfig:
    """Agent-specific configuration for local deployment"""
    max_retries: int = 3
    retry_delay: float = 1.0
    query_timeout: int = 30
    max_context_length: int = 4000
    confidence_threshold: float = 0.85
    enable_query_validation: bool = True
    enable_sql_explanation: bool = True
    max_query_limit: int = 1000
    cache_enabled: bool = True
    cache_ttl: int = 3600

class SchemaManager:
    """Manages schema.json loading and context generation"""
    
    def __init__(self, schema_file_path: str = "schema.json"):
        self.schema_file_path = schema_file_path
        self.schema_data = None
        self.last_loaded = None
        self.load_schema()
    
    def load_schema(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load schema from schema.json file"""
        try:
            schema_path = Path(self.schema_file_path)
            
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema file not found: {self.schema_file_path}")
            
            # Check if reload is needed
            if not force_reload and self.schema_data and self.last_loaded:
                file_mtime = schema_path.stat().st_mtime
                if file_mtime <= self.last_loaded:
                    return self.schema_data
            
            # Load schema
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.schema_data = json.load(f)
            
            self.last_loaded = time.time()
            logger.info(f"Schema loaded successfully from {self.schema_file_path}")
            
            return self.schema_data
            
        except Exception as e:
            logger.error(f"Error loading schema: {e}")
            raise
    
    def get_all_tables(self) -> List[str]:
        """Get list of all table names"""
        if not self.schema_data:
            return []
        
        return list(self.schema_data.get('tables', {}).keys())
    
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific table"""
        if not self.schema_data:
            return None
        
        if isinstance(self.schema_data, dict) and table_name in self.schema_data:
            # Direct table access in flat schema format
            return self.schema_data.get(table_name)
            
        tables = self.schema_data.get('tables', {})
        if tables:
            return tables.get(table_name)
            
        # If we get here, try direct access as a fallback
        return self.schema_data.get(table_name)
        
    def get_key_columns(self, table_name: str, max_columns: int = 5) -> List[str]:
        """Get the most important columns for a table (primary keys + descriptive columns)"""
        table_info = self.get_table_info(table_name)
        if not table_info:
            return []
            
        result_columns = []
        
        # First add primary keys
        if 'primary_keys' in table_info:
            result_columns.extend(table_info['primary_keys'])
            
        # Then add foreign keys
        if 'foreign_keys' in table_info:
            for fk_col in table_info['foreign_keys'].keys():
                if fk_col not in result_columns and len(result_columns) < max_columns:
                    result_columns.append(fk_col)
        
        # Then add descriptive columns (name, title, description, etc.)
        descriptive_patterns = ['name', 'title', 'description', 'label', 'type', 'code', 'status', 'email']
        if 'columns' in table_info:
            for col in table_info['columns']:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in descriptive_patterns):
                    if col not in result_columns and len(result_columns) < max_columns:
                        result_columns.append(col)
        
        # If we still need more columns, add any remaining ones up to max_columns
        if 'columns' in table_info and len(result_columns) < max_columns:
            for col in table_info['columns']:
                if col not in result_columns and len(result_columns) < max_columns:
                    result_columns.append(col)
                    
        return result_columns
    
    def find_relevant_tables(self, query_text: str) -> List[str]:
        """Find tables relevant to the user query using keyword matching"""
        if not self.schema_data:
            return []
        
        query_lower = query_text.lower()
        query_words = set(query_lower.split())
        relevant_tables = set()
        table_mentions = {}  # Track number of mentions for each table
        
        # First check if this is a specific query about tables
        explicit_table_pattern = r'tell me about (?:the\s+)?(\w+)(?:\s+and\s+(\w+))? tables?'
        explicit_match = re.search(explicit_table_pattern, query_lower)
        if explicit_match:
            explicit_tables = [name for name in explicit_match.groups() if name]
            # Try to map these explicit mentions to actual table names
            for mention in explicit_tables:
                exact_match_found = False
                
                # Try direct matches
                for table_name in self.schema_data.get('tables', {}):
                    if table_name.lower() == mention.lower():
                        relevant_tables.add(table_name)
                        exact_match_found = True
                        break
                
                # If no direct match, try variations
                if not exact_match_found:
                    for table_name, table_info in self.schema_data.get('tables', {}).items():
                        variations = table_info.get('variations', [])
                        if variations and any(v.lower() == mention.lower() for v in variations):
                            relevant_tables.add(table_name)
                            break
        
        # Standard keyword matching approach
        for table_name, table_info in self.schema_data.get('tables', {}).items():
            mention_score = 0
            
            # Check for exact table name
            if table_name.lower() in query_lower:
                mention_score += 3  # Higher score for direct name match
            
            # Check natural language variations with clear word boundaries
            variations = table_info.get('variations', [])
            if variations:
                for variation in variations:
                    var_lower = variation.lower()
                    # Look for complete words (with boundaries)
                    if re.search(r'\b' + re.escape(var_lower) + r'\b', query_lower):
                        mention_score += 2
                        break
            
            # Check column names for keywords
            column_matches = 0
            columns = table_info.get('columns', [])
            for col_name in columns:
                col_lower = col_name.lower()
                # Only count distinct column matches
                if col_lower in query_lower or col_lower.replace('_', ' ') in query_lower:
                    column_matches += 1
                    if column_matches >= 3:  # Limit to avoid over-counting
                        break
            
            # Add score based on column matches, but with diminishing returns
            mention_score += min(column_matches, 3) * 0.5
                
            # If the table has any mentions, add it to our tracking
            if mention_score > 0:
                table_mentions[table_name] = mention_score
                # Only add tables with significant mentions
                if mention_score >= 2:
                    relevant_tables.add(table_name)
        
        # If we have no strong matches but some weak ones, include the top scoring table
        if not relevant_tables and table_mentions:
            top_table = max(table_mentions.items(), key=lambda x: x[1])[0]
            relevant_tables.add(top_table)
        
        # Add related tables through foreign keys for tables already identified
        related_tables = set()
        for table_name in relevant_tables:
            # Only add related tables when we have few direct matches
            if len(relevant_tables) <= 2:
                table_info = self.schema_data.get('tables', {}).get(table_name, {})
                foreign_keys = table_info.get('foreign_keys', {})
                
                for fk_target in foreign_keys.values():
                    related_table = fk_target.split('.')[0]
                    # Only add the related table if it's not already in our direct matches
                    if related_table not in relevant_tables:
                        related_tables.add(related_table)
        
        # Combine direct matches with related tables
        all_relevant = relevant_tables.union(related_tables)
        
        return list(all_relevant)
    
    def generate_schema_context(self, table_names: List[str] = None, max_tables: int = 10) -> str:
        """Generate schema context for LLM prompt - optimized for speed"""
        if not self.schema_data:
            return "No schema information available."
        
        # Use cached context if available for the same tables
        cache_key = "_".join(sorted(table_names)) if table_names else "all_tables"
        if hasattr(self, '_schema_context_cache') and cache_key in self._schema_context_cache:
            return self._schema_context_cache[cache_key]
        
        # Initialize cache if not exists
        if not hasattr(self, '_schema_context_cache'):
            self._schema_context_cache = {}
            
        context_parts = []
        
        # Add database info
        db_name = self.schema_data.get('database_name', 'Database')
        context_parts.append(f"DATABASE: {db_name}")
        
        # Determine which tables to include - limit for speed
        tables_to_include = table_names if table_names else list(self.schema_data.get('tables', {}).keys())[:max_tables]
        
        # If we have multiple tables, highlight their relationships
        if table_names and len(table_names) > 1:
            relationship_info = []
            relationship_info.append("\nTABLE RELATIONSHIPS (IMPORTANT FOR JOINS):")
            
            # Find and highlight relationships between the selected tables
            for table_name in table_names:
                if table_name not in self.schema_data.get('tables', {}):
                    continue
                    
                table_info = self.schema_data['tables'].get(table_name, {})
                foreign_keys = table_info.get('foreign_keys', {})
                
                if foreign_keys:
                    for fk_col, fk_target in foreign_keys.items():
                        target_table = fk_target.split('.')[0]
                        target_col = fk_target.split('.')[1]
                        
                        if target_table in table_names:
                            relationship_info.append(f"  • {table_name}.{fk_col} -> {target_table}.{target_col}")
            
            if len(relationship_info) > 1:  # More than just the header
                context_parts.extend(relationship_info)
        
        # Limit to max tables for performance
        tables_to_include = tables_to_include[:max_tables]
        
        context_parts.append(f"\nAVAILABLE TABLES ({len(tables_to_include)}):")
        
        for table_name in tables_to_include:
            table_info = self.schema_data['tables'].get(table_name, {})
            
            # Table header - simplified for performance
            context_parts.append(f"\nTABLE: {table_name}")
            
            # Columns - only include essential columns for performance
            columns = table_info.get('columns', {})
            primary_keys = table_info.get('primary_keys', [])
            
            # Find most important columns - primary keys, foreign keys, and first few columns
            important_cols = set()
            
            # Add primary keys
            for pk in primary_keys:
                important_cols.add(pk)
                
            # Add foreign keys
            foreign_keys = table_info.get('foreign_keys', {})
            for fk in foreign_keys:
                important_cols.add(fk)
                
            # Add up to 5 more columns if available
            col_count = 0
            if isinstance(columns, list):
                for col in columns:
                    if col not in important_cols and col_count < 5:
                        important_cols.add(col)
                        col_count += 1
            elif isinstance(columns, dict):
                for col in columns.keys():
                    if col not in important_cols and col_count < 5:
                        important_cols.add(col)
                        col_count += 1
                        
            # Format columns as compact string
            col_list = list(important_cols)
            if len(col_list) > 0:
                cols_str = ", ".join(col_list[:10])  # Limit to 10 columns
                context_parts.append(f"   Columns: {cols_str}")
                if len(col_list) > 10:
                    context_parts.append(f"   ...and {len(col_list) - 10} more columns")
            
            # Only include essential relationships
            if foreign_keys:
                fk_list = []
                for fk_col, fk_target in foreign_keys.items():
                    fk_list.append(f"{fk_col} → {fk_target}")
                if fk_list:
                    context_parts.append(f"   Relations: {'; '.join(fk_list[:3])}")
        
        # Create the final context
        final_context = '\n'.join(context_parts)
        
        # Cache the result for future use
        self._schema_context_cache[cache_key] = final_context
        
        return final_context

class DatabaseManager:
    """Manages AWS SQL Server database connections and query execution"""
    
    def __init__(self, config):
        self.config = config
        self.connection_string = self._build_connection_string()
        self.engine = None
        self._initialize_connection()
    
    def _build_connection_string(self) -> str:
        """Build SQL Server connection string from config"""
        # For SQL Server with pyodbc
        driver = self.config.get('driver', 'ODBC Driver 17 for SQL Server')
        server = self.config.get('server', '10.0.0.45')
        database = self.config.get('database', '146_36156520-AC21-435A-9C9B-1EC9145A9090')
        username = self.config.get('username', 'usr_mohsin')
        password = self.config.get('password', 'blY|5K:3pe10')
        port = self.config.get('port', 1433)
        
        connection_string = (
            f"mssql+pyodbc://{username}:{password}@{server}:{port}/{database}"
            f"?driver={driver.replace(' ', '+')}&TrustServerCertificate=yes"
        )
        
        return connection_string
    
    def _initialize_connection(self):
        """Initialize database connection"""
        try:
            self.engine = create_engine(
                self.connection_string,
                echo=False,  # Set to True for SQL debugging
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={
                    "timeout": 30,
                    "autocommit": True
                }
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(sql_text("SELECT @@VERSION as version, DB_NAME() as db_name"))
                row = result.fetchone()
                logger.info(f"Connected to SQL Server: {row.db_name}")
                
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def execute_query(self, query: str, max_rows: int = 1000) -> Dict[str, Any]:
        """Execute SQL query with comprehensive error handling"""
        start_time = time.time()
        
        try:
            # Validate query
            self._validate_query(query)
            
            with self.engine.connect() as conn:
                # Execute query with timeout
                result = conn.execute(sql_text(query))
                
                if result.returns_rows:
                    # Fetch results
                    columns = list(result.keys())
                    rows = []
                    
                    row_count = 0
                    for row in result:
                        if max_rows and row_count >= max_rows:
                            break
                        
                        # Convert row to list with proper formatting
                        row_values = [self._format_value(val) for val in row]
                        rows.append(row_values)
                        row_count += 1
                    
                    execution_time = time.time() - start_time
                    
                    return {
                        "success": True,
                        "columns": columns,
                        "rows": rows,
                        "row_count": row_count,
                        "execution_time": execution_time,
                        "query": query,
                        "timestamp": datetime.now().isoformat(),
                        "truncated": max_rows and row_count >= max_rows
                    }
                else:
                    # Non-SELECT query (shouldn't happen with our restrictions)
                    execution_time = time.time() - start_time
                    return {
                        "success": True,
                        "message": "Query executed successfully (no results returned)",
                        "execution_time": execution_time,
                        "query": query,
                        "timestamp": datetime.now().isoformat()
                    }
                    
        except SQLAlchemyError as e:
            execution_time = time.time() - start_time
            error_msg = str(e.orig) if hasattr(e, 'orig') else str(e)
            
            logger.error(f"SQL execution error: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "error_type": type(e).__name__,
                "query": query,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Unexpected error executing query: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "query": query,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_query(self, query: str):
        """Validate SQL query for security"""
        query_upper = query.upper().strip()
        
        # Only allow SELECT statements
        if not query_upper.startswith('SELECT') and not query_upper.startswith('WITH'):
            raise ValueError("Only SELECT queries are allowed")
        
        # Block dangerous keywords
        blocked_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 
            'TRUNCATE', 'EXEC', 'EXECUTE', 'SP_', 'XP_', 'OPENROWSET',
            'OPENDATASOURCE', 'BULK', 'BACKUP', 'RESTORE'
        ]
        
        for keyword in blocked_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', query_upper):
                raise ValueError(f"Blocked keyword detected: {keyword}")
    
    def _format_value(self, value):
        """Format database values for display"""
        if value is None:
            return None
        
        if isinstance(value, datetime):
            return value.isoformat()
        
        if isinstance(value, (bytes, bytearray)):
            return f"<BINARY_DATA:{len(value)}_bytes>"
        
        if isinstance(value, (int, float)) and abs(value) > 1e15:
            return f"{value:e}"
        
        return value
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sql_text("SELECT 1 as test"))
                result.fetchone()
                return True, "Connection successful"
        except Exception as e:
            return False, str(e)

class LocalSQLTranslator:
    """Local SQL translator using Ollama + Mistral"""
    
    def __init__(self, schema_manager: SchemaManager, config: Dict[str, Any]):
        self.schema_manager = schema_manager
        self.config = config
        
        # Initialize local Ollama + Mistral
        ollama_base_url = config.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        ollama_model = config.get('OLLAMA_MODEL', 'mistral')
        
        self.llm = OllamaLLM(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.1,        # Slightly higher temperature for better creative solutions
            num_ctx=2048,           # Larger context window for better schema understanding
            num_thread=4,           # Balanced thread count for most machines
            num_gpu=1,              # Use GPU if available
            repeat_penalty=1.1,     # Lower penalty for faster generation
            top_k=40,               # Less restrictive for better solutions
            top_p=0.9,              # Good balance between deterministic and creative outputs
            stop=["```", "SQL Query:", ";"], # Stop tokens to improve extraction
            timeout=15              # Reasonable timeout
        )
        
        # Create optimized prompts
        self.sql_prompt = self._create_sql_prompt()
        self.validation_prompt = self._create_validation_prompt()
        
        logger.info(f"Local SQL Translator initialized with {ollama_model}")
    
    def _create_sql_prompt(self) -> ChatPromptTemplate:
        """Create optimized SQL generation prompt for ANY tables in the schema"""
        template = """You are an expert SQL Server database analyst specializing in precise query generation. Your job is to convert natural language questions into SQL Server queries that exactly match the provided schema.

DATABASE SCHEMA:
{schema_context}

UNIVERSAL SQL GENERATION RULES (FOR ANY TABLES):
1. Generate ONLY SQL Server compatible SELECT queries, never INSERT/UPDATE/DELETE
2. ALWAYS use EXACT table and column names from the schema (case-sensitive)
3. Use TOP instead of LIMIT for row limiting (e.g., SELECT TOP 15)
4. ALWAYS use [] brackets around ALL table and column names (e.g., [USER], [RECORD_ID])
5. Use proper SQL Server functions (GETDATE(), DATEDIFF(), etc.)
6. Handle NULL values with ISNULL() or COALESCE()
7. Use single-letter table aliases consistently for ALL tables (e.g., u for USER, a for ACCOUNT)
8. Always include primary keys and essential identifying columns from each table
9. ALWAYS limit results to TOP 15 rows unless otherwise specified
10. NEVER use reserved keywords as column names without brackets
11. NEVER join a table with itself unless specifically requested
12. NEVER use hardcoded table assumptions - rely ONLY on schema information

ROBUST MULTI-TABLE QUERY RULES (CRITICAL FOR ANY TABLES):
1. NEVER generate separate SELECT statements - use ONE single query with proper JOINs
2. ALWAYS use appropriate JOIN types (INNER JOIN, LEFT JOIN) based on the question
3. NEVER use CROSS JOIN between any tables - they create cartesian products
4. Use foreign key relationships from schema for ALL table JOINs, not just specific tables
5. ALWAYS include explicit join conditions: ON t1.foreign_key = t2.primary_key
6. For "tell me about" queries, show key columns from ALL mentioned tables
7. ALWAYS fully qualify ALL column references with table aliases
8. If no direct foreign key exists, look for columns with similar names or naming patterns:
   - [TableName]_ID columns often relate to [TableName].RECORD_ID
   - Columns with identical names across tables often indicate relationships
   - Look for singular/plural variants (e.g., USER_ID in ACCOUNTS relates to USER.RECORD_ID)

UNIVERSAL TABLE RELATIONSHIP DETECTION:
- ALL tables have relationship information in the schema - use it for ALL tables equally
- Use 'foreign_keys' section of each table to find relationships between ANY tables
- JOIN format for ANY tables: FROM [TableA] a JOIN [TableB] b ON a.foreign_key_column = b.primary_key_column
- For ANY two tables without explicit foreign keys, look for common naming patterns:
  - Column named [TableB]_ID in TableA likely joins to TableB.RECORD_ID or TableB.ID
  - Common ID columns with same name might be joinable
- NEVER rely on special-case logic for specific tables - use universal rules

PATTERN-SPECIFIC RULES:
- For "tell me about TABLE" queries: SELECT TOP 15 key_columns FROM [TABLE]
- For multiple tables overview: JOIN tables on their relationships, select important columns from each
- For counts: SELECT COUNT(*) FROM [table_name] WHERE condition
- For aggregations: SELECT column, SUM/AVG/MAX/MIN(column) FROM [table_name] GROUP BY column
- For filtering: Include proper WHERE clauses with appropriate operators
- For text search: Use WHERE column LIKE '%search_term%'
- For non-exact matches: Use UPPER(), LOWER(), or CHARINDEX() functions

PERFORMANCE & SAFETY RULES:
1. Avoid SELECT * with multiple tables (causes column ambiguity)
2. Always specify TOP clause to limit result size (15-20 rows is good)
3. Use column aliases for clarity in output
4. Always qualify columns with table aliases when multiple tables are involved
5. Use COUNT(*) with GROUP BY only when aggregation is needed

USER QUESTION: {question}

Generate the SQL query that answers this question. Return ONLY the SQL query without any explanations, markdown, or additional text.

SQL Query:"""

        return ChatPromptTemplate.from_template(template)
    
    def _create_validation_prompt(self) -> ChatPromptTemplate:
        """Create SQL validation prompt for ANY tables in the schema"""
        template = """Review this SQL Server query for correctness:

QUERY TO VALIDATE:
{sql_query}

SCHEMA CONTEXT:
{schema_context}

ORIGINAL QUESTION:
{original_question}

Check for ALL tables in the schema (not just specific ones):
1. Correct table and column names (must match schema exactly)
2. SQL Server syntax compliance (TOP not LIMIT, proper functions)
3. Logical correctness for the question asked
4. ALWAYS verify JOINs are used when multiple tables are mentioned
5. Check that tables are properly joined using their relationship keys
6. Make sure no CROSS JOINs are used when proper relationships exist
7. Verify table and column references use proper [] brackets
8. Ensure column references are qualified with table names/aliases
9. Check for correct JOIN order based on the logical flow of data
10. Make sure there are NO separate SELECT statements - everything should be in ONE query with JOINs

If the query is correct, respond with: VALID

If the query needs corrections, provide the corrected SQL query only, without explanations.

Response:"""

        return ChatPromptTemplate.from_template(template)
    
    def translate(self, question: str, use_rag: bool = True) -> Dict[str, Any]:
        """Translate natural language to SQL with optional RAG enhancement
        
        Args:
            question: Natural language question to translate
            use_rag: Whether to use RAG enhancement (default: True)
            
        Returns:
            Dict containing SQL query and metadata
        """
        start_time = time.time()
        rag_metadata = {}  # Store RAG-related metadata
        
        try:
            # Try RAG-enhanced retrieval first if available and enabled
            if use_rag and RAG_AVAILABLE:
                try:
                    rag_engine = get_rag_engine()
                    rag_result = rag_engine.enhance_query(question)
                    
                    # Use RAG-enhanced context and tables
                    schema_context = rag_result.get('enhanced_context', '')
                    relevant_tables = rag_result.get('relevant_tables', [])
                    
                    # Store RAG metadata for response
                    rag_metadata = {
                        'rag_enabled': True,
                        'confidence_scores': rag_result.get('confidence_scores', {}),
                        'similar_examples': rag_result.get('similar_examples', []),
                        'relevant_columns': rag_result.get('relevant_columns', [])
                    }
                    
                    logger.info(f"RAG enhancement: Found {len(relevant_tables)} relevant tables with confidence: {rag_result.get('confidence_scores', {})}")
                    
                except Exception as rag_error:
                    logger.warning(f"RAG enhancement failed, falling back to standard context: {rag_error}")
                    # Fallback to standard schema context
                    relevant_tables = self.schema_manager.find_relevant_tables(question)
                    schema_context = self.schema_manager.generate_schema_context(relevant_tables) if relevant_tables else self.schema_manager.generate_schema_context(max_tables=3)
                    rag_metadata = {'rag_enabled': False, 'fallback_reason': str(rag_error)}
            else:
                # Standard schema context without RAG
                relevant_tables = self.schema_manager.find_relevant_tables(question)
                
                if relevant_tables:
                    schema_context = self.schema_manager.generate_schema_context(relevant_tables)
                else:
                    # If no specific tables found, use all tables (limited)
                    schema_context = self.schema_manager.generate_schema_context(max_tables=3)
                
                rag_metadata = {'rag_enabled': False, 'reason': 'RAG disabled or not available'}
            
            # Generate SQL
            sql_result = self._generate_sql(question, schema_context)
            sql_query = sql_result['sql']
            
            # Check for multiple disconnected SELECT statements immediately
            if len(relevant_tables) > 1 and sql_query.upper().count("SELECT") > 1 and "JOIN" not in sql_query.upper():
                logger.warning("Multiple tables detected but no JOINs found in query. Attempting to fix...")
                fixed_query = self._fix_multiple_selects(sql_query, schema_context)
                if fixed_query != sql_query:
                    logger.info("Fixed multiple disconnected SELECT statements")
                    sql_query = fixed_query
            
            # Validate SQL if enabled
            if self.config.get('enable_query_validation', True):
                validated_sql = self._validate_sql(sql_query, schema_context, question)
                if validated_sql != sql_query:
                    logger.info("SQL query was corrected during validation")
                    sql_query = validated_sql
            
            # Post-process SQL
            sql_query = self._post_process_sql(sql_query)
            
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "sql": sql_query,
                "tables_used": relevant_tables,
                "execution_time": execution_time,
                "schema_context_length": len(schema_context),
                "raw_response": sql_result.get('raw_response', ''),
                "rag_metadata": rag_metadata  # Include RAG enhancement metadata
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"SQL translation failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": execution_time,
                "question": question
            }
    
    def _generate_sql(self, question: str, schema_context: str) -> Dict[str, Any]:
        """Generate SQL using local LLM with smart fallbacks for ALL tables in schema"""
        try:
            start_time = time.time()
            
            # Get all tables from schema manager
            all_tables = self.schema_manager.get_all_tables()
            
            # Extract tables mentioned explicitly in the question using word boundary detection
            # This ensures we only match complete words, not partial strings
            mentioned_tables_set = set()
            
            # First, check for explicit "tell me about TABLE" patterns
            explicit_patterns = [
                r"tell me about (?:the\s+)?(\w+)(?:\s+and\s+(\w+))?(?:\s+and\s+(\w+))?(?:\s+and\s+(\w+))?\s*tables?",
                r"show (?:the\s+)?(\w+)(?:\s+and\s+(\w+))?(?:\s+and\s+(\w+))?(?:\s+and\s+(\w+))?\s*tables?",
                r"describe (?:the\s+)?(\w+)(?:\s+and\s+(\w+))?(?:\s+and\s+(\w+))?(?:\s+and\s+(\w+))?\s*tables?"
            ]
            
            # Try each pattern
            explicit_matches = []
            for pattern in explicit_patterns:
                matches = re.search(pattern, question.lower())
                if matches:
                    explicit_matches = [m for m in matches.groups() if m]
                    break
            
            # Process explicit matches first if they exist
            if explicit_matches:
                logger.info(f"Explicit table references found: {explicit_matches}")
                for table_mention in explicit_matches:
                    for table in all_tables:
                        if table.lower() == table_mention.lower():
                            mentioned_tables_set.add(table)
                            break
            
            # If no explicit matches, perform comprehensive search using word boundaries
            if not mentioned_tables_set:
                for table in all_tables:
                    # Use word boundary check for direct matches
                    pattern = r'\b' + re.escape(table.lower()) + r'\b'
                    if re.search(pattern, question.lower()):
                        mentioned_tables_set.add(table)
                        continue
                    
                    # Use word boundaries for singulars/plurals of the table name
                    table_singular = table.lower().rstrip('s')
                    if len(table_singular) > 3:  # Only check meaningful words
                        pattern = r'\b' + re.escape(table_singular) + r'\b'
                        if re.search(pattern, question.lower()):
                            mentioned_tables_set.add(table)
                            continue
                    
                    # Add plural version check
                    table_plural = table.lower() + 's' if not table.lower().endswith('s') else table.lower()
                    pattern = r'\b' + re.escape(table_plural) + r'\b'
                    if re.search(pattern, question.lower()):
                        mentioned_tables_set.add(table)
                        continue
            
            # Convert to list with unique values
            mentioned_tables = list(mentioned_tables_set)
            
            # Find tables with distinctive columns mentioned in the question
            # Only use this if we don't have explicit table mentions
            if not mentioned_tables:
                words_in_question = set(question.lower().split())
                column_related_tables = {}  # Table -> score mapping
                
                # Preprocess question to split on special characters for better matching
                processed_question = re.sub(r'[^\w\s]', ' ', question.lower())
                processed_words = set(processed_question.split())
                
                # Create set of unique column names across all tables for distinctiveness check
                all_column_names = set()
                for table in all_tables:
                    table_info = self.schema_manager.get_table_info(table)
                    if table_info and 'columns' in table_info:
                        for col in table_info['columns']:
                            all_column_names.add(col.lower())
                
                # Find distinctive columns for each table
                for table in all_tables:
                    table_info = self.schema_manager.get_table_info(table)
                    if not table_info or 'columns' not in table_info:
                        continue
                    
                    table_score = 0
                    for column in table_info['columns']:
                        col_lower = column.lower()
                        
                        # Check exact column name with word boundaries
                        col_pattern = r'\b' + re.escape(col_lower) + r'\b'
                        if re.search(col_pattern, processed_question):
                            # If this column is unique to this table, give it higher weight
                            distinctiveness = sum(1 for t in all_tables if t != table and 
                                              col_lower in [c.lower() for c in 
                                                self.schema_manager.get_table_info(t).get('columns', [])])
                            
                            if distinctiveness == 0:  # Completely unique column
                                table_score += 3
                            else:
                                table_score += 1
                        
                        # Check for column parts in question
                        col_words = col_lower.replace('_', ' ').split()
                        matches = sum(1 for word in col_words if len(word) > 3 and word in processed_words)
                        if matches > 0:
                            table_score += matches * 0.5
                    
                    if table_score > 0:
                        column_related_tables[table] = table_score
                
                # Take the top 2 tables by score if we found any
                if column_related_tables:
                    top_tables = sorted(column_related_tables.items(), key=lambda x: x[1], reverse=True)[:2]
                    column_related_tables = set(table for table, score in top_tables)
            
            # Combine direct mentions with column-related tables
            all_relevant_tables = list(set(mentioned_tables) | column_related_tables)
            
            # If no tables detected, try to find them from column mentions
            if not all_relevant_tables:
                column_detected_tables = self._detect_tables_from_columns(question)
                if column_detected_tables:
                    logger.info(f"Detected tables from columns: {', '.join(column_detected_tables)}")
                    all_relevant_tables = column_detected_tables
            
            # Last resort, use the schema manager's generic table finder
            if not all_relevant_tables:
                all_relevant_tables = self.schema_manager.find_relevant_tables(question)
            
            # Get information about table relationships for smart JOINs
            # This is critical for producing good queries for ANY tables
            table_relationships = self._get_table_relationships(all_relevant_tables)
            
            # Enhance question with specific guidance based on identified tables
            enhanced_question = question
            
            # Flag for multiple tables
            multiple_tables = len(all_relevant_tables) > 1
            
            # Create a comprehensive prompt that will work for ANY tables in the schema
            if multiple_tables:
                # First identify direct relationships between the relevant tables
                direct_relationships = []
                indirect_relationships = []
                
                for i, source in enumerate(all_relevant_tables):
                    for target in all_relevant_tables[i+1:]:
                        relationship_found = False
                        
                        # Check direct relationship source → target
                        if source in table_relationships and target in table_relationships[source]:
                            source_col, target_col = table_relationships[source][target]
                            direct_relationships.append(f"[{source}].[{source_col}] = [{target}].[{target_col}]")
                            relationship_found = True
                        
                        # Check direct relationship target → source
                        elif target in table_relationships and source in table_relationships[target]:
                            target_col, source_col = table_relationships[target][source]
                            direct_relationships.append(f"[{source}].[{source_col}] = [{target}].[{target_col}]")
                            relationship_found = True
                        
                        # If no direct relationship, try to find intermediate tables
                        if not relationship_found:
                            for intermediate in all_tables:
                                if intermediate in all_relevant_tables:
                                    continue
                                    
                                if ((source in table_relationships and intermediate in table_relationships[source]) and
                                    (target in table_relationships and intermediate in table_relationships[target])):
                                    # Found a potential bridge table
                                    source_col, int_col1 = table_relationships[source][intermediate]
                                    target_col, int_col2 = table_relationships[target][intermediate]
                                    indirect_relationships.append(
                                        f"[{source}].[{source_col}] = [{intermediate}].[{int_col1}] AND " +
                                        f"[{intermediate}].[{int_col2}] = [{target}].[{target_col}]"
                                    )
                
                # Build enhanced instructions with relationship information
                relationship_hints = ""
                if direct_relationships:
                    relationship_hints += f" Use these direct join conditions: {'; '.join(direct_relationships)}."
                elif indirect_relationships:
                    relationship_hints += f" Consider these indirect relationships through bridge tables: {'; '.join(indirect_relationships)}."
                
                # Add primary keys information for each table
                pk_info = []
                for table in all_relevant_tables:
                    table_info = self.schema_manager.get_table_info(table)
                    if table_info and 'primary_keys' in table_info and table_info['primary_keys']:
                        pk = table_info['primary_keys'][0]
                        pk_info.append(f"{table}.{pk}")
                
                pk_hint = ""
                if pk_info:
                    pk_hint = f" Primary keys: {', '.join(pk_info)}."
                
                enhanced_question = (
                    f"{question} - IMPORTANT: Create a single efficient SQL query for tables "
                    f"{', '.join(all_relevant_tables)} using proper SQL Server syntax. "
                    f"Always use table aliases (e.g., 't1', 't2') and proper JOIN conditions. "
                    f"Never use CROSS JOIN or separate SELECT statements.{relationship_hints}{pk_hint} "
                    f"Limit results to TOP 15 rows and select key identifying columns from each table."
                )
            
            # If asking about table structure, provide extra guidance
            if "tell me about" in question.lower() or "describe" in question.lower() or "show me" in question.lower():
                enhanced_question += " For each table, show only the most important and identifying columns."
            
            # Invoke LLM with enhanced context - with simpler error handling
            try:
                # Use a more direct approach for speed
                logger.info(f"Starting SQL generation for question: '{question[:50]}...'")
                chain = self.sql_prompt | self.llm | StrOutputParser()
                
                # Try with a timeout (rely on LLM's internal timeout)
                response = chain.invoke({
                    "question": enhanced_question,
                    "schema_context": schema_context
                })
                
                # Extract SQL from response
                sql_query = self._extract_sql(response)
                original_query = sql_query
                
                # Only fix common problems for multi-table queries to save time
                if len(all_relevant_tables) > 1:
                    # Analyze and fix common SQL problems
                    sql_query = self._fix_common_sql_problems(sql_query, all_relevant_tables, question, table_relationships)
                    
                    # Log if we made changes
                    if sql_query != original_query:
                        logger.info(f"SQL query was fixed. Original had issues.")
                
                # Rate the quality with simple heuristic - faster
                if "SELECT" in sql_query and "FROM" in sql_query:
                    confidence_score = 0.9
                else:
                    confidence_score = 0.5
            except Exception as e:
                logger.error(f"Error during SQL generation: {e}")
                # Fallback to a simple SELECT query for the first table as emergency response
                if all_relevant_tables:
                    first_table = all_relevant_tables[0]
                    # Get some key columns for this table
                    table_info = self.schema_manager.get_table_info(first_table)
                    columns = "*"  # Default to all columns
                    
                    if table_info and 'columns' in table_info:
                        # Try to get important columns only
                        important_cols = []
                        # Add primary key
                        if 'primary_keys' in table_info and table_info['primary_keys']:
                            important_cols.extend(table_info['primary_keys'])
                        
                        # Add some other columns
                        all_cols = table_info['columns']
                        for col in all_cols:
                            if len(important_cols) < 5 and col not in important_cols:
                                important_cols.append(col)
                        
                        if important_cols:
                            columns = ", ".join([f"[{col}]" for col in important_cols])
                            
                    # Create a simple query with the table
                    sql_query = f"SELECT TOP 10 {columns} FROM [{first_table}]"
                    confidence_score = 0.3
                    logger.info(f"Using fallback query for table {first_table}")
                else:
                    sql_query = "SELECT 'No tables detected' AS Message"
                    confidence_score = 0.1
            
            generation_time = time.time() - start_time
            logger.info(f"SQL generation completed in {generation_time:.2f}s with confidence {confidence_score:.2f}")
            
            return {
                "sql": sql_query,
                "raw_response": response,
                "confidence": confidence_score,
                "generation_time": generation_time,
                "tables_detected": all_relevant_tables
            }
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise
    
    def _find_common_columns(self, table1: str, table2: str) -> Optional[Tuple[str, str]]:
        """
        Find potential columns that might relate two tables even when not defined in foreign keys
        Returns (table1_column, table2_column) if a likely match is found, None otherwise
        """
        table1_info = self.schema_manager.get_table_info(table1)
        table2_info = self.schema_manager.get_table_info(table2)
        
        if not table1_info or not table2_info:
            return None
            
        # Get columns for both tables
        table1_columns = table1_info.get('columns', []) if isinstance(table1_info, dict) else []
        table2_columns = table2_info.get('columns', []) if isinstance(table2_info, dict) else []
        
        # Look for ID columns that might match between tables
        for col1 in table1_columns:
            col1_lower = col1.lower()
            # Check for ID-like columns
            if col1_lower == 'record_id' or col1_lower.endswith('_id'):
                # If table2's name (singular) + '_id' matches col1, that's a strong hint
                table2_singular = table2[:-1] if table2.lower().endswith('s') else table2
                if f"{table2_singular.lower()}_id" == col1_lower:
                    # Look for primary key in table2
                    for col2 in table2_columns:
                        if col2.lower() == 'record_id' or col2.lower() == 'id':
                            return (col1, col2)
                            
        # Check for exact column name matches (e.g. both tables have customer_id)
        for col1 in table1_columns:
            if col1.lower().endswith('_id') and col1 in table2_columns:
                return (col1, col1)
        
        # Look for table name in column names
        table1_name_singular = table1[:-1] if table1.lower().endswith('s') else table1
        table2_name_singular = table2[:-1] if table2.lower().endswith('s') else table2
        
        for col2 in table2_columns:
            col2_lower = col2.lower()
            # Check if column name contains table1's name
            if table1_name_singular.lower() + "_id" == col2_lower:
                # Find matching primary key
                for col1 in table1_columns:
                    if col1.lower() == 'record_id' or col1.lower() == 'id':
                        return (col1, col2)
                        
        # No likely relationship found
        return None
        
    def _get_table_relationships(self, tables: List[str]) -> Dict[str, Dict[str, Tuple[str, str]]]:
        """
        Build a comprehensive map of table relationships for ANY tables in the schema
        Returns: {source_table: {target_table: (source_column, target_column)}}
        """
        relationships = {}
        
        try:
            # Get all tables to ensure we can find relationships even with tables not in the query
            all_tables = self.schema_manager.get_all_tables()
            
            # Process ALL tables to build a complete relationship map
            # This helps with finding indirect relationships too
            for table in all_tables:
                table_info = self.schema_manager.get_table_info(table)
                if not table_info:
                    continue
                
                # Initialize entry for this table
                if table not in relationships:
                    relationships[table] = {}
                
                # Check if we have foreign keys
                foreign_keys = {}
                
                # Handle different schema formats
                if 'foreign_keys' in table_info:
                    foreign_keys = table_info['foreign_keys']
                elif isinstance(table_info, dict):
                    # Try to detect foreign key fields by naming conventions if not explicitly defined
                    for col in table_info.get('columns', []):
                        col_lower = col.lower()
                        # Look for columns that might be foreign keys
                        if (col_lower.endswith('_id') or col_lower.endswith('record_id')) and col_lower != 'record_id':
                            # Guess the target table from column name
                            target_table = col_lower.replace('_id', '').replace('record_', '').upper()
                            if target_table in all_tables:
                                foreign_keys[col] = f"{target_table}.RECORD_ID"
                
                # Process foreign keys
                for source_col, fk_reference in foreign_keys.items():
                    try:
                        parts = fk_reference.split('.')
                        if len(parts) != 2:
                            continue
                            
                        target_table, target_col = parts
                        
                        # Add relationship in both directions for comprehensive mapping
                        relationships[table][target_table] = (source_col, target_col)
                        
                        # Also add reverse relationship
                        if target_table not in relationships:
                            relationships[target_table] = {}
                        relationships[target_table][table] = (target_col, source_col)
                        
                    except Exception as e:
                        logger.warning(f"Error processing foreign key {fk_reference} in table {table}: {e}")
            
            # As a backup, look for similar column names for tables that have no defined relationships
            for i, table1 in enumerate(tables):
                if table1 not in relationships:
                    relationships[table1] = {}
                    
                for table2 in tables[i+1:]:
                    if table1 == table2:
                        continue
                        
                    # If no relationship exists between these tables, try to find one
                    if table2 not in relationships.get(table1, {}) and table1 not in relationships.get(table2, {}):
                        common_column = self._find_common_columns(table1, table2)
                        if common_column:
                            # Add inferred relationship
                            if table1 not in relationships:
                                relationships[table1] = {}
                            relationships[table1][table2] = common_column
                            
                            if table2 not in relationships:
                                relationships[table2] = {}
                            relationships[table2][table1] = (common_column[1], common_column[0])
        
        except Exception as e:
            logger.error(f"Error building table relationships: {e}")
            
        # Log relationship stats for monitoring
        related_table_count = sum(1 for t in tables if t in relationships and relationships[t])
        logger.info(f"Found relationships for {related_table_count} of {len(tables)} tables in query")
                    
        return relationships
        
    def _fix_common_sql_problems(self, sql: str, tables: List[str], question: str, 
                                table_relationships: Dict[str, Dict[str, Tuple[str, str]]]) -> str:
        """Fix common problems in generated SQL queries for ANY tables in the schema"""
        sql_upper = sql.upper()
        original_sql = sql
        
        # Track if we made any fixes
        fixes_applied = []
        
        # 0. First check if this is querying the same table multiple times unnecessarily
        unique_tables = set(tables)
        if len(unique_tables) < len(tables):
            logger.warning(f"Duplicate tables detected in query: {tables}. Using unique set: {unique_tables}")
            tables = list(unique_tables)
        
        # 1. Fix CROSS JOINs when tables have relationships - critical for ANY two tables
        if "CROSS JOIN" in sql_upper and len(tables) > 1:
            # Try to fix all cross joins
            cross_join_pairs = re.findall(r'\[(\w+)\]\s+(\w+)\s+CROSS\s+JOIN\s+\[(\w+)\]\s+(\w+)', sql, flags=re.IGNORECASE)
            
            for match in cross_join_pairs:
                source_table, source_alias, target_table, target_alias = match
                
                # Skip if same table (can't join with itself meaningfully)
                if source_table == target_table:
                    continue
                
                # Look for direct relationships between these tables
                join_condition = None
                
                # Check source → target direction
                if (source_table in table_relationships and 
                    target_table in table_relationships[source_table]):
                    source_col, target_col = table_relationships[source_table][target_table]
                    join_condition = f"{source_alias}.[{source_col}] = {target_alias}.[{target_col}]"
                
                # Check target → source direction
                elif (target_table in table_relationships and 
                      source_table in table_relationships[target_table]):
                    target_col, source_col = table_relationships[target_table][source_table]
                    join_condition = f"{source_alias}.[{source_col}] = {target_alias}.[{target_col}]"
                
                # If no direct relationship but both tables have primary keys, use those as a last resort
                if not join_condition:
                    source_info = self.schema_manager.get_table_info(source_table)
                    target_info = self.schema_manager.get_table_info(target_table)
                    
                    if (source_info and target_info and 
                        'primary_keys' in source_info and source_info['primary_keys'] and
                        'primary_keys' in target_info and target_info['primary_keys']):
                        # Try to use the primary keys if they have similar names
                        source_pk = source_info['primary_keys'][0]
                        target_pk = target_info['primary_keys'][0]
                        
                        if source_pk.lower() == target_pk.lower() or (
                            'id' in source_pk.lower() and 'id' in target_pk.lower()):
                            join_condition = f"{source_alias}.[{source_pk}] = {target_alias}.[{target_pk}]"
                
                # Apply the fix if we found a join condition
                if join_condition:
                    old_pattern = f"[{source_table}] {source_alias} CROSS JOIN [{target_table}] {target_alias}"
                    new_join = f"[{source_table}] {source_alias} JOIN [{target_table}] {target_alias} ON {join_condition}"
                    sql = sql.replace(old_pattern, new_join)
                    fixes_applied.append(f"Replaced CROSS JOIN with JOIN ON {join_condition}")
                else:
                    # If we can't find a relationship, at least limit the cross join
                    old_pattern = f"[{source_table}] {source_alias} CROSS JOIN [{target_table}] {target_alias}"
                    new_join = f"(SELECT TOP 5 * FROM [{source_table}]) {source_alias} CROSS JOIN (SELECT TOP 5 * FROM [{target_table}]) {target_alias}"
                    sql = sql.replace(old_pattern, new_join)
                    fixes_applied.append(f"Limited CROSS JOIN result size for {source_table} × {target_table}")
        
        # 2. Fix multiple SELECT statements (should be a single query)
        if sql_upper.count("SELECT") > 1 and "UNION" not in sql_upper and "WITH" not in sql_upper:
            # This is a complex fix that depends on the specific SQL, but we can try a simple approach
            logger.warning("Multiple SELECT statements found - attempting to fix by creating a single query")
            
            # For descriptive queries, create a single query showing key columns from all tables
            if "tell me about" in question.lower() or "describe" in question.lower():
                # Generate a better query showing key columns from each table
                fixed_query = self._generate_descriptive_query(tables)
                return fixed_query
        
        # 3. Fix LIMIT vs TOP syntax (SQL Server uses TOP, not LIMIT)
        if "LIMIT" in sql_upper:
            limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)
            if limit_match:
                limit_val = limit_match.group(1)
                # Replace SELECT with SELECT TOP N
                sql = re.sub(r'SELECT', f'SELECT TOP {limit_val}', sql, count=1, flags=re.IGNORECASE)
                # Remove the LIMIT clause
                sql = re.sub(r'LIMIT\s+\d+', '', sql, flags=re.IGNORECASE)
        
        # 4. Fix missing table aliases in multi-table queries
        if len(tables) > 1:
            for table in tables:
                # Check if table is used without alias
                table_pattern = rf'\[?{table}\]?\s+(?![a-z])'
                if re.search(table_pattern, sql, re.IGNORECASE):
                    alias = table[0].lower()  # Use first character as alias
                    sql = re.sub(table_pattern, f'[{table}] {alias} ', sql, flags=re.IGNORECASE)
        
        # 5. Fix JOIN conditions if tables are joined incorrectly
        if len(tables) > 1 and "JOIN" in sql_upper:
            for source_table in tables:
                for target_table in tables:
                    if source_table == target_table:
                        continue
                        
                    # Only fix if we know the correct relationship
                    if (source_table in table_relationships and 
                        target_table in table_relationships[source_table]):
                        
                        source_col, target_col = table_relationships[source_table][target_table]
                        source_alias = source_table[0].lower()
                        target_alias = target_table[0].lower()
                        
                        # Look for JOIN between these tables with incorrect ON condition
                        join_pattern = rf"\[?{source_table}\]?\s+{source_alias}\s+JOIN\s+\[?{target_table}\]?\s+{target_alias}\s+ON\s+(.+?)(?:\s+WHERE|\s+ORDER|\s+GROUP|\s+HAVING|$)"
                        join_match = re.search(join_pattern, sql, re.IGNORECASE | re.DOTALL)
                        
                        if join_match:
                            on_condition = join_match.group(1).strip()
                            correct_condition = f"{source_alias}.{source_col} = {target_alias}.{target_col}"
                            
                            # If ON condition doesn't match the correct foreign key relationship
                            if correct_condition.lower() not in on_condition.lower():
                                sql = sql.replace(on_condition, correct_condition)
        
        # 6. Add TOP clause if missing and no WHERE clause exists (for safer execution)
        if "TOP" not in sql_upper and "WHERE" not in sql_upper and "COUNT" not in sql_upper:
            sql = re.sub(r'SELECT', 'SELECT TOP 100', sql, count=1, flags=re.IGNORECASE)
        
        return sql
    
    def _generate_descriptive_query(self, tables: List[str]) -> str:
        """Generate a descriptive query showing key columns for multiple tables"""
        if not tables:
            return "SELECT 'No tables identified' AS Message"
            
        if len(tables) == 1:
            table = tables[0]
            key_columns = self.schema_manager.get_key_columns(table, 8)
            if key_columns:
                columns_str = ", ".join(f"[{col}]" for col in key_columns)
                return f"SELECT TOP 15 {columns_str} FROM [{table}]"
            else:
                return f"SELECT TOP 15 * FROM [{table}]"
            
        # For multiple tables, create a query that shows key columns from each table with proper JOINs
        table_info = {}
        for table in tables:
            key_columns = self.schema_manager.get_key_columns(table, 5)
            info = self.schema_manager.get_table_info(table)
            
            if key_columns and info:
                table_info[table] = {
                    'columns': key_columns,
                    'foreign_keys': info.get('foreign_keys', {})
                }
        
        # Determine table relationships to build JOINs
        relationships = self._get_table_relationships(tables)
        
        # Choose a main table (table mentioned first or with most relationships)
        main_table = tables[0]
        for table in tables:
            if table in relationships and len(relationships[table]) > len(relationships.get(main_table, {})):
                main_table = table
        
        # Build the SELECT clause
        select_parts = []
        for table in tables:
            if table in table_info:
                alias = table[0].lower()
                for col in table_info[table]['columns']:
                    select_parts.append(f"{alias}.{col} AS {table}_{col}")
        
        # Build the FROM clause with JOINs
        from_clause = f"[{main_table}] {main_table[0].lower()}"
        processed_tables = {main_table}
        
        # Keep adding JOINs until all tables are included
        while len(processed_tables) < len(tables):
            for source in processed_tables.copy():
                if source not in relationships:
                    continue
                    
                for target, join_cols in relationships[source].items():
                    if target not in processed_tables:
                        source_col, target_col = join_cols
                        source_alias = source[0].lower()
                        target_alias = target[0].lower()
                        from_clause += f"\nJOIN [{target}] {target_alias} ON {source_alias}.{source_col} = {target_alias}.{target_col}"
                        processed_tables.add(target)
            
            # If no more tables can be joined through relationships,
            # add the remaining tables with CROSS JOIN (as a fallback)
            remaining = set(tables) - processed_tables
            if remaining and len(processed_tables) == len(tables) - len(remaining):
                for table in remaining:
                    from_clause += f"\nCROSS JOIN [{table}] {table[0].lower()}"
                    processed_tables.add(table)
        
        # Build the final query
        query = f"SELECT TOP 15 {', '.join(select_parts)}\nFROM {from_clause}"
        return query

    def _detect_tables_from_columns(self, query: str) -> List[str]:
        """
        Detect tables by finding columns mentioned in the query
        This is a fallback method when direct table name detection fails
        Works for ANY tables in the schema, not just specific ones
        """
        detected_tables = {}  # Table -> score dict
        all_tables = self.schema_manager.get_all_tables()
        
        # Process query for better matching
        query_lower = query.lower()
        # Split on punctuation and special characters
        processed_query = re.sub(r'[^\w\s]', ' ', query_lower)
        query_words = set(processed_query.split())
        
        # Create a map of column names to tables
        column_to_tables = {}
        # Also track column frequency across tables for uniqueness weighting
        column_frequency = {}
        
        # First pass - build column maps
        for table in all_tables:
            table_info = self.schema_manager.get_table_info(table)
            if table_info and 'columns' in table_info:
                for column in table_info['columns']:
                    col_lower = column.lower()
                    if col_lower not in column_to_tables:
                        column_to_tables[col_lower] = []
                        column_frequency[col_lower] = 0
                    column_to_tables[col_lower].append(table)
                    column_frequency[col_lower] += 1
        
        # Second pass - look for column matches in the query
        for col_name, tables in column_to_tables.items():
            # Check for exact column name with word boundaries
            pattern = r'\b' + re.escape(col_name) + r'\b'
            exact_match = re.search(pattern, query_lower)
            
            # Check for column parts in the query for longer column names
            col_parts = col_name.replace('_', ' ').split()
            part_matches = sum(1 for part in col_parts if len(part) > 3 and part in query_words)
            
            # Calculate match score based on match type and column uniqueness
            match_score = 0
            if exact_match:
                # Higher score for exact matches
                match_score = 3.0 / column_frequency[col_name]  # More unique columns get higher scores
            elif part_matches > 0 and len(col_parts) > 1:
                # Partial matches get lower scores
                match_score = (part_matches / len(col_parts)) / column_frequency[col_name]
            
            # Only consider meaningful matches
            if match_score > 0:
                # Add score to each table that has this column
                for table in tables:
                    if table not in detected_tables:
                        detected_tables[table] = 0
                    detected_tables[table] += match_score
        
        # Sort tables by score and return top candidates
        scored_tables = sorted(detected_tables.items(), key=lambda x: x[1], reverse=True)
        result_tables = [table for table, score in scored_tables[:3] if score >= 0.5]
        
        # Log what we found
        if result_tables:
            logger.info(f"Column-based table detection found: {', '.join(result_tables)}")
            
        return result_tables
    
    def _rate_sql_quality(self, sql: str, question: str, tables: List[str]) -> float:
        """Rate the quality of the generated SQL on a scale of 0 to 1"""
        score = 1.0  # Start with perfect score
        deductions = []
        
        # 1. Check for CROSS JOINs (usually bad unless specifically needed)
        if "CROSS JOIN" in sql.upper() and len(tables) > 1:
            score -= 0.3
            deductions.append("Uses CROSS JOIN between tables")
        
        # 2. Check for missing JOINs when multiple tables are mentioned
        if len(tables) > 1 and "JOIN" not in sql.upper():
            score -= 0.4
            deductions.append("Missing JOINs between multiple tables")
        
        # 3. Check for correct SELECT syntax
        if not re.search(r'SELECT\s+', sql, re.IGNORECASE):
            score -= 0.8
            deductions.append("Invalid SELECT syntax")
        
        # 4. Check for wrong LIMIT syntax (SQL Server uses TOP)
        if "LIMIT" in sql.upper():
            score -= 0.2
            deductions.append("Uses LIMIT instead of TOP (SQL Server syntax)")
        
        # 5. Check for multiple SELECT statements
        if sql.upper().count("SELECT") > 1 and "UNION" not in sql.upper() and "WITH" not in sql.upper():
            score -= 0.5
            deductions.append("Contains multiple separate SELECT statements")
        
        # 6. Check for proper table references
        for table in tables:
            if table not in sql and f"[{table}]" not in sql:
                score -= 0.2
                deductions.append(f"Missing table reference: {table}")
        
        # 7. Check for dangerous operations in informational queries
        if "tell me about" in question.lower() or "describe" in question.lower():
            for dangerous in ["DELETE", "DROP", "UPDATE", "INSERT", "TRUNCATE"]:
                if dangerous in sql.upper():
                    score -= 0.9
                    deductions.append(f"Contains dangerous {dangerous} operation in an informational query")
        
        # 8. Check for proper WHERE clause for filtering queries
        filter_keywords = ["where", "which", "find", "search", "only", "specific"]
        if any(keyword in question.lower() for keyword in filter_keywords) and "WHERE" not in sql.upper():
            score -= 0.3
            deductions.append("Missing WHERE clause for a filtering query")
        
        # 9. Check for column ambiguity in multi-table queries
        if len(tables) > 1 and "SELECT *" in sql.upper():
            score -= 0.2
            deductions.append("Uses SELECT * with multiple tables (potential column ambiguity)")
        
        # Log deductions if any
        if deductions:
            logger.info(f"SQL quality deductions: {', '.join(deductions)}")
        
        # Ensure score is between 0 and 1
        return max(0.0, min(score, 1.0))
    
    def _extract_sql(self, response: str) -> str:
        """Extract clean SQL from LLM response with enhanced detection for ANY table query"""
        # Remove markdown code blocks
        sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            sql = sql_match.group(1)
        else:
            # Try to extract any code block
            code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if code_match:
                sql = code_match.group(1)
            else:
                # Try to find a SELECT statement directly if no code blocks
                select_match = re.search(r'\b(SELECT\s+.*?)(;|\Z)', response, re.DOTALL | re.IGNORECASE)
                if select_match:
                    sql = select_match.group(1)
                else:
                    sql = response
        
        # Clean the SQL
        sql = sql.strip()
        
        # Remove common prefixes
        sql = re.sub(r'^(SQL Query:|Query:|SQL:)', '', sql, flags=re.IGNORECASE).strip()
        sql = re.sub(r';?\s*$', '', sql)  # Remove trailing semicolon
        
        # Check if SQL looks like a valid query (must start with SELECT or WITH)
        if not re.match(r'\s*(SELECT|WITH)\s+', sql, re.IGNORECASE):
            logger.warning("Extracted text doesn't appear to be valid SQL")
            # Look for a SELECT statement inside what we extracted
            select_match = re.search(r'\b(SELECT\s+.*?)(\Z|;)', sql, re.DOTALL | re.IGNORECASE)
            if select_match:
                sql = select_match.group(1)
                logger.info("Found and extracted SELECT statement from response")
        
        # If there are multiple SELECT statements, try to combine them using JOINs
        if sql.count("SELECT") > 1 and "JOIN" not in sql.upper():
            # Extract individual statements and attempt to create a JOIN
            statements = re.findall(r'SELECT.*?FROM\s+\[?(\w+)\]?', sql, re.DOTALL | re.IGNORECASE)
            if len(statements) > 1:
                logger.warning(f"Found multiple SELECT statements without JOINs: {statements}")
                # We'll let the validation step handle this - don't modify here
                
        # Ensure it starts with SELECT or WITH
        if not re.match(r'^\s*(SELECT|WITH)', sql, re.IGNORECASE):
            # Try to find SELECT in the response
            select_match = re.search(r'(SELECT.*?)(?:\n\n|$)', sql, re.DOTALL | re.IGNORECASE)
            if select_match:
                sql = select_match.group(1)
            else:
                raise ValueError("Generated response does not contain a valid SQL query")
        
        return sql.strip()
    
    def _validate_sql(self, sql: str, schema_context: str, question: str) -> str:
        """Validate and potentially correct SQL"""
        # First check for multiple disconnected SELECTs
        if sql.upper().count("SELECT") > 1 and "JOIN" not in sql.upper():
            fixed_query = self._fix_multiple_selects(sql, schema_context)
            if fixed_query != sql:
                logger.info("Fixed multiple disconnected SELECT statements")
                sql = fixed_query
        
        try:
            chain = self.validation_prompt | self.llm | StrOutputParser()
            
            response = chain.invoke({
                "sql_query": sql,
                "schema_context": schema_context,
                "original_question": question
            })
            
            response = response.strip()
            
            if response.upper() == "VALID":
                return sql
            else:
                # Try to extract corrected SQL
                corrected_sql = self._extract_sql(response)
                if corrected_sql and corrected_sql != sql:
                    return corrected_sql
                
        except Exception as e:
            logger.error(f"Error in SQL validation: {e}")
        
        return sql
    
    def _fix_multiple_selects(self, sql: str, schema_context: str) -> str:
        """Attempt to fix multiple disconnected SELECT statements by converting to a JOIN query"""
        # If we have multiple SELECT statements, create a prompt to convert them to a proper JOIN
        if sql.upper().count("SELECT") <= 1 or "JOIN" in sql.upper():
            return sql
            
        try:
            # Create a specialized prompt for fixing disconnected SELECTs
            template = """You are an expert SQL Server database analyst. Fix this SQL query that has multiple separate SELECT statements.
            
DATABASE SCHEMA:
{schema_context}

CURRENT INCORRECT QUERY WITH MULTIPLE SELECT STATEMENTS:
{sql_query}

INSTRUCTIONS:
1. Convert the multiple SELECT statements into a SINGLE query using appropriate JOINs
2. Identify the relationship between tables based on the schema
3. Use proper JOIN syntax with the correct join conditions
4. Return ONLY the fixed SQL query without any explanation

FIXED SQL QUERY:"""

            fix_prompt = ChatPromptTemplate.from_template(template)
            chain = fix_prompt | self.llm | StrOutputParser()
            
            # Generate response
            response = chain.invoke({
                "sql_query": sql,
                "schema_context": schema_context
            })
            
            # Extract corrected SQL
            corrected_sql = self._extract_sql(response)
            
            # Check if it's better - should have JOINs now
            if "JOIN" in corrected_sql.upper() and corrected_sql.upper().count("SELECT") < sql.upper().count("SELECT"):
                return corrected_sql
            
        except Exception as e:
            logger.error(f"Error fixing multiple SELECTs: {e}")
        
        return sql  # Return original on failure
    
    def _post_process_sql(self, sql: str) -> str:
        """Post-process SQL for SQL Server compatibility"""
        # Ensure SQL Server specific syntax
        sql = re.sub(r'\bLIMIT\s+(\d+)', r'TOP \1', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bNOW\(\)', 'GETDATE()', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bCURDATE\(\)', 'CAST(GETDATE() AS DATE)', sql, flags=re.IGNORECASE)
        
        return sql.strip()

class ConversationMemory:
    """Simple conversation memory for context"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.messages: List[BaseMessage] = []
        self.query_history: List[Dict[str, Any]] = []
        self.conversation_id = f"conv_{int(time.time())}"
        
        # Initialize with system message
        system_message = self._create_system_message()
        self.messages.append(system_message)
        
        # LangChain memory for compatibility
        self.langchain_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def _create_system_message(self) -> SystemMessage:
        """Create system message for context"""
        content = f"""You are QueryMancer, a local AI-powered SQL assistant.

Session ID: {self.conversation_id}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CAPABILITIES:
✓ Natural Language to SQL Server Translation
✓ Local Processing (Ollama + Mistral)
✓ Schema-Aware Query Generation
✓ AWS SQL Server Database Execution
✓ Secure, No External APIs

GUIDELINES:
- Generate precise SQL Server queries
- Use exact table/column names from schema
- Provide clear explanations
- Handle errors gracefully
- Maintain conversation context

Ready to help with your database queries!"""

        return SystemMessage(content=content)
    
    def add_message(self, message: BaseMessage):
        """Add message to memory"""
        self.messages.append(message)
        
        # Update LangChain memory
        if isinstance(message, HumanMessage):
            self.langchain_memory.chat_memory.add_user_message(message.content)
        elif isinstance(message, AIMessage):
            self.langchain_memory.chat_memory.add_ai_message(message.content)
        
        # Maintain history limit
        if len(self.messages) > self.max_history + 1:  # +1 for system message
            system_msg = self.messages[0]
            self.messages = [system_msg] + self.messages[-self.max_history:]
    
    def add_query_result(self, query: str, sql: str, success: bool, 
                        execution_time: float = None, error: str = None):
        """Track query results"""
        self.query_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'sql': sql,
            'success': success,
            'execution_time': execution_time,
            'error': error
        })
        
        # Keep recent queries only
        if len(self.query_history) > 20:
            self.query_history = self.query_history[-20:]
    
    def get_recent_queries(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent query history"""
        return self.query_history[-limit:]
    
    def clear(self):
        """Clear conversation memory"""
        system_msg = self.messages[0] if self.messages else None
        self.messages.clear()
        self.query_history.clear()
        
        if system_msg:
            self.messages.append(system_msg)
        else:
            self.messages.append(self._create_system_message())
        
        self.langchain_memory.clear()

def format_results_as_markdown(columns: List[str], rows: List[Any], 
                              max_display_rows: int = 100) -> str:
    """Format query results as markdown table"""
    try:
        if not columns or not rows:
            return "*No results found*"
        
        total_rows = len(rows)
        display_rows = rows[:max_display_rows]
        
        # Format header
        header = "| " + " | ".join([str(col).replace("|", "\\|") for col in columns]) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        
        # Format data rows
        formatted_rows = []
        for row in display_rows:
            formatted_row = []
            for cell in row:
                if cell is None:
                    cell_str = "*NULL*"
                elif isinstance(cell, bool):
                    cell_str = "✓" if cell else "✗"
                elif isinstance(cell, (int, float)):
                    cell_str = f"{cell:,}" if isinstance(cell, int) else f"{cell:.2f}"
                elif isinstance(cell, datetime):
                    cell_str = cell.strftime("%Y-%m-%d %H:%M")
                else:
                    cell_str = str(cell).replace("|", "\\|").replace("\n", " ")
                    if len(cell_str) > 50:
                        cell_str = cell_str[:47] + "..."
                
                formatted_row.append(cell_str)
            
            formatted_rows.append("| " + " | ".join(formatted_row) + " |")
        
        # Build table
        table = f"{header}\n{separator}\n" + "\n".join(formatted_rows)
        
        # Add footer info
        if total_rows > max_display_rows:
            table += f"\n\n*Showing {max_display_rows} of {total_rows:,} total rows*"
        else:
            table += f"\n\n*Total rows: {total_rows:,}*"
        
        return table
        
    except Exception as e:
        logger.error(f"Error formatting results: {e}")
        return f"*Error formatting results: {str(e)}*"

class QueryMancerAgent:
    """Main agent class for local SQL chatbot"""
    
    def __init__(self, config: Dict[str, Any], schema_file: str = "schema.json"):
        self.config = config
        self.agent_config = AgentConfig()
        
        # Initialize components
        self.schema_manager = SchemaManager(schema_file)
        self.db_manager = DatabaseManager(config)
        self.translator = LocalSQLTranslator(self.schema_manager, config)
        
        # Simple in-memory cache
        self.query_cache = {}
        
        # Session stats
        self.session_stats = {
            'queries_processed': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'cache_hits': 0,
            'session_start': time.time()
        }
        
        logger.info("QueryMancer Agent initialized for local deployment")
    
    def process_query(self, query_text: str, memory: ConversationMemory = None) -> Dict[str, Any]:
        """Process natural language query"""
        if not memory:
            memory = ConversationMemory()
        
        start_time = time.time()
        self.session_stats['queries_processed'] += 1
        
        try:
            # Input validation
            if not query_text or not query_text.strip():
                raise ValueError("Empty query provided")
            
            query_text = query_text.strip()
            
            # Check simple cache
            cache_key = hashlib.md5(query_text.lower().encode()).hexdigest()
            if self.agent_config.cache_enabled and cache_key in self.query_cache:
                self.session_stats['cache_hits'] += 1
                cached_result = self.query_cache[cache_key].copy()
                cached_result['from_cache'] = True
                logger.info("Using cached query result")
                return cached_result
            
            # Translate to SQL
            translation_result = self.translator.translate(query_text)
            
            if not translation_result['success']:
                self.session_stats['failed_queries'] += 1
                memory.add_query_result(query_text, '', False, error=translation_result.get('error'))
                return self._format_error_response(query_text, '', translation_result)
            
            sql_query = translation_result['sql']
            
            # Execute SQL
            execution_result = self.db_manager.execute_query(sql_query, self.agent_config.max_query_limit)
            
            # Format response
            if execution_result['success']:
                response = self._format_success_response(
                    query_text, sql_query, execution_result, translation_result
                )
                self.session_stats['successful_queries'] += 1
                
                # Cache successful result
                if self.agent_config.cache_enabled:
                    self.query_cache[cache_key] = response.copy()
                
                memory.add_query_result(
                    query_text, sql_query, True, execution_result.get('execution_time')
                )
                
                return response
            else:
                self.session_stats['failed_queries'] += 1
                memory.add_query_result(
                    query_text, sql_query, False, error=execution_result.get('error')
                )
                return self._format_error_response(query_text, sql_query, execution_result)
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.session_stats['failed_queries'] += 1
            
            logger.error(f"Error processing query: {e}")
            
            error_result = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'query_text': query_text,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
            
            if memory:
                memory.add_query_result(query_text, '', False, error=str(e))
            
            return error_result
    
    def _format_success_response(self, query_text: str, sql_query: str, 
                               execution_result: Dict[str, Any],
                               translation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format successful query response"""
        
        response_parts = []
        
        # Success header
        exec_time = execution_result.get('execution_time', 0)
        row_count = execution_result.get('row_count', 0)
        
        response_parts.append(f"✅ **Query Executed Successfully**")
        response_parts.append(f"⚡ **Execution Time:** {exec_time:.3f}s | 📊 **Rows:** {row_count:,}")
        
        # Show generated SQL
        response_parts.append(f"\n**Generated SQL:**")
        response_parts.append(f"```sql\n{sql_query}\n```")
        
        # Format results
        if "columns" in execution_result and "rows" in execution_result:
            if execution_result["rows"]:
                formatted_results = format_results_as_markdown(
                    execution_result["columns"],
                    execution_result["rows"]
                )
                response_parts.append(f"\n**Results:**\n{formatted_results}")
            else:
                response_parts.append("\n*No data found matching your criteria.*")
        
        # Add query explanation if complex
        tables_used = translation_result.get('tables_used', [])
        if len(tables_used) > 1:
            explanation = self._explain_query(sql_query, tables_used)
            if explanation:
                response_parts.append(f"\n💡 **Query Explanation:**\n{explanation}")
        
        # Show truncation warning
        if execution_result.get('truncated'):
            response_parts.append(f"\n⚠️ *Results limited to {self.agent_config.max_query_limit:,} rows*")
        
        return {
            'success': True,
            'response': "\n\n".join(response_parts),
            'query_text': query_text,
            'sql': sql_query,
            'execution_result': execution_result,
            'translation_result': translation_result,
            'tables_used': tables_used,
            'timestamp': datetime.now().isoformat(),
            'session_stats': self.get_session_stats()
        }
    
    def _format_error_response(self, query_text: str, sql_query: str,
                             error_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format error response with helpful information"""
        
        error_msg = error_result.get('error', 'Unknown error')
        error_type = error_result.get('error_type', 'Error')
        
        response_parts = []
        response_parts.append(f"❌ **Query Failed**")
        response_parts.append(f"**Error:** {error_msg}")
        
        # Show SQL if it was generated
        if sql_query:
            response_parts.append(f"\n**Generated SQL:**\n```sql\n{sql_query}\n```")
        
        # Provide helpful suggestions
        suggestions = self._get_error_suggestions(error_msg, query_text)
        if suggestions:
            response_parts.append(f"\n💡 **Suggestions:**\n{suggestions}")
        
        # Show available tables if table not found
        if 'invalid object' in error_msg.lower() or 'not found' in error_msg.lower():
            available_tables = self.schema_manager.get_all_tables()[:5]
            if available_tables:
                response_parts.append(f"\n📋 **Available Tables:** {', '.join(available_tables)}")
        
        return {
            'success': False,
            'error': error_msg,
            'error_type': error_type,
            'response': "\n\n".join(response_parts),
            'query_text': query_text,
            'sql': sql_query,
            'timestamp': datetime.now().isoformat()
        }
    
    def _explain_query(self, sql_query: str, tables_used: List[str]) -> str:
        """Generate simple query explanation"""
        try:
            explanation_parts = []
            
            # Identify query type
            sql_upper = sql_query.upper()
            
            if 'COUNT(' in sql_upper:
                explanation_parts.append("This query counts records")
            elif any(func in sql_upper for func in ['SUM(', 'AVG(', 'MAX(', 'MIN(']):
                explanation_parts.append("This query calculates aggregate values")
            elif 'GROUP BY' in sql_upper:
                explanation_parts.append("This query groups data and shows summaries")
            elif 'ORDER BY' in sql_upper:
                explanation_parts.append("This query sorts the results")
            elif 'JOIN' in sql_upper:
                explanation_parts.append("This query combines data from multiple tables")
            else:
                explanation_parts.append("This query retrieves data")
            
            # Mention tables
            if tables_used:
                if len(tables_used) == 1:
                    explanation_parts.append(f"from the {tables_used[0]} table")
                else:
                    explanation_parts.append(f"from tables: {', '.join(tables_used)}")
            
            # Check for filters
            if 'WHERE' in sql_upper:
                explanation_parts.append("with specific filtering conditions")
            
            return " ".join(explanation_parts) + "."
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return ""
    
    def _get_error_suggestions(self, error_msg: str, query_text: str) -> str:
        """Get helpful suggestions based on error"""
        suggestions = []
        error_lower = error_msg.lower()
        
        if 'invalid object' in error_lower or 'invalid column' in error_lower:
            suggestions.append("- Check table and column names for typos")
            suggestions.append("- Use exact names as they appear in the database")
            suggestions.append("- Try rephrasing your question")
        
        elif 'syntax error' in error_lower:
            suggestions.append("- The generated SQL may have syntax issues")
            suggestions.append("- Try asking your question differently")
            suggestions.append("- Be more specific about what you want to find")
        
        elif 'timeout' in error_lower:
            suggestions.append("- Query took too long to execute")
            suggestions.append("- Try adding more specific filters")
            suggestions.append("- Consider asking for a smaller subset of data")
        
        elif 'permission' in error_lower or 'access denied' in error_lower:
            suggestions.append("- Check database connection and permissions")
            suggestions.append("- Only SELECT queries are allowed")
        
        else:
            # General suggestions
            suggestions.append("- Try rephrasing your question")
            suggestions.append("- Be more specific about what data you need")
            suggestions.append("- Check if the table/column names are correct")
        
        return "\n".join(suggestions) if suggestions else ""
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        total_queries = self.session_stats['queries_processed']
        successful_queries = self.session_stats['successful_queries']
        success_rate = successful_queries / total_queries if total_queries > 0 else 0
        
        session_duration = time.time() - self.session_stats['session_start']
        
        return {
            'queries_processed': total_queries,
            'successful_queries': successful_queries,
            'failed_queries': self.session_stats['failed_queries'],
            'cache_hits': self.session_stats['cache_hits'],
            'success_rate': success_rate,
            'session_duration_minutes': session_duration / 60,
            'cache_size': len(self.query_cache)
        }
    
    def get_available_tables(self) -> List[str]:
        """Get list of available database tables"""
        return self.schema_manager.get_all_tables()
    
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific table"""
        return self.schema_manager.get_table_info(table_name)
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test database connection"""
        return self.db_manager.test_connection()
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def reload_schema(self) -> bool:
        """Reload schema from file"""
        try:
            self.schema_manager.load_schema(force_reload=True)
            logger.info("Schema reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error reloading schema: {e}")
            return False
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate agent setup"""
        validation = {
            'schema_loaded': False,
            'database_connected': False,
            'ollama_available': False,
            'overall_status': 'unknown'
        }
        
        try:
            # Test schema
            tables = self.schema_manager.get_all_tables()
            validation['schema_loaded'] = len(tables) > 0
            
            # Test database
            db_success, _ = self.db_manager.test_connection()
            validation['database_connected'] = db_success
            
            # Test Ollama
            try:
                test_response = self.translator.llm.invoke("Test")
                validation['ollama_available'] = bool(test_response)
            except Exception:
                validation['ollama_available'] = False
            
            # Overall status
            if all([validation['schema_loaded'], validation['database_connected'], validation['ollama_available']]):
                validation['overall_status'] = 'ready'
            elif validation['schema_loaded'] and validation['database_connected']:
                validation['overall_status'] = 'partial'
            else:
                validation['overall_status'] = 'failed'
                
        except Exception as e:
            validation['error'] = str(e)
            validation['overall_status'] = 'error'
        
        return validation

# Main processing function for external use
def process_natural_language_query(query_text: str, config: Dict[str, Any], 
                                 memory: ConversationMemory = None) -> Tuple[str, float]:
    """Main entry point for processing natural language queries"""
    
    agent = QueryMancerAgent(config)
    
    if not memory:
        memory = ConversationMemory()
    
    start_time = time.time()
    
    try:
        result = agent.process_query(query_text, memory)
        processing_time = time.time() - start_time
        
        if result['success']:
            return result['response'], processing_time
        else:
            return result.get('response', f"❌ Error: {result.get('error', 'Unknown error')}"), processing_time
            
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error in main processing: {e}")
        return f"❌ System Error: {str(e)}", processing_time

# Utility functions
def validate_agent_setup(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate agent setup without creating full agent"""
    try:
        agent = QueryMancerAgent(config)
        return agent.validate_setup()
    except Exception as e:
        return {
            'overall_status': 'error',
            'error': str(e),
            'schema_loaded': False,
            'database_connected': False,
            'ollama_available': False
        }

def get_database_tables(config: Dict[str, Any]) -> List[str]:
    """Get list of available database tables"""
    try:
        schema_manager = SchemaManager()
        return schema_manager.get_all_tables()
    except Exception as e:
        logger.error(f"Error getting tables: {e}")
        return []

def test_database_connection(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Test database connection"""
    try:
        db_manager = DatabaseManager(config)
        return db_manager.test_connection()
    except Exception as e:
        return False, str(e)

def execute_sql_directly(sql_query: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute SQL query directly"""
    try:
        db_manager = DatabaseManager(config)
        return db_manager.execute_query(sql_query)
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# Example usage and testing
if __name__ == "__main__":
    print("🚀 QueryMancer Local Agent - Ollama + Mistral Edition")
    print("=" * 60)
    
    # Test configuration (replace with actual config)
    test_config = {
        'SQL_SERVER': '10.0.0.45',
        'SQL_DATABASE': '146_36156520-AC21-435A-9C9B-1EC9145A9090',
        'SQL_USERNAME': 'usr_mohsin',
        'SQL_PASSWORD': 'blY|5K:3pe10',
        'OLLAMA_BASE_URL': 'http://localhost:11434',
        'OLLAMA_MODEL': 'mistral',
        'enable_query_validation': True
    }
    
    # Validate setup
    print("🔧 Validating setup...")
    validation = validate_agent_setup(test_config)
    
    for component, status in validation.items():
        if component not in ['error', 'overall_status']:
            icon = "✅" if status else "❌"
            print(f"  {icon} {component.replace('_', ' ').title()}: {status}")
    
    print(f"\n📊 Overall Status: {validation['overall_status'].upper()}")
    
    if validation.get('error'):
        print(f"❌ Error: {validation['error']}")
    
    if validation['overall_status'] == 'ready':
        print("\n✨ QueryMancer Agent is ready!")
        print("💬 You can now ask natural language questions about your database")
        
        # Example queries
        example_queries = [
            "Show me all records from XERO_EXPORT",
            "How many transactions are there in total?",
            "What is the sum of all transaction amounts?",
            "Show me recent invoices sorted by date"
        ]
        
        print(f"\n📝 Example queries you can try:")
        for i, query in enumerate(example_queries, 1):
            print(f"  {i}. \"{query}\"")
    
    else:
        print(f"\n⚠️  Setup incomplete. Please check the issues above.")
        if not validation.get('ollama_available', False):
            print("💡 Make sure Ollama is running: ollama serve")
            print("💡 And Mistral is installed: ollama pull mistral")
    
    print("\n🔒 Security Features:")
    print("  ✓ Local processing only (no external APIs)")
    print("  ✓ Schema-based SQL generation")
    print("  ✓ SQL injection protection")
    print("  ✓ Read-only database access")
    print("  ✓ Query validation and sanitization")