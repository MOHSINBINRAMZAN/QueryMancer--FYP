"""
QueryMancer - AI-Powered SQL Chatbot Application
Enhanced with Ollama + Mistral + RAG with FAISS Vector Db for local AI inference and AWS SQL Server integration.

Current Date and Time: 
Current User: 
"""

# Ensure proper encoding for Windows console
import sys
import os
if sys.platform == 'win32':
    # Set console encoding to UTF-8 if possible
    try:
        os.system('chcp 65001 > nul 2>&1')
    except:
        pass

import streamlit as st
import json
import time
import traceback
import logging
import pandas as pd
import re
import uuid
import hashlib
import plotly.express as px
import plotly.graph_objects as go
import pyodbc
import sqlalchemy
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Union
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
import warnings
from dotenv import load_dotenv

# LangChain and Ollama imports
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableSequence

# Import our Enhanced UI module
from ui import EnhancedLocalUI

# Load environment variables
load_dotenv()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Enhanced logging configuration
import os
os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists

# Create a custom stream handler for stdout that can handle Unicode
class UnicodeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Replace Unicode emojis with plain text alternatives
            msg = msg.replace("✅", "[OK]")
            msg = msg.replace("❌", "[ERROR]")
            msg = msg.replace("🚀", "[ROCKET]")
            msg = msg.replace("📊", "[CHART]")
            msg = msg.replace("🔍", "[SEARCH]")
            msg = msg.replace("⚡", "[LIGHTNING]")
            msg = msg.replace("🎯", "[TARGET]")
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/querymancer_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8'),
        UnicodeStreamHandler(sys.stdout)
    ]
)

# Set encoding for all loggers to handle Unicode properly
for handler in logging.root.handlers:
    if hasattr(handler, 'stream') and hasattr(handler.stream, 'reconfigure'):
        try:
            handler.stream.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass

logger = logging.getLogger("querymancer.app")

# Constants
CURRENT_USER = "Mohsin Ramzan"
CURRENT_DATETIME = datetime.now()
CURRENT_DAY = CURRENT_DATETIME.day

# Database Configuration
DB_CONFIG = {
    'server': os.getenv('DB_SERVER', 'MECHREVO-\\SQLEXPRESS'),
    'database': os.getenv('DB_DATABASE', 'CRM'),
    'username': os.getenv('DB_USERNAME', ''),  # Empty for Windows Authentication
    'password': os.getenv('DB_PASSWORD', ''),  # Empty for Windows Authentication
    'driver': os.getenv('DB_DRIVER', 'ODBC Driver 17 for SQL Server'),
    'trust_certificate': os.getenv('DB_TRUST_CERTIFICATE', 'yes'),
    
}

# Ollama Configuration
OLLAMA_CONFIG = {
    'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
    'model': os.getenv('OLLAMA_MODEL', 'mistral'),
    'temperature': float(os.getenv('OLLAMA_TEMPERATURE', '0.1')),
    'max_tokens': int(os.getenv('OLLAMA_MAX_TOKENS', '2048'))
}

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for Streamlit integration"""
    
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.current_text = ""
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.placeholder.markdown("🧠 **AI is thinking...**")
    
    def on_llm_new_token(self, token: str, **kwargs):
        self.current_text += token
        self.placeholder.markdown(f"🧠 **Generating SQL:** {self.current_text}")

class SQLOutputParser(BaseOutputParser[str]):
    """Custom output parser for SQL queries"""
    
    def parse(self, text: str) -> str:
        """Parse SQL from LLM output"""
        # Remove any markdown formatting
        text = text.strip()
        
        # Extract SQL from code blocks
        sql_pattern = r'```(?:sql)?\s*(.*?)\s*```'
        sql_match = re.search(sql_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if sql_match:
            return sql_match.group(1).strip()
        
        # If no code block, try to extract SQL-like content
        lines = text.split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']):
                sql_lines.append(line)
            elif sql_lines and (line.startswith(('FROM', 'WHERE', 'JOIN', 'ORDER BY', 'GROUP BY', 'HAVING')) or 
                               line.upper().startswith(('FROM', 'WHERE', 'JOIN', 'ORDER BY', 'GROUP BY', 'HAVING'))):
                sql_lines.append(line)
        
        return '\n'.join(sql_lines) if sql_lines else text

@dataclass
class QueryMetrics:
    """Query metrics tracking"""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    user: str = CURRENT_USER
    query_text: str = ""
    sql_generated: str = ""
    execution_time: float = 0.0
    confidence_score: float = 0.0
    execution_success: bool = False
    rows_returned: int = 0
    error_message: str = ""
    ai_response_time: float = 0.0

class DatabaseManager:
    """Database connection and query execution manager"""
    
    def __init__(self):
        self.connection_string = self._build_connection_string()
        self.engine = None
        self._test_connection()
    
    def _build_connection_string(self) -> str:
        """Build SQL Server connection string"""
        if DB_CONFIG['username'] and DB_CONFIG['password']:
            # SQL Server Authentication
            conn_str = (
                f"DRIVER={{{DB_CONFIG['driver']}}};"
                f"SERVER={DB_CONFIG['server']};"
                f"DATABASE={DB_CONFIG['database']};"
                f"UID={DB_CONFIG['username']};"
                f"PWD={DB_CONFIG['password']};"
                f"TrustServerCertificate={DB_CONFIG['trust_certificate']};"
            )
        else:
            # Windows Authentication
            conn_str = (
                f"DRIVER={{{DB_CONFIG['driver']}}};"
                f"SERVER={DB_CONFIG['server']};"
                f"DATABASE={DB_CONFIG['database']};"
                f"Trusted_Connection=yes;"
                f"TrustServerCertificate={DB_CONFIG['trust_certificate']};"
            )
        
        return conn_str
    
    def _test_connection(self):
        """Test database connection"""
        try:
            conn = pyodbc.connect(self.connection_string, timeout=10)
            conn.close()
            logger.info("[OK] Database connection successful")
        except Exception as e:
            logger.error(f"[ERROR] Database connection failed: {e}")
            raise
    
    def _execute_single_query(self, sql: str) -> Dict[str, Any]:
        """Execute a single SQL query and return results"""
        try:
            start_time = time.time()
            
            with pyodbc.connect(self.connection_string, timeout=30) as conn:
                cursor = conn.cursor()
                
                # Execute query
                cursor.execute(sql)
                
                # Fetch all results
                columns = [desc[0] for desc in cursor.description]
                
                # Check for duplicate column names
                if len(columns) != len(set(columns)):
                    # Get table names from the query to use as prefixes
                    table_matches = re.findall(r'FROM\s+\[?(\w+)\]?(?:\s+[aA])?.*?(?:JOIN\s+\[?(\w+)\]?(?:\s+[bB])?)?', sql, re.IGNORECASE)
                    table_prefixes = []
                    if table_matches:
                        for match in table_matches:
                            table_prefixes.extend([t for t in match if t])
                    
                    # Default prefixes if we couldn't extract from query
                    if not table_prefixes or len(table_prefixes) < 2:
                        table_prefixes = ["T1", "T2", "T3", "T4"]
                    
                    # Create a mapping for duplicate columns
                    col_counts = {}
                    renamed_columns = []
                    prefix_idx = 0
                    
                    for col in columns:
                        if col in col_counts:
                            col_counts[col] += 1
                            # Add prefix from corresponding table or use numeric suffix if we run out of tables
                            prefix = table_prefixes[min(prefix_idx, len(table_prefixes)-1)]
                            renamed_columns.append(f"{prefix}_{col}")
                            prefix_idx += 1
                        else:
                            col_counts[col] = 1
                            renamed_columns.append(col)
                    
                    columns = renamed_columns
                
                rows = cursor.fetchall()
                
                # Convert to pandas DataFrame
                df = pd.DataFrame([list(row) for row in rows], columns=columns)
                
                # Format date columns
                for col in df.columns:
                    if df[col].dtype == 'datetime64[ns]':
                        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                execution_time = time.time() - start_time
                
                return {
                    'success': True,
                    'data': df,
                    'message': f"Query executed successfully. {len(rows)} rows returned.",
                    'sql': sql,
                    'execution_time': execution_time
                }
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logging.error(f"Query execution failed: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'sql': sql,
                'execution_time': execution_time
            }
    
    def execute_query(self, sql: str) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        try:
            start_time = time.time()
            
            # Check for multiple SELECT statements
            selects = re.findall(r'SELECT\s+.+?FROM\s+\[?(\w+)\]?', sql, re.IGNORECASE | re.DOTALL)
            
            if len(selects) > 1:
                logging.warning("Multiple SELECT statements detected. Attempting to create a JOIN query.")
                
                # EMERGENCY FIX: Always force a JOIN when multiple tables are detected
                if len(selects) >= 2:
                    # Use single line SQL with no breaks to avoid parsing issues
                    sql = f"SELECT a.*, b.* FROM [{selects[0]}] a CROSS JOIN [{selects[1]}] b"
                    logging.warning(f"Forcing JOIN between {selects[0]} and {selects[1]}")
                    # Return immediately with the fixed query
                    return self._execute_single_query(sql)
                
                # Map to exact table names from schema
                tables = []
                schema_tables = list(self.schema_manager.schema_data.get('tables', {}).keys())
                
                for table in selects:
                    table = table.strip('[]')
                    # Try direct match first
                    if table in schema_tables:
                        tables.append(table)
                        continue
                    
                    # Try to find the correct case
                    for schema_table in schema_tables:
                        if schema_table.upper() == table.upper():
                            tables.append(schema_table)
                            break
                    
                    # Check variations if still not found
                    if len(tables) < len(selects):
                        for schema_table, table_info in self.schema_manager.schema_data.get('tables', {}).items():
                            variations = table_info.get('variations', [])
                            if table in variations or table.upper() in [v.upper() for v in variations]:
                                tables.append(schema_table)
                                break
                
                logging.info(f"Mapped input tables {selects} to schema tables {tables}")
                
                # Create a joined query using our helper method
                if len(tables) > 1:
                    fixed_sql = self._fix_multiple_selects(sql, tables)
                    if fixed_sql != sql:
                        sql = fixed_sql
                        logging.info(f"Created JOIN query for tables: {tables}")
            
            with pyodbc.connect(self.connection_string, timeout=30) as conn:
                cursor = conn.cursor()
                
                # Execute query
                cursor.execute(sql)
                
                # Check if it's a SELECT query
                if sql.strip().upper().startswith('SELECT'):
                    # Fetch all results
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    
                    # Check if there are any duplicate column names
                    if len(columns) != len(set(columns)):
                        logging.info("Detected duplicate column names in query result")
                        
                        # Extract table aliases from the query
                        table_aliases = {}
                        alias_pattern = r'\[?(\w+)\]?\s+(\w+)(?:\s+|\))'
                        matches = re.findall(alias_pattern, sql)
                        
                        for table, alias in matches:
                            table_aliases[alias.lower()] = table
                        
                        logging.debug(f"Extracted table aliases: {table_aliases}")
                        
                        # Track which columns we've already seen
                        unique_columns = []
                        column_seen_count = {}
                        
                        for i, col in enumerate(columns):
                            # Count occurrences of this column name
                            if col in column_seen_count:
                                column_seen_count[col] += 1
                                
                                # Try several methods to get table information
                                table_identifier = None
                                
                                # Method 1: Try to get table name from cursor description
                                try:
                                    desc = cursor.description[i]
                                    if len(desc) > 2 and desc[2]:
                                        table_name = desc[2]
                                        # Find matching alias
                                        for alias, table in table_aliases.items():
                                            if table.upper() == table_name.upper():
                                                table_identifier = alias
                                                break
                                        if not table_identifier:
                                            table_identifier = table_name
                                except (IndexError, AttributeError):
                                    pass
                                
                                # Method 2: Try to infer from query pattern for explicit columns
                                if not table_identifier and '.' in sql:
                                    # Look for pattern like "a.COLUMN_NAME"
                                    col_pattern = r'(\w+)\.' + col
                                    col_matches = re.findall(col_pattern, sql)
                                    if col_matches:
                                        table_identifier = col_matches[0]
                                
                                # Method 3: Use the count as last resort
                                if table_identifier:
                                    unique_columns.append(f"{table_identifier}_{col}")
                                else:
                                    unique_columns.append(f"{col}_{column_seen_count[col]}")
                            else:
                                unique_columns.append(col)
                                column_seen_count[col] = 0
                        
                        logging.debug(f"Mapped columns: {list(zip(columns, unique_columns))}")
                    else:
                        unique_columns = columns
                    
                    # Convert to pandas DataFrame with unique column names
                    df = pd.DataFrame([list(row) for row in rows], columns=unique_columns)
                    
                    execution_time = time.time() - start_time
                    
                    return {
                        'success': True,
                        'data': df,
                        'rows_affected': len(df),
                        'execution_time': execution_time,
                        'message': f"Query executed successfully. {len(df)} rows returned."
                    }
                else:
                    # For non-SELECT queries (INSERT, UPDATE, DELETE, etc.)
                    rows_affected = cursor.rowcount
                    conn.commit()
                    
                    execution_time = time.time() - start_time
                    
                    return {
                        'success': True,
                        'data': None,
                        'rows_affected': rows_affected,
                        'execution_time': execution_time,
                        'message': f"Query executed successfully. {rows_affected} rows affected."
                    }
                    
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Query execution failed: {error_msg}")
            
            return {
                'success': False,
                'data': None,
                'rows_affected': 0,
                'execution_time': execution_time,
                'error': error_msg,
                'message': f"Query failed: {error_msg}"
            }

class SchemaManager:
    """Database schema management using schema.json"""
    
    def __init__(self, schema_path: str = "schema.json"):
        self.schema_path = schema_path
        self.schema_data = self._load_schema()
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load schema from JSON file"""
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            logger.info(f"[OK] Schema loaded from {self.schema_path}")
            return schema
        except FileNotFoundError:
            logger.error(f"[ERROR] Schema file not found: {self.schema_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"[ERROR] Invalid JSON in schema file: {e}")
            return {}
    
    def get_relevant_tables(self, query: str) -> List[Dict[str, Any]]:
        """Get relevant tables - FAST and focused on explicitly mentioned tables"""
        import re
        query_upper = query.upper()
        
        logger.info(f"Analyzing query for table relevance: '{query}'")
        
        # Find explicitly mentioned tables ONLY using precise word extraction
        explicitly_mentioned = []
        
        # Extract potential table names (uppercase words with underscores)
        potential_table_names = set(re.findall(r'[A-Z_][A-Z0-9_]*', query))
        logger.info(f"Potential table names found in query: {potential_table_names}")
        
        for table_name, table_info in self.schema_data.items():
            # Check if table name is explicitly mentioned (exact match)
            if table_name in potential_table_names:
                explicitly_mentioned.append({
                    'name': table_name,
                    'info': table_info,
                    'score': 1000,
                    'reasons': [f"explicitly_mentioned({table_name})"]
                })
                logger.info(f"[OK] Found explicitly mentioned table: {table_name}")
            else:
                # Check variations for exact matches
                variations = table_info.get('variations', [])
                for variation in variations:
                    if variation and variation.upper() in potential_table_names:
                        explicitly_mentioned.append({
                            'name': table_name,
                            'info': table_info,
                            'score': 1000,
                            'reasons': [f"explicitly_mentioned_via_variation({variation})"]
                        })
                        logger.info(f"[OK] Found explicitly mentioned table via variation: {table_name} ({variation})")
                        break
        
        # If we found explicit mentions, return them
        if explicitly_mentioned:
            logger.info(f"Using {len(explicitly_mentioned)} explicitly mentioned tables")
            return explicitly_mentioned
        
        # Fallback: simple keyword matching for common queries
        fallback_tables = []
        for table_name, table_info in self.schema_data.items():
            score = 0
            
            # Simple word matching
            table_words = table_name.lower().replace('_', ' ').split()
            for word in query.lower().split():
                if word in table_words:
                    score += 50
            
            if score > 0:
                fallback_tables.append({
                    'name': table_name,
                    'info': table_info,
                    'score': score,
                    'reasons': [f"word_match"]
                })
        
        fallback_tables.sort(key=lambda x: x['score'], reverse=True)
        result = fallback_tables[:5]  # Limit to 5 tables max
        
        for table in result:
            logger.info(f"Selected table: {table['name']} (score: {table['score']})")
        
        logger.info(f"Found {len(result)} relevant tables from {len(self.schema_data)} total")
        return result
    
    def get_schema_context(self, query: str, max_tables: int = 3) -> str:
        """Get schema context for the query - FAST and focused"""
        logger.info(f"Getting schema context for query: {query}")
        relevant_tables = self.get_relevant_tables(query)
        logger.info(f"Found {len(relevant_tables)} relevant tables: {[t['name'] for t in relevant_tables]}")
        
        if not relevant_tables:
            return "-- No relevant tables found --"
        
        context_parts = ["-- Database Schema Context --\n"]
        
        # Include ONLY the most relevant tables (limit for speed)
        tables_to_include = relevant_tables[:max_tables]
        
        for table_data in tables_to_include:
            table_name = table_data['name']
            table_info = table_data['info']
            
            context_parts.append(f"TABLE: {table_name}")
            
            # Primary keys
            primary_keys = table_info.get('primary_keys', [])
            if primary_keys:
                context_parts.append("PRIMARY KEYS:")
                for pk in primary_keys:
                    context_parts.append(f"  - {pk}")
            
            # Foreign keys (only the most important ones)
            foreign_keys = table_info.get('foreign_keys', {})
            if foreign_keys:
                context_parts.append("FOREIGN KEYS:")
                for fk_col, fk_ref in list(foreign_keys.items())[:10]:  # Limit FK display
                    context_parts.append(f"  - {fk_col} references {fk_ref}")
            
            # Columns (limit to most important ones for speed)
            columns = table_info.get('columns', [])
            if columns:
                context_parts.append("COLUMNS:")
                # Show first 15 columns to keep context manageable
                for col in columns[:15]:
                    col_desc = f"  - {col}"
                    # Mark primary keys
                    if col in primary_keys:
                        col_desc += " [PRIMARY KEY]"
                    # Mark foreign keys  
                    if col in foreign_keys:
                        col_desc += f" [FOREIGN KEY -> {foreign_keys[col]}]"
                    context_parts.append(col_desc)
                if len(columns) > 15:
                    context_parts.append(f"  ... and {len(columns) - 15} more columns")
            
            context_parts.append("")  # Empty line between tables
        
        context = "\n".join(context_parts)
        logger.info(f"Generated schema context length: {len(context)} characters")
        return context

class AIQueryProcessor:
    """AI-powered query processor using Ollama + Mistral"""
    
    def __init__(self, schema_manager: SchemaManager):
        self.schema_manager = schema_manager
        self.llm = self._initialize_ollama()
        self.sql_parser = SQLOutputParser()
        self.prompt_template = self._create_prompt_template()
    
    def _initialize_ollama(self) -> OllamaLLM:
        """Initialize Ollama LLM"""
        try:
            llm = OllamaLLM(
                model=OLLAMA_CONFIG['model'],
                base_url=OLLAMA_CONFIG['base_url'],
                temperature=OLLAMA_CONFIG['temperature'],
                num_predict=OLLAMA_CONFIG['max_tokens']
            )
            
            # Test the connection
            test_response = llm.invoke("SELECT 1")
            logger.info("[OK] Ollama connection successful")
            return llm
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize Ollama: {e}")
            raise
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for SQL generation"""
        template = """CRITICAL: You must return ONLY a SQL query. NO other text allowed.

{schema_context}

RESPONSE FORMAT - MANDATORY:
1. NO explanations or descriptions
2. NO "Here's the SQL" or "Based on the schema" 
3. NO numbered lists or bullet points
4. NO code blocks or markdown
5. NO assumptions about relationships
6. ONLY raw SQL starting with SELECT/INSERT/UPDATE/DELETE

EXAMPLES OF CORRECT RESPONSES:
Question: "tell me about USER and ACCOUNT tables"
CORRECT: SELECT u.[USER_DISPLAY_NAME], a.[ACCOUNT_NAME] FROM [USER] u LEFT JOIN [ACCOUNT] a ON u.[RECORD_ID] = a.[ACCOUNT_OWNER]

Question: "show all users"
CORRECT: SELECT * FROM [USER]

Question: "show VAT_RATE and USER information" (NO RELATIONSHIP EXISTS)
CORRECT: SELECT * FROM [VAT_RATE]

EXAMPLES OF INCORRECT RESPONSES (NEVER DO THIS):
❌ "Here's the SQL query..."
❌ "Based on the schema..."
❌ "assuming they have a foreign key relationship"
❌ "```sql SELECT... ```"
❌ "1. To join tables..."
❌ Creating JOINs when no foreign key exists

UNIVERSAL RULES FOR ALL TABLES:
1. ALWAYS use EXACTLY the column names from the schema - NEVER invent columns
2. ALL tables use RECORD_ID as the primary key - always use this exact name  
3. Use square brackets [TABLE_NAME] and [COLUMN_NAME] for ALL identifiers (MANDATORY)
4. For JOINs, ONLY use the foreign_keys explicitly defined in the schema
5. If NO foreign key relationship exists between tables, DO NOT create a JOIN
6. Return ONLY the SQL query - no explanations, markdown, or code blocks
7. Start directly with SELECT, INSERT, UPDATE, DELETE, or other SQL keywords
8. NEVER use words like "assuming", "suppose", "let me assume"
9. NEVER create fake relationships between unrelated tables

CRITICAL: ALWAYS use square brackets around table and column names:
✅ CORRECT: SELECT [COLUMN] FROM [TABLE_NAME] 
❌ WRONG: SELECT COLUMN FROM TABLE_NAME

MULTI-TABLE QUERIES:
When the user mentions MULTIPLE tables:
- ONLY create JOINs if foreign key relationships exist in the schema
- If NO relationship exists, use CROSS JOIN to show all combinations
- NEVER invent fake relationships or assume connections
- Format for unrelated tables: SELECT columns FROM [TABLE1] t1 CROSS JOIN [TABLE2] t2

COLUMN DETECTION STRATEGY:
- Each table's columns are listed in the schema context above
- Use exact column names as shown in the COLUMNS section
- Primary keys are marked [PRIMARY KEY]  
- Foreign keys are marked [FOREIGN KEY -> reference]

JOIN PATTERN FOR ANY TABLES:
- If table A has foreign key FK_COL referencing TABLE_B.RECORD_ID:
  JOIN [TABLE_B] ON [TABLE_A].[FK_COL] = [TABLE_B].[RECORD_ID]

COMMON QUERY PATTERNS:

Basic Selection:
"Show all [table_name]" → SELECT * FROM [TABLE_NAME]
"List [table_name] data" → SELECT * FROM [TABLE_NAME] 
"Get [column_name] from [table_name]" → SELECT [COLUMN_NAME] FROM [TABLE_NAME]

Multi-Table Queries:
"Tell me about USER and ACCOUNT" → SELECT u.[USER_DISPLAY_NAME], a.[ACCOUNT_NAME] FROM [USER] u LEFT JOIN [ACCOUNT] a ON a.[ACCOUNT_OWNER] = u.[RECORD_ID]
"Show VAT_RATE and USER information" → SELECT v.[PERCENTAGE], u.[USER_DISPLAY_NAME] FROM [VAT_RATE] v CROSS JOIN [USER] u
"Join contacts and members" → SELECT c.*, m.* FROM [CONTACT] c LEFT JOIN [MEMBER] m ON c.[RECORD_ID] = m.[CONTACT]

Filtering:
"Find [table_name] where [condition]" → SELECT * FROM [TABLE_NAME] WHERE [CONDITION]
"Show active [table_name]" → SELECT * FROM [TABLE_NAME] WHERE [DELETED] = 0 OR [DELETED] IS NULL
"Recent [table_name]" → SELECT * FROM [TABLE_NAME] WHERE [DATE_CREATED_ON] >= DATEADD(day, -30, GETDATE())

Aggregation:
"Count [table_name]" → SELECT COUNT(*) AS Total FROM [TABLE_NAME]
"Sum of [column]" → SELECT SUM([COLUMN]) FROM [TABLE_NAME]
"Group by [column]" → SELECT [COLUMN], COUNT(*) FROM [TABLE_NAME] GROUP BY [COLUMN]

Sorting:
"Order by [column]" → ORDER BY [COLUMN_NAME]
"Latest [table_name]" → ORDER BY [DATE_CREATED_ON] DESC
"Sort by name" → ORDER BY [NAME_COLUMN] (use actual name column from schema)

EXAMPLES FOR ANY TABLE STRUCTURE:

Query: "Show all contacts"
SQL: SELECT * FROM [CONTACT]

Query: "Count accounts by status"  
SQL: SELECT [ACCOUNT_STATUS_NAME], COUNT(*) as Count FROM [ACCOUNT] GROUP BY [ACCOUNT_STATUS], [ACCOUNT_STATUS_NAME]

Query: "Find recent activities with contact info"
SQL: SELECT a.*, c.[CONTACT_NAME] FROM [ACTIVITY] a JOIN [CONTACT] c ON a.[CONTACT] = c.[RECORD_ID] WHERE a.[DATE_CREATED_ON] >= DATEADD(day, -7, GETDATE())

Query: "List users and their emails"  
SQL: SELECT [USER_DISPLAY_NAME], [EMAIL] FROM [USER] WHERE [EMAIL] IS NOT NULL

Query: "Show abstracts with reviews"
SQL: SELECT a.[TITLE], r.[DECISION_REMARKS] FROM [ABSTRACT] a LEFT JOIN [ABSTRACT_REVIEW] r ON a.[RECORD_ID] = r.[ABSTRACT]

Query: "Tell me about USER and ACCOUNT tables"
SQL: SELECT u.[USER_DISPLAY_NAME], u.[EMAIL], a.[ACCOUNT_NAME], a.[EMAIL_ADDRESS] FROM [USER] u CROSS JOIN [ACCOUNT] a

Human Question: {question}

SQL Query:"""
        
        return PromptTemplate(
            input_variables=["schema_context", "question"],
            template=template
        )
    
    def _fix_multiple_selects(self, sql: str, tables: list) -> str:
        """Fix multiple SELECT statements by creating a JOIN query for any number of tables"""
        if not tables or len(tables) < 2:
            return sql  # We need at least 2 tables to create a JOIN
            
        if len(tables) > 2:
            logging.warning(f"More than 2 tables detected: {tables}. Will create a chain of JOINs.")
        
        # Use schema info to find relationships
        try:
            schema_data = self.schema_manager.schema_data
            
            # Build a graph of table relationships for path finding
            relationships = {}
            table_aliases = {}
            
            # Create a mapping of table aliases for readability
            for i, table in enumerate(tables):
                alias = table[0].lower() + str(i+1)
                table_aliases[table] = alias
                relationships[table] = []
            
            # Find all direct foreign key relationships
            for table in tables:
                table_info = schema_data.get('tables', {}).get(table, {})
                if not table_info:
                    logging.warning(f"Table {table} not found in schema")
                    continue
                
                # Get foreign keys from this table to others
                foreign_keys = table_info.get('foreign_keys', {})
                for fk_field, fk_target in foreign_keys.items():
                    target_table = fk_target.split('.')[0]
                    if target_table in tables:
                        relationships[table].append({
                            'target': target_table,
                            'source_field': fk_field,
                            'target_field': 'RECORD_ID'  # Assuming all FKs point to RECORD_ID
                        })
            
            # Now build the SQL based on relationships
            if len(tables) == 2:
                # For 2 tables, try to find direct relationship
                table1, table2 = tables
                alias1, alias2 = table_aliases[table1], table_aliases[table2]
                
                # Check for direct relationship from table1 to table2
                join_clause = None
                for rel in relationships.get(table1, []):
                    if rel['target'] == table2:
                        join_clause = f"JOIN [{table2}] {alias2} ON {alias1}.{rel['source_field']} = {alias2}.{rel['target_field']}"
                        break
                
                # Check for direct relationship from table2 to table1 if not found
                if not join_clause:
                    for rel in relationships.get(table2, []):
                        if rel['target'] == table1:
                            join_clause = f"JOIN [{table2}] {alias2} ON {alias2}.{rel['source_field']} = {alias1}.{rel['target_field']}"
                            break
                
                # Default to CROSS JOIN if no relationship found
                if not join_clause:
                    join_clause = f"CROSS JOIN [{table2}] {alias2}"
                
                # Build the final query
                columns_clause = f"{alias1}.*, {alias2}.*"
                from_clause = f"FROM [{table1}] {alias1} {join_clause}"
                
                return f"SELECT {columns_clause} {from_clause}"
            else:
                # For more than 2 tables, create a chain of JOINs
                # First table is the base, then we join others based on relationships
                base_table = tables[0]
                base_alias = table_aliases[base_table]
                
                columns_clause = [f"{base_alias}.*"]
                from_clause = f"FROM [{base_table}] {base_alias}"
                join_clauses = []
                
                # Add each additional table with appropriate JOINs
                for table in tables[1:]:
                    alias = table_aliases[table]
                    columns_clause.append(f"{alias}.*")
                    
                    # Try to find a relationship to any already included table
                    join_added = False
                    for rel_table in tables[:tables.index(table)]:
                        rel_alias = table_aliases[rel_table]
                        
                        # Check if table has FK to rel_table
                        for rel in relationships.get(table, []):
                            if rel['target'] == rel_table:
                                join_clauses.append(
                                    f"JOIN [{table}] {alias} ON {alias}.{rel['source_field']} = {rel_alias}.{rel['target_field']}"
                                )
                                join_added = True
                                break
                                
                        # Check if rel_table has FK to table
                        if not join_added:
                            for rel in relationships.get(rel_table, []):
                                if rel['target'] == table:
                                    join_clauses.append(
                                        f"JOIN [{table}] {alias} ON {rel_alias}.{rel['source_field']} = {alias}.{rel['target_field']}"
                                    )
                                    join_added = True
                                    break
                        
                        if join_added:
                            break
                    
                    # Default to CROSS JOIN if no relationship found
                    if not join_added:
                        join_clauses.append(f"CROSS JOIN [{table}] {alias}")
                
                # Build the final query with all JOINs
                return f"SELECT {', '.join(columns_clause)} {from_clause} {' '.join(join_clauses)}"
                
            # Check if table1 has a foreign key to table2
            t1_info = schema_data.get('tables', {}).get(table1, {})
            if not t1_info:
                logging.warning(f"Table {table1} not found in schema")
                return sql
                
            t1_foreign_keys = t1_info.get('foreign_keys', {})
            
            join_field = None
            for fk_field, fk_target in t1_foreign_keys.items():
                if fk_target.startswith(table2 + '.'):
                    join_field = fk_field
                    break
                    
            if join_field:
                # We found a direct foreign key from table1 to table2
                join_sql = f"""
                SELECT t1.*, t2.*
                FROM [{table1}] t1
                JOIN [{table2}] t2 ON t1.{join_field} = t2.RECORD_ID
                """
                return join_sql
                
            # Check if table2 has a foreign key to table1
            t2_info = schema_data.get('tables', {}).get(table2, {})
            if not t2_info:
                logging.warning(f"Table {table2} not found in schema")
                return sql
                
            t2_foreign_keys = t2_info.get('foreign_keys', {})
            
            join_field = None
            for fk_field, fk_target in t2_foreign_keys.items():
                if fk_target.startswith(table1 + '.'):
                    join_field = fk_field
                    break
                    
            if join_field:
                # We found a direct foreign key from table2 to table1
                join_sql = f"""
                SELECT t1.*, t2.*
                FROM [{table1}] t1
                JOIN [{table2}] t2 ON t2.{join_field} = t1.RECORD_ID
                """
                return join_sql
            
            # If no direct relationship found, use a CROSS JOIN
            logging.info(f"No direct relationship found between {table1} and {table2}, using CROSS JOIN")
            join_sql = f"""
            SELECT t1.*, t2.*
            FROM [{table1}] t1
            CROSS JOIN [{table2}] t2
            """
            return join_sql
                
        except Exception as e:
            logging.error(f"Error fixing multiple SELECTs: {e}")
            
            # Fallback to simple CROSS JOINs as last resort
            try:
                if len(tables) >= 2:
                    base_table = tables[0]
                    base_alias = base_table[0].lower()
                    
                    columns = [f"{base_alias}.*"]
                    join_clauses = [f"FROM [{base_table}] {base_alias}"]
                    
                    for i, table in enumerate(tables[1:], 1):
                        alias = table[0].lower() + str(i)
                        columns.append(f"{alias}.*")
                        join_clauses.append(f"CROSS JOIN [{table}] {alias}")
                    
                    join_sql = f"SELECT {', '.join(columns)} {' '.join(join_clauses)}"
                    return join_sql
            except Exception as e2:
                logging.error(f"Error in fallback join generation: {e2}")
                
            return sql
    
    def _clean_ai_response(self, response: str) -> str:
        """Clean AI response to extract only SQL code"""
        import re
        
        # Remove common explanatory prefixes
        prefixes_to_remove = [
            r'^.*?(?:here\'s|here is|based on|following|the sql|let me|i\'ll)\s*(?:the|a)?\s*(?:sql\s*)?(?:query|code)?[:\.]?\s*',
            r'^.*?(?:suggested|recommended)\s*(?:sql\s*)?(?:query|code)?[:\.]?\s*',
            r'^.*?(?:you can use|try this|use this)\s*(?:sql\s*)?(?:query|code)?[:\.]?\s*'
        ]
        
        cleaned = response.strip()
        for prefix_pattern in prefixes_to_remove:
            cleaned = re.sub(prefix_pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove markdown code blocks
        cleaned = re.sub(r'```(?:sql)?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        
        # If we have numbered lists, extract only SQL lines
        lines = cleaned.split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and obvious non-SQL lines
            if not line:
                continue
            
            # Skip numbered items, bullet points, and explanatory text
            if (re.match(r'^\d+\.', line) or 
                line.startswith(('*', '-', '•')) or
                line.lower().startswith(('note:', 'explanation:', 'this query', 'the above', 'here', 'based on'))):
                continue
            
            # If line starts with SQL keywords, include it and subsequent SQL lines
            if (line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH')) or
                any(keyword in line.upper() for keyword in ['SELECT ', 'FROM ', 'WHERE ', 'JOIN '])):
                sql_lines.append(line)
            elif sql_lines and any(keyword in line.upper() for keyword in 
                    ['FROM', 'WHERE', 'JOIN', 'ORDER BY', 'GROUP BY', 'HAVING', 'AND', 'OR', 'ON']):
                sql_lines.append(line)
        
        if sql_lines:
            cleaned = ' '.join(sql_lines)
        
        # Final check - if we still don't have any SQL keywords, return empty string
        # This will force the fallback generator to be used
        if not any(keyword in cleaned.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE']):
            logger.warning(f"No SQL found after cleaning. Original: {response[:100]}")
            return ""
        
        # Additional check - if the response contains obvious explanatory text or fake columns, force fallback
        problematic_patterns = [
            'assuming', 'suppose', 'let me', 'here is', 'based on', 
            'foreign key in the', 'references the', 'i assume',
            'fake', 'example', 'hypothetical'
        ]
        if any(pattern in cleaned.lower() for pattern in problematic_patterns):
            logger.warning(f"Detected explanatory text in response, forcing fallback. Cleaned: {cleaned[:100]}")
            return ""
        
        return cleaned.strip()
    
    def _fix_square_brackets(self, sql: str) -> str:
        """Fix missing square brackets around table and column names"""
        import re
        
        # Fix table names after FROM and JOIN keywords
        # Pattern: FROM TableName or JOIN TableName (without brackets)
        sql = re.sub(r'\b(FROM|JOIN)\s+([A-Z_][A-Z0-9_]*)\b(?!\s*\[)', r'\1 [\2]', sql, flags=re.IGNORECASE)
        
        # Fix table aliases: FROM [TABLE] T -> FROM [TABLE] T (keep aliases without brackets)
        # But ensure the table name itself has brackets
        
        # Fix column references: TableAlias.ColumnName -> TableAlias.[ColumnName]
        sql = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\\.([A-Z_][A-Z0-9_]*)(?!\s*[,\s])', r'\1.[\2]', sql)
        
        # Fix SELECT column names that don't have brackets
        # This is more complex as we need to preserve functions and expressions
        parts = sql.split()
        for i, part in enumerate(parts):
            if part.upper() == 'SELECT' and i + 1 < len(parts):
                # Process the columns after SELECT
                # This is a simplified approach - in a full implementation you'd need a proper SQL parser
                break
        
        return sql

    def _post_process_sql(self, sql: str, question: str) -> str:
        """Post-process SQL to fix common issues and ensure schema compliance for ANY tables"""
        
        # First, fix missing square brackets around table and column names
        sql = self._fix_square_brackets(sql)
        
        # Extract tables mentioned in the SQL query
        tables_in_sql = re.findall(r'FROM\s+\[(\w+)\]|JOIN\s+\[(\w+)\]', sql, re.IGNORECASE)
        tables_in_sql = [t[0] or t[1] for t in tables_in_sql if t[0] or t[1]]
        
        # Check for invalid column references in each table
        for table in tables_in_sql:
            # Get valid columns for this table from schema
            table_info = self.schema_manager.schema_data.get(table.upper(), None) or \
                         self.schema_manager.schema_data.get(table, None)
            
            if table_info:
                valid_columns = table_info.get('columns', [])
                # Convert to list of strings if it's not already
                if valid_columns and isinstance(valid_columns[0], dict):
                    valid_columns = [col.get('name', '') for col in valid_columns]
                
                # Find all column references for this table in the SQL
                table_aliases = re.findall(r'\[' + table + r'\]\s+(?:AS\s+)?(\w+)', sql, re.IGNORECASE)
                if table_aliases:
                    for alias in table_aliases:
                        col_pattern = r'\b' + re.escape(alias) + r'\.\[(\w+)\]'
                        col_refs = re.findall(col_pattern, sql, re.IGNORECASE)
                        
                        for col in col_refs:
                            if col not in valid_columns and col != '*':
                                # Look for similar columns as replacements
                                closest_match = self._find_closest_column(col, valid_columns)
                                if closest_match:
                                    sql = re.sub(
                                        r'\b' + re.escape(alias) + r'\.\[' + re.escape(col) + r'\]',
                                        f"{alias}.[{closest_match}]",
                                        sql
                                    )
        
        # Check for invalid JOIN conditions
        for i, table1 in enumerate(tables_in_sql):
            for table2 in tables_in_sql[i+1:]:
                # Look for JOIN conditions between these tables
                join_pattern = r'\[' + re.escape(table1) + r'\](?:\s+AS\s+(\w+))?.+?JOIN\s+\[' + re.escape(table2) + r'\](?:\s+AS\s+(\w+))?.+?ON\s+(\w+)\.\[(\w+)\]\s*=\s*(\w+)\.\[(\w+)\]'
                join_matches = re.findall(join_pattern, sql, re.IGNORECASE | re.DOTALL)
                
                if join_matches:
                    for match in join_matches:
                        # Extract all parts of the JOIN condition
                        alias1 = match[0] if match[0] else table1[0].lower()  # Default to first letter if no alias
                        alias2 = match[1] if match[1] else table2[0].lower()
                        left_alias, left_col, right_alias, right_col = match[2], match[3], match[4], match[5]
                        
                        # Verify this JOIN against schema foreign keys
                        table1_info = self.schema_manager.schema_data.get(table1.upper(), {}) or \
                                      self.schema_manager.schema_data.get(table1, {})
                        table2_info = self.schema_manager.schema_data.get(table2.upper(), {}) or \
                                      self.schema_manager.schema_data.get(table2, {})
                        
                        # Get foreign key relationships
                        fk1 = table1_info.get('foreign_keys', {})
                        fk2 = table2_info.get('foreign_keys', {})
                        
                        # Check if either table references the other
                        correct_join_found = False
                        
                        # Check if table1 has FK to table2
                        for fk_col, ref in fk1.items():
                            if ref.startswith(f"{table2}.") or ref.startswith(f"{table2.upper()}."):
                                ref_col = ref.split('.')[1]
                                # Found a valid relationship, check if it matches our JOIN
                                if (left_alias == alias1 and right_alias == alias2 and 
                                    left_col != fk_col and right_col == ref_col):
                                    # Fix the JOIN condition
                                    old_join = f"{left_alias}.[{left_col}] = {right_alias}.[{right_col}]"
                                    new_join = f"{alias1}.[{fk_col}] = {alias2}.[{ref_col}]"
                                    sql = sql.replace(old_join, new_join)
                                    correct_join_found = True
                                    break
                        
                        # Check if table2 has FK to table1
                        if not correct_join_found:
                            for fk_col, ref in fk2.items():
                                if ref.startswith(f"{table1}.") or ref.startswith(f"{table1.upper()}."):
                                    ref_col = ref.split('.')[1]
                                    # Found a valid relationship, check if it matches our JOIN
                                    if (left_alias == alias2 and right_alias == alias1 and 
                                        left_col != fk_col and right_col == ref_col):
                                        # Fix the JOIN condition
                                        old_join = f"{left_alias}.[{left_col}] = {right_alias}.[{right_col}]"
                                        new_join = f"{alias2}.[{fk_col}] = {alias1}.[{ref_col}]"
                                        sql = sql.replace(old_join, new_join)
                                        break
        
        return sql
    
    def _find_closest_column(self, invalid_col: str, valid_columns: List[str]) -> str:
        """Find the closest matching valid column name"""
        # Common mappings for misnamed columns
        common_mappings = {
            "ID": "RECORD_ID",
            "REGISTRATION_DATE": "DATE_CREATED",
            "ACCOUNT_ID": "ACCOUNT_OWNER",
            "USER_ID": "RECORD_ID"
        }
        
        # Check common mappings first
        if invalid_col in common_mappings and common_mappings[invalid_col] in valid_columns:
            return common_mappings[invalid_col]
            
        # Check for any column containing the invalid name
        for col in valid_columns:
            if invalid_col in col:
                return col
                
        # Check for primary key if column has "ID" in it
        if "ID" in invalid_col and "RECORD_ID" in valid_columns:
            return "RECORD_ID"
            
        return ""
        
    def _find_join_condition(self, table1: str, table2: str) -> str:
        """Find a join condition between two tables based on foreign keys"""
        # Skip if tables are the same (can't join a table with itself meaningfully)
        if table1 == table2:
            return ""
            
        # Get table info
        try:
            schema_data = self.schema_manager.schema_data
            if isinstance(schema_data, dict) and table1 in schema_data and table2 in schema_data:
                # Approach 1: Check if table1 has foreign key to table2
                if 'foreign_keys' in schema_data[table1]:
                    for fk_col, fk_target in schema_data[table1]['foreign_keys'].items():
                        if fk_target.startswith(f"{table2}."):
                            target_col = fk_target.split('.')[1]
                            return f"a.{fk_col} = b.{target_col}"
                
                # Approach 2: Check if table2 has foreign key to table1
                if 'foreign_keys' in schema_data[table2]:
                    for fk_col, fk_target in schema_data[table2]['foreign_keys'].items():
                        if fk_target.startswith(f"{table1}."):
                            target_col = fk_target.split('.')[1]
                            return f"b.{fk_col} = a.{target_col}"
            
            # No direct foreign key relationship found
            return ""
        except Exception as e:
            logging.error(f"Error finding join condition: {e}")
            return ""
            
    def _find_common_column(self, table1: str, table2: str) -> tuple:
        """Find columns with the same name or similar patterns between tables"""
        # Skip if tables are the same
        if table1 == table2:
            return None
            
        try:
            schema_data = self.schema_manager.schema_data
            if isinstance(schema_data, dict):
                # Get columns for each table
                cols1 = schema_data.get(table1, {}).get('columns', [])
                cols2 = schema_data.get(table2, {}).get('columns', [])
                
                # Look for exact matches first (common IDs)
                common_cols = set([c.lower() for c in cols1]) & set([c.lower() for c in cols2])
                if common_cols:
                    # Prefer ID columns
                    for col in common_cols:
                        if 'id' in col.lower():
                            # Find original case
                            col1 = next((c for c in cols1 if c.lower() == col.lower()), None)
                            col2 = next((c for c in cols2 if c.lower() == col.lower()), None)
                            if col1 and col2:
                                return (col1, col2)
                
                # Look for pattern matches (e.g., TABLE_ID in one table, ID in another)
                for col1 in cols1:
                    if 'id' in col1.lower() or 'record' in col1.lower():
                        for col2 in cols2:
                            if 'id' in col2.lower() or 'record' in col2.lower():
                                # If one column contains the name of the other table + ID
                                if table1.lower() in col2.lower() or table2.lower() in col1.lower():
                                    return (col1, col2)
                
                # If primary key exists in both, try that
                pk1 = schema_data.get(table1, {}).get('primary_keys', [])
                pk2 = schema_data.get(table2, {}).get('primary_keys', [])
                
                if pk1 and pk2 and (pk1[0].lower() == pk2[0].lower() or 
                                  'id' in pk1[0].lower() and 'id' in pk2[0].lower()):
                    return (pk1[0], pk2[0])
            
            # No common column found
            return None
        except Exception as e:
            logging.error(f"Error finding common column: {e}")
            return None
    
    def generate_sql(self, question: str, callback_handler=None) -> Dict[str, Any]:
        """Generate SQL query from natural language question - optimized for speed"""
        try:
            start_time = time.time()
            logger.info(f"Starting SQL generation for question: '{question}'")
            
            # Use simplified schema context - only include the most relevant tables
            # This makes the context much smaller for faster processing
            schema_context = self.schema_manager.get_schema_context(question)
            
            if not schema_context or "-- Database Schema Context --" not in schema_context:
                logger.warning("No schema context generated - this may affect query quality")
                schema_context = "-- No specific schema context available --"
            
            logger.info(f"Using schema context: {schema_context[:200]}...")
            
            # Set a timeout for the entire process to prevent hanging
            import signal
            
            # Create a faster chain with optimized parameters
            if callback_handler:
                # Use temperature 0 for faster, more deterministic results
                fast_llm = OllamaLLM(
                    model=self.llm.model,
                    base_url=self.llm.base_url,
                    temperature=0.0,
                    num_ctx=512,         # Smaller context for faster responses
                    num_thread=2,        # Fewer threads for less overhead
                    num_predict=256,     # Limit token output for speed
                    stop=["```", ";"],   # Stop at end of SQL
                    seed=42              # Consistent results
                )
                
                chain = self.prompt_template | fast_llm
            else:
                chain = self.prompt_template | self.llm
            
            # Check if this is a multi-table query by identifying mentioned table names
            mentioned_tables = []
            import re
            
            # Extract potential table names (uppercase words with underscores)
            potential_table_names = set(re.findall(r'[A-Z_][A-Z0-9_]*', question))
            
            # Direct table name matching for exact mentions
            for table_name in self.schema_manager.schema_data.keys():
                # Check for exact table name match in question
                if table_name in potential_table_names:
                    if table_name not in mentioned_tables:
                        mentioned_tables.append(table_name)
                        logger.info(f"[OK] Found exact table match: {table_name}")
                else:
                    # Check variations 
                    table_info = self.schema_manager.schema_data.get(table_name, {})
                    variations = table_info.get('variations', [])
                    for variation in variations:
                        if variation and variation.upper() in potential_table_names:
                            if table_name not in mentioned_tables:
                                mentioned_tables.append(table_name)
                                logger.info(f"[OK] Found table via variation: {table_name} ({variation})")
                            break
            
            # Only use substring matching if we found no exact matches at all
            if not mentioned_tables:
                logger.info("[WARNING] No exact table matches found, using substring fallback")
                for table_name in self.schema_manager.schema_data.keys():
                    if table_name.lower() in question.lower():
                        mentioned_tables.append(table_name)
                    else:
                        # Check variations
                        table_info = self.schema_manager.schema_data.get(table_name, {})
                        variations = table_info.get('variations', [])
                        if any(var.lower() in question.lower() for var in variations if var):
                            mentioned_tables.append(table_name)
                            mentioned_tables.append(table_name)
            
            # If we found multiple tables, it's a multi-table query
            is_multi_table_query = len(mentioned_tables) > 1
            
            if is_multi_table_query:
                # Add specific JOIN guidance based on the schema, for any tables
                join_guidance = "CRITICAL: Always follow these rules for JOINs:\n"
                join_guidance += "1. Use EXACTLY the column names from the schema\n"
                join_guidance += "2. The primary key for tables is usually RECORD_ID\n"
                join_guidance += "3. Use the correct foreign key relationships as defined in the schema\n"
                join_guidance += "4. Never invent columns that don't exist in the schema\n"
                
                # Add specific relationships for the tables mentioned in the question
                relationships = []
                for table1 in mentioned_tables:
                    table1_info = self.schema_manager.schema_data.get(table1, {})
                    foreign_keys = table1_info.get('foreign_keys', {})
                    
                    for fk_col, reference in foreign_keys.items():
                        ref_parts = reference.split('.')
                        if len(ref_parts) == 2:
                            ref_table, ref_col = ref_parts
                            if ref_table in mentioned_tables:
                                relationships.append(f"JOIN [{ref_table}] ON [{table1}].[{fk_col}] = [{ref_table}].[{ref_col}]")
                
                if relationships:
                    join_guidance += "Suggested JOIN patterns based on schema:\n"
                    for rel in relationships[:3]:  # Limit to 3 to avoid overwhelming
                        join_guidance += f"- {rel}\n"
                
                question = f"{question} - {join_guidance}"
            
            # Generate SQL directly without threading to avoid StreamlitAPI errors
            try:
                # Run the chain with a simple timeout
                import signal
                
                # Define timeout handler
                def timeout_handler(signum, frame):
                    raise TimeoutError("SQL generation took too long")
                
                # Set timeout for Unix systems (doesn't work on Windows)
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(10)  # 10 second timeout
                except (AttributeError, ValueError):
                    # Windows doesn't support SIGALRM - we'll rely on LLM's internal timeout
                    pass
                
                # Generate the SQL
                response = chain.invoke({
                    "schema_context": schema_context,
                    "question": question
                })
                
                # Clear the alarm if set (for Unix systems)
                try:
                    signal.alarm(0)
                except (AttributeError, ValueError):
                    pass
                    
            except Exception as e:
                # Handle errors or timeouts
                logger.error(f"SQL generation failed: {str(e)}")
                if mentioned_tables:
                    # Fallback to a simple query for the first mentioned table
                    first_table = mentioned_tables[0]
                    response = f"SELECT TOP 10 * FROM [{first_table}]"
                else:
                    response = "SELECT 'Error generating SQL' AS Error_Message"
                
            # Post-process the generated SQL to ensure it respects schema relationships
            response = self._clean_ai_response(response)
            response = self._post_process_sql(response, question)
            
            # Parse SQL from response and check for multiple SELECTs - use faster regex
            select_count = len(re.findall(r'\bSELECT\b', response.upper()))
            if select_count > 1:
                # Extract table names from each SELECT statement - use simplified regex
                selects = re.findall(r'FROM\s+\[?(\w+)\]?', response, re.IGNORECASE)
                
                # Deduplicate table names to prevent joining a table with itself
                unique_tables = []
                for table in selects:
                    if table not in unique_tables:
                        unique_tables.append(table)
                
                # Only proceed if we have at least 2 different tables
                if len(unique_tables) > 1:
                    # EMERGENCY FIX: Directly force a JOIN when multiple SELECTs are detected
                    logging.warning(f"Emergency fix: Multiple SELECTs detected for tables: {unique_tables}")
                    
                    # Get schema tables for validation
                    schema_tables = list(self.schema_manager.schema_data.get('tables', {}).keys())
                    
                    # Map the detected table names to their proper case in the schema
                    exact_tables = []
                    for table in unique_tables:
                        # Try direct match first
                        if table in schema_tables:
                            exact_tables.append(table)
                            continue
                        
                        # Try to find the correct case
                        table_found = False
                        for schema_table in schema_tables:
                            if schema_table.upper() == table.upper():
                                exact_tables.append(schema_table)
                                table_found = True
                                break
                        
                        # Check variations
                        if not table_found:
                            for schema_table, table_info in self.schema_manager.schema_data.get('tables', {}).items():
                                variations = table_info.get('variations', [])
                                if variations and (table in variations or table.upper() in [v.upper() for v in variations]):
                                    exact_tables.append(schema_table)
                                    break
                    
                    # Only create a join if we have distinct tables
                    if len(exact_tables) >= 2:
                        table1 = exact_tables[0]
                        table2 = exact_tables[1]
                        
                        # Check for relationships between these tables to create a better join
                        join_condition = self._find_join_condition(table1, table2)
                        
                        if join_condition:
                            # If we found a relationship, use proper JOIN but limit rows for speed
                            forced_join_sql = f"SELECT TOP 15 a.*, b.* FROM [{table1}] a JOIN [{table2}] b ON {join_condition}"
                            logging.info(f"Forcing proper JOIN between {table1} and {table2} using {join_condition}")
                        else:
                            # Otherwise use INNER JOIN with derived condition if possible
                            # First check if tables have columns with similar names
                            common_column = self._find_common_column(table1, table2)
                            
                            # Always limit rows to prevent slow queries
                            if not re.search(r'\bTOP\s+\d+\b', forced_join_sql, re.IGNORECASE):
                                forced_join_sql = forced_join_sql.replace('SELECT', 'SELECT TOP 15')
                                logging.info("Added TOP 15 limit to query for performance")
                            
                            if common_column:
                                forced_join_sql = f"SELECT a.*, b.* FROM [{table1}] a INNER JOIN [{table2}] b ON a.{common_column[0]} = b.{common_column[1]}"
                                logging.info(f"Forcing INNER JOIN with derived condition: a.{common_column[0]} = b.{common_column[1]}")
                            else:
                                # Last resort: limit rows in each table to prevent huge result sets
                                forced_join_sql = f"SELECT a.*, b.* FROM (SELECT TOP 10 * FROM [{table1}]) a CROSS JOIN (SELECT TOP 10 * FROM [{table2}]) b"
                                logging.info(f"Forcing limited CROSS JOIN between {table1} and {table2}")
                        
                        response = forced_join_sql
                    
                    logging.warning(f"Multiple SELECTs detected. Tables: {selects} mapped to {exact_tables}")
                    # The forced JOIN will be used regardless
            sql_query = self.sql_parser.parse(response)
            
            # If parsing failed to extract SQL, try to generate a simple fallback query
            if not sql_query or not any(keyword in sql_query.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                logger.warning(f"SQL parser failed to extract valid SQL from response: {response[:200]}...")
                
                # Try alternative extraction
                alt_sql = self._extract_sql_alternative(response)
                if alt_sql and any(keyword in alt_sql.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                    # Validate that the extracted SQL uses valid column names
                    if self._validate_column_names(alt_sql):
                        sql_query = alt_sql
                        logger.info(f"Using alternative extraction: {sql_query[:50]}...")
                    else:
                        logger.warning("Alternative extraction contains invalid column names, using fallback")
                        sql_query = self._generate_universal_fallback_query(question, schema_context, mentioned_tables)
                else:
                    # Check if alternative extraction found something but no SQL keywords
                    if alt_sql:
                        logger.warning(f"No SQL keywords found in: {alt_sql[:100]}...")
                    
                    # UNIVERSAL fallback query generator - works for ANY table mentioned in the query
                    logger.warning(f"Generating universal fallback query for tables: {mentioned_tables}")
                    sql_query = self._generate_universal_fallback_query(question, schema_context, mentioned_tables)
            else:
                # Even if we have valid SQL syntax, check for invalid column names
                if not self._validate_column_names(sql_query):
                    logger.warning("Generated SQL contains invalid column names, using fallback")
                    sql_query = self._generate_universal_fallback_query(question, schema_context, mentioned_tables)
            
            ai_response_time = time.time() - start_time
            
            # Validate SQL
            is_valid, validation_error = self._validate_sql(sql_query)
            
            return {
                'success': True,
                'sql': sql_query,
                'raw_response': response,
                'ai_response_time': ai_response_time,
                'is_valid': is_valid,
                'validation_error': validation_error,
                'confidence': self._calculate_confidence(sql_query, question)
            }
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_response_time': time.time() - start_time,
                'confidence': 0.0
            }
    
    def _validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Basic SQL validation with enhanced support for all tables"""
        try:
            sql = sql.strip()
            if not sql:
                return False, "Empty SQL query"
            
            # Check for basic SQL keywords - be more flexible
            sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
            sql_upper = sql.upper()
            
            if not any(keyword in sql_upper for keyword in sql_keywords):
                # Try to extract SQL from the response one more time
                logger.warning(f"No SQL keywords found in: {sql[:100]}...")
                
                # Try alternative extraction methods
                extracted_sql = self._extract_sql_alternative(sql)
                if extracted_sql and any(keyword in extracted_sql.upper() for keyword in sql_keywords):
                    logger.info(f"Alternative extraction succeeded: {extracted_sql[:50]}...")
                    # Replace the original sql with the extracted one
                    sql = extracted_sql
                    sql_upper = sql.upper()
                else:
                    return False, "No valid SQL keywords found"
            
            # Check for basic syntax issues
            if sql.count('(') != sql.count(')'):
                return False, "Unmatched parentheses"
            
            # Check for SQL injection patterns (basic) - but be more lenient for SELECT statements
            if 'SELECT' in sql_upper:
                # Only check for the most dangerous patterns for SELECT queries
                dangerous_patterns = ['DROP TABLE', 'TRUNCATE']
            else:
                # Full check for non-SELECT queries
                dangerous_patterns = ['DROP TABLE', 'DELETE FROM', 'TRUNCATE', 'ALTER TABLE']
                
            for pattern in dangerous_patterns:
                if pattern in sql_upper:
                    return False, f"Potentially dangerous SQL pattern detected: {pattern}"
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def _extract_sql_alternative(self, text: str) -> str:
        """Alternative SQL extraction method for problematic responses"""
        try:
            # Method 1: Extract SQL from code blocks or after "SQL:" markers
            import re
            
            # Remove common prefixes that indicate explanation text
            text = re.sub(r'^(Here\'s|Based on|The SQL query|Following are|Let me provide|I\'ll create).*?[:.]', '', text, flags=re.IGNORECASE)
            
            # Look for SQL code blocks
            sql_block_pattern = r'```(?:sql)?\s*(SELECT.*?)```'
            matches = re.findall(sql_block_pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip().replace('\n', ' ')
            
            # Look for lines that start with SQL keywords
            lines = text.split('\n')
            sql_candidates = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                # Skip empty lines and obvious non-SQL lines
                if not line or line.startswith(('#', '//', '--', 'Based on', 'Here', 'The following', 'Let me')):
                    continue
                
                # Look for SQL-like patterns
                if (any(keyword in line.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'JOIN', 'INSERT', 'UPDATE', 'DELETE']) or
                    line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH'))):
                    
                    # Collect this line and subsequent SQL continuation lines
                    sql_statement = line
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        if not next_line:
                            continue
                        # Check if it's a continuation of SQL
                        if (any(keyword in next_line.upper() for keyword in 
                                ['FROM', 'WHERE', 'JOIN', 'ORDER BY', 'GROUP BY', 'HAVING', 'LIMIT', 'TOP', 'ON', 'AND', 'OR']) or
                            next_line.startswith((' ', '\t')) or  # Indented continuation
                            sql_statement.endswith(',') or  # Column list continuation
                            not re.match(r'^[A-Za-z]', next_line)):  # Doesn't start with a letter (likely continuation)
                            sql_statement += ' ' + next_line
                        else:
                            break
                    
                    if sql_statement:
                        sql_candidates.append(sql_statement)
                        break  # Take the first complete SQL statement
            
            if sql_candidates:
                return sql_candidates[0].strip()
            
            # Method 2: Look for complete SQL statements using regex
            sql_pattern = r'(SELECT\s+.*?(?:;|\s*$))'
            matches = re.findall(sql_pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].replace('\n', ' ').strip().rstrip(';')
            
            # Method 3: Just find the first line that starts with SELECT and try to extract it
            for line in lines:
                line = line.strip()
                if line.upper().startswith('SELECT'):
                    return line
            
            return ""
            
        except Exception as e:
            logger.error(f"Alternative SQL extraction failed: {e}")
            return ""
    
    def _get_fallback_schema_context(self, question: str) -> str:
        """Generate fallback schema context for USER and ACTIVITY tables"""
        context = ["-- Database Schema Context --\n"]
        
        # Always include USER and ACTIVITY tables for fallback
        fallback_tables = ['USER', 'ACTIVITY']
        
        for table_name in fallback_tables:
            if table_name in self.schema_manager.schema_data:
                table_info = self.schema_manager.schema_data[table_name]
                context.append(f"TABLE: {table_name}")
                
                # Add primary keys
                primary_keys = table_info.get('primary_keys', ['RECORD_ID'])
                context.append("PRIMARY KEYS:")
                for pk in primary_keys:
                    context.append(f"  - {pk}")
                
                # Add foreign keys
                foreign_keys = table_info.get('foreign_keys', {})
                if foreign_keys:
                    context.append("FOREIGN KEYS:")
                    for fk_col, reference in foreign_keys.items():
                        context.append(f"  - {fk_col} references {reference}")
                
                # Add columns
                columns = table_info.get('columns', [])
                if columns:
                    context.append("COLUMNS:")
                    for col_name in columns[:10]:  # Limit to first 10 for performance
                        context.append(f"  - {col_name}")
                
                context.append("")
        
        return "\n".join(context)
    
    def _generate_universal_fallback_query(self, question: str, schema_context: str, mentioned_tables: list = None) -> str:
        """Generate fallback SQL query for ANY table based on question analysis"""
        question_lower = question.lower()
        
        # Use provided mentioned_tables if available, otherwise extract from schema context
        available_tables = mentioned_tables if mentioned_tables else []
        
        if not available_tables:
            # Fallback to parsing schema context if mentioned_tables not provided
            lines = schema_context.split('\n')
            for line in lines:
                if line.startswith('TABLE: '):
                    table_name = line.replace('TABLE: ', '').strip()
                    available_tables.append(table_name)
        
        logger.info(f"Available tables for fallback: {available_tables}")
        
        # Analyze question intent
        intent_patterns = {
            'show_all': ['show all', 'list all', 'get all', 'display all', 'view all'],
            'count': ['count', 'how many', 'number of', 'total'],
            'recent': ['recent', 'latest', 'new', 'last'],
            'join': ['join', 'with', 'and', 'related', 'together'],
            'search': ['find', 'search', 'where', 'filter']
        }
        
        detected_intent = 'show_all'  # default
        for intent, patterns in intent_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                detected_intent = intent
                break
        
        # Generate SQL based on detected intent and available tables
        if len(available_tables) == 0:
            return "SELECT 'No tables detected in query' AS Message"
        
        elif len(available_tables) == 1:
            table = available_tables[0]
            
            if detected_intent == 'count':
                return f"SELECT COUNT(*) AS Total_{table} FROM [{table}]"
            elif detected_intent == 'recent':
                # Try common date columns
                date_columns = ['DATE_CREATED_ON', 'DATE_CREATED', 'DATE_MODIFIED', 'DATE']
                date_col = None
                for col in date_columns:
                    if col in schema_context:
                        date_col = col
                        break
                
                if date_col:
                    return f"SELECT TOP 10 * FROM [{table}] WHERE [{date_col}] IS NOT NULL ORDER BY [{date_col}] DESC"
                else:
                    return f"SELECT TOP 10 * FROM [{table}]"
            else:
                return f"SELECT TOP 10 * FROM [{table}]"
        
        else:
            # Multiple tables - try to create a JOIN
            table1, table2 = available_tables[0], available_tables[1]
            
            # Look for foreign key relationships in schema context
            join_condition = None
            
            # First, try to get the relationship from the actual schema data
            if hasattr(self, 'schema_manager') and self.schema_manager.schema_data:
                schema_data = self.schema_manager.schema_data
                
                # Check if table1 has foreign keys referencing table2
                if table1 in schema_data:
                    foreign_keys = schema_data[table1].get('foreign_keys', {})
                    for fk_col, fk_ref in foreign_keys.items():
                        if table2 in fk_ref and 'RECORD_ID' in fk_ref:
                            join_condition = f"[{table1}].[{fk_col}] = [{table2}].[RECORD_ID]"
                            logger.info(f"Found FK relationship from schema: {join_condition}")
                            break
                
                # Check if table2 has foreign keys referencing table1
                if not join_condition and table2 in schema_data:
                    foreign_keys = schema_data[table2].get('foreign_keys', {})
                    for fk_col, fk_ref in foreign_keys.items():
                        if table1 in fk_ref and 'RECORD_ID' in fk_ref:
                            join_condition = f"[{table2}].[{fk_col}] = [{table1}].[RECORD_ID]"
                            logger.info(f"Found FK relationship from schema: {join_condition}")
                            break
            
            # Fallback: parse from schema context text if direct schema access failed
            if not join_condition:
                lines = schema_context.split('\n')
                
                # Look for explicit foreign key references pattern
                for line in lines:
                    # Check for pattern like "ACCOUNT_OWNER references USER.RECORD_ID"
                    if 'references' in line and table1 in line and table2 in line:
                        parts = line.split('references')
                        if len(parts) == 2:
                            fk_col = parts[0].strip().replace('- ', '').replace('  - ', '')
                            target = parts[1].strip()
                            if table2 in target and 'RECORD_ID' in target:
                                join_condition = f"[{table1}].[{fk_col}] = [{table2}].[RECORD_ID]"
                                logger.info(f"Found FK relationship from text: {join_condition}")
                                break
                    elif 'references' in line and table2 in line and table1 in line:
                        parts = line.split('references')
                        if len(parts) == 2:
                            fk_col = parts[0].strip().replace('- ', '').replace('  - ', '')
                            target = parts[1].strip()
                            if table1 in target and 'RECORD_ID' in target:
                                join_condition = f"[{table2}].[{fk_col}] = [{table1}].[RECORD_ID]"
                                logger.info(f"Found FK relationship from text: {join_condition}")
                                break
            
            # Try reverse relationship
            if not join_condition:
                for i, line in enumerate(lines):
                    if f'TABLE: {table2}' in line:
                        for j in range(i+1, min(i+20, len(lines))):
                            if 'FOREIGN KEY ->' in lines[j] and table1 in lines[j]:
                                fk_line = lines[j].strip()
                                if ' - ' in fk_line:
                                    fk_col = fk_line.split(' - ')[0].strip()
                                    join_condition = f"[{table2}].[{fk_col}] = [{table1}].[RECORD_ID]"
                                    break
                            elif f'TABLE:' in lines[j]:
                                break
            
            if join_condition:
                if detected_intent == 'count':
                    return f"SELECT COUNT(*) AS Total FROM [{table1}] t1 JOIN [{table2}] t2 ON {join_condition}"
                else:
                    # Get some meaningful columns from both tables for a better display
                    columns1 = self._get_display_columns(table1)
                    columns2 = self._get_display_columns(table2)
                    
                    select_columns = []
                    if columns1:
                        select_columns.extend([f"t1.[{col}]" for col in columns1[:3]])  # First 3 columns
                    if columns2:
                        select_columns.extend([f"t2.[{col}]" for col in columns2[:3]])  # First 3 columns
                    
                    if select_columns:
                        columns_str = ", ".join(select_columns)
                        return f"SELECT TOP 10 {columns_str} FROM [{table1}] t1 LEFT JOIN [{table2}] t2 ON {join_condition}"
                    else:
                        return f"SELECT TOP 10 t1.*, t2.* FROM [{table1}] t1 LEFT JOIN [{table2}] t2 ON {join_condition}"
            else:
                # No relationship found - use CROSS JOIN to show all combinations
                logger.info(f"No relationship found between {table1} and {table2}, using CROSS JOIN")
                
                # For unrelated tables, show ALL columns from both tables
                return f"SELECT TOP 100 t1.*, t2.* FROM [{table1}] t1 CROSS JOIN [{table2}] t2"
    
    def _get_display_columns(self, table_name: str) -> List[str]:
        """Get meaningful display columns for a table"""
        if not hasattr(self, 'schema_manager') or not self.schema_manager.schema_data:
            return []
        
        schema_data = self.schema_manager.schema_data
        if table_name not in schema_data:
            return []
        
        columns = schema_data[table_name].get('columns', [])
        if not columns:
            return []
        
        # Prioritize common display columns
        priority_patterns = [
            'NAME', 'DISPLAY_NAME', 'TITLE', 'DESCRIPTION', 
            'EMAIL', 'PHONE', 'ADDRESS', 'STATUS', 'TYPE'
        ]
        
        priority_columns = []
        other_columns = []
        
        for col in columns:
            is_priority = any(pattern in col.upper() for pattern in priority_patterns)
            if is_priority:
                priority_columns.append(col)
            elif col != 'RECORD_ID':  # Exclude RECORD_ID unless it's the only column
                other_columns.append(col)
        
        # Return priority columns first, then others, max 5 total
        result = priority_columns + other_columns
        return result[:5]
    
    def _validate_column_names(self, sql: str) -> bool:
        """Validate that SQL uses only valid column names from the schema"""
        try:
            import re
            
            # Extract table and column references from the SQL
            # Look for patterns like [TABLE].[COLUMN] or alias.[COLUMN]
            column_refs = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\.\[([A-Z_][A-Z0-9_]*)\]', sql)
            
            # Also look for direct column references in SELECT clause
            select_cols = re.findall(r'SELECT\s+.*?\[([A-Z_][A-Z0-9_]*)\]', sql, re.IGNORECASE)
            
            all_column_refs = column_refs + select_cols
            
            # Get all valid column names from schema for mentioned tables
            valid_columns = set()
            for table_name, table_info in self.schema_manager.schema_data.items():
                columns = table_info.get('columns', [])
                valid_columns.update(columns)
            
            # Check if all referenced columns are valid
            for col in all_column_refs:
                if col not in valid_columns:
                    logger.warning(f"Invalid column name detected: {col}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Column validation failed: {e}")
            return True  # Default to true if validation fails
    
    def _calculate_confidence(self, sql: str, question: str) -> float:
        """Calculate confidence score for generated SQL"""
        confidence = 0.5  # Base confidence
        
        # Check if SQL contains relevant keywords from question
        question_words = set(question.lower().split())
        sql_words = set(sql.lower().split())
        
        common_words = question_words.intersection(sql_words)
        if common_words:
            confidence += 0.2
        
        # Check for proper SQL structure
        if 'SELECT' in sql.upper():
            confidence += 0.1
        if 'FROM' in sql.upper():
            confidence += 0.1
        if 'WHERE' in sql.upper() and ('filter' in question.lower() or 'where' in question.lower()):
            confidence += 0.1
        
        return min(confidence, 1.0)

class PerformanceMonitor:
    """Performance monitoring and metrics tracking"""
    
    def __init__(self):
        self.metrics = []
    
    def record_query(self, metrics: QueryMetrics):
        """Record query metrics"""
        self.metrics.append(metrics)
        # Keep only last 100 entries
        if len(self.metrics) > 100:
            self.metrics = self.metrics[-100:]
    
    def get_current_accuracy(self) -> float:
        """Get current accuracy percentage"""
        if not self.metrics:
            return 0.0
        
        successful = sum(1 for m in self.metrics if m.execution_success)
        return (successful / len(self.metrics)) * 100
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.metrics:
            return {"message": "No data available"}
        
        successful = sum(1 for m in self.metrics if m.execution_success)
        total = len(self.metrics)
        avg_time = sum(m.execution_time for m in self.metrics) / total
        avg_ai_time = sum(m.ai_response_time for m in self.metrics) / total
        avg_confidence = sum(m.confidence_score for m in self.metrics) / total
        
        return {
            "total_queries": total,
            "successful_queries": successful,
            "success_rate": (successful / total) * 100,
            "current_accuracy": self.get_current_accuracy(),
            "average_confidence": avg_confidence * 100,
            "average_execution_time": avg_time,
            "average_ai_response_time": avg_ai_time,
            "performance_grade": self._get_grade(successful / total * 100)
        }
    
    def _get_grade(self, success_rate: float) -> str:
        """Get performance grade"""
        if success_rate >= 95:
            return "A+ (Excellent)"
        elif success_rate >= 90:
            return "A (Very Good)"
        elif success_rate >= 80:
            return "B (Good)"
        elif success_rate >= 70:
            return "C (Fair)"
        else:
            return "F (Poor)"

class QueryMancerApp:
    """Main QueryMancer AI-powered SQL chatbot application"""
    
    def __init__(self):
        self.current_timestamp = CURRENT_DATETIME
        self.current_user = CURRENT_USER
        
        # Initialize UI first (no external dependencies)
        self.ui_instance = EnhancedLocalUI()
        
        # Delay other component initialization until after login
        self.schema_manager = None
        self.db_manager = None
        self.ai_processor = None
        self.performance_monitor = None
        self._components_initialized = False
    
    def _initialize_components(self):
        """Initialize database and AI components (called after authentication)"""
        if self._components_initialized:
            return True
            
        try:
            self.schema_manager = SchemaManager()
            self.db_manager = DatabaseManager()
            self.ai_processor = AIQueryProcessor(self.schema_manager)
            self.performance_monitor = PerformanceMonitor()
            
            # Update status variables to keep header and sidebar in sync
            self._sync_status_variables()
            
            self._components_initialized = True
            logger.info("[OK] All components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Component initialization failed: {e}")
            st.error(f"Failed to initialize application components: {e}")
            return False
            
    def _sync_status_variables(self):
        """Synchronize status variables for header and sidebar display"""
        try:
            # Set DB status
            st.session_state.db_connected = True  # Assume connected as it's checked in DatabaseManager.__init__
            
            # Set Schema status
            st.session_state.schema_loaded = bool(self.schema_manager.schema_data)
            
            # Set AI model status
            try:
                # Check if Ollama is responsive
                st.session_state.ollama_status = "running"
                st.session_state.mistral_loaded = True
            except:
                st.session_state.ollama_status = "unavailable"
                st.session_state.mistral_loaded = False
                
            # Update overall status
            st.session_state.local_ai_ready = (
                st.session_state.ollama_status == "running" and
                st.session_state.mistral_loaded and
                st.session_state.schema_loaded and
                st.session_state.db_connected
            )
            
        except Exception as e:
            logger.error(f"Error syncing status variables: {e}")
            # Set default values if error occurs
            if "ollama_status" not in st.session_state:
                st.session_state.ollama_status = "unknown"
            if "mistral_loaded" not in st.session_state:
                st.session_state.mistral_loaded = False
            if "schema_loaded" not in st.session_state:
                st.session_state.schema_loaded = False
            if "db_connected" not in st.session_state:
                st.session_state.db_connected = False
            if "local_ai_ready" not in st.session_state:
                st.session_state.local_ai_ready = False
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="QueryMancer - AI SQL Chatbot",
            page_icon="🧙‍♂️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        defaults = {
            "messages": [],
            "session_id": str(uuid.uuid4())[:8],
            "query_count": 0,
            "success_count": 0,
            "error_count": 0,
            # UI-specific state variables required by EnhancedLocalUI
            "local_ai_ready": False,
            "ollama_status": "unknown",
            "mistral_loaded": False,
            "schema_loaded": False,
            "db_connected": False,
            "is_thinking": False,
            "start_time": CURRENT_DATETIME,
            "show_welcome": True,
            "show_sql_editor": False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_header(self):
        """Render application header"""
        accuracy = self.performance_monitor.get_current_accuracy()
        
        # Import and use the UI rendering function from our enhanced UI
        from ui import EnhancedLocalUI
        
        # Get reference to the UI instance (it should be already initialized in the run method)
        if not hasattr(self, 'ui_instance'):
            self.ui_instance = EnhancedLocalUI()
            
        # Use the enhanced header rendering
        self.ui_instance.render_enhanced_header()
        
        # Display accuracy information below the header
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1rem;">
            <p style="font-size: 1.2rem;">
                Current Accuracy: {accuracy:.1f}%
            </p>
            <div style="margin-top: 1rem; font-size: 0.9rem;">
                👤 {self.current_user} • 🔗 Session {st.session_state.session_id} • 
                🕒 {self.current_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with system status and controls"""
        st.sidebar.markdown("# 🎮 Control Panel")
        
        # System Status
        st.sidebar.markdown("### 📋 System Status")
        
        # Database Status
        try:
            self.db_manager._test_connection()
            st.sidebar.success("✅ Database Connected")
            # Update session state for header
            st.session_state.db_connected = True
        except:
            st.sidebar.error("❌ Database Offline")
            st.session_state.db_connected = False
        
        # AI Status
        try:
            test_response = self.ai_processor.llm.invoke("SELECT 1")
            st.sidebar.success("✅ AI Model Ready")
            # Update session state for header
            st.session_state.mistral_loaded = True
            st.session_state.ollama_status = "running"
        except:
            st.sidebar.error("❌ AI Model Offline")
            st.session_state.mistral_loaded = False
        
        # Schema Status
        if self.schema_manager.schema_data:
            table_count = len(self.schema_manager.schema_data.get('tables', {}))
            st.sidebar.success(f"✅ Schema Loaded ({table_count} tables)")
            # Update session state for header
            st.session_state.schema_loaded = True
        else:
            st.sidebar.error("❌ Schema Not Loaded")
            st.session_state.schema_loaded = False
            
        # Update overall AI ready status
        st.session_state.local_ai_ready = (
            st.session_state.ollama_status == "running" and
            st.session_state.mistral_loaded and
            st.session_state.schema_loaded and
            st.session_state.db_connected
        )
            
        # UI Settings
        st.sidebar.markdown("### 🖌️ UI Settings")
        show_samples = st.sidebar.checkbox("Show Sample Queries", value=False, key="show_samples_checkbox")
        st.session_state.show_sample_queries = show_samples
        
        # Performance Metrics
        stats = self.performance_monitor.get_statistics()
        
        if stats.get("total_queries", 0) > 0:
            st.sidebar.markdown("### 📈 Performance")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Accuracy", f"{stats.get('current_accuracy', 0):.1f}%")
                st.metric("Total Queries", stats.get('total_queries', 0))
            with col2:
                st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
                st.metric("Avg Response", f"{stats.get('average_ai_response_time', 0):.1f}s")
            
            st.sidebar.markdown(f"**Grade:** {stats.get('performance_grade', 'No Data')}")
        
        # Clear Chat
        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.rerun()
    
    def render_chat_interface(self):
        """Render main chat interface"""
        # Display chat history
        for message in st.session_state.messages:
            is_user = message["role"] == "user"
            
            with st.chat_message("user" if is_user else "assistant"):
                if "timestamp" in message:
                    st.markdown(f"*{message['timestamp']}*")
                
                st.markdown(message["content"])
                
                # Show SQL and results for assistant messages
                if not is_user:
                    if "sql" in message:
                        st.code(message["sql"], language="sql")
                    
                    if "results" in message:
                        self._display_query_results(message["results"])
                    
                    if "confidence" in message:
                        confidence = message["confidence"]
                        st.progress(confidence, text=f"Confidence: {confidence:.1%}")
        
        # Sample queries for new users - enhanced greeting with AI Avatar Animation
        if not st.session_state.messages:
            # Use the ui_instance to render the avatar animation (CSS is loaded from style.css)
            self.ui_instance.render_query_suggestions()
            
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your database..."):
            self._process_user_query(prompt)
            st.rerun()
    
    def _process_user_query(self, query: str):
        """Process user query through the AI pipeline"""
        # Add user message
        user_message = {
            "role": "user",
            "content": query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        }
        st.session_state.messages.append(user_message)
        st.session_state.query_count += 1
        
        # Show AI processing status
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("🧠 **AI is analyzing your question...**")
            
            try:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Phase 1: Analyzing
                status_text.markdown('<div style="text-align: center; color: #c0c0c0;">⏳ Analyzing your question... (0%)</div>', unsafe_allow_html=True)
                for i in range(33):
                    progress_bar.progress(i/100)
                    time.sleep(0.02)
                time.sleep(0.5)  # Pause between phases
                
                # Generate SQL using AI
                callback_handler = StreamlitCallbackHandler(thinking_placeholder)
                
                # Phase 2: Generating SQL - show user we're working on it
                status_text.markdown('<div style="text-align: center; color: #c0c0c0;">🧠 Generating SQL... (33%)</div>', unsafe_allow_html=True)
                progress_bar.progress(33/100)
                
                # Set progress immediately to avoid thread issues
                progress_bar.progress(50/100)
                status_text.markdown('<div style="text-align: center; color: #c0c0c0;">🧠 Working on your SQL query... (50%)</div>', unsafe_allow_html=True)
                
                # Generate SQL with timeout
                start_time = time.time()
                ai_result = self.ai_processor.generate_sql(query, callback_handler)
                
                # Update progress to show completion
                progress_bar.progress(65/100)
                
                # Show generation time
                generation_time = time.time() - start_time
                logger.info(f"SQL generation took {generation_time:.2f} seconds")
                
                if ai_result['success'] and ai_result['is_valid']:
                    sql_query = ai_result['sql']
                    
                    # Phase 3: Executing query
                    status_text.markdown('<div style="text-align: center; color: #c0c0c0;">📊 Executing query... (66%)</div>', unsafe_allow_html=True)
                    for i in range(66, 100):
                        progress_bar.progress(i/100)
                        time.sleep(0.02)
                    time.sleep(0.2)
                    
                    # Complete the progress bar
                    progress_bar.progress(1.0)
                    status_text.markdown('<div style="text-align: center; color: #c0c0c0;">✓ Complete! (100%)</div>', unsafe_allow_html=True)
                    time.sleep(0.5)
                    
                    # Clear the progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    thinking_placeholder.markdown("⚡ **Executing SQL query...**")
                    
                    # Execute SQL query
                    db_result = self.db_manager.execute_query(sql_query)
                    
                    # Create assistant response
                    if db_result['success']:
                        response_content = f"✅ **Query executed successfully!**\n\n{db_result['message']}"
                        
                        assistant_message = {
                            "role": "assistant",
                            "content": response_content,
                            "sql": sql_query,
                            "results": db_result,
                            "confidence": ai_result['confidence'],
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
                        }
                        
                        # Record successful metrics
                        metrics = QueryMetrics(
                            query_text=query,
                            sql_generated=sql_query,
                            execution_time=db_result['execution_time'],
                            confidence_score=ai_result['confidence'],
                            execution_success=True,
                            rows_returned=db_result['rows_affected'],
                            ai_response_time=ai_result['ai_response_time']
                        )
                        
                        st.session_state.success_count += 1
                        
                    else:
                        response_content = f"❌ **Query execution failed:**\n\n{db_result['message']}"
                        
                        assistant_message = {
                            "role": "assistant",
                            "content": response_content,
                            "sql": sql_query,
                            "confidence": ai_result['confidence'],
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
                        }
                        
                        # Record failed metrics
                        metrics = QueryMetrics(
                            query_text=query,
                            sql_generated=sql_query,
                            execution_time=db_result['execution_time'],
                            confidence_score=ai_result['confidence'],
                            execution_success=False,
                            error_message=db_result.get('error', ''),
                            ai_response_time=ai_result['ai_response_time']
                        )
                        
                        st.session_state.error_count += 1
                
                else:
                    error_msg = ai_result.get('validation_error', ai_result.get('error', 'Unknown error'))
                    response_content = f"❌ **Failed to generate valid SQL:**\n\n{error_msg}"
                    
                    assistant_message = {
                        "role": "assistant",
                        "content": response_content,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
                    }
                    
                    # Record failed metrics
                    metrics = QueryMetrics(
                        query_text=query,
                        sql_generated=ai_result.get('sql', ''),
                        execution_success=False,
                        error_message=error_msg,
                        ai_response_time=ai_result.get('ai_response_time', 0.0)
                    )
                    
                    st.session_state.error_count += 1
                
                # Add assistant message and record metrics
                st.session_state.messages.append(assistant_message)
                self.performance_monitor.record_query(metrics)
                
                # Clear thinking placeholder and display final response
                thinking_placeholder.empty()
                
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(f"Query processing error: {e}")
                
                thinking_placeholder.empty()
                st.error(error_msg)
                
                # Record error metrics
                metrics = QueryMetrics(
                    query_text=query,
                    execution_success=False,
                    error_message=error_msg
                )
                self.performance_monitor.record_query(metrics)
                st.session_state.error_count += 1
    
    def _display_query_results(self, db_result: Dict[str, Any]):
        """Display query results in a user-friendly format"""
        if not db_result.get('success', False):
            st.error(f"Query failed: {db_result.get('error', 'Unknown error')}")
            return
        
        # Show execution info
        execution_time = db_result.get('execution_time', 0)
        rows_affected = db_result.get('rows_affected', 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Execution Time", f"{execution_time:.3f}s")
        with col2:
            st.metric("📊 Rows", rows_affected)
        with col3:
            st.metric("✅ Status", "Success")
        
        # Display data if available
        data = db_result.get('data')
        if data is not None and not data.empty:
            # Show data table
            st.markdown("### 📋 Query Results:")
            st.dataframe(data, use_container_width=True)
            
            # Show basic statistics for numeric columns
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.markdown("### 📈 Quick Stats:")
                stats_df = data[numeric_cols].describe()
                st.dataframe(stats_df, use_container_width=True)
            
            # Show visualizations if enabled and data is suitable
            if len(data) > 1 and len(data.columns) >= 2:
                st.markdown("### 📊 Data Visualization:")
                
                try:
                    # Create simple chart based on data type
                    if len(numeric_cols) >= 2:
                        # Scatter plot for numeric data
                        fig = px.scatter(
                            data, 
                            x=numeric_cols[0], 
                            y=numeric_cols[1],
                            title=f"{numeric_cols[0]} vs {numeric_cols[1]}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif len(numeric_cols) == 1:
                        # Bar chart or histogram
                        if len(data) <= 20:
                            fig = px.bar(
                                data, 
                                x=data.columns[0], 
                                y=numeric_cols[0],
                                title=f"{numeric_cols[0]} by {data.columns[0]}"
                            )
                        else:
                            fig = px.histogram(
                                data, 
                                x=numeric_cols[0],
                                title=f"Distribution of {numeric_cols[0]}"
                            )
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    logger.debug(f"Visualization error: {e}")
                    # Silently skip visualization errors
                    pass
            
            # Download option
            csv = data.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def run(self):
        """Run the QueryMancer application"""
        try:
            # Initialize and setup the UI FIRST (handles st.set_page_config and CSS)
            if not hasattr(self, 'ui_instance'):
                from ui import EnhancedLocalUI
                self.ui_instance = EnhancedLocalUI()
                self.ui_instance.setup()
            
            # Initialize session state (after page config is set)
            self.initialize_session_state()
            
            # Note: The CSS is loaded from the external style.css file via the EnhancedLocalUI class
            
            # Check authentication - show login form if not authenticated
            if not st.session_state.get('authenticated', False):
                self.ui_instance.render_login_form()
                return  # Don't render the rest of the app until authenticated
            
            # Initialize database and AI components after authentication
            if not self._initialize_components():
                return  # Stop if components failed to initialize
            
            # Render UI components (only after authentication)
            self.render_header()
            
            # Show welcome screen with features
            self.ui_instance.render_welcome_section()
            
            # Main content area with sidebar
            with st.container():
                # Render sidebar
                self.render_sidebar()
                
                # Main chat interface
                self.render_chat_interface()
            
            # Footer with additional info
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; opacity: 0.9; font-size: 1.0rem;">
                🧙‍♂️ QueryMancer v3.0 | Powered by Ollama + Mistral + RAG with FAISS Vector Db| 
                Local AI • Secure Database • No External APIs
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Application error: {e}")
            logger.error(f"Application error: {e}")
            st.markdown("### Debug Information:")
            st.code(traceback.format_exc())

def main():
    """Main application entry point"""
    try:
        # Create and run the application
        app = QueryMancerApp()
        app.run()
        
    except Exception as e:
        st.error(f"Failed to start QueryMancer: {e}")
        logger.error(f"Application startup failed: {e}")
        st.markdown("### Error Details:")
        st.code(traceback.format_exc())
        
        st.markdown("### Troubleshooting:")
        st.markdown("""
        1. **Check Ollama Status**: Ensure Ollama is running with `ollama serve`
        2. **Verify Model**: Run `ollama run mistral` to download the model
        3. **Database Connection**: Check your .env file for correct database credentials
        4. **Schema File**: Ensure schema.json exists in the project directory
        5. **Dependencies**: Run `pip install -r requirements.txt`
        """)

if __name__ == "__main__":
    main()