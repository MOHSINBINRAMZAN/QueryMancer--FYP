"""
Enhanced utils.py for Querymancer - Local AI-Powered SQL Chatbot
Compatible with existing project structure and focused on local Ollama + Mistral integration
"""

import json
import re
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import difflib
from fuzzywuzzy import fuzz, process
import pandas as pd
from contextlib import contextmanager

# Import from your existing __init__.py
from . import (
    app_config, logger, SchemaManager, ModelConfig, SQLServerConfig,
    DatabaseError, SQLExecutionError, SchemaError, AccuracyError
)

# Define custom exceptions
class ModelError(Exception):
    """Exception raised for model-related errors"""
    pass

@dataclass
class QueryResult:
    """Enhanced query result container"""
    success: bool
    data: Optional[pd.DataFrame] = None
    sql_query: str = ""
    execution_time: float = 0.0
    row_count: int = 0
    error_message: str = ""
    confidence_score: float = 0.0
    suggested_fixes: List[str] = field(default_factory=list)
    schema_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "success": self.success,
            "sql_query": self.sql_query,
            "execution_time": self.execution_time,
            "row_count": self.row_count,
            "error_message": self.error_message,
            "confidence_score": self.confidence_score,
            "suggested_fixes": self.suggested_fixes,
            "data": self.data.to_dict('records') if self.data is not None else None,
            "columns": list(self.data.columns) if self.data is not None else [],
            "schema_context": self.schema_context
        }

class SchemaContextBuilder:
    """Enhanced schema context builder for local LLM prompting"""
    
    def __init__(self, schema_manager: SchemaManager):
        self.schema_manager = schema_manager
        self._table_cache = {}
        self._column_cache = {}
        
    def get_relevant_context(self, query: str, max_tables: int = 5) -> Dict[str, Any]:
        """Get relevant schema context based on natural language query"""
        try:
            schema = self.schema_manager.load_schema()
            
            # Find relevant tables using fuzzy matching and keywords
            relevant_tables = self._find_relevant_tables(query, schema, max_tables)
            
            # Build comprehensive context
            context = {
                "database_name": schema.get('database_name', 'Unknown'),
                "relevant_tables": {},
                "table_relationships": {},
                "suggested_joins": [],
                "query_hints": []
            }
            
            for table_name in relevant_tables:
                table_info = self.schema_manager.get_table_info(table_name)
                if table_info:
                    context["relevant_tables"][table_name] = self._format_table_context(table_info)
                    context["table_relationships"][table_name] = self.schema_manager.get_table_relationships(table_name)
            
            # Add suggested joins if multiple tables are involved
            if len(relevant_tables) > 1:
                context["suggested_joins"] = self._suggest_joins(relevant_tables)
            
            # Add query-specific hints
            context["query_hints"] = self._generate_query_hints(query, relevant_tables)
            
            return context
            
        except Exception as e:
            logger.error(f"Error building schema context: {e}")
            return {"error": str(e)}
    
    def _find_relevant_tables(self, query: str, schema: Dict[str, Any], max_tables: int) -> List[str]:
        """Find relevant tables using multiple matching strategies"""
        query_lower = query.lower()
        tables = list(schema.get('tables', {}).keys())
        
        # Strategy 1: Direct table name matches
        direct_matches = []
        for table in tables:
            if table.lower() in query_lower or any(
                alias.lower() in query_lower 
                for alias in schema['tables'][table].get('aliases', [])
            ):
                direct_matches.append(table)
        
        # Strategy 2: Natural language matches
        nl_matches = self.schema_manager.find_tables_by_natural_language(query)
        
        # Strategy 3: Column name matches
        column_matches = []
        for table in tables:
            table_info = schema['tables'][table]
            for column in table_info.get('columns', {}):
                if column.lower() in query_lower:
                    column_matches.append(table)
                    break
        
        # Strategy 4: Fuzzy matching for table names
        fuzzy_matches = []
        table_names = [t.lower() for t in tables]
        query_words = re.findall(r'\b\w+\b', query_lower)
        
        for word in query_words:
            matches = process.extractBests(word, table_names, limit=2, score_cutoff=70)
            for match, score in matches:
                original_table = next(t for t in tables if t.lower() == match)
                fuzzy_matches.append((original_table, score))
        
        # Combine and rank matches
        all_matches = set()
        
        # Prioritize direct matches
        all_matches.update(direct_matches)
        
        # Add natural language matches
        all_matches.update(nl_matches)
        
        # Add column matches
        all_matches.update(column_matches)
        
        # Add fuzzy matches (sorted by score)
        fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
        for table, score in fuzzy_matches[:max_tables]:
            all_matches.add(table)
        
        # If no matches found, return most commonly referenced tables
        if not all_matches:
            # Return tables with most columns or relationships as fallback
            table_weights = []
            for table in tables:
                weight = len(schema['tables'][table].get('columns', {}))
                weight += len(schema['tables'][table].get('foreign_keys', []))
                table_weights.append((table, weight))
            
            table_weights.sort(key=lambda x: x[1], reverse=True)
            all_matches.update([t[0] for t in table_weights[:max_tables]])
        
        return list(all_matches)[:max_tables]
    
    def _format_table_context(self, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format table information for LLM context"""
        formatted = {
            "description": table_info.get('description', ''),
            "columns": {},
            "primary_keys": table_info.get('primary_keys', []),
            "indexes": table_info.get('indexes', [])
        }
        
        # Format column information
        for col_name, col_info in table_info.get('columns', {}).items():
            if isinstance(col_info, dict):
                formatted["columns"][col_name] = {
                    "type": col_info.get('type', 'Unknown'),
                    "nullable": col_info.get('nullable', True),
                    "description": col_info.get('description', ''),
                    "primary_key": col_info.get('primary_key', False),
                    "foreign_key": col_info.get('foreign_key', False),
                    "unique": col_info.get('unique', False),
                    "default": col_info.get('default', None)
                }
            else:
                formatted["columns"][col_name] = {
                    "type": str(col_info),
                    "nullable": True,
                    "description": ""
                }
        
        return formatted
    
    def _suggest_joins(self, tables: List[str]) -> List[Dict[str, str]]:
        """Suggest JOIN clauses based on foreign key relationships"""
        joins = []
        
        for table in tables:
            relationships = self.schema_manager.get_table_relationships(table)
            
            for fk in relationships.get('foreign_keys', []):
                if isinstance(fk, dict):
                    ref_table = fk.get('references', '').split('.')[0]
                    if ref_table in tables:
                        joins.append({
                            "type": "INNER JOIN",
                            "from_table": table,
                            "to_table": ref_table,
                            "condition": f"{table}.{fk.get('column')} = {fk.get('references')}"
                        })
        
        return joins
    
    def _generate_query_hints(self, query: str, tables: List[str]) -> List[str]:
        """Generate query-specific hints based on analysis"""
        hints = []
        query_lower = query.lower()
        
        # Aggregation hints
        if any(word in query_lower for word in ['total', 'sum', 'count', 'average', 'max', 'min']):
            hints.append("Consider using GROUP BY for aggregations")
            hints.append("Use appropriate aggregate functions (SUM, COUNT, AVG, MAX, MIN)")
        
        # Date/time hints
        if any(word in query_lower for word in ['date', 'year', 'month', 'day', 'time', '2024', '2023']):
            hints.append("Use date functions like YEAR(), MONTH(), or date ranges in WHERE clause")
            hints.append("Consider using BETWEEN for date ranges")
        
        # Sorting hints
        if any(word in query_lower for word in ['sort', 'order', 'latest', 'recent', 'oldest']):
            hints.append("Use ORDER BY clause for sorting results")
            hints.append("Use DESC for descending order, ASC for ascending")
        
        # Filtering hints
        if any(word in query_lower for word in ['where', 'filter', 'completed', 'active', 'status']):
            hints.append("Use WHERE clause for filtering conditions")
            hints.append("Consider using IN operator for multiple values")
        
        # Performance hints
        if len(tables) > 2:
            hints.append("Consider query performance with multiple table joins")
            hints.append("Use appropriate indexes for better performance")
        
        return hints

class SQLValidator:
    """Enhanced SQL validator with schema awareness"""
    
    def __init__(self, schema_manager: SchemaManager):
        self.schema_manager = schema_manager
        
    def validate_query(self, sql: str) -> Tuple[bool, List[str], float]:
        """Validate SQL query against schema and return confidence score"""
        errors = []
        warnings = []
        confidence = 1.0
        
        try:
            # Basic SQL syntax validation
            if not self._basic_syntax_check(sql):
                errors.append("Invalid SQL syntax")
                confidence *= 0.1
            
            # Schema validation
            schema_errors, schema_confidence = self._validate_against_schema(sql)
            errors.extend(schema_errors)
            confidence *= schema_confidence
            
            # Security validation
            security_errors = self._security_check(sql)
            errors.extend(security_errors)
            if security_errors:
                confidence *= 0.0  # Security issues are critical
            
            # Performance warnings
            perf_warnings = self._performance_check(sql)
            warnings.extend(perf_warnings)
            
            is_valid = len(errors) == 0
            all_issues = errors + warnings
            
            return is_valid, all_issues, confidence
            
        except Exception as e:
            logger.error(f"Error validating SQL: {e}")
            return False, [f"Validation error: {str(e)}"], 0.0
    
    def _basic_syntax_check(self, sql: str) -> bool:
        """Basic SQL syntax validation"""
        sql = sql.strip()
        
        # Check for empty query
        if not sql:
            return False
        
        # Check for balanced parentheses
        if sql.count('(') != sql.count(')'):
            return False
        
        # Check for basic SQL structure
        sql_upper = sql.upper()
        if not any(sql_upper.startswith(cmd) for cmd in ['SELECT', 'WITH']):
            return False
        
        # Check for basic required clauses in SELECT
        if sql_upper.startswith('SELECT'):
            if 'FROM' not in sql_upper:
                return False
        
        return True
    
    def _validate_against_schema(self, sql: str) -> Tuple[List[str], float]:
        """Validate SQL against loaded schema"""
        errors = []
        confidence = 1.0
        
        try:
            schema = self.schema_manager.load_schema()
            tables = set(schema.get('tables', {}).keys())
            
            # Extract table names from SQL
            used_tables = self._extract_table_names(sql)
            
            # Check if tables exist
            missing_tables = []
            for table in used_tables:
                if table not in tables:
                    # Try fuzzy matching
                    matches = difflib.get_close_matches(table, tables, n=1, cutoff=0.8)
                    if matches:
                        errors.append(f"Table '{table}' not found. Did you mean '{matches[0]}'?")
                        confidence *= 0.7
                    else:
                        missing_tables.append(table)
                        confidence *= 0.3
            
            if missing_tables:
                errors.append(f"Unknown tables: {', '.join(missing_tables)}")
            
            # Check column names for existing tables
            valid_tables = [t for t in used_tables if t in tables]
            for table in valid_tables:
                column_errors, col_confidence = self._validate_columns(sql, table, schema['tables'][table])
                errors.extend(column_errors)
                confidence *= col_confidence
            
            return errors, confidence
            
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return [f"Schema validation error: {str(e)}"], 0.5
    
    def _extract_table_names(self, sql: str) -> List[str]:
        """Extract table names from SQL query"""
        tables = []
        
        # Remove comments and normalize whitespace
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        sql = ' '.join(sql.split())
        
        # Extract FROM clause tables
        from_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        from_matches = re.findall(from_pattern, sql, re.IGNORECASE)
        tables.extend(from_matches)
        
        # Extract JOIN clause tables
        join_pattern = r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        join_matches = re.findall(join_pattern, sql, re.IGNORECASE)
        tables.extend(join_matches)
        
        # Remove duplicates and return
        return list(set(tables))
    
    def _validate_columns(self, sql: str, table_name: str, table_schema: Dict[str, Any]) -> Tuple[List[str], float]:
        """Validate column references for a specific table"""
        errors = []
        confidence = 1.0
        
        try:
            available_columns = set(table_schema.get('columns', {}).keys())
            
            # Extract column references (simplified approach)
            # This is a basic implementation - could be enhanced with proper SQL parsing
            sql_upper = sql.upper()
            table_upper = table_name.upper()
            
            # Look for table.column references
            pattern = rf'\b{re.escape(table_upper)}\.([a-zA-Z_][a-zA-Z0-9_]*)'
            column_refs = re.findall(pattern, sql, re.IGNORECASE)
            
            for col_ref in column_refs:
                if col_ref.lower() not in [c.lower() for c in available_columns]:
                    # Try fuzzy matching
                    matches = difflib.get_close_matches(col_ref, available_columns, n=1, cutoff=0.8)
                    if matches:
                        errors.append(f"Column '{table_name}.{col_ref}' not found. Did you mean '{matches[0]}'?")
                        confidence *= 0.8
                    else:
                        errors.append(f"Unknown column: {table_name}.{col_ref}")
                        confidence *= 0.6
            
            return errors, confidence
            
        except Exception as e:
            return [f"Column validation error: {str(e)}"], 0.7
    
    def _security_check(self, sql: str) -> List[str]:
        """Check for potential security issues"""
        errors = []
        sql_upper = sql.upper()
        
        # Check for dangerous keywords
        dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE',
            'EXEC', 'EXECUTE', 'SP_', 'XP_'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                errors.append(f"Dangerous keyword detected: {keyword}")
        
        # Check for SQL injection patterns
        injection_patterns = [
            r"['\"];.*--",  # Comment injection
            r"UNION\s+SELECT",  # Union injection
            r";\s*\w+",  # Multiple statements
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                errors.append("Potential SQL injection pattern detected")
        
        return errors
    
    def _performance_check(self, sql: str) -> List[str]:
        """Check for potential performance issues"""
        warnings = []
        sql_upper = sql.upper()
        
        # Check for SELECT *
        if re.search(r'\bSELECT\s+\*', sql, re.IGNORECASE):
            warnings.append("Consider specifying columns instead of using SELECT *")
        
        # Check for missing WHERE clause on large tables
        if 'FROM' in sql_upper and 'WHERE' not in sql_upper and 'LIMIT' not in sql_upper:
            warnings.append("Consider adding WHERE clause to limit results")
        
        # Check for Cartesian products (missing JOIN conditions)
        from_count = len(re.findall(r'\bFROM\b', sql_upper))
        join_count = len(re.findall(r'\bJOIN\b', sql_upper))
        comma_joins = len(re.findall(r',\s*[a-zA-Z_]', sql))
        
        total_tables = from_count + join_count + comma_joins
        if total_tables > 1 and 'ON' not in sql_upper and 'WHERE' not in sql_upper:
            warnings.append("Possible Cartesian product - ensure proper JOIN conditions")
        
        return warnings

class QueryOptimizer:
    """Query optimizer with schema awareness"""
    
    def __init__(self, schema_manager: SchemaManager):
        self.schema_manager = schema_manager
    
    def optimize_query(self, sql: str) -> Tuple[str, List[str]]:
        """Optimize SQL query and return optimized version with suggestions"""
        suggestions = []
        optimized_sql = sql
        
        try:
            # Add LIMIT if missing and no aggregation
            if not self._has_aggregation(sql) and not self._has_limit(sql):
                optimized_sql = self._add_default_limit(optimized_sql)
                suggestions.append("Added LIMIT clause for better performance")
            
            # Suggest indexes based on WHERE clauses
            index_suggestions = self._suggest_indexes(sql)
            suggestions.extend(index_suggestions)
            
            # Optimize JOIN order
            optimized_joins, join_suggestions = self._optimize_joins(optimized_sql)
            optimized_sql = optimized_joins
            suggestions.extend(join_suggestions)
            
            return optimized_sql, suggestions
            
        except Exception as e:
            logger.error(f"Query optimization error: {e}")
            return sql, [f"Optimization error: {str(e)}"]
    
    def _has_aggregation(self, sql: str) -> bool:
        """Check if query has aggregation functions"""
        agg_functions = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'GROUP BY']
        sql_upper = sql.upper()
        return any(func in sql_upper for func in agg_functions)
    
    def _has_limit(self, sql: str) -> bool:
        """Check if query has LIMIT clause"""
        return 'LIMIT' in sql.upper() or 'TOP' in sql.upper()
    
    def _add_default_limit(self, sql: str) -> str:
        """Add default LIMIT to query"""
        # For SQL Server, use TOP instead of LIMIT
        if sql.upper().startswith('SELECT'):
            # Insert TOP after SELECT
            return re.sub(
                r'^(\s*SELECT)\s+',
                r'\1 TOP 1000 ',
                sql,
                flags=re.IGNORECASE
            )
        return sql
    
    def _suggest_indexes(self, sql: str) -> List[str]:
        """Suggest indexes based on WHERE clause analysis"""
        suggestions = []
        
        # Extract WHERE conditions
        where_match = re.search(r'\bWHERE\b(.*?)(?:\bORDER\s+BY\b|\bGROUP\s+BY\b|\bHAVING\b|$)', 
                               sql, re.IGNORECASE | re.DOTALL)
        
        if where_match:
            where_clause = where_match.group(1)
            
            # Extract column references in WHERE clause
            column_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\b'
            columns = re.findall(column_pattern, where_clause)
            
            if columns:
                suggestions.append(f"Consider indexes on: {', '.join(set(columns))}")
        
        return suggestions
    
    def _optimize_joins(self, sql: str) -> Tuple[str, List[str]]:
        """Optimize JOIN order and suggest improvements"""
        suggestions = []
        
        # This is a simplified optimization
        # In a production system, you'd use query statistics and cardinality estimates
        
        # Count number of JOINs
        join_count = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
        
        if join_count > 3:
            suggestions.append("Consider breaking complex joins into smaller queries or using CTEs")
        
        # Check for proper JOIN conditions
        if 'JOIN' in sql.upper() and sql.count('ON') < join_count:
            suggestions.append("Ensure all JOINs have proper ON conditions")
        
        return sql, suggestions

class ResultFormatter:
    """Format query results for display"""
    
    @staticmethod
    def format_dataframe(df: pd.DataFrame, max_rows: int = 100) -> Dict[str, Any]:
        """Format DataFrame for display with metadata"""
        if df is None or df.empty:
            return {
                "data": [],
                "columns": [],
                "row_count": 0,
                "message": "No data returned"
            }
        
        # Limit rows for display
        display_df = df.head(max_rows) if len(df) > max_rows else df
        
        # Convert to display format
        formatted_data = []
        for _, row in display_df.iterrows():
            formatted_row = {}
            for col in display_df.columns:
                value = row[col]
                if pd.isna(value):
                    formatted_row[col] = None
                elif isinstance(value, (datetime, pd.Timestamp)):
                    formatted_row[col] = value.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(value, (float, int)):
                    formatted_row[col] = float(value) if isinstance(value, float) else int(value)
                else:
                    formatted_row[col] = str(value)
            formatted_data.append(formatted_row)
        
        return {
            "data": formatted_data,
            "columns": list(display_df.columns),
            "row_count": len(df),
            "displayed_rows": len(display_df),
            "truncated": len(df) > max_rows,
            "column_types": {col: str(df[col].dtype) for col in df.columns}
        }
    
    @staticmethod
    def format_error(error: Exception, sql: str = "") -> Dict[str, Any]:
        """Format error for display"""
        return {
            "success": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "sql_query": sql,
            "timestamp": datetime.now().isoformat()
        }

class QueryCache:
    """Simple query result cache"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._access_times = {}
    
    def get_key(self, sql: str) -> str:
        """Generate cache key for SQL query"""
        return hashlib.md5(sql.encode()).hexdigest()
    
    def get(self, sql: str) -> Optional[QueryResult]:
        """Get cached result"""
        key = self.get_key(sql)
        
        if key not in self._cache:
            return None
        
        # Check TTL
        if time.time() - self._access_times[key] > self.ttl_seconds:
            del self._cache[key]
            del self._access_times[key]
            return None
        
        # Update access time
        self._access_times[key] = time.time()
        return self._cache[key]
    
    def set(self, sql: str, result: QueryResult) -> None:
        """Cache query result"""
        # Only cache successful results
        if not result.success:
            return
        
        key = self.get_key(sql)
        
        # Cleanup old entries if cache is full
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._access_times.keys(), 
                           key=lambda k: self._access_times[k])
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
        
        self._cache[key] = result
        self._access_times[key] = time.time()
    
    def clear(self) -> None:
        """Clear cache"""
        self._cache.clear()
        self._access_times.clear()

# Utility functions for common operations

def clean_sql_query(sql: str) -> str:
    """Clean and normalize SQL query"""
    if not sql:
        return ""
    
    # Remove extra whitespace
    sql = ' '.join(sql.split())
    
    # Remove trailing semicolon
    sql = sql.rstrip(';')
    
    # Ensure proper capitalization of keywords
    keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 
               'OUTER', 'ON', 'GROUP', 'BY', 'ORDER', 'HAVING', 'LIMIT', 'TOP']
    
    for keyword in keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        sql = re.sub(pattern, keyword, sql, flags=re.IGNORECASE)
    
    return sql

def extract_sql_from_response(response: str) -> str:
    """Extract SQL query from LLM response"""
    # Try to find SQL in code blocks
    sql_block_pattern = r'```(?:sql|SQL)?\s*(.*?)\s*```'
    matches = re.findall(sql_block_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return clean_sql_query(matches[0])
    
    # Try to find SELECT statements
    select_pattern = r'(SELECT\b.*?)(?:\n\n|$)'
    matches = re.findall(select_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return clean_sql_query(matches[0])
    
    # Return cleaned response as fallback
    return clean_sql_query(response)

def validate_connection_string(conn_string: str) -> bool:
    """Validate SQL Server connection string format"""
    required_parts = ['DRIVER=', 'SERVER=', 'DATABASE=']
    return all(part in conn_string.upper() for part in required_parts)

def get_table_sample_data(schema_manager: SchemaManager, table_name: str, 
                         limit: int = 5) -> Dict[str, Any]:
    """Get sample data structure for a table (for prompting)"""
    try:
        table_info = schema_manager.get_table_info(table_name)
        if not table_info:
            return {}
        
        sample = {
            "table_name": table_name,
            "columns": [],
            "sample_values": {}
        }
        
        for col_name, col_info in table_info.get('columns', {}).items():
            if isinstance(col_info, dict):
                col_type = col_info.get('type', 'Unknown')
                sample["columns"].append({
                    "name": col_name,
                    "type": col_type,
                    "nullable": col_info.get('nullable', True)
                })
                
                # Generate sample values based on type
                sample["sample_values"][col_name] = _generate_sample_value(col_type)
        
        return sample
        
    except Exception as e:
        logger.error(f"Error generating table sample: {e}")
        return {}

def _generate_sample_value(col_type: str) -> str:
    """Generate sample value based on column type"""
    col_type_lower = col_type.lower()
    
    if 'int' in col_type_lower:
        return "123"
    elif 'varchar' in col_type_lower or 'char' in col_type_lower:
        return "'sample_text'"
    elif 'date' in col_type_lower:
        return "'2024-01-15'"
    elif 'datetime' in col_type_lower:
        return "'2024-01-15 10:30:00'"
    elif 'decimal' in col_type_lower or 'float' in col_type_lower:
        return "123.45"
    elif 'bit' in col_type_lower or 'boolean' in col_type_lower:
        return "1"
    else:
        return "'sample_value'"

# Export all utility classes and functions
__all__ = [
    'QueryResult',
    'SchemaContextBuilder', 
    'SQLValidator',
    'QueryOptimizer',
    'ResultFormatter',
    'QueryCache',
    'clean_sql_query',
    'extract_sql_from_response',
    'validate_connection_string',
    'get_table_sample_data'
]

class DatabaseConnectionManager:
    """Enhanced database connection manager with connection pooling"""
    
    def __init__(self, config: SQLServerConfig):
        self.config = config
        self._connection_pool = []
        self._pool_lock = threading.Lock()
        self._max_connections = config.POOL_SIZE
        
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup"""
        import pyodbc
        connection = None
        try:
            connection = pyodbc.connect(
                self.config.get_connection_string(),
                timeout=self.config.CONNECTION_TIMEOUT
            )
            connection.timeout = self.config.COMMAND_TIMEOUT
            yield connection
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise DatabaseError(f"Failed to connect to database: {str(e)}")
        finally:
            if connection:
                try:
                    connection.close()
                except:
                    pass
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT @@VERSION, DB_NAME()")
                result = cursor.fetchone()
                version_info = result[0][:100] if result and result[0] else "Unknown"
                db_name = result[1] if result and result[1] else "Unknown"
                return True, f"Connected to {db_name}: {version_info}"
        except Exception as e:
            return False, str(e)

class LocalLLMManager:
    """Manager for local Ollama LLM interactions"""
    
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self._client = None
        
    def initialize(self) -> bool:
        """Initialize Ollama client"""
        try:
            from langchain_ollama import OllamaLLM
            
            base_url = self.config.base_url or "http://localhost:11434"
            
            self._client = OllamaLLM(
                model=self.config.name,
                temperature=self.config.temperature,
                base_url=base_url,
                timeout=self.config.timeout
            )
            
            # Test the connection
            response = self._client.invoke("SELECT 1")
            logger.info(f"Ollama client initialized successfully with model: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            return False
    
    def generate_sql(self, prompt: str) -> Tuple[str, float]:
        """Generate SQL from natural language prompt"""
        if not self._client:
            if not self.initialize():
                raise ModelError("Failed to initialize Ollama client")
        
        try:
            start_time = time.time()
            
            response = self._client.invoke(prompt)
            
            execution_time = time.time() - start_time
            
            # Extract SQL from response
            sql_query = extract_sql_from_response(response)
            
            logger.info(f"SQL generated in {execution_time:.2f}s")
            return sql_query, execution_time
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise ModelError(f"Failed to generate SQL: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            import requests
            base_url = self.config.base_url or "http://localhost:11434"
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

class PromptTemplateManager:
    """Manages SQL generation prompts with schema context"""
    
    def __init__(self, schema_manager: SchemaManager):
        self.schema_manager = schema_manager
        
    def create_sql_prompt(self, user_query: str, schema_context: Dict[str, Any], 
                         conversation_history: List[Dict] = None) -> str:
        """Create comprehensive SQL generation prompt"""
        
        # Base system prompt for SQL generation
        system_prompt = """You are an expert SQL Server database analyst. Your task is to convert natural language questions into precise, executable SQL Server queries.

CRITICAL REQUIREMENTS:
1. Generate ONLY valid SQL Server (T-SQL) syntax
2. Use the provided schema information accurately
3. Always include proper JOIN conditions when using multiple tables
4. Use appropriate WHERE clauses for filtering
5. Include ORDER BY for sorting when requested
6. Use TOP N instead of LIMIT for SQL Server
7. Handle date/datetime comparisons properly with SQL Server functions
8. Return only the SQL query without explanations unless specifically asked

SAFETY RULES:
- Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)
- Use parameterized approaches where possible
- Avoid dynamic SQL construction
- Always validate table and column names against the schema"""

        # Schema context section
        schema_section = self._build_schema_section(schema_context)
        
        # Query examples section
        examples_section = self._build_examples_section(schema_context)
        
        # User query section
        query_section = f"""
USER QUESTION: {user_query}

Based on the schema above, generate a precise SQL Server query that answers this question.
Remember to:
- Use exact table and column names from the schema
- Include appropriate JOINs if multiple tables are needed
- Add WHERE clauses for filtering
- Use proper SQL Server syntax (TOP instead of LIMIT, etc.)
- Format the query for readability

SQL Query:"""

        # Combine all sections
        full_prompt = f"{system_prompt}\n\n{schema_section}\n\n{examples_section}\n\n{query_section}"
        
        return full_prompt
    
    def _build_schema_section(self, schema_context: Dict[str, Any]) -> str:
        """Build schema information section"""
        if not schema_context or "relevant_tables" not in schema_context:
            return "DATABASE SCHEMA: No schema information available."
        
        schema_parts = [
            f"DATABASE: {schema_context.get('database_name', '146_36156520-AC21-435A-9C9B-1EC9145A9090')}",
            "\nTABLE SCHEMAS:"
        ]
        
        for table_name, table_info in schema_context["relevant_tables"].items():
            schema_parts.append(f"\nTable: {table_name}")
            
            if table_info.get("description"):
                schema_parts.append(f"Description: {table_info['description']}")
            
            schema_parts.append("Columns:")
            for col_name, col_info in table_info.get("columns", {}).items():
                col_type = col_info.get("type", "Unknown")
                flags = []
                
                if col_info.get("primary_key"):
                    flags.append("PK")
                if col_info.get("foreign_key"):
                    flags.append("FK")
                if not col_info.get("nullable", True):
                    flags.append("NOT NULL")
                if col_info.get("unique"):
                    flags.append("UNIQUE")
                
                flag_str = f" ({', '.join(flags)})" if flags else ""
                desc = f" - {col_info.get('description')}" if col_info.get('description') else ""
                
                schema_parts.append(f"  - {col_name}: {col_type}{flag_str}{desc}")
            
            # Add primary keys
            if table_info.get("primary_keys"):
                schema_parts.append(f"Primary Keys: {', '.join(table_info['primary_keys'])}")
        
        # Add relationship information
        if schema_context.get("suggested_joins"):
            schema_parts.append("\nSUGGESTED JOINS:")
            for join in schema_context["suggested_joins"]:
                schema_parts.append(f"  {join['from_table']} {join['type']} {join['to_table']} ON {join['condition']}")
        
        return "\n".join(schema_parts)
    
    def _build_examples_section(self, schema_context: Dict[str, Any]) -> str:
        """Build query examples section"""
        if not schema_context.get("relevant_tables"):
            return ""
        
        table_names = list(schema_context["relevant_tables"].keys())
        
        examples = [
            "QUERY EXAMPLES:",
            f"-- Get all records from a table:",
            f"SELECT TOP 100 * FROM {table_names[0]} ORDER BY {table_names[0]}_id;",
            "",
            f"-- Filter with WHERE clause:",
            f"SELECT * FROM {table_names[0]} WHERE status = 'active';",
            "",
            f"-- Aggregate functions:",
            f"SELECT COUNT(*) as total_count FROM {table_names[0]};",
            ""
        ]
        
        # Add JOIN example if multiple tables
        if len(table_names) > 1:
            examples.extend([
                f"-- JOIN multiple tables:",
                f"SELECT t1.*, t2.* FROM {table_names[0]} t1",
                f"INNER JOIN {table_names[1]} t2 ON t1.id = t2.{table_names[0]}_id;",
                ""
            ])
        
        # Add date filtering example
        examples.extend([
            "-- Date filtering (SQL Server):",
            f"SELECT * FROM {table_names[0]} WHERE created_date >= '2024-01-01'",
            f"  AND created_date < DATEADD(year, 1, '2024-01-01');",
            ""
        ])
        
        return "\n".join(examples)
    
    def create_error_recovery_prompt(self, original_query: str, error_message: str, 
                                   schema_context: Dict[str, Any]) -> str:
        """Create prompt for error recovery and SQL correction"""
        
        prompt = f"""The following SQL query generated an error. Please analyze and correct it:

ORIGINAL QUERY:
{original_query}

ERROR MESSAGE:
{error_message}

SCHEMA CONTEXT:
{self._build_schema_section(schema_context)}

Please provide a corrected SQL query that:
1. Fixes the specific error mentioned above
2. Uses correct table and column names from the schema
3. Follows proper SQL Server syntax
4. Maintains the original intent of the query

CORRECTED SQL Query:"""

        return prompt

class QueryExecutor:
    """Enhanced query executor with comprehensive error handling"""
    
    def __init__(self, db_manager: DatabaseConnectionManager, 
                 validator: SQLValidator, cache: QueryCache):
        self.db_manager = db_manager
        self.validator = validator
        self.cache = cache
        
    def execute_query(self, sql: str, use_cache: bool = True) -> QueryResult:
        """Execute SQL query with comprehensive error handling and caching"""
        
        # Check cache first
        if use_cache:
            cached_result = self.cache.get(sql)
            if cached_result:
                logger.info("Returning cached query result")
                return cached_result
        
        # Create result object
        result = QueryResult(success=False, sql_query=sql)
        
        try:
            start_time = time.time()
            
            # Validate query
            is_valid, issues, confidence = self.validator.validate_query(sql)
            result.confidence_score = confidence
            
            if not is_valid:
                result.error_message = f"Query validation failed: {'; '.join(issues)}"
                result.suggested_fixes = issues
                return result
            
            # Execute query
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql(sql, conn)
                
                result.data = df
                result.row_count = len(df)
                result.execution_time = time.time() - start_time
                result.success = True
                
                logger.info(f"Query executed successfully: {result.row_count} rows in {result.execution_time:.2f}s")
                
                # Cache successful result
                if use_cache:
                    self.cache.set(sql, result)
                
                return result
                
        except Exception as e:
            result.execution_time = time.time() - start_time
            result.error_message = str(e)
            
            # Analyze error and provide suggestions
            result.suggested_fixes = self._analyze_error(str(e), sql)
            
            logger.error(f"Query execution failed: {e}")
            return result
    
    def _analyze_error(self, error_message: str, sql: str) -> List[str]:
        """Analyze error message and provide helpful suggestions"""
        suggestions = []
        error_lower = error_message.lower()
        
        # Common SQL Server errors
        if "invalid object name" in error_lower:
            suggestions.append("Check table name spelling and ensure table exists")
            suggestions.append("Verify you have access to the specified table")
            
        elif "invalid column name" in error_lower:
            suggestions.append("Check column name spelling")
            suggestions.append("Verify column exists in the specified table")
            
        elif "syntax error" in error_lower:
            suggestions.append("Check SQL syntax - missing commas, parentheses, or keywords")
            suggestions.append("Verify proper use of SQL Server T-SQL syntax")
            
        elif "timeout" in error_lower:
            suggestions.append("Query took too long - consider adding WHERE clause to limit results")
            suggestions.append("Check if appropriate indexes exist")
            
        elif "permission" in error_lower or "access" in error_lower:
            suggestions.append("Insufficient permissions to access the requested data")
            suggestions.append("Contact administrator for proper database access")
            
        elif "connection" in error_lower:
            suggestions.append("Database connection issue - check network connectivity")
            suggestions.append("Verify database server is running and accessible")
        
        # Add generic suggestions if no specific error patterns matched
        if not suggestions:
            suggestions.append("Review query syntax and table/column names")
            suggestions.append("Check database connection and permissions")
        
        return suggestions

class ConversationManager:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history = []
        
    def add_interaction(self, user_query: str, sql_query: str, result: QueryResult):
        """Add interaction to conversation history"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "sql_query": sql_query,
            "success": result.success,
            "row_count": result.row_count,
            "execution_time": result.execution_time,
            "error_message": result.error_message if not result.success else None
        }
        
        self.history.append(interaction)
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_recent_context(self, count: int = 3) -> List[Dict[str, Any]]:
        """Get recent conversation context"""
        return self.history[-count:] if len(self.history) >= count else self.history
    
    def clear_history(self):
        """Clear conversation history"""
        self.history.clear()
        
    def export_history(self) -> Dict[str, Any]:
        """Export conversation history"""
        return {
            "exported_at": datetime.now().isoformat(),
            "total_interactions": len(self.history),
            "history": self.history
        }

# Threading support
import threading

class AsyncQueryExecutor:
    """Asynchronous query executor for non-blocking operations"""
    
    def __init__(self, query_executor: QueryExecutor):
        self.executor = query_executor
        self._executor_pool = threading.ThreadPoolExecutor(max_workers=3)
        
    def execute_async(self, sql: str, callback=None) -> threading.Future:
        """Execute query asynchronously"""
        future = self._executor_pool.submit(self.executor.execute_query, sql)
        
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
            
        return future
    
    def shutdown(self):
        """Shutdown async executor"""
        self._executor_pool.shutdown(wait=True)

# Performance monitoring
class PerformanceMonitor:
    """Monitor query performance and system metrics"""
    
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_execution_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self._execution_times = []
        
    def record_query(self, result: QueryResult, from_cache: bool = False):
        """Record query execution metrics"""
        self.metrics["total_queries"] += 1
        
        if result.success:
            self.metrics["successful_queries"] += 1
            self._execution_times.append(result.execution_time)
            
            # Update average execution time
            self.metrics["average_execution_time"] = sum(self._execution_times) / len(self._execution_times)
        else:
            self.metrics["failed_queries"] += 1
            
        if from_cache:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.metrics.copy()
        
        if stats["total_queries"] > 0:
            stats["success_rate"] = stats["successful_queries"] / stats["total_queries"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_queries"]
        else:
            stats["success_rate"] = 0.0
            stats["cache_hit_rate"] = 0.0
            
        return stats
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_execution_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self._execution_times.clear()

# Main utility class that orchestrates everything
class QueryMancerUtils:
    """Main utility class that orchestrates all components"""
    
    def __init__(self, config=None):
        # Use global config if none provided
        self.config = config or app_config
        
        # Initialize core components
        self.schema_manager = self.config.schema_manager
        self.context_builder = SchemaContextBuilder(self.schema_manager)
        self.validator = SQLValidator(self.schema_manager)
        self.optimizer = QueryOptimizer(self.schema_manager)
        self.cache = QueryCache(max_size=100, ttl_seconds=300)
        self.db_manager = DatabaseConnectionManager(self.config.sqlserver_config)
        self.llm_manager = LocalLLMManager(self.config.model_config)
        self.prompt_manager = PromptTemplateManager(self.schema_manager)
        self.query_executor = QueryExecutor(self.db_manager, self.validator, self.cache)
        self.conversation_manager = ConversationManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize async executor
        self.async_executor = AsyncQueryExecutor(self.query_executor)
        
        logger.info("QueryMancerUtils initialized successfully")
    
    def process_natural_language_query(self, user_query: str) -> QueryResult:
        """Main method to process natural language query and return results"""
        try:
            # Get schema context
            schema_context = self.context_builder.get_relevant_context(user_query)
            
            # Create prompt
            prompt = self.prompt_manager.create_sql_prompt(user_query, schema_context)
            
            # Generate SQL
            sql_query,generation_time = self.llm_manager.generate_sql(prompt)
            
            if not sql_query:
                return QueryResult(
                    success=False,
                    error_message="Failed to generate SQL query",
                    sql_query=""
                )
            
            # Optimize query
            optimized_sql, optimization_suggestions = self.optimizer.optimize_query(sql_query)
            
            # Execute query
            result = self.query_executor.execute_query(optimized_sql)
            result.schema_context = schema_context
            
            # Add optimization suggestions to result
            if optimization_suggestions:
                result.suggested_fixes.extend(optimization_suggestions)
            
            # Record interaction
            self.conversation_manager.add_interaction(user_query, optimized_sql, result)
            self.performance_monitor.record_query(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return QueryResult(
                success=False,
                error_message=f"Processing error: {str(e)}",
                sql_query=""
            )
    
    def retry_with_error_correction(self, user_query: str, original_result: QueryResult) -> QueryResult:
        """Retry query with error correction"""
        try:
            schema_context = self.context_builder.get_relevant_context(user_query)
            
            # Create error recovery prompt
            correction_prompt = self.prompt_manager.create_error_recovery_prompt(
                original_result.sql_query, 
                original_result.error_message,
                schema_context
            )
            
            # Generate corrected SQL
            corrected_sql, _ = self.llm_manager.generate_sql(correction_prompt)
            
            if corrected_sql:
                # Execute corrected query
                result = self.query_executor.execute_query(corrected_sql)
                result.schema_context = schema_context
                
                # Record interaction
                self.conversation_manager.add_interaction(user_query, corrected_sql, result)
                self.performance_monitor.record_query(result)
                
                return result
            
            return original_result
            
        except Exception as e:
            logger.error(f"Error in retry attempt: {e}")
            return original_result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "database_connection": self.db_manager.test_connection(),
            "llm_available": self.llm_manager.is_available(),
            "schema_info": self._get_schema_info(),
            "performance_stats": self.performance_monitor.get_statistics(),
            "cache_size": len(self.cache._cache),
            "conversation_history_length": len(self.conversation_manager.history),
            "config": {
                "model": self.config.model_config.name,
                "database": self.config.sqlserver_config.DATABASE,
                "rag_enabled": self.config.rag_config.ENABLED,
                "target_accuracy": self.config.accuracy_config.TARGET_ACCURACY
            }
        }
    
    def shutdown(self):
        """Gracefully shutdown all components"""
        try:
            self.async_executor.shutdown()
            self.cache.clear()
            logger.info("QueryMancerUtils shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            
    def _get_schema_info(self) -> Dict[str, Any]:
        """Get summary information about the loaded schema"""
        try:
            schema = self.schema_manager.load_schema()
            
            return {
                "database_name": schema.get('database_name', 'Unknown'),
                "table_count": len(schema.get('tables', {})),
                "loaded": bool(schema),
                "last_updated": schema.get('last_updated', 'Unknown'),
                "table_names": list(schema.get('tables', {}).keys())[:10]  # List first 10 tables
            }
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            return {"error": str(e), "loaded": False}

# Create global instance for easy access
query_mancer_utils = QueryMancerUtils()

# Updated exports
__all__ = [
    'QueryResult',
    'SchemaContextBuilder', 
    'SQLValidator',
    'QueryOptimizer',
    'ResultFormatter',
    'QueryCache',
    'DatabaseConnectionManager',
    'LocalLLMManager',
    'PromptTemplateManager',
    'QueryExecutor',
    'ConversationManager',
    'AsyncQueryExecutor',
    'PerformanceMonitor',
    'QueryMancerUtils',
    'query_mancer_utils',
    'clean_sql_query',
    'extract_sql_from_response',
    'validate_connection_string',
    'get_table_sample_data'
]

if __name__ == "__main__":
    # Test the utilities
    print("🔧 Testing QueryMancer Utils...")
    
    # Test system status
    status = query_mancer_utils.get_system_status()
    print(f"System Status: {json.dumps(status, indent=2)}")
    
    # Test a simple query
    test_query = "Show me all users"
    print(f"\nTesting query: {test_query}")
    
    result = query_mancer_utils.process_natural_language_query(test_query)
    print(f"Result: Success={result.success}, Rows={result.row_count}")
    
    if not result.success:
        print(f"Error: {result.error_message}")
        print(f"Suggestions: {result.suggested_fixes}")
    
    print("✅ Utils testing completed!")