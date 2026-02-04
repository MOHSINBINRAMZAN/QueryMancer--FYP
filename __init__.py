"""
Querymancer - Local AI-Powered SQL Query Generation Tool

Enhanced for local Ollama + Mistral integration with manual schema injection
for secure, high-accuracy SQL generation without external API dependencies.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import os
import socket
import platform
import time
import json
import hashlib
from dotenv import load_dotenv

# Load .env file
load_dotenv()

__version__ = "2.0.0"
__author__ = "MOHSINBINRAMZAN"
__email__ = "mohsin.ramzan@oomi.co.uk"
__description__ = "Local AI-Powered SQL Server Query Generation Tool with Ollama + Mistral"

MINIMUM_PYTHON_VERSION = (3, 8)

def check_python_version() -> bool:
    return sys.version_info >= MINIMUM_PYTHON_VERSION

if not check_python_version():
    raise RuntimeError(
        f"Python {'.'.join(map(str, MINIMUM_PYTHON_VERSION))} or higher is required. "
        f"Current version: {'.'.join(map(str, sys.version_info[:2]))}"
    )

# === Model and DB Configuration ===

class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    LOCAL = "local"

@dataclass
class ModelConfig:
    name: str = "mistral"
    temperature: float = 0.05  # Very low for precise SQL generation
    provider: ModelProvider = ModelProvider.OLLAMA
    max_tokens: int = 4000
    context_window: int = 8192
    description: str = "Mistral via Ollama - Local SQL generation"
    timeout: int = 60
    base_url: str = "http://localhost:11434"
    supports_streaming: bool = True
    sql_focused: bool = True
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0 <= self.temperature <= 2):
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
            
        # Set default model kwargs for Ollama
        if not self.model_kwargs:
            self.model_kwargs = {
                "num_ctx": self.context_window,
                "temperature": self.temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

@dataclass
class SchemaConfig:
    """Configuration for manual schema injection"""
    SCHEMA_DIR: str = "schemas"
    SCHEMA_FILE: str = "schema.json"  # Changed to match your file
    AUTO_RELOAD: bool = True
    CACHE_SCHEMA: bool = True
    VALIDATE_SCHEMA: bool = True
    SCHEMA_VERSION: str = "1.0"
    NATURAL_LANGUAGE_MAPPINGS: bool = True
    FUZZY_MATCHING: bool = True
    FUZZY_THRESHOLD: float = 0.8
    COLUMN_TYPE_VALIDATION: bool = True
    FOREIGN_KEY_VALIDATION: bool = True
    
    @property
    def schema_path(self) -> Path:
        return Path(self.SCHEMA_DIR) / self.SCHEMA_FILE
    
    def to_dict(self) -> Dict[str, Any]:
        return {k.lower(): v for k, v in self.__dict__.items() if not k.startswith('_')}

@dataclass
class SQLServerConfig:
    """Configuration for AWS SQL Server connections"""
    HOST: str = field(default_factory=lambda: os.getenv("DB_HOST", "10.0.0.45"))
    DATABASE: str = field(default_factory=lambda: os.getenv("DB_NAME", "146_36156520-AC21-435A-9C9B-1EC9145A9090"))
    USERNAME: str = field(default_factory=lambda: os.getenv("DB_USERNAME", "usr_mohsin"))
    PASSWORD: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", "blY|5K:3pe10"))
    DRIVER: str = field(default_factory=lambda: os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server"))
    CONNECTION_TIMEOUT: int = 60
    COMMAND_TIMEOUT: int = 60
    PORT: int = 1433
    TRUST_CERT: bool = True
    POOL_SIZE: int = 5
    MAX_OVERFLOW: int = 10
    POOL_TIMEOUT: int = 30
    POOL_RECYCLE: int = 3600

    def get_connection_string(self) -> str:
        """Get ODBC connection string for AWS SQL Server"""
        return (
            f"DRIVER={{{self.DRIVER}}};"
            f"SERVER={self.HOST},{self.PORT};"
            f"DATABASE={self.DATABASE};"
            f"UID={self.USERNAME};"
            f"PWD={self.PASSWORD};"
            f"TrustServerCertificate={'yes' if self.TRUST_CERT else 'no'};"
            f"Connection Timeout={self.CONNECTION_TIMEOUT};"
            f"Command Timeout={self.COMMAND_TIMEOUT};"
        )
        
    def get_sqlalchemy_uri(self) -> str:
        """Get SQLAlchemy connection URI for AWS SQL Server"""
        from urllib.parse import quote_plus
        
        # URL-encode password to handle special characters
        encoded_password = quote_plus(self.PASSWORD)
        driver_encoded = self.DRIVER.replace(' ', '+')
        
    
        trust_cert_param = "yes" if self.TRUST_CERT else "no"
        
        return (
            f"mssql+pyodbc://{self.USERNAME}:{encoded_password}@"
            f"{self.HOST}:{self.PORT}/{self.DATABASE}?"
            f"driver={driver_encoded}&"
            f"TrustServerCertificate={trust_cert_param}&"
            f"connection_timeout={self.CONNECTION_TIMEOUT}&"
            f"command_timeout={self.COMMAND_TIMEOUT}"
        )
        
    def get_pool_params(self) -> Dict[str, Any]:
        """Get connection pooling parameters"""
        return {
            "pool_size": self.POOL_SIZE,
            "max_overflow": self.MAX_OVERFLOW,
            "pool_timeout": self.POOL_TIMEOUT,
            "pool_recycle": self.POOL_RECYCLE,
            "pool_pre_ping": True
        }

@dataclass
class SQLGuardConfig:
    """Enhanced SQL safety and validation with schema awareness"""
    ENABLED: bool = True
    MAX_ROWS: int = 1000
    MAX_EXECUTION_TIME: int = 30  # Increased for AWS latency
    ALLOWED_QUERY_TYPES: List[str] = field(default_factory=lambda: ["SELECT", "WITH"])
    BLOCKED_KEYWORDS: List[str] = field(default_factory=lambda: [
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE",
        "EXEC", "sp_", "xp_", "sys.", "SHUTDOWN", "RESTORE", "BACKUP"
    ])
    REGEX_PATTERNS: Dict[str, str] = field(default_factory=lambda: {
        "comment_after_semicolon": r";.*--",
        "multiple_statements": r";\s*\w+",
        "system_objects": r"\bsys\.\w+",
        "extended_procedures": r"\bxp_\w+",
        "dynamic_sql": r"EXEC\s*\(\s*@"
    })
    VALIDATE_AGAINST_SCHEMA: bool = True
    STRICT_COLUMN_VALIDATION: bool = True
    STRICT_TABLE_VALIDATION: bool = True
    AUTO_FIX_COLUMN_NAMES: bool = True

@dataclass
class AccuracyConfig:
    """Configuration for improving SQL generation accuracy"""
    TARGET_ACCURACY: float = 0.99
    ENABLE_FUZZY_MATCHING: bool = True
    FUZZY_THRESHOLD: float = 0.85
    ENABLE_SEMANTIC_SEARCH: bool = True
    ENABLE_COLUMN_ALIASING: bool = True
    ENABLE_TABLE_ALIASING: bool = True
    VALIDATE_JOINS: bool = True
    SUGGEST_CORRECTIONS: bool = True
    ENABLE_QUERY_OPTIMIZATION: bool = True
    CONFIDENCE_THRESHOLD: float = 0.9
    FALLBACK_TO_SCHEMA: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {k.lower(): v for k, v in self.__dict__.items()}

# === Enhanced Schema Management ===

class SchemaManager:
    """Manages database schema loading, caching, and validation for local AI SQL generation"""
    
    def __init__(self, config: SchemaConfig):
        self.config = config
        self._schema_cache: Optional[Dict[str, Any]] = None
        self._schema_hash: Optional[str] = None
        self._last_load_time: Optional[float] = None
        self._table_mappings: Dict[str, str] = {}
        self._column_mappings: Dict[str, Dict[str, str]] = {}
        self._natural_language_mappings: Dict[str, List[str]] = {}
        
    def load_schema(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load schema from JSON file with caching"""
        schema_path = self.config.schema_path
        
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        # Check if we need to reload
        if not force_reload and self._schema_cache is not None:
            if not self.config.AUTO_RELOAD:
                return self._schema_cache
                
            # Check if file has been modified
            current_hash = self._get_file_hash(schema_path)
            if current_hash == self._schema_hash:
                return self._schema_cache
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
            
            if self.config.VALIDATE_SCHEMA:
                self._validate_schema(schema_data)
            
            self._schema_cache = schema_data
            self._schema_hash = self._get_file_hash(schema_path)
            self._last_load_time = time.time()
            
            # Build mappings for faster lookups
            self._build_mappings(schema_data)
            
            logger.info(f"Schema loaded successfully from {schema_path}")
            return schema_data
            
        except Exception as e:
            logger.error(f"Error loading schema: {e}")
            raise
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of file for change detection"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _validate_schema(self, schema: Dict[str, Any]) -> None:
        """Validate schema structure"""
        required_keys = ['tables']
        for key in required_keys:
            if key not in schema:
                raise ValueError(f"Missing required key in schema: {key}")
        
        if not isinstance(schema['tables'], dict):
            raise ValueError("'tables' must be a dictionary")
        
        for table_name, table_info in schema['tables'].items():
            self._validate_table_schema(table_name, table_info)
    
    def _validate_table_schema(self, table_name: str, table_info: Dict[str, Any]) -> None:
        """Validate individual table schema"""
        required_keys = ['columns']
        for key in required_keys:
            if key not in table_info:
                raise ValueError(f"Missing required key '{key}' in table '{table_name}'")
        
        if not isinstance(table_info['columns'], dict):
            raise ValueError(f"'columns' must be a dictionary in table '{table_name}'")
    
    def _build_mappings(self, schema: Dict[str, Any]) -> None:
        """Build lookup mappings for faster access"""
        self._table_mappings = {}
        self._column_mappings = {}
        self._natural_language_mappings = {}
        
        for table_name, table_info in schema['tables'].items():
            # Table mappings (case-insensitive)
            self._table_mappings[table_name.lower()] = table_name
            
            # Add table aliases if available
            if 'aliases' in table_info:
                for alias in table_info['aliases']:
                    self._table_mappings[alias.lower()] = table_name
            
            # Column mappings
            self._column_mappings[table_name] = {}
            for col_name, col_info in table_info['columns'].items():
                self._column_mappings[table_name][col_name.lower()] = col_name
                
                # Add column aliases/variations if available
                if isinstance(col_info, dict) and 'aliases' in col_info:
                    for alias in col_info['aliases']:
                        self._column_mappings[table_name][alias.lower()] = col_name
                
                # Add variations if available
                if isinstance(col_info, dict) and 'variations' in col_info:
                    for variation in col_info['variations']:
                        self._column_mappings[table_name][variation.lower()] = col_name
            
            # Natural language mappings
            if 'natural_language' in table_info:
                nl_terms = table_info['natural_language']
                if isinstance(nl_terms, list):
                    self._natural_language_mappings[table_name] = [term.lower() for term in nl_terms]
    
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific table"""
        if not self._schema_cache:
            self.load_schema()
        
        # Try exact match first
        if table_name in self._schema_cache['tables']:
            return self._schema_cache['tables'][table_name]
        
        # Try case-insensitive match
        table_key = self._table_mappings.get(table_name.lower())
        if table_key:
            return self._schema_cache['tables'][table_key]
        
        return None
    
    def get_column_info(self, table_name: str, column_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific column"""
        table_info = self.get_table_info(table_name)
        if not table_info:
            return None
        
        columns = table_info.get('columns', {})
        
        # Try exact match first
        if column_name in columns:
            return columns[column_name]
        
        # Try case-insensitive match through mappings
        if table_name in self._column_mappings:
            col_key = self._column_mappings[table_name].get(column_name.lower())
            if col_key and col_key in columns:
                return columns[col_key]
        
        return None
    
    def find_tables_by_natural_language(self, query: str) -> List[str]:
        """Find tables based on natural language description"""
        query_lower = query.lower()
        matches = []
        
        # Direct table name matches
        for table_name in self._table_mappings.keys():
            if table_name in query_lower:
                actual_table = self._table_mappings[table_name]
                if actual_table not in matches:
                    matches.append(actual_table)
        
        # Natural language term matches
        for table_name, nl_terms in self._natural_language_mappings.items():
            for term in nl_terms:
                if term in query_lower and table_name not in matches:
                    matches.append(table_name)
                    break
        
        return matches
    
    def get_all_tables(self) -> List[str]:
        """Get list of all table names"""
        if not self._schema_cache:
            self.load_schema()
        return list(self._schema_cache['tables'].keys())
    
    def get_table_relationships(self, table_name: str) -> Dict[str, Any]:
        """Get foreign key relationships for a table"""
        table_info = self.get_table_info(table_name)
        if not table_info:
            return {}
        
        return {
            'foreign_keys': table_info.get('foreign_keys', []),
            'primary_keys': table_info.get('primary_keys', []),
            'referenced_by': table_info.get('referenced_by', [])
        }
    
    def generate_schema_context(self, relevant_tables: List[str] = None, query: str = "") -> str:
        """Generate schema context for LLM prompting with query-specific table detection"""
        if not self._schema_cache:
            self.load_schema()
        
        # If no relevant tables specified, try to detect from query
        if not relevant_tables and query:
            relevant_tables = self.find_tables_by_natural_language(query)
        
        # If still no relevant tables, include top 5 most important tables
        if not relevant_tables:
            all_tables = self.get_all_tables()
            relevant_tables = all_tables[:5]  # Limit to avoid overwhelming the LLM
        
        context_parts = [
            f"Database: {self._schema_cache.get('database_name', '146_36156520-AC21-435A-9C9B-1EC9145A9090')}",
            f"Schema Version: {self._schema_cache.get('version', '1.0')}",
            "\n=== RELEVANT TABLES AND COLUMNS ==="
        ]
        
        for table_name in relevant_tables:
            table_info = self.get_table_info(table_name)
            if not table_info:
                continue
            
            # Table description
            table_desc = table_info.get('description', '')
            context_parts.append(f"\nTable: {table_name}")
            if table_desc:
                context_parts.append(f"Description: {table_desc}")
            
            # Columns with detailed information
            columns = table_info.get('columns', {})
            context_parts.append("Columns:")
            
            for col_name, col_info in columns.items():
                if isinstance(col_info, dict):
                    col_type = col_info.get('type', 'Unknown')
                    is_pk = col_info.get('primary_key', False)
                    is_fk = col_info.get('foreign_key', False)
                    nullable = col_info.get('nullable', True)
                    col_desc = col_info.get('description', '')
                    
                    flags = []
                    if is_pk: flags.append("PRIMARY KEY")
                    if is_fk: flags.append("FOREIGN KEY")
                    if not nullable: flags.append("NOT NULL")
                    
                    flag_str = f" ({', '.join(flags)})" if flags else ""
                    desc_str = f" - {col_desc}" if col_desc else ""
                    
                    context_parts.append(f"  • {col_name}: {col_type}{flag_str}{desc_str}")
                else:
                    context_parts.append(f"  • {col_name}: {col_info}")
            
            # Relationships
            relationships = self.get_table_relationships(table_name)
            if relationships.get('foreign_keys'):
                context_parts.append("Foreign Key Relationships:")
                for fk in relationships['foreign_keys']:
                    if isinstance(fk, dict):
                        context_parts.append(f"  → {fk.get('column', '')} references {fk.get('references', '')}")
                    else:
                        context_parts.append(f"  → {fk}")
        
        context_parts.append("\n=== IMPORTANT NOTES ===")
        context_parts.append("• Use proper table and column names exactly as shown above")
        context_parts.append("• Consider foreign key relationships for JOINs")
        context_parts.append("• Use appropriate WHERE clauses for filtering")
        context_parts.append("• Always include LIMIT for large result sets")
        
        return "\n".join(context_parts)
    
    def get_similar_columns(self, search_term: str, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """Get columns similar to search term using fuzzy matching"""
        if not self.config.FUZZY_MATCHING:
            return []
        
        try:
            from difflib import SequenceMatcher
            
            matches = []
            search_term_lower = search_term.lower()
            
            for table_name, columns in self._column_mappings.items():
                for column_key, column_name in columns.items():
                    similarity = SequenceMatcher(None, search_term_lower, column_key).ratio()
                    if similarity >= threshold:
                        matches.append((table_name, column_name, similarity))
            
            # Sort by similarity score (highest first)
            return sorted(matches, key=lambda x: x[2], reverse=True)
        
        except ImportError:
            logger.warning("difflib not available for fuzzy matching")
            return []

@dataclass
class LogConfig:
    """Enhanced logging configuration"""
    LEVEL: str = "INFO"
    FILE: str = "querymancer.log"
    MAX_SIZE: int = 10 * 1024 * 1024  # 10 MB
    BACKUP_COUNT: int = 5
    CONSOLE: bool = True
    SQL_LOG: bool = True
    MODEL_LOG: bool = True
    SCHEMA_LOG: bool = True
    ACCURACY_LOG: bool = True

@dataclass
class AppConfig:
    """Enhanced master application configuration for local AI SQL chatbot"""
    model_config: ModelConfig
    sqlserver_config: SQLServerConfig
    schema_config: SchemaConfig = field(default_factory=SchemaConfig)
    sql_guard_config: SQLGuardConfig = field(default_factory=SQLGuardConfig)
    accuracy_config: AccuracyConfig = field(default_factory=AccuracyConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    debug_mode: bool = False
    cache_enabled: bool = True
    max_history_length: int = 50  # Reduced for chatbot UI
    db_timeout: int = 30  # Increased for AWS latency
    agent_timeout: int = 60  # Increased for local model inference
    default_query_limit: int = 100
    max_query_limit: int = 1000
    rate_limit_requests: int = 10  # Local rate limiting
    rate_limit_window: int = 60
    seed: int = 42
    version: str = __version__
    
    def __post_init__(self):
        # Setup system information
        self.system_info = {
            "hostname": socket.gethostname(),
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "start_time": time.time(),
            "model_provider": self.model_config.provider.value,
            "local_inference": True
        }
        
        # Initialize schema manager
        self.schema_manager = SchemaManager(self.schema_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary excluding sensitive information"""
        result = {
            "model": self.model_config.to_dict(),
            "schema": self.schema_config.to_dict(),
            "accuracy": self.accuracy_config.to_dict(),
            "debug": self.debug_mode,
            "cache_enabled": self.cache_enabled,
            "max_history": self.max_history_length,
            "agent_timeout": self.agent_timeout,
            "version": self.version,
            "system_info": self.system_info
        }
        
        # Add database info with password masked
        db_info = {
            "host": self.sqlserver_config.HOST,
            "database": self.sqlserver_config.DATABASE,
            "username": self.sqlserver_config.USERNAME,
            "driver": self.sqlserver_config.DRIVER,
            "password": "***MASKED***"
        }
        result["database"] = db_info
        
        return result

class ConfigManager:
    """Enhanced configuration manager for local AI SQL chatbot"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._config: Optional[AppConfig] = None

    def load_config(self) -> AppConfig:
        """Load configuration from environment or file"""
        if self.config_file and Path(self.config_file).exists():
            return self._load_from_file()
        return self._load_from_environment()

    def _load_from_environment(self) -> AppConfig:
        """Load configuration from environment variables"""
        
        # Model configuration for local Ollama + Mistral
        model_config = ModelConfig(
            name=os.getenv("MODEL_NAME", "mistral"),
            temperature=float(os.getenv("MODEL_TEMPERATURE", "0.05")),
            provider=ModelProvider.OLLAMA,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            max_tokens=int(os.getenv("MODEL_MAX_TOKENS", "4000")),
            context_window=int(os.getenv("MODEL_CONTEXT_WINDOW", "8192")),
            timeout=int(os.getenv("MODEL_TIMEOUT", "60")),
            description="Local Mistral via Ollama for SQL generation"
        )

        # AWS SQL Server configuration from environment
        sql_config = SQLServerConfig(
            HOST=os.getenv("DB_HOST", "10.0.0.45"),
            DATABASE=os.getenv("DB_NAME", "146_36156520-AC21-435A-9C9B-1EC9145A9090"),
            USERNAME=os.getenv("DB_USERNAME", "usr_mohsin"),
            PASSWORD=os.getenv("DB_PASSWORD", "blY|5K:3pe10"),
            DRIVER=os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server"),
            CONNECTION_TIMEOUT=int(os.getenv("DB_CONNECTION_TIMEOUT", "60")),
            COMMAND_TIMEOUT=int(os.getenv("DB_COMMAND_TIMEOUT", "60")),
            PORT=int(os.getenv("DB_PORT", "1433")),
            TRUST_CERT=os.getenv("DB_TRUST_CERT", "true").lower() == "true"
        )

        # Schema configuration
        schema_config = SchemaConfig(
            SCHEMA_DIR=os.getenv("SCHEMA_DIR", "schemas"),
            SCHEMA_FILE=os.getenv("SCHEMA_FILE", "schema.json"),
            AUTO_RELOAD=os.getenv("SCHEMA_AUTO_RELOAD", "true").lower() == "true",
            CACHE_SCHEMA=os.getenv("CACHE_SCHEMA", "true").lower() == "true",
            VALIDATE_SCHEMA=os.getenv("VALIDATE_SCHEMA", "true").lower() == "true",
            FUZZY_MATCHING=os.getenv("FUZZY_MATCHING", "true").lower() == "true",
            FUZZY_THRESHOLD=float(os.getenv("FUZZY_THRESHOLD", "0.8"))
        )
        
        # SQL Guard configuration
        sql_guard = SQLGuardConfig(
            ENABLED=os.getenv("ENABLE_QUERY_VALIDATION", "true").lower() == "true",
            MAX_ROWS=int(os.getenv("MAX_QUERY_RESULTS", "1000")),
            MAX_EXECUTION_TIME=int(os.getenv("MAX_EXECUTION_TIME", "30")),
            VALIDATE_AGAINST_SCHEMA=os.getenv("VALIDATE_AGAINST_SCHEMA", "true").lower() == "true",
            STRICT_COLUMN_VALIDATION=os.getenv("STRICT_COLUMN_VALIDATION", "true").lower() == "true",
            STRICT_TABLE_VALIDATION=os.getenv("STRICT_TABLE_VALIDATION", "true").lower() == "true"
        )

        # Accuracy configuration
        accuracy_config = AccuracyConfig(
            TARGET_ACCURACY=float(os.getenv("TARGET_ACCURACY", "0.99")),
            ENABLE_FUZZY_MATCHING=os.getenv("ENABLE_FUZZY_MATCHING", "true").lower() == "true",
            FUZZY_THRESHOLD=float(os.getenv("ACCURACY_FUZZY_THRESHOLD", "0.85")),
            ENABLE_SEMANTIC_SEARCH=os.getenv("ENABLE_SEMANTIC_SEARCH", "true").lower() == "true",
            VALIDATE_JOINS=os.getenv("VALIDATE_JOINS", "true").lower() == "true",
            SUGGEST_CORRECTIONS=os.getenv("SUGGEST_CORRECTIONS", "true").lower() == "true"
        )

        # Logging configuration
        log_config = LogConfig(
            LEVEL=os.getenv("LOG_LEVEL", "INFO"),
            FILE=os.getenv("LOG_FILE", "querymancer.log"),
            CONSOLE=os.getenv("CONSOLE_LOG", "true").lower() == "true",
            SCHEMA_LOG=os.getenv("SCHEMA_LOG", "true").lower() == "true"
        )

        return AppConfig(
            model_config=model_config,
            sqlserver_config=sql_config,
            schema_config=schema_config,
            sql_guard_config=sql_guard,
            accuracy_config=accuracy_config,
            log_config=log_config,
            debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
            cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            max_history_length=int(os.getenv("MAX_HISTORY", "50")),
            db_timeout=int(os.getenv("DB_TIMEOUT", "30")),
            agent_timeout=int(os.getenv("AGENT_TIMEOUT", "60"))
        )
        
    def _load_from_file(self) -> AppConfig:
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
                
            # Extract configurations
            model_data = config_data.get("model", {})
            model_config = ModelConfig(
                name=model_data.get("name", "mistral"),
                temperature=model_data.get("temperature", 0.05),
                provider=ModelProvider.OLLAMA,
                base_url=model_data.get("base_url", "http://localhost:11434"),
                max_tokens=model_data.get("max_tokens", 4000),
                context_window=model_data.get("context_window", 8192),
                timeout=model_data.get("timeout", 60)
            )
            
            # SQL Server config
            sql_data = config_data.get("sqlserver", {})
            sql_config = SQLServerConfig(
                HOST=sql_data.get("host", "10.0.0.45"),
                DATABASE=sql_data.get("database", "146_36156520-AC21-435A-9C9B-1EC9145A9090"),
                USERNAME=sql_data.get("username", "usr_mohsin"),
                PASSWORD=sql_data.get("password", "blY|5K:3pe10"),
                DRIVER=sql_data.get("driver", "ODBC Driver 17 for SQL Server"),
                CONNECTION_TIMEOUT=sql_data.get("connection_timeout", 30),
                COMMAND_TIMEOUT=sql_data.get("command_timeout", 60),
                PORT=sql_data.get("port", 1433)
            )
            
            # Schema config
            schema_data = config_data.get("schema", {})
            schema_config = SchemaConfig(
                SCHEMA_DIR=schema_data.get("schema_dir", "schemas"),
                SCHEMA_FILE=schema_data.get("schema_file", "schema.json"),
                AUTO_RELOAD=schema_data.get("auto_reload", True),
                CACHE_SCHEMA=schema_data.get("cache_schema", True),
                VALIDATE_SCHEMA=schema_data.get("validate_schema", True),
                FUZZY_MATCHING=schema_data.get("fuzzy_matching", True),
                FUZZY_THRESHOLD=schema_data.get("fuzzy_threshold", 0.8)
            )
            
            return AppConfig(
                model_config=model_config,
                sqlserver_config=sql_config,
                schema_config=schema_config,
                debug_mode=config_data.get("debug_mode", False),
                cache_enabled=config_data.get("cache_enabled", True),
                max_history_length=config_data.get("max_history_length", 50)
            )
            
        except Exception as e:
            logger.error(f"Error loading configuration from file: {e}")
            return self._load_from_environment()

    def save_config(self, config: AppConfig, file_path: Optional[str] = None) -> bool:
        """Save configuration to file"""
        if not file_path and not self.config_file:
            return False
            
        save_path = file_path or self.config_file
        try:
            config_dict = config.to_dict()
            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            return False

# === Enhanced Exceptions ===

class QueryMancerError(Exception): 
    """Base exception for Querymancer"""
    pass

class ModelError(QueryMancerError): 
    """Model-related errors"""
    pass

class DatabaseError(QueryMancerError): 
    """Database connection/query errors"""
    pass

class ConfigurationError(QueryMancerError): 
    """Configuration errors"""
    pass

class TimeoutError(QueryMancerError): 
    """Timeout errors"""
    pass

class SQLExecutionError(QueryMancerError): 
    """SQL execution errors"""
    pass

class SchemaError(QueryMancerError): 
    """Schema loading/validation errors"""
    pass

class AccuracyError(QueryMancerError): 
    """SQL accuracy/validation errors"""
    pass

# === Enhanced Diagnostics ===

def validate_environment() -> Dict[str, Union[bool, str]]:
    """Enhanced environment validation for local AI SQL chatbot"""
    results = {
        "python": check_python_version(),
        "sql_driver": False,
        "ollama_available": False,
        "ollama_models": [],
        "schema_file_exists": False,
        "schema_valid": False,
        "langchain_available": False,
        "database_connection": False
    }
    
    # Check SQL Server ODBC driver
    try:
        import pyodbc
        drivers = pyodbc.drivers()
        results["sql_driver"] = any("SQL Server" in d for d in drivers)
        results["available_drivers"] = drivers
        if not results["sql_driver"]:
            results["sql_driver_error"] = "No SQL Server ODBC driver found"
    except ImportError:
        results["sql_driver_error"] = "pyodbc not installed - run: pip install pyodbc"
    except Exception as e:
        results["sql_driver_error"] = str(e)
        
    # Check Ollama availability and models
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            results["ollama_available"] = True
            models_data = response.json().get("models", [])
            results["ollama_models"] = [m["name"] for m in models_data]
            
            # Check if Mistral is available
            mistral_available = any("mistral" in model.lower() for model in results["ollama_models"])
            results["mistral_available"] = mistral_available
            
            if not mistral_available:
                results["ollama_warning"] = "Mistral model not found. Run: ollama pull mistral"
        else:
            results["ollama_error"] = f"Ollama responded with status {response.status_code}"
    except ImportError:
        results["ollama_error"] = "requests not installed - run: pip install requests"
    except requests.ConnectionError:
        results["ollama_error"] = "Cannot connect to Ollama. Is it running? Start with: ollama serve"
    except requests.Timeout:
        results["ollama_error"] = "Ollama connection timeout"
    except Exception as e:
        results["ollama_error"] = str(e)
        
    # Check LangChain availability
    try:
        import langchain
        from langchain_ollama import OllamaLLM
        results["langchain_available"] = True
        results["langchain_version"] = langchain.__version__
    except ImportError as e:
        results["langchain_error"] = f"LangChain not properly installed: {e}"
        results["langchain_available"] = False
    
    # Check schema file
    schema_path = Path("schemas") / "schema.json"
    results["schema_file_exists"] = schema_path.exists()
    results["schema_path"] = str(schema_path)
    
    if results["schema_file_exists"]:
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
            
            # Basic validation
            results["schema_valid"] = 'tables' in schema_data
            
            if results["schema_valid"]:
                results["schema_tables"] = len(schema_data.get('tables', {}))
                results["schema_version"] = schema_data.get('version', 'Unknown')
                results["database_name"] = schema_data.get('database_name', 'Unknown')
            else:
                results["schema_error"] = "Schema missing required 'tables' key"
                
        except json.JSONDecodeError as e:
            results["schema_error"] = f"Invalid JSON in schema file: {e}"
            results["schema_valid"] = False
        except Exception as e:
            results["schema_error"] = f"Error reading schema file: {e}"
            results["schema_valid"] = False
    else:
        results["schema_error"] = "Schema file not found. Please create schemas/schema.json"
        
    # Test database connection if credentials are available
    if os.getenv("DB_HOST") and os.getenv("DB_USERNAME") and os.getenv("DB_PASSWORD"):
        try:
            success, message = test_database_connection()
            results["database_connection"] = success
            if not success:
                results["database_error"] = message
        except Exception as e:
            results["database_error"] = str(e)
    else:
        results["database_error"] = "Database credentials not found in environment variables"
        
    return results

def setup_logging(config: Optional[LogConfig] = None) -> logging.Logger:
    """Enhanced logging setup for local AI SQL chatbot"""
    if config is None:
        config = LogConfig()
        
    log_level = getattr(logging, config.LEVEL.upper(), logging.INFO)
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    handlers = []
    
    # Console handler
    if config.CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    
    # File handler with rotation
    if config.FILE:
        try:
            from logging.handlers import RotatingFileHandler
            
            # Ensure log directory exists
            log_path = Path(config.FILE)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                config.FILE,
                maxBytes=config.MAX_SIZE,
                backupCount=config.BACKUP_COUNT,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                '%(asctime)s | %(name)-12s | %(levelname)-8s | %(funcName)-15s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not set up file logging: {e}")
            
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )
    
    # Create and configure main logger
    logger = logging.getLogger("querymancer")
    logger.setLevel(log_level)
    
    # Set third-party loggers to WARNING to reduce noise
    for noisy_logger in ['urllib3', 'requests', 'httpx', 'httpcore']:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    
    return logger

def test_database_connection() -> Tuple[bool, str]:
    """Test AWS SQL Server database connection"""
    try:
        import pyodbc
        conn_string = app_config.sqlserver_config.get_connection_string()
        
        logger.info("Testing database connection...")
        conn = pyodbc.connect(conn_string)
        cursor = conn.cursor()
        
        # Test basic connectivity and get server info
        cursor.execute("""
            SELECT 
                @@VERSION as server_version,
                DB_NAME() as database_name,
                SYSTEM_USER as current_user,
                GETDATE() as current_time
        """)
        result = cursor.fetchone()
        
        if result:
            server_info = {
                "version": result.server_version.split('\n')[0] if result.server_version else "Unknown",
                "database": result.database_name,
                "user": result.current_user,
                "time": result.current_time.strftime("%Y-%m-%d %H:%M:%S") if result.current_time else "Unknown"
            }
            
            cursor.close()
            conn.close()
            
            success_msg = (
                f"✅ Connected to {server_info['database']} as {server_info['user']} "
                f"at {server_info['time']}"
            )
            logger.info(success_msg)
            return True, success_msg
        else:
            return False, "No response from database"
            
    except ImportError:
        error_msg = "pyodbc not installed. Run: pip install pyodbc"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Database connection failed: {str(e)}"
        logger.error(error_msg)
        
        # Provide helpful error messages for common issues
        error_str = str(e).lower()
        if "driver" in error_str:
            error_msg += "\n💡 Tip: Install SQL Server ODBC driver from Microsoft"
        elif "timeout" in error_str or "network" in error_str:
            error_msg += "\n💡 Tip: Check network connectivity and firewall settings"
        elif "login failed" in error_str or "authentication" in error_str:
            error_msg += "\n💡 Tip: Verify username and password in .env file"
        elif "server" in error_str:
            error_msg += "\n💡 Tip: Check server hostname and port in .env file"
            
        return False, error_msg

def test_schema_loading() -> Tuple[bool, str]:
    """Test schema loading and validation"""
    try:
        schema_data = app_config.schema_manager.load_schema()
        table_count = len(schema_data.get('tables', {}))
        db_name = schema_data.get('database_name', 'Unknown')
        version = schema_data.get('version', 'Unknown')
        
        # Test natural language mapping
        test_query = "show me users"
        relevant_tables = app_config.schema_manager.find_tables_by_natural_language(test_query)
        
        success_msg = (
            f"✅ Schema loaded: {table_count} tables from '{db_name}' (v{version}). "
            f"Natural language test found {len(relevant_tables)} relevant tables."
        )
        logger.info(success_msg)
        return True, success_msg
        
    except FileNotFoundError:
        error_msg = "❌ Schema file not found at schemas/schema.json"
        logger.error(error_msg)
        return False, error_msg
    except json.JSONDecodeError as e:
        error_msg = f"❌ Invalid JSON in schema file: {e}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"❌ Schema loading failed: {e}"
        logger.error(error_msg)
        return False, error_msg

def test_ollama_connection() -> Tuple[bool, str]:
    """Test Ollama connection and Mistral model availability"""
    try:
        import requests
        
        # Test Ollama server
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code != 200:
            return False, f"Ollama server responded with status {response.status_code}"
        
        models_data = response.json().get("models", [])
        model_names = [m["name"] for m in models_data]
        
        # Check for Mistral
        mistral_models = [m for m in model_names if "mistral" in m.lower()]
        
        if mistral_models:
            success_msg = f"✅ Ollama connected. Mistral models available: {', '.join(mistral_models)}"
            logger.info(success_msg)
            return True, success_msg
        else:
            warning_msg = f"⚠️ Ollama connected but Mistral not found. Available: {', '.join(model_names[:3])}"
            logger.warning(warning_msg)
            return False, warning_msg + "\n💡 Run: ollama pull mistral"
            
    except ImportError:
        return False, "requests library not available"
    except requests.ConnectionError:
        return False, "Cannot connect to Ollama. Start with: ollama serve"
    except requests.Timeout:
        return False, "Ollama connection timeout"
    except Exception as e:
        return False, f"Ollama test failed: {e}"

def create_sample_schema() -> bool:
    """Create a sample schema file for reference"""
    sample_schema = {
        "database_name": "146_36156520-AC21-435A-9C9B-1EC9145A9090",
        "version": "1.0",
        "description": "Sample AWS SQL Server database schema for AI SQL chatbot",
        "tables": {
            "XERO_EXPORT": {
                "description": "Financial transaction exports from Xero",
                "columns": {
                    "TransactionID": {
                        "type": "varchar(50)",
                        "primary_key": True,
                        "nullable": False,
                        "description": "Unique transaction identifier"
                    },
                    "Date": {
                        "type": "datetime",
                        "nullable": False,
                        "description": "Transaction date"
                    },
                    "Amount": {
                        "type": "decimal(18,2)",
                        "nullable": False,
                        "description": "Transaction amount"
                    },
                    "Description": {
                        "type": "varchar(255)",
                        "nullable": True,
                        "description": "Transaction description"
                    },
                    "AccountCode": {
                        "type": "varchar(20)",
                        "nullable": True,
                        "description": "Account code reference"
                    },
                    "ContactName": {
                        "type": "varchar(100)",
                        "nullable": True,
                        "description": "Contact or customer name"
                    },
                    "Reference": {
                        "type": "varchar(100)",
                        "nullable": True,
                        "description": "Transaction reference number"
                    }
                },
                "primary_keys": ["TransactionID"],
                "indexes": ["Date", "AccountCode", "ContactName"],
                "natural_language": ["transactions", "xero", "exports", "financial", "accounting"],
                "aliases": ["xero", "transactions", "financial_data"]
            },
            "Invoices": {
                "description": "Customer invoices and billing information",
                "columns": {
                    "InvoiceID": {
                        "type": "int",
                        "primary_key": True,
                        "nullable": False,
                        "description": "Unique invoice identifier"
                    },
                    "CustomerID": {
                        "type": "int",
                        "foreign_key": True,
                        "references": "Customers.CustomerID",
                        "nullable": False,
                        "description": "Customer reference"
                    },
                    "InvoiceNumber": {
                        "type": "varchar(50)",
                        "nullable": False,
                        "unique": True,
                        "description": "Human-readable invoice number"
                    },
                    "InvoiceDate": {
                        "type": "datetime",
                        "nullable": False,
                        "description": "Invoice creation date"
                    },
                    "DueDate": {
                        "type": "datetime",
                        "nullable": True,
                        "description": "Payment due date"
                    },
                    "TotalAmount": {
                        "type": "decimal(18,2)",
                        "nullable": False,
                        "description": "Total invoice amount"
                    },
                    "Status": {
                        "type": "varchar(20)",
                        "nullable": False,
                        "default": "Draft",
                        "values": ["Draft", "Sent", "Paid", "Overdue", "Cancelled"],
                        "description": "Invoice status"
                    }
                },
                "primary_keys": ["InvoiceID"],
                "foreign_keys": [
                    {
                        "column": "CustomerID",
                        "references": "Customers.CustomerID",
                        "on_delete": "CASCADE"
                    }
                ],
                "indexes": ["InvoiceNumber", "CustomerID", "InvoiceDate", "Status"],
                "natural_language": ["invoices", "bills", "billing", "payments"],
                "aliases": ["invoice", "bill", "billing"]
            },
            "Customers": {
                "description": "Customer information and contact details",
                "columns": {
                    "CustomerID": {
                        "type": "int",
                        "primary_key": True,
                        "nullable": False,
                        "description": "Unique customer identifier"
                    },
                    "CustomerName": {
                        "type": "varchar(100)",
                        "nullable": False,
                        "description": "Customer full name or company name"
                    },
                    "Email": {
                        "type": "varchar(100)",
                        "nullable": True,
                        "description": "Customer email address"
                    },
                    "Phone": {
                        "type": "varchar(20)",
                        "nullable": True,
                        "description": "Customer phone number"
                    },
                    "Address": {
                        "type": "varchar(255)",
                        "nullable": True,
                        "description": "Customer address"
                    },
                    "CreatedDate": {
                        "type": "datetime",
                        "nullable": False,
                        "default": "GETDATE()",
                        "description": "Customer creation date"
                    }
                },
                "primary_keys": ["CustomerID"],
                "indexes": ["CustomerName", "Email"],
                "natural_language": ["customers", "clients", "contacts", "people"],
                "aliases": ["customer", "client", "contact"]
            }
        }
    }
    
    try:
        schema_path = Path("schemas") / "sample_schema.json"
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(sample_schema, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Sample schema created at: {schema_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating sample schema: {e}")
        return False

def create_env_template() -> bool:
    """Create .env.example template file"""
    env_template = """# Querymancer Local AI SQL Chatbot Configuration

# === Database Configuration (AWS SQL Server) ===
DB_HOST=10.0.0.45
DB_NAME=146_36156520-AC21-435A-9C9B-1EC9145A9090
DB_USERNAME=usr_mohsin
DB_PASSWORD=blY|5K:3pe10
DB_DRIVER=ODBC Driver 17 for SQL Server
DB_PORT=1433
DB_TRUST_CERT=true
DB_CONNECTION_TIMEOUT=30
DB_COMMAND_TIMEOUT=60

# === Ollama Model Configuration ===
MODEL_NAME=mistral
MODEL_TEMPERATURE=0.05
MODEL_MAX_TOKENS=4000
MODEL_CONTEXT_WINDOW=8192
MODEL_TIMEOUT=60
OLLAMA_BASE_URL=http://localhost:11434

# === Schema Configuration ===
SCHEMA_DIR=schemas
SCHEMA_FILE=schema.json
SCHEMA_AUTO_RELOAD=true
CACHE_SCHEMA=true
VALIDATE_SCHEMA=true
FUZZY_MATCHING=true
FUZZY_THRESHOLD=0.8

# === Query Safety Configuration ===
ENABLE_QUERY_VALIDATION=true
MAX_QUERY_RESULTS=1000
MAX_EXECUTION_TIME=30
VALIDATE_AGAINST_SCHEMA=true
STRICT_COLUMN_VALIDATION=true
STRICT_TABLE_VALIDATION=true

# === Application Configuration ===
DEBUG_MODE=false
LOG_LEVEL=INFO
LOG_FILE=logs/querymancer.log
CONSOLE_LOG=true
CACHE_ENABLED=true
MAX_HISTORY=50

# === Accuracy Configuration ===
TARGET_ACCURACY=0.99
ENABLE_FUZZY_MATCHING=true
ACCURACY_FUZZY_THRESHOLD=0.85
ENABLE_SEMANTIC_SEARCH=true
VALIDATE_JOINS=true
SUGGEST_CORRECTIONS=true
"""
    
    try:
        env_path = Path(".env.example")
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_template)
        
        logger.info(f"✅ Environment template created at: {env_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating .env template: {e}")
        return False

# Initialize configuration manager and load config
config_manager = ConfigManager()
app_config = config_manager.load_config()

# Setup logging
logger = setup_logging(app_config.log_config)
logger.info(f"🔮 Querymancer v{__version__} - Local AI SQL Chatbot Initialized")

# Create necessary directories
def ensure_directories():
    """Ensure necessary directories exist"""
    dirs = ["logs", "cache", "schemas", "temp"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

ensure_directories()

# Run comprehensive environment validation
env_status = validate_environment()

# Critical dependency checks
critical_checks = ["python", "sql_driver", "ollama_available", "schema_file_exists", "langchain_available"]
failed_checks = [k for k in critical_checks if k in env_status and not env_status[k]]

if failed_checks:
    logger.warning("⚠️ Some critical dependencies failed:")
    for check in failed_checks:
        logger.warning(f"  ❌ {check}")
        
    # Provide specific guidance
    if "ollama_available" in failed_checks:
        logger.error("❌ Ollama not available!")
        logger.info("💡 To fix: Start Ollama with 'ollama serve' and install Mistral with 'ollama pull mistral'")
        
    if "schema_file_exists" in failed_checks:
        logger.error("❌ Schema file not found!")
        logger.info("💡 Creating sample schema file...")
        if create_sample_schema():
            logger.info("✅ Sample created. Copy it to schemas/schema.json and customize")
            
    if "langchain_available" in failed_checks:
        logger.error("❌ LangChain not properly installed!")
        logger.info("💡 Install with: pip install langchain langchain-ollama")
        
    if "sql_driver" in failed_checks:
        logger.error("❌ SQL Server ODBC driver not found!")
        logger.info("💡 Install from: https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server")
        
else:
    logger.info("✅ All critical dependencies validated successfully")
    
    # Additional success information
    if env_status.get("schema_tables"):
        logger.info(f"📊 Schema loaded: {env_status['schema_tables']} tables found")
        
    if env_status.get("mistral_available"):
        logger.info("🤖 Mistral model available in Ollama")
    elif env_status.get("ollama_models"):
        logger.warning(f"⚠️ Ollama available but Mistral not found. Available: {', '.join(env_status['ollama_models'][:3])}")

# Create environment template if it doesn't exist
if not Path(".env.example").exists():
    logger.info("📝 Creating .env.example template...")
    create_env_template()

# Export public API for other modules
__all__ = [
    # Core configuration
    "app_config", "config_manager", "logger",
    
    # Configuration classes
    "ModelConfig", "SQLServerConfig", "SchemaConfig", 
    "SQLGuardConfig", "AccuracyConfig", "LogConfig", "AppConfig",
    
    # Schema management
    "SchemaManager",
    
    # Enums
    "ModelProvider",
    
    # Testing functions
    "validate_environment", "test_database_connection", 
    "test_schema_loading", "test_ollama_connection",
    
    # Utility functions
    "create_sample_schema", "create_env_template", "ensure_directories",
    
    # Exceptions
    "QueryMancerError", "ModelError", "DatabaseError", 
    "ConfigurationError", "TimeoutError", "SQLExecutionError", 
    "SchemaError", "AccuracyError"
]

# Run startup diagnostics if executed directly
if __name__ == "__main__":
    print("=" * 80)
    print(f"🔮 Querymancer v{__version__} - Local AI SQL Chatbot")
    print("=" * 80)
    
    print("\n📋 System Status:")
    status_icons = {"true": "✅", "false": "❌", True: "✅", False: "❌"}
    
    for key, value in env_status.items():
        if isinstance(value, (bool, str)) and str(value).lower() in status_icons:
            icon = status_icons[str(value).lower()]
            print(f"  {icon} {key.replace('_', ' ').title()}")
        elif key.endswith('_error'):
            print(f"  ⚠️ {key.replace('_', ' ').title()}: {value}")
        elif key in ['ollama_models', 'available_drivers'] and isinstance(value, list):
            if value:
                print(f"  📋 {key.replace('_', ' ').title()}: {', '.join(value[:3])}")
    
    print("\n⚙️ Configuration:")
    config_dict = app_config.to_dict()
    print(f"  🤖 Model: {config_dict['model']['name']} via {config_dict['model']['provider']}")
    print(f"  🌡️ Temperature: {config_dict['model']['temperature']}")
    print(f"  🗄️ Database: {config_dict['database']['database']} on {config_dict['database']['host']}")
    print(f"  🎯 Target Accuracy: {config_dict['accuracy']['target_accuracy']*100}%")
    print(f"  📊 Schema File: {app_config.schema_config.schema_path}")
    
    print("\n🔍 Connection Tests:")
    
    # Test Ollama
    ollama_success, ollama_msg = test_ollama_connection()
    print(f"  {'✅' if ollama_success else '❌'} Ollama: {ollama_msg}")
    
    # Test Database
    db_success, db_msg = test_database_connection()
    print(f"  {'✅' if db_success else '❌'} Database: {db_msg}")
    
    # Test Schema
    schema_success, schema_msg = test_schema_loading()
    print(f"  {'✅' if schema_success else '❌'} Schema: {schema_msg}")
    
    print("\n🚀 Ready Status:")
    all_ready = ollama_success and schema_success and env_status.get('langchain_available', False)
    
    if all_ready:
        print("  🎉 Querymancer is ready for local AI SQL generation!")
        print("  💬 Start the chatbot with: streamlit run app.py")
        print("  📝 Ask questions like:")
        print("    - 'Show me total transaction amount for 2024 from XERO_EXPORT'")
        print("    - 'List all completed invoices sorted by date'")
        print("    - 'Find customers with overdue payments'")
    else:
        print("  ⚠️ Querymancer needs attention before use:")
        if not ollama_success:
            print("    - Fix Ollama connection")
        if not schema_success:
            print("    - Fix schema loading")
        if not env_status.get('langchain_available', False):
            print("    - Install LangChain: pip install langchain langchain-ollama")
        if not db_success:
            print("    - Configure database connection in .env file")
    
    print("\n📚 Next Steps:")
    print("  1. Copy .env.example to .env and configure your AWS SQL Server credentials")
    print("  2. Update schemas/schema.json with your actual database schema")
    print("  3. Ensure Ollama is running: ollama serve")
    print("  4. Pull Mistral model: ollama pull mistral")
    print("  5. Start the chatbot: streamlit run app.py")
    
    print("\n🔧 Troubleshooting:")
    print("  • Database issues: Check credentials and network connectivity")
    print("  • Ollama issues: Ensure service is running and Mistral is pulled")
    print("  • Schema issues: Validate JSON format and required fields")
    print("  • Performance: Adjust temperature and context window in config")
    
    print("\n" + "=" * 80)