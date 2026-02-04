"""
QueryMancer Enhanced Configuration - Local AI + Secure Database Integration

This module provides comprehensive configuration management for:
- Ollama + Mistral local AI integration
- AWS SQL Server secure connections
- Schema.json-based retrieval (no vector DB)
- Performance monitoring and accuracy tracking

Author: Mohsin Ramzan
Updated: 2025-08-20
"""

import os
import logging
import json
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from enum import Enum
import socket
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')

# Enhanced logging setup
# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/config_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("querymancer.config")

class SystemStatus(Enum):
    """System component status"""
    READY = "ready"
    FAILED = "failed"
    PARTIAL = "partial" 
    UNKNOWN = "unknown"

class DatabaseType(Enum):
    """Supported database types"""
    SQL_SERVER = "sql_server"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"

@dataclass
class OllamaConfig:
    """Ollama + Mistral AI configuration"""
    
    # Ollama connection settings
    base_url: str = field(default_factory=lambda: os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'))
    model: str = field(default_factory=lambda: os.getenv('OLLAMA_MODEL', 'mistral'))
    
    # Model parameters
    temperature: float = field(default_factory=lambda: float(os.getenv('OLLAMA_TEMPERATURE', '0.1')))
    max_tokens: int = field(default_factory=lambda: int(os.getenv('OLLAMA_MAX_TOKENS', '2048')))
    top_p: float = field(default_factory=lambda: float(os.getenv('OLLAMA_TOP_P', '0.9')))
    top_k: int = field(default_factory=lambda: int(os.getenv('OLLAMA_TOP_K', '40')))
    
    # Performance settings
    num_predict: int = field(default_factory=lambda: int(os.getenv('OLLAMA_NUM_PREDICT', '2048')))
    repeat_penalty: float = field(default_factory=lambda: float(os.getenv('OLLAMA_REPEAT_PENALTY', '1.1')))
    
    # Timeout settings
    request_timeout: int = field(default_factory=lambda: int(os.getenv('OLLAMA_TIMEOUT', '120')))
    connection_timeout: int = field(default_factory=lambda: int(os.getenv('OLLAMA_CONN_TIMEOUT', '30')))
    
    # Retry settings
    max_retries: int = field(default_factory=lambda: int(os.getenv('OLLAMA_MAX_RETRIES', '3')))
    retry_delay: float = field(default_factory=lambda: float(os.getenv('OLLAMA_RETRY_DELAY', '1.0')))
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.base_url:
            raise ValueError("OLLAMA_BASE_URL must be set")
        if not self.model:
            raise ValueError("OLLAMA_MODEL must be set")
        
        # Validate temperature range
        if not 0.0 <= self.temperature <= 2.0:
            logger.warning(f"Temperature {self.temperature} is outside recommended range [0.0, 2.0]")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Ollama connection and model availability"""
        import requests
        
        test_result = {
            "success": False,
            "model_available": False,
            "server_running": False,
            "response_time": 0.0,
            "error": None,
            "model_info": None
        }
        
        try:
            start_time = time.time()
            
            # Test if Ollama server is running
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.connection_timeout
            )
            
            if response.status_code == 200:
                test_result["server_running"] = True
                models = response.json().get("models", [])
                
                # Check if our model is available
                model_names = [model.get("name", "") for model in models]
                test_result["model_available"] = any(self.model in name for name in model_names)
                
                if test_result["model_available"]:
                    # Test simple generation
                    test_payload = {
                        "model": self.model,
                        "prompt": "SELECT 1",
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 50
                        }
                    }
                    
                    gen_response = requests.post(
                        f"{self.base_url}/api/generate",
                        json=test_payload,
                        timeout=self.request_timeout
                    )
                    
                    if gen_response.status_code == 200:
                        test_result["success"] = True
                        test_result["model_info"] = gen_response.json()
                
                test_result["response_time"] = time.time() - start_time
            
        except requests.exceptions.ConnectionError:
            test_result["error"] = "Ollama server not running or not accessible"
        except requests.exceptions.Timeout:
            test_result["error"] = "Connection timeout - server may be overloaded"
        except Exception as e:
            test_result["error"] = str(e)
        
        return test_result
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get optimized model parameters for SQL generation"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_predict": self.num_predict,
            "repeat_penalty": self.repeat_penalty,
            "stop": ["```", "Human:", "User:", "\n\n\n"]  # Stop tokens for SQL generation
        }

@dataclass
class DatabaseConfig:
    """Enhanced AWS SQL Server configuration"""
    
    # Connection parameters
    server: str = field(default_factory=lambda: os.getenv('DB_SERVER', '10.0.0.45'))
    database: str = field(default_factory=lambda: os.getenv('DB_DATABASE', '146_36156520-AC21-435A-9C9B-1EC9145A9090'))
    username: str = field(default_factory=lambda: os.getenv('DB_USERNAME', ''))
    password: str = field(default_factory=lambda: os.getenv('DB_PASSWORD', ''))
    port: int = field(default_factory=lambda: int(os.getenv('DB_PORT', '1433')))
    trusted_connection: str = field(default_factory=lambda: os.getenv('DB_TRUSTED_CONNECTION', 'no'))
    
    # Driver configuration
    driver: str = field(default_factory=lambda: os.getenv('DB_DRIVER', 'ODBC Driver 17 for SQL Server'))
    preferred_drivers: List[str] = field(default_factory=lambda: [
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server", 
        "ODBC Driver 13 for SQL Server",
        "SQL Server Native Client 11.0",
        "SQL Server"
    ])
    
    # Security settings
    
    trust_certificate: str = field(default_factory=lambda: os.getenv('DB_TRUST_CERTIFICATE', 'yes'))
    
    # Timeout settings
    connection_timeout: int = field(default_factory=lambda: int(os.getenv('DB_CONNECTION_TIMEOUT', '30')))
    command_timeout: int = field(default_factory=lambda: int(os.getenv('DB_COMMAND_TIMEOUT', '120')))
    query_timeout: int = field(default_factory=lambda: int(os.getenv('DB_QUERY_TIMEOUT', '300')))
    
    # Pool settings
    pool_size: int = field(default_factory=lambda: int(os.getenv('DB_POOL_SIZE', '10')))
    max_overflow: int = field(default_factory=lambda: int(os.getenv('DB_MAX_OVERFLOW', '20')))
    pool_timeout: int = field(default_factory=lambda: int(os.getenv('DB_POOL_TIMEOUT', '60')))
    
    # Retry settings
    max_retries: int = field(default_factory=lambda: int(os.getenv('DB_MAX_RETRIES', '3')))
    retry_delay: float = field(default_factory=lambda: float(os.getenv('DB_RETRY_DELAY', '2.0')))
    
    def __post_init__(self):
        """Validate database configuration"""
        # For Windows Authentication, username/password are not required
        if self.trusted_connection.lower() != 'yes':
            required_fields = ['server', 'database', 'username', 'password']
            missing_fields = [field for field in required_fields if not getattr(self, field)]
            
            if missing_fields:
                raise ValueError(f"Missing required database fields: {missing_fields}")
        else:
            # Only server and database are required for Windows Auth
            required_fields = ['server', 'database']
            missing_fields = [field for field in required_fields if not getattr(self, field)]
            
            if missing_fields:
                raise ValueError(f"Missing required database fields: {missing_fields}")
    
    def get_available_drivers(self) -> List[str]:
        """Get available ODBC drivers"""
        try:
            import pyodbc
            available_drivers = list(pyodbc.drivers())
            sql_drivers = [d for d in available_drivers if 'SQL Server' in d]
            return sql_drivers
        except ImportError:
            logger.error("pyodbc not available - install with: pip install pyodbc")
            return []
        except Exception as e:
            logger.error(f"Error getting ODBC drivers: {e}")
            return []
    
    def get_best_driver(self) -> str:
        """Get the best available ODBC driver"""
        available_drivers = self.get_available_drivers()
        
        if not available_drivers:
            raise RuntimeError("No SQL Server ODBC drivers found")
        
        # Try preferred drivers in order
        for preferred in self.preferred_drivers:
            if preferred in available_drivers:
                return preferred
        
        # Use first available
        return available_drivers[0]
    
    def generate_connection_string(self) -> str:
        """Generate optimized connection string"""
        driver = self.get_best_driver()
        
        # Build base connection string
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
        )
        
        # Use Windows Authentication (Trusted Connection) or SQL Server Authentication
        if self.trusted_connection.lower() == 'yes':
            conn_str += "Trusted_Connection=yes;"
        else:
            conn_str += (
                f"UID={self.username};"
                f"PWD={self.password};"
            )
        
        conn_str += (
            f"TrustServerCertificate={self.trust_certificate};"
            f"Connection Timeout={self.connection_timeout};"
            f"Command Timeout={self.command_timeout};"
        )
        
        return conn_str
    
    def test_connection(self) -> Dict[str, Any]:
        """Test database connection with comprehensive diagnostics"""
        test_result = {
            "success": False,
            "connection_time": 0.0,
            "server_info": None,
            "error": None,
            "network_reachable": False
        }
        
        try:
            # For named instances (like SQLEXPRESS), skip port test - go straight to connection test
            start_time = time.time()
            
            # Only test port for non-named instances (instances without backslash)
            if '\\' not in self.server:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                
                result = sock.connect_ex((self.server, self.port))
                sock.close()
                
                if result == 0:
                    test_result["network_reachable"] = True
                else:
                    test_result["error"] = f"Network unreachable - port {self.port} closed"
                    return test_result
            else:
                # Named instance - assume network is reachable for local instances
                test_result["network_reachable"] = True
            
            # Test database connection
            import pyodbc
            
            conn_str = self.generate_connection_string()
            conn = pyodbc.connect(conn_str, timeout=self.connection_timeout)
            
            # Test with simple query
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    @@VERSION as version,
                    @@SERVERNAME as server_name,
                    DB_NAME() as database_name,
                    GETDATE() as current_time,
                    USER_NAME() as current_user
            """)
            
            result = cursor.fetchone()
            if result:
                test_result["server_info"] = {
                    "version": result[0],
                    "server_name": result[1], 
                    "database_name": result[2],
                    "current_time": result[3].isoformat() if result[3] else None,
                    "current_user": result[4]
                }
                test_result["success"] = True
            
            test_result["connection_time"] = time.time() - start_time
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            test_result["error"] = str(e)
            test_result["connection_time"] = time.time() - start_time
        
        return test_result

@dataclass
class SchemaConfig:
    """Schema.json configuration for local retrieval"""
    
    # Schema file settings
    schema_file: str = field(default_factory=lambda: os.getenv('SCHEMA_FILE', 'schema.json'))
    schema_encoding: str = field(default_factory=lambda: os.getenv('SCHEMA_ENCODING', 'utf-8'))
    
    # Cache settings
    cache_ttl: int = field(default_factory=lambda: int(os.getenv('SCHEMA_CACHE_TTL', '3600')))  # 1 hour
    auto_refresh: bool = field(default_factory=lambda: os.getenv('SCHEMA_AUTO_REFRESH', 'true').lower() == 'true')
    
    # Validation settings
    enable_validation: bool = field(default_factory=lambda: os.getenv('SCHEMA_VALIDATION', 'true').lower() == 'true')
    strict_mode: bool = field(default_factory=lambda: os.getenv('SCHEMA_STRICT_MODE', 'false').lower() == 'true')
    
    # Search settings
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv('SCHEMA_SIMILARITY_THRESHOLD', '0.6')))
    max_suggestions: int = field(default_factory=lambda: int(os.getenv('SCHEMA_MAX_SUGGESTIONS', '3')))
    
    def __post_init__(self):
        """Validate schema configuration"""
        schema_path = Path(self.schema_file)
        if not schema_path.exists():
            logger.warning(f"Schema file not found: {self.schema_file}")
    
    def load_schema(self) -> Dict[str, Any]:
        """Load schema from JSON file"""
        try:
            schema_path = Path(self.schema_file)
            
            if not schema_path.exists():
                logger.error(f"Schema file not found: {self.schema_file}")
                return {}
            
            with open(schema_path, 'r', encoding=self.schema_encoding) as f:
                schema_data = json.load(f)
            
            # Validate schema structure
            if self.enable_validation:
                validation_result = self._validate_schema_structure(schema_data)
                if not validation_result["valid"]:
                    if self.strict_mode:
                        raise ValueError(f"Schema validation failed: {validation_result['errors']}")
                    else:
                        logger.warning(f"Schema validation warnings: {validation_result['warnings']}")
            
            logger.info(f"Schema loaded successfully: {len(schema_data.get('tables', {}))} tables")
            return schema_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in schema file: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading schema: {e}")
            return {}
    
    def _validate_schema_structure(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate schema structure"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required top-level keys
        required_keys = ['tables']
        for key in required_keys:
            if key not in schema_data:
                validation_result["errors"].append(f"Missing required key: {key}")
                validation_result["valid"] = False
        
        # Validate tables structure
        if 'tables' in schema_data:
            tables = schema_data['tables']
            
            if not isinstance(tables, dict):
                validation_result["errors"].append("'tables' must be a dictionary")
                validation_result["valid"] = False
            else:
                for table_name, table_info in tables.items():
                    # Validate table structure
                    if not isinstance(table_info, dict):
                        validation_result["warnings"].append(f"Table '{table_name}' info is not a dictionary")
                        continue
                    
                    # Check for columns
                    if 'columns' not in table_info:
                        validation_result["warnings"].append(f"Table '{table_name}' has no columns defined")
                    elif not isinstance(table_info['columns'], list):
                        validation_result["warnings"].append(f"Table '{table_name}' columns should be a list")
        
        return validation_result
    
    def get_table_context(self, query: str, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant table context based on query"""
        context = {
            "relevant_tables": [],
            "all_tables": list(schema_data.get('tables', {}).keys()),
            "suggestions": []
        }
        
        query_lower = query.lower()
        tables = schema_data.get('tables', {})
        
        # Find directly mentioned tables
        for table_name, table_info in tables.items():
            if table_name.lower() in query_lower:
                context["relevant_tables"].append({
                    "name": table_name,
                    "info": table_info,
                    "match_type": "direct"
                })
                continue
            
            # Check variations
            variations = table_info.get('variations', [])
            if any(var.lower() in query_lower for var in variations):
                context["relevant_tables"].append({
                    "name": table_name,
                    "info": table_info,
                    "match_type": "variation"
                })
                continue
            
            # Check column names
            columns = table_info.get('columns', [])
            column_names = [col.get('name', '') for col in columns if isinstance(col, dict)]
            if any(col.lower() in query_lower for col in column_names):
                context["relevant_tables"].append({
                    "name": table_name,
                    "info": table_info,
                    "match_type": "column"
                })
        
        # If no relevant tables found, include most common tables
        if not context["relevant_tables"]:
            # Sort by table size or importance (if available)
            sorted_tables = sorted(tables.items(), key=lambda x: len(x[1].get('columns', [])), reverse=True)
            
            for table_name, table_info in sorted_tables[:3]:  # Top 3 tables
                context["relevant_tables"].append({
                    "name": table_name,
                    "info": table_info,
                    "match_type": "fallback"
                })
        
        return context

@dataclass 
class ApplicationConfig:
    """Main application configuration"""
    
    # Component configurations
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    
    # Application settings
    app_name: str = "QueryMancer"
    version: str = "3.0.0"
    debug: bool = field(default_factory=lambda: os.getenv('DEBUG', 'false').lower() == 'true')
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    
    # Performance settings
    max_query_rows: int = field(default_factory=lambda: int(os.getenv('MAX_QUERY_ROWS', '1000')))
    query_cache_size: int = field(default_factory=lambda: int(os.getenv('QUERY_CACHE_SIZE', '100')))
    enable_metrics: bool = field(default_factory=lambda: os.getenv('ENABLE_METRICS', 'true').lower() == 'true')
    
    # Security settings
    enable_query_validation: bool = field(default_factory=lambda: os.getenv('ENABLE_QUERY_VALIDATION', 'true').lower() == 'true')
    dangerous_keywords: List[str] = field(default_factory=lambda: [
        'DROP TABLE', 'DELETE FROM', 'TRUNCATE', 'ALTER TABLE', 'CREATE TABLE'
    ])
    
    # UI settings
    ui_theme: str = field(default_factory=lambda: os.getenv('UI_THEME', 'dark'))
    enable_visualizations: bool = field(default_factory=lambda: os.getenv('ENABLE_VISUALIZATIONS', 'true').lower() == 'true')
    max_visualization_rows: int = field(default_factory=lambda: int(os.getenv('MAX_VIZ_ROWS', '500')))
    
    def __post_init__(self):
        """Initialize application configuration"""
        # Set up logging level
        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(numeric_level)
        
        # Create required directories
        required_dirs = ['logs', 'cache', 'schemas']
        for dir_name in required_dirs:
            Path(dir_name).mkdir(exist_ok=True)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate entire application configuration"""
        validation_result = {
            "valid": True,
            "components": {},
            "overall_status": SystemStatus.UNKNOWN,
            "errors": [],
            "warnings": []
        }
        
        # Test Ollama
        logger.info("Testing Ollama configuration...")
        ollama_test = self.ollama.test_connection()
        validation_result["components"]["ollama"] = {
            "status": SystemStatus.READY if ollama_test["success"] else SystemStatus.FAILED,
            "details": ollama_test
        }
        
        if not ollama_test["success"]:
            validation_result["errors"].append(f"Ollama: {ollama_test.get('error', 'Connection failed')}")
            validation_result["valid"] = False
        
        # Test Database
        logger.info("Testing database configuration...")
        db_test = self.database.test_connection()
        validation_result["components"]["database"] = {
            "status": SystemStatus.READY if db_test["success"] else SystemStatus.FAILED,
            "details": db_test
        }
        
        if not db_test["success"]:
            validation_result["errors"].append(f"Database: {db_test.get('error', 'Connection failed')}")
            validation_result["valid"] = False
        
        # Test Schema
        logger.info("Testing schema configuration...")
        schema_data = self.schema.load_schema()
        schema_status = SystemStatus.READY if schema_data else SystemStatus.FAILED
        validation_result["components"]["schema"] = {
            "status": schema_status,
            "details": {
                "tables_count": len(schema_data.get('tables', {})),
                "file_exists": Path(self.schema.schema_file).exists()
            }
        }
        
        if not schema_data:
            validation_result["warnings"].append("Schema: No schema data loaded")
        
        # Determine overall status
        component_statuses = [comp["status"] for comp in validation_result["components"].values()]
        
        if all(status == SystemStatus.READY for status in component_statuses):
            validation_result["overall_status"] = SystemStatus.READY
        elif any(status == SystemStatus.READY for status in component_statuses):
            validation_result["overall_status"] = SystemStatus.PARTIAL
        else:
            validation_result["overall_status"] = SystemStatus.FAILED
        
        return validation_result
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        import platform
        import sys
        
        return {
            "application": {
                "name": self.app_name,
                "version": self.version,
                "debug_mode": self.debug
            },
            "system": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "architecture": platform.architecture()[0]
            },
            "configuration": {
                "ollama_url": self.ollama.base_url,
                "ollama_model": self.ollama.model,
                "database_server": self.database.server,
                "database_name": self.database.database,
                "schema_file": self.schema.schema_file
            },
            "timestamp": datetime.now().isoformat()
        }

# Global configuration instance
config = ApplicationConfig()

def diagnose_system() -> Dict[str, Any]:
    """Run comprehensive system diagnostics"""
    logger.info("🔍 Running QueryMancer system diagnostics...")
    
    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "system_info": config.get_system_info(),
        "validation": config.validate_configuration(),
        "recommendations": []
    }
    
    # Generate recommendations based on validation results
    validation = diagnostics["validation"]
    
    if validation["overall_status"] == SystemStatus.FAILED:
        diagnostics["recommendations"].extend([
            "❌ Critical system components are not working",
            "🔧 Please check the error messages above and resolve issues"
        ])
    elif validation["overall_status"] == SystemStatus.PARTIAL:
        diagnostics["recommendations"].extend([
            "⚠️ Some components need attention",
            "🔧 System may work with limited functionality"
        ])
    else:
        diagnostics["recommendations"].extend([
            "✅ All systems operational!",
            "🚀 QueryMancer is ready for AI-powered SQL generation"
        ])
    
    # Component-specific recommendations
    components = validation.get("components", {})
    
    if components.get("ollama", {}).get("status") == SystemStatus.FAILED:
        diagnostics["recommendations"].extend([
            "🤖 Ollama Issues:",
            "   • Start Ollama server: ollama serve",
            "   • Install Mistral model: ollama run mistral",
            "   • Check OLLAMA_BASE_URL in .env file"
        ])
    
    if components.get("database", {}).get("status") == SystemStatus.FAILED:
        diagnostics["recommendations"].extend([
            "🗄️ Database Issues:",
            "   • Verify AWS SQL Server credentials in .env",
            "   • Check network connectivity to database server",
            "   • Ensure ODBC drivers are installed"
        ])
    
    if components.get("schema", {}).get("status") == SystemStatus.FAILED:
        diagnostics["recommendations"].extend([
            "📋 Schema Issues:",
            "   • Create schema.json file in project root",
            "   • Verify JSON syntax and structure",
            "   • Check SCHEMA_FILE path in .env"
        ])
    
    return diagnostics

def test_complete_pipeline() -> Dict[str, Any]:
    """Test the complete AI to SQL pipeline"""
    logger.info("🧪 Testing complete AI → SQL pipeline...")
    
    test_result = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_test": "incomplete",
        "steps": {},
        "overall_success": False
    }
    
    try:
        # Step 1: Test schema loading
        logger.info("Step 1: Testing schema loading...")
        schema_data = config.schema.load_schema()
        test_result["steps"]["schema_loading"] = {
            "success": bool(schema_data),
            "tables_loaded": len(schema_data.get('tables', {}))
        }
        
        # Step 2: Test Ollama connection
        logger.info("Step 2: Testing Ollama connection...")
        ollama_test = config.ollama.test_connection()
        test_result["steps"]["ollama_connection"] = ollama_test
        
        # Step 3: Test database connection
        logger.info("Step 3: Testing database connection...")
        db_test = config.database.test_connection()
        test_result["steps"]["database_connection"] = db_test
        
        # Step 4: Test AI SQL generation (if Ollama is working)
        if ollama_test.get("success"):
            logger.info("Step 4: Testing AI SQL generation...")
            try:
                from langchain_ollama import OllamaLLM
                
                llm = OllamaLLM(
                    model=config.ollama.model,
                    base_url=config.ollama.base_url,
                    temperature=0.1
                )
                
                test_prompt = "Generate a simple SQL query to select all records from a users table"
                response = llm.invoke(test_prompt)
                
                test_result["steps"]["ai_generation"] = {
                    "success": bool(response),
                    "response_length": len(response) if response else 0,
                    "contains_sql": "SELECT" in response.upper() if response else False
                }
                
            except Exception as e:
                test_result["steps"]["ai_generation"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Determine overall success
        successful_steps = sum(1 for step in test_result["steps"].values() 
                             if step.get("success", False))
        total_steps = len(test_result["steps"])
        
        test_result["overall_success"] = successful_steps >= 3  # At least 3 critical steps working
        test_result["success_rate"] = (successful_steps / total_steps) * 100 if total_steps > 0 else 0
        
        logger.info(f"Pipeline test completed: {successful_steps}/{total_steps} steps successful")
        
    except Exception as e:
        test_result["pipeline_test"] = "failed"
        test_result["error"] = str(e)
        logger.error(f"Pipeline test failed: {e}")
    
    return test_result

if __name__ == "__main__":
    print("🚀 QueryMancer Configuration Diagnostics")
    print("=" * 50)
    print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👤 User: Mohsin Ramzan")