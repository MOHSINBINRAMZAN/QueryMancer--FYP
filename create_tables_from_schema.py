"""
Script to automatically create all tables from schema.json in the connected SQL Server database.
This script handles foreign key relationships and creates tables in the correct order.
"""

import json
import pyodbc
from collections import defaultdict
from config import DatabaseConfig
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_schema(schema_path: str = "schema.json") -> dict:
    """Load the schema from JSON file."""
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_connection():
    """Get database connection using config."""
    config = DatabaseConfig()
    conn_str = config.generate_connection_string()
    logger.info(f"Connecting to database: {config.server}/{config.database}")
    return pyodbc.connect(conn_str)

def get_sql_type(column_name: str) -> str:
    """
    Infer SQL data type based on column name patterns.
    This is a heuristic approach since schema.json only contains column names.
    """
    col_upper = column_name.upper()
    
    # ID/Key columns
    if col_upper == 'RECORD_ID' or col_upper.endswith('_ID'):
        return "INT"
    
    # Date columns
    if any(date_word in col_upper for date_word in ['DATE', '_ON', 'TIME', 'TIMESTAMP']):
        if 'TIME' in col_upper and 'DATE' not in col_upper:
            return "TIME"
        return "DATETIME"
    
    # Boolean/Flag columns
    if any(bool_word in col_upper for bool_word in ['DELETED', 'ACTIVE', 'APPROVED', 'COMPLETED', 
                                                      'CONFIRMED', 'CEASED', 'CHECKED', 'WINNER',
                                                      'FINALIST', 'ENTRANT', 'INVITED', 'ORAL', 
                                                      'POSTER', 'MAIN_', 'IS_', 'HAS_', 'CAN_',
                                                      'ACCEPT_', 'SHOW_', 'SEND_', 'STOP_',
                                                      'DO_NOT_', 'USE_', 'VAT_REGISTERED',
                                                      'CHARITY', 'INDEPENDENT', 'LIMITED_COMPANY',
                                                      'PROSPECT', 'BOUNCED', 'EXCLUDE_']):
        return "BIT"
    
    # Numeric columns
    if any(num_word in col_upper for num_word in ['NUMBER', 'COUNT', 'TOTAL', 'AMOUNT', 
                                                   'QUANTITY', 'SCORE', 'PERCENTAGE', 'LIMIT',
                                                   'BALANCE', 'CREDIT', 'DEBIT', 'PRICE',
                                                   'COST', 'FEE', 'VALUE', 'RATE', 'VOTES',
                                                   'LATITUDE', 'LONGITUDE', 'DISCOUNT']):
        if 'AMOUNT' in col_upper or 'PRICE' in col_upper or 'COST' in col_upper or 'VALUE' in col_upper or 'FEE' in col_upper or 'BALANCE' in col_upper:
            return "DECIMAL(18,2)"
        if 'LATITUDE' in col_upper or 'LONGITUDE' in col_upper or 'PERCENTAGE' in col_upper or 'RATE' in col_upper:
            return "DECIMAL(18,6)"
        return "INT"
    
    # Text/Content columns - use larger types
    if any(text_word in col_upper for text_word in ['DESCRIPTION', 'REMARKS', 'NOTE', 'COMMENT',
                                                     'BODY', 'CONTENT', 'BIO', 'ABSTRACT', 
                                                     'SUMMARY', 'DETAILS', 'MESSAGE', 'TEXT']):
        return "NVARCHAR(MAX)"
    
    # File/Image columns
    if any(file_word in col_upper for file_word in ['FILE', 'IMAGE', 'LOGO', 'ATTACHMENT', 
                                                     'PHOTO', 'PICTURE', 'DOCUMENT']):
        return "NVARCHAR(500)"
    
    # Email columns
    if 'EMAIL' in col_upper:
        return "NVARCHAR(255)"
    
    # URL/Website columns
    if any(url_word in col_upper for url_word in ['WEBSITE', 'URL', 'LINK', 'FACEBOOK', 
                                                   'TWITTER', 'LINKEDIN', 'YOUTUBE', 'GOOGLE']):
        return "NVARCHAR(500)"
    
    # Phone/Fax columns
    if any(phone_word in col_upper for phone_word in ['TELEPHONE', 'PHONE', 'MOBILE', 'FAX', 'SMS']):
        return "NVARCHAR(50)"
    
    # Address columns
    if any(addr_word in col_upper for addr_word in ['ADDRESS', 'STREET', 'CITY', 'TOWN', 
                                                     'POSTCODE', 'ZIP', 'COUNTY', 'COUNTRY']):
        return "NVARCHAR(255)"
    
    # Name columns - moderate length
    if 'NAME' in col_upper or 'TITLE' in col_upper:
        return "NVARCHAR(255)"
    
    # Color columns
    if col_upper == 'COLOR' or col_upper == 'COLOUR':
        return "NVARCHAR(50)"
    
    # Default to NVARCHAR(255) for unknown columns
    return "NVARCHAR(255)"

def clean_column_name(col_name: str) -> str:
    """Clean column name and wrap in brackets if needed."""
    # Remove any existing brackets
    col_name = col_name.strip().strip('[]')
    # Wrap in brackets to handle special characters
    return f"[{col_name}]"

def get_table_dependencies(schema: dict) -> dict:
    """
    Build a dependency graph showing which tables depend on which.
    Returns a dict where keys are table names and values are sets of tables they depend on.
    """
    dependencies = defaultdict(set)
    
    for table_name, table_info in schema.items():
        if isinstance(table_info, dict) and 'foreign_keys' in table_info:
            for fk_col, fk_ref in table_info['foreign_keys'].items():
                if isinstance(fk_ref, str) and '.' in fk_ref:
                    ref_table = fk_ref.split('.')[0].strip()
                    if ref_table != table_name:  # Exclude self-references
                        dependencies[table_name].add(ref_table)
    
    return dependencies

def topological_sort(schema: dict) -> list:
    """
    Sort tables so that dependent tables come after their dependencies.
    Uses Kahn's algorithm for topological sorting.
    """
    dependencies = get_table_dependencies(schema)
    all_tables = set(schema.keys())
    
    # Tables with no dependencies
    no_deps = [t for t in all_tables if t not in dependencies or not dependencies[t]]
    sorted_tables = []
    
    while no_deps:
        table = no_deps.pop(0)
        sorted_tables.append(table)
        
        # Remove this table from others' dependencies
        for other_table in list(dependencies.keys()):
            dependencies[other_table].discard(table)
            if not dependencies[other_table]:
                if other_table not in sorted_tables and other_table not in no_deps:
                    no_deps.append(other_table)
    
    # Add any remaining tables (might have circular dependencies)
    remaining = all_tables - set(sorted_tables)
    sorted_tables.extend(remaining)
    
    return sorted_tables

def generate_create_table_sql(table_name: str, table_info: dict, include_fk: bool = False) -> str:
    """Generate CREATE TABLE SQL statement for a table."""
    if not isinstance(table_info, dict):
        return None
    
    columns = table_info.get('columns', [])
    if not columns:
        return None
    
    schema_name = table_info.get('schema', 'dbo')
    primary_keys = table_info.get('primary_keys', [])
    
    # Normalize primary_keys to list
    if isinstance(primary_keys, str):
        primary_keys = [primary_keys]
    
    # Build column definitions
    col_defs = []
    for col in columns:
        col_clean = clean_column_name(col)
        col_type = get_sql_type(col)
        
        # Make RECORD_ID NOT NULL if it's a primary key
        if col.upper() == 'RECORD_ID' or col in primary_keys:
            col_defs.append(f"    {col_clean} {col_type} NOT NULL")
        else:
            col_defs.append(f"    {col_clean} {col_type} NULL")
    
    # Add primary key constraint
    if primary_keys:
        pk_cols = ', '.join([clean_column_name(pk) for pk in primary_keys])
        col_defs.append(f"    CONSTRAINT [PK_{table_name}] PRIMARY KEY CLUSTERED ({pk_cols})")
    
    columns_sql = ',\n'.join(col_defs)
    
    create_sql = f"""
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{table_name}' AND schema_id = SCHEMA_ID('{schema_name}'))
BEGIN
    CREATE TABLE [{schema_name}].[{table_name}] (
{columns_sql}
    );
    PRINT 'Created table: {schema_name}.{table_name}';
END
ELSE
BEGIN
    PRINT 'Table already exists: {schema_name}.{table_name}';
END
"""
    return create_sql

def generate_foreign_key_sql(table_name: str, table_info: dict, all_tables: set) -> list:
    """Generate ALTER TABLE statements to add foreign keys."""
    if not isinstance(table_info, dict):
        return []
    
    foreign_keys = table_info.get('foreign_keys', {})
    if not foreign_keys:
        return []
    
    schema_name = table_info.get('schema', 'dbo')
    fk_statements = []
    
    for fk_col, fk_ref in foreign_keys.items():
        if not isinstance(fk_ref, str) or '.' not in fk_ref:
            continue
        
        ref_parts = fk_ref.split('.')
        ref_table = ref_parts[0].strip()
        ref_col = ref_parts[1].strip() if len(ref_parts) > 1 else 'RECORD_ID'
        
        # Only add FK if referenced table exists in schema
        if ref_table not in all_tables:
            logger.warning(f"Skipping FK: {table_name}.{fk_col} -> {ref_table}.{ref_col} (table not found)")
            continue
        
        fk_name = f"FK_{table_name}_{fk_col}"
        
        fk_sql = f"""
IF NOT EXISTS (SELECT * FROM sys.foreign_keys WHERE name = '{fk_name}')
BEGIN
    BEGIN TRY
        ALTER TABLE [{schema_name}].[{table_name}]
        ADD CONSTRAINT [{fk_name}]
        FOREIGN KEY ([{fk_col}]) REFERENCES [{schema_name}].[{ref_table}]([{ref_col}]);
        PRINT 'Created FK: {fk_name}';
    END TRY
    BEGIN CATCH
        PRINT 'Failed to create FK: {fk_name} - ' + ERROR_MESSAGE();
    END CATCH
END
"""
        fk_statements.append(fk_sql)
    
    return fk_statements

def main():
    """Main function to create all tables from schema."""
    logger.info("="*60)
    logger.info("Starting table creation from schema.json")
    logger.info("="*60)
    
    # Load schema
    logger.info("Loading schema.json...")
    schema = load_schema()
    logger.info(f"Found {len(schema)} table definitions")
    
    # Get sorted table order
    logger.info("Analyzing table dependencies...")
    sorted_tables = topological_sort(schema)
    all_tables = set(schema.keys())
    
    # Connect to database
    logger.info("Connecting to database...")
    conn = get_connection()
    cursor = conn.cursor()
    
    # Phase 1: Create all tables (without foreign keys)
    logger.info("="*60)
    logger.info("PHASE 1: Creating tables...")
    logger.info("="*60)
    
    tables_created = 0
    tables_skipped = 0
    tables_failed = 0
    
    for table_name in sorted_tables:
        table_info = schema[table_name]
        create_sql = generate_create_table_sql(table_name, table_info)
        
        if create_sql:
            try:
                cursor.execute(create_sql)
                conn.commit()
                tables_created += 1
                if tables_created % 50 == 0:
                    logger.info(f"Progress: {tables_created} tables processed...")
            except Exception as e:
                logger.error(f"Failed to create table {table_name}: {str(e)}")
                tables_failed += 1
        else:
            tables_skipped += 1
    
    logger.info(f"Tables created/verified: {tables_created}")
    logger.info(f"Tables skipped: {tables_skipped}")
    logger.info(f"Tables failed: {tables_failed}")
    
    # Phase 2: Add foreign keys
    logger.info("="*60)
    logger.info("PHASE 2: Adding foreign key constraints...")
    logger.info("="*60)
    
    fk_created = 0
    fk_failed = 0
    
    for table_name in sorted_tables:
        table_info = schema[table_name]
        fk_statements = generate_foreign_key_sql(table_name, table_info, all_tables)
        
        for fk_sql in fk_statements:
            try:
                cursor.execute(fk_sql)
                conn.commit()
                fk_created += 1
            except Exception as e:
                logger.error(f"Failed to create FK for {table_name}: {str(e)}")
                fk_failed += 1
    
    logger.info(f"Foreign keys created/verified: {fk_created}")
    logger.info(f"Foreign keys failed: {fk_failed}")
    
    # Close connection
    cursor.close()
    conn.close()
    
    logger.info("="*60)
    logger.info("Table creation complete!")
    logger.info("="*60)
    
    return {
        'tables_created': tables_created,
        'tables_skipped': tables_skipped,
        'tables_failed': tables_failed,
        'fk_created': fk_created,
        'fk_failed': fk_failed
    }

if __name__ == "__main__":
    try:
        results = main()
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Tables Created/Verified: {results['tables_created']}")
        print(f"Tables Skipped: {results['tables_skipped']}")
        print(f"Tables Failed: {results['tables_failed']}")
        print(f"Foreign Keys Created: {results['fk_created']}")
        print(f"Foreign Keys Failed: {results['fk_failed']}")
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise
