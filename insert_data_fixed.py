"""
Execute INSERT statements to populate the CMS database
Fixed version - handles IDENTITY columns properly
"""

import pyodbc
import json
from datetime import datetime

# Database connection
SERVER = r'MECHREVO-\SQLEXPRESS'
DATABASE = 'CMS'

def get_connection():
    """Get database connection"""
    conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;'
    return pyodbc.connect(conn_str, autocommit=False)

def load_schema():
    """Load schema from JSON file"""
    with open('schema.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def get_existing_tables(cursor):
    """Get list of existing tables in database"""
    cursor.execute("""
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA = 'dbo'
    """)
    return set(row[0] for row in cursor.fetchall())

def get_table_info(cursor, table_name):
    """Get columns and identity info for a table"""
    cursor.execute(f"""
        SELECT 
            c.COLUMN_NAME, 
            c.DATA_TYPE, 
            c.CHARACTER_MAXIMUM_LENGTH,
            c.IS_NULLABLE,
            COLUMNPROPERTY(OBJECT_ID('dbo.' + c.TABLE_NAME), c.COLUMN_NAME, 'IsIdentity') as IsIdentity
        FROM INFORMATION_SCHEMA.COLUMNS c
        WHERE c.TABLE_NAME = ? AND c.TABLE_SCHEMA = 'dbo'
        ORDER BY c.ORDINAL_POSITION
    """, (table_name,))
    return cursor.fetchall()

def topological_sort(schema, existing_tables):
    """Sort tables based on foreign key dependencies"""
    dependencies = {}
    for table_name in existing_tables:
        if table_name in schema:
            fks = schema[table_name].get('foreign_keys', {})
            deps = set()
            for fk_col, fk_ref in fks.items():
                parent_table = fk_ref.split('.')[0]
                if parent_table in existing_tables and parent_table != table_name:
                    deps.add(parent_table)
            dependencies[table_name] = deps
        else:
            dependencies[table_name] = set()
    
    sorted_tables = []
    no_deps = [t for t, deps in dependencies.items() if not deps]
    
    while no_deps:
        table = no_deps.pop(0)
        sorted_tables.append(table)
        
        for t, deps in dependencies.items():
            if table in deps:
                deps.remove(table)
                if not deps and t not in sorted_tables and t not in no_deps:
                    no_deps.append(t)
    
    for table in existing_tables:
        if table not in sorted_tables:
            sorted_tables.append(table)
    
    return sorted_tables

def generate_value(col_name, data_type, max_length, record_num, table_name):
    """Generate sample value based on column name and type"""
    col_upper = col_name.upper()
    
    # Boolean/bit fields
    if data_type == 'bit':
        return 0
    
    # Date fields
    if data_type in ['datetime', 'datetime2', 'date', 'smalldatetime']:
        if 'DELETED' in col_upper:
            return None
        elif 'CREATED' in col_upper:
            return f'2025-01-{15 + record_num:02d}'
        elif 'MODIFIED' in col_upper:
            return f'2025-01-{16 + record_num:02d}'
        elif 'START' in col_upper:
            return f'2025-02-{record_num:02d} 09:00:00'
        elif 'END' in col_upper:
            return f'2025-02-{record_num:02d} 17:00:00'
        else:
            return f'2025-01-{20 + record_num:02d}'
    
    # Numeric fields
    if data_type in ['int', 'bigint', 'smallint', 'tinyint']:
        if col_upper in ['DELETED', 'ACTIVE', 'APPROVED', 'ENABLED']:
            return 0
        elif 'AMOUNT' in col_upper or 'TOTAL' in col_upper:
            return record_num * 100
        elif 'COUNT' in col_upper or 'NUMBER' in col_upper:
            return record_num * 5
        else:
            return record_num
    
    if data_type in ['decimal', 'numeric', 'money', 'float', 'real']:
        if 'AMOUNT' in col_upper or 'PRICE' in col_upper:
            return record_num * 99.99
        else:
            return record_num * 10.0
    
    # String fields
    if data_type in ['varchar', 'nvarchar', 'char', 'nchar', 'text', 'ntext']:
        max_len = min(max_length or 50, 100) if max_length and max_length > 0 else 50
        
        if col_upper == 'COLOR':
            colors = ['#FF5733', '#33FF57', '#3357FF']
            return colors[(record_num - 1) % len(colors)]
        elif col_upper in ['CREATED_BY', 'MODIFIED_BY', 'DELETED_BY']:
            return 'admin'
        elif 'EMAIL' in col_upper:
            return f'user{record_num}@example.com'
        elif 'PHONE' in col_upper or 'TELEPHONE' in col_upper or 'MOBILE' in col_upper:
            return f'+1-555-{100 + record_num:03d}-{1000 + record_num:04d}'
        elif col_upper == 'NAME' or col_upper.endswith('_NAME'):
            base = col_upper.replace('_NAME', '').replace('_', ' ').title()
            return f'{base} {record_num}'[:max_len]
        elif 'DESCRIPTION' in col_upper or 'REMARKS' in col_upper:
            return f'Sample {col_upper.lower()} {record_num}'[:max_len]
        elif 'TITLE' in col_upper:
            return f'Title {record_num}'[:max_len]
        elif 'CODE' in col_upper:
            return f'{table_name[:3]}{record_num:04d}'[:max_len]
        else:
            return f'Sample {record_num}'[:max_len]
    
    # GUID
    if data_type == 'uniqueidentifier':
        return None  # Let SQL Server generate it
    
    return None

def insert_records(cursor, table_name, schema, num_records=2):
    """Insert sample records into a table"""
    columns = get_table_info(cursor, table_name)
    
    if not columns:
        return 0, 0
    
    # Find identity column
    identity_col = None
    for col_name, data_type, max_len, nullable, is_identity in columns:
        if is_identity:
            identity_col = col_name
            break
    
    # Get FK info
    fk_info = {}
    if table_name in schema:
        fk_info = {k.upper(): v for k, v in schema[table_name].get('foreign_keys', {}).items()}
    
    success = 0
    errors = 0
    
    for record_num in range(1, num_records + 1):
        col_names = []
        values = []
        placeholders = []
        
        for col_name, data_type, max_len, nullable, is_identity in columns:
            col_upper = col_name.upper()
            
            # Skip identity columns (unless it's RECORD_ID which we want to control)
            if is_identity and col_upper != 'RECORD_ID':
                continue
            
            # Handle RECORD_ID specially
            if col_upper == 'RECORD_ID':
                col_names.append(f'[{col_name}]')
                values.append(record_num)
                placeholders.append('?')
                continue
            
            # FK columns
            if col_upper in fk_info:
                col_names.append(f'[{col_name}]')
                values.append(min(record_num, 2))  # Reference record 1 or 2
                placeholders.append('?')
                continue
            
            # Generate value
            value = generate_value(col_name, data_type, max_len, record_num, table_name)
            col_names.append(f'[{col_name}]')
            values.append(value)
            placeholders.append('?')
        
        if col_names:
            sql = f"INSERT INTO [dbo].[{table_name}] ({', '.join(col_names)}) VALUES ({', '.join(placeholders)})"
            
            try:
                # Handle IDENTITY_INSERT if RECORD_ID is identity
                if identity_col and identity_col.upper() == 'RECORD_ID':
                    cursor.execute(f"SET IDENTITY_INSERT [dbo].[{table_name}] ON")
                
                cursor.execute(sql, values)
                
                if identity_col and identity_col.upper() == 'RECORD_ID':
                    cursor.execute(f"SET IDENTITY_INSERT [dbo].[{table_name}] OFF")
                
                success += 1
            except Exception as e:
                err = str(e).lower()
                if 'duplicate' not in err and 'violation of primary' not in err:
                    errors += 1
                    # Re-raise for debugging (first few errors only)
                    if errors <= 5:
                        print(f"    Error in {table_name}: {str(e)[:100]}")
    
    return success, errors

def main():
    print("=" * 70)
    print("INSERTING SAMPLE DATA INTO CMS DATABASE")
    print("=" * 70)
    print()
    
    # Load schema
    print("Loading schema...")
    schema = load_schema()
    
    # Connect
    print("Connecting to database...")
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get tables
    print("Getting tables...")
    existing_tables = get_existing_tables(cursor)
    print(f"Found {len(existing_tables)} tables")
    
    # Sort by dependencies
    print("Sorting by dependencies...")
    sorted_tables = topological_sort(schema, existing_tables)
    
    print(f"\nInserting 2 records per table...")
    print()
    
    total_success = 0
    total_errors = 0
    tables_with_data = 0
    
    for i, table_name in enumerate(sorted_tables):
        success, errors = insert_records(cursor, table_name, schema, num_records=2)
        
        if success > 0:
            conn.commit()
            total_success += success
            tables_with_data += 1
        
        total_errors += errors
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(sorted_tables)} tables ({total_success} records inserted)")
    
    conn.commit()
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Tables processed: {len(sorted_tables)}")
    print(f"✅ Records inserted: {total_success}")
    print(f"✅ Tables with data: {tables_with_data}")
    print(f"❌ Errors: {total_errors}")
    
    # Verify
    print("\nVerifying data...")
    cursor.execute("""
        SELECT t.name AS TableName, 
               SUM(p.rows) AS RowCnt
        FROM sys.tables t
        INNER JOIN sys.partitions p ON t.object_id = p.object_id
        WHERE p.index_id IN (0, 1)
        GROUP BY t.name
        HAVING SUM(p.rows) > 0
        ORDER BY RowCnt DESC
    """)
    
    results = cursor.fetchall()
    print(f"\n📊 Tables with data: {len(results)}")
    
    if results:
        print("\nSample of tables with data:")
        for table_name, row_count in results[:15]:
            print(f"  • {table_name}: {row_count} records")
        
        total_records = sum(row[1] for row in results)
        print(f"\n📈 Total records in database: {total_records}")
    
    cursor.close()
    conn.close()
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
