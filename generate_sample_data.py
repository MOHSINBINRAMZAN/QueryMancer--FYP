"""
Generate Sample Data INSERT Statements for CMS Database
This script generates 2 sample records per table in the correct order
to satisfy foreign key constraints.
"""

import json
import pyodbc
from datetime import datetime, timedelta
import random
import string

# Database connection
SERVER = r'MECHREVO-\SQLEXPRESS'
DATABASE = 'CMS'

def get_connection():
    """Get database connection"""
    conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;'
    return pyodbc.connect(conn_str)

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

def get_table_columns(cursor, table_name):
    """Get actual columns from database table"""
    cursor.execute(f"""
        SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = 'dbo'
        ORDER BY ORDINAL_POSITION
    """)
    return cursor.fetchall()

def topological_sort(schema, existing_tables):
    """Sort tables based on foreign key dependencies"""
    # Build dependency graph
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
    
    # Topological sort using Kahn's algorithm
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
    
    # Add remaining tables (circular dependencies)
    for table in existing_tables:
        if table not in sorted_tables:
            sorted_tables.append(table)
    
    return sorted_tables

def generate_sample_value(col_name, data_type, max_length, record_num, table_name):
    """Generate appropriate sample value based on column name and type"""
    col_upper = col_name.upper()
    
    # Primary key - RECORD_ID
    if col_upper == 'RECORD_ID':
        return record_num
    
    # Foreign key references (will be set to 1 or 2 based on record_num)
    if col_upper.endswith('_ID') and col_upper != 'RECORD_ID':
        return min(record_num, 2)
    
    # Boolean/bit fields
    if data_type == 'bit' or col_upper in ['DELETED', 'ACTIVE', 'APPROVED', 'ENABLED', 'IS_ACTIVE', 'IS_DELETED']:
        return 0
    
    # Date fields
    if data_type in ['datetime', 'datetime2', 'date', 'smalldatetime']:
        if 'CREATED' in col_upper or 'DATE_CREATED' in col_upper:
            return f"'2025-01-{15 + record_num:02d}'"
        elif 'MODIFIED' in col_upper:
            return f"'2025-01-{16 + record_num:02d}'"
        elif 'DELETED' in col_upper:
            return 'NULL'
        elif 'START' in col_upper:
            return f"'2025-02-{record_num:02d} 09:00:00'"
        elif 'END' in col_upper:
            return f"'2025-02-{record_num:02d} 17:00:00'"
        else:
            return f"'2025-01-{20 + record_num:02d}'"
    
    # Numeric fields
    if data_type in ['int', 'bigint', 'smallint', 'tinyint']:
        if 'AMOUNT' in col_upper or 'TOTAL' in col_upper or 'VALUE' in col_upper:
            return record_num * 100
        elif 'COUNT' in col_upper or 'NUMBER' in col_upper or 'QUANTITY' in col_upper:
            return record_num * 5
        elif 'YEAR' in col_upper:
            return 2025
        elif 'MONTH' in col_upper:
            return record_num
        elif 'PERCENTAGE' in col_upper or 'RATE' in col_upper:
            return record_num * 10
        else:
            return record_num
    
    if data_type in ['decimal', 'numeric', 'money', 'float', 'real']:
        if 'AMOUNT' in col_upper or 'PRICE' in col_upper or 'COST' in col_upper or 'VALUE' in col_upper:
            return f"{record_num * 99.99:.2f}"
        elif 'PERCENTAGE' in col_upper or 'RATE' in col_upper:
            return f"{record_num * 5.5:.2f}"
        else:
            return f"{record_num * 10.00:.2f}"
    
    # String fields
    if data_type in ['varchar', 'nvarchar', 'char', 'nchar', 'text', 'ntext']:
        max_len = min(max_length or 50, 100) if max_length and max_length > 0 else 50
        
        # Name fields
        if col_upper == 'NAME' or col_upper.endswith('_NAME'):
            base_name = col_upper.replace('_NAME', '').replace('_', ' ').title()
            value = f"{base_name} {record_num}"
            return f"N'{value[:max_len]}'"
        
        # Email fields
        if 'EMAIL' in col_upper:
            return f"N'user{record_num}@example.com'"
        
        # Phone fields
        if 'PHONE' in col_upper or 'TELEPHONE' in col_upper or 'MOBILE' in col_upper or 'FAX' in col_upper:
            return f"N'+1-555-{100 + record_num:03d}-{1000 + record_num:04d}'"
        
        # Address fields
        if 'ADDRESS' in col_upper and 'LINE' in col_upper:
            return f"N'{record_num * 100} Sample Street'"
        if 'POSTCODE' in col_upper or 'ZIP' in col_upper:
            return f"N'{10000 + record_num}'"
        if 'TOWN' in col_upper or 'CITY' in col_upper:
            cities = ['London', 'Manchester', 'Birmingham', 'Leeds', 'Glasgow']
            return f"N'{cities[(record_num - 1) % len(cities)]}'"
        if 'COUNTY' in col_upper and 'NAME' not in col_upper:
            return f"N'County {record_num}'"
        if 'COUNTRY' in col_upper and 'NAME' not in col_upper:
            return f"N'United Kingdom'"
        
        # Description/remarks
        if 'DESCRIPTION' in col_upper or 'REMARKS' in col_upper or 'NOTE' in col_upper or 'COMMENT' in col_upper:
            value = f"Sample {col_upper.lower().replace('_', ' ')} for record {record_num}"
            return f"N'{value[:max_len]}'"
        
        # Title fields
        if 'TITLE' in col_upper:
            return f"N'Sample Title {record_num}'"
        
        # Code fields
        if 'CODE' in col_upper:
            return f"N'{table_name[:3].upper()}{record_num:04d}'"
        
        # Status fields
        if 'STATUS' in col_upper and 'NAME' not in col_upper:
            return f"N'Active'"
        
        # Type fields
        if 'TYPE' in col_upper and 'NAME' not in col_upper:
            return f"N'Type{record_num}'"
        
        # Created/Modified by
        if col_upper in ['CREATED_BY', 'MODIFIED_BY', 'DELETED_BY']:
            return f"N'admin'"
        
        # Color
        if col_upper == 'COLOR':
            colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33F5', '#F5FF33']
            return f"N'{colors[(record_num - 1) % len(colors)]}'"
        
        # Default string
        value = f"Sample {record_num}"
        return f"N'{value[:max_len]}'"
    
    # GUID/uniqueidentifier
    if data_type == 'uniqueidentifier':
        return f"NEWID()"
    
    # Default
    return 'NULL'

def generate_insert_for_table(cursor, table_name, schema, record_num):
    """Generate INSERT statement for a single record"""
    columns = get_table_columns(cursor, table_name)
    
    if not columns:
        return None
    
    col_names = []
    col_values = []
    
    # Get foreign key info from schema
    fk_info = {}
    if table_name in schema:
        fk_info = schema[table_name].get('foreign_keys', {})
    
    for col_name, data_type, max_length, is_nullable in columns:
        col_upper = col_name.upper()
        
        # Skip identity columns (except RECORD_ID which we'll handle)
        # Most systems use RECORD_ID as primary key
        
        # Check if this is a foreign key
        is_fk = col_name in fk_info or col_upper in [k.upper() for k in fk_info.keys()]
        
        # Generate value
        if is_fk and col_upper != 'RECORD_ID':
            # For FK, use 1 or 2 based on record number
            value = min(record_num, 2)
        else:
            value = generate_sample_value(col_name, data_type, max_length, record_num, table_name)
        
        col_names.append(f"[{col_name}]")
        col_values.append(str(value))
    
    if col_names:
        return f"INSERT INTO [dbo].[{table_name}] ({', '.join(col_names)}) VALUES ({', '.join(col_values)});"
    return None

def main():
    print("=" * 70)
    print("GENERATING SAMPLE DATA FOR CMS DATABASE")
    print("=" * 70)
    print()
    
    # Load schema
    print("Loading schema...")
    schema = load_schema()
    
    # Connect to database
    print("Connecting to database...")
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get existing tables
    print("Getting existing tables...")
    existing_tables = get_existing_tables(cursor)
    print(f"Found {len(existing_tables)} tables in database")
    
    # Sort tables by dependencies
    print("Sorting tables by foreign key dependencies...")
    sorted_tables = topological_sort(schema, existing_tables)
    
    # Generate SQL file
    output_file = 'insert_sample_data.sql'
    print(f"\nGenerating INSERT statements to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("-- ============================================\n")
        f.write("-- SAMPLE DATA INSERT STATEMENTS FOR CMS DATABASE\n")
        f.write(f"-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-- 2 records per table\n")
        f.write("-- ============================================\n\n")
        f.write("USE [CMS];\nGO\n\n")
        f.write("SET NOCOUNT ON;\n")
        f.write("SET IDENTITY_INSERT OFF;\n\n")
        
        success_count = 0
        skip_count = 0
        
        for table_name in sorted_tables:
            f.write(f"\n-- Table: {table_name}\n")
            f.write(f"PRINT 'Inserting into {table_name}...';\n")
            
            # Check if table has RECORD_ID as identity
            cursor.execute(f"""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = '{table_name}' 
                AND TABLE_SCHEMA = 'dbo'
                AND COLUMNPROPERTY(OBJECT_ID(TABLE_SCHEMA + '.' + TABLE_NAME), COLUMN_NAME, 'IsIdentity') = 1
            """)
            identity_col = cursor.fetchone()
            
            if identity_col:
                f.write(f"SET IDENTITY_INSERT [dbo].[{table_name}] ON;\n")
            
            f.write("BEGIN TRY\n")
            
            # Generate 2 records per table
            for record_num in [1, 2]:
                insert_sql = generate_insert_for_table(cursor, table_name, schema, record_num)
                if insert_sql:
                    f.write(f"    {insert_sql}\n")
                    success_count += 1
            
            f.write("END TRY\n")
            f.write("BEGIN CATCH\n")
            f.write(f"    PRINT 'Error inserting into {table_name}: ' + ERROR_MESSAGE();\n")
            f.write("END CATCH\n")
            
            if identity_col:
                f.write(f"SET IDENTITY_INSERT [dbo].[{table_name}] OFF;\n")
            
            f.write("GO\n")
        
        f.write("\n-- ============================================\n")
        f.write("-- SUMMARY\n")
        f.write("-- ============================================\n")
        f.write(f"-- Total INSERT statements generated: {success_count}\n")
        f.write("PRINT 'Sample data insertion complete!';\n")
    
    print(f"\n✅ Generated {success_count} INSERT statements")
    print(f"📄 Output saved to: {output_file}")
    print("\nTo execute, run in SQL Server Management Studio:")
    print(f"   Open {output_file} and execute")
    
    # Also generate a quick preview
    print("\n" + "=" * 70)
    print("PREVIEW (First 5 tables):")
    print("=" * 70)
    
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        preview_lines = []
        table_count = 0
        for line in lines:
            if '-- Table:' in line:
                table_count += 1
                if table_count > 5:
                    break
            if table_count <= 5:
                preview_lines.append(line)
        print(''.join(preview_lines[:100]))
    
    cursor.close()
    conn.close()
    
    print("\n✅ Script complete!")

if __name__ == "__main__":
    main()
