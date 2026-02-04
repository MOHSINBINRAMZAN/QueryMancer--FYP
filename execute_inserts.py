"""
Execute the INSERT statements to populate the CMS database
"""

import pyodbc
from datetime import datetime

# Database connection
SERVER = r'MECHREVO-\SQLEXPRESS'
DATABASE = 'CMS'

def get_connection():
    """Get database connection"""
    conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;'
    return pyodbc.connect(conn_str)

def main():
    print("=" * 70)
    print("EXECUTING SAMPLE DATA INSERTS FOR CMS DATABASE")
    print("=" * 70)
    print()
    
    # Connect to database
    print("Connecting to database...")
    conn = get_connection()
    cursor = conn.cursor()
    
    # Read the SQL file
    print("Reading insert_sample_data.sql...")
    with open('insert_sample_data.sql', 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    # Split by GO statements and execute each batch
    batches = sql_content.split('\nGO\n')
    
    success_count = 0
    error_count = 0
    current_table = ""
    
    print(f"\nExecuting {len(batches)} batches...")
    print()
    
    for i, batch in enumerate(batches):
        batch = batch.strip()
        if not batch or batch.startswith('--'):
            continue
        
        # Extract table name for progress
        if '-- Table:' in batch:
            lines = batch.split('\n')
            for line in lines:
                if '-- Table:' in line:
                    current_table = line.replace('-- Table:', '').strip()
                    break
        
        try:
            # Remove PRINT statements and handle IDENTITY_INSERT
            statements = []
            for line in batch.split('\n'):
                line = line.strip()
                if line.startswith('PRINT'):
                    continue
                if line.startswith('BEGIN TRY') or line.startswith('END TRY'):
                    continue
                if line.startswith('BEGIN CATCH') or line.startswith('END CATCH'):
                    continue
                if line.startswith('USE '):
                    continue
                if line.startswith('SET NOCOUNT'):
                    continue
                if 'ERROR_MESSAGE' in line:
                    continue
                if line:
                    statements.append(line)
            
            batch_sql = '\n'.join(statements)
            
            if batch_sql.strip():
                # Execute each INSERT separately
                for stmt in batch_sql.split(';'):
                    stmt = stmt.strip()
                    if stmt and stmt.upper().startswith(('INSERT', 'SET IDENTITY')):
                        try:
                            cursor.execute(stmt)
                            if stmt.upper().startswith('INSERT'):
                                success_count += 1
                        except Exception as e:
                            error_str = str(e)
                            # Ignore duplicate key errors - data might already exist
                            if 'duplicate' in error_str.lower() or 'violation' in error_str.lower():
                                pass
                            else:
                                error_count += 1
                                if error_count <= 20:  # Only show first 20 errors
                                    print(f"  ❌ {current_table}: {error_str[:80]}")
                
                conn.commit()
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(batches)} batches... ({success_count} inserts, {error_count} errors)")
                
        except Exception as e:
            error_count += 1
            if error_count <= 20:
                print(f"  ❌ Batch error at {current_table}: {str(e)[:80]}")
    
    conn.commit()
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Successful inserts: {success_count}")
    print(f"❌ Errors: {error_count}")
    
    # Verify by counting records
    print("\nVerifying data in tables...")
    cursor.execute("""
        SELECT t.name AS TableName, 
               SUM(p.rows) AS RowCount
        FROM sys.tables t
        INNER JOIN sys.partitions p ON t.object_id = p.object_id
        WHERE p.index_id IN (0, 1)
        GROUP BY t.name
        HAVING SUM(p.rows) > 0
        ORDER BY RowCount DESC
    """)
    
    tables_with_data = cursor.fetchall()
    print(f"\n📊 Tables with data: {len(tables_with_data)}")
    
    if tables_with_data:
        print("\nTop 20 tables by record count:")
        for i, (table_name, row_count) in enumerate(tables_with_data[:20]):
            print(f"  {i+1:3}. {table_name}: {row_count} records")
    
    total_records = sum(row[1] for row in tables_with_data)
    print(f"\n📈 Total records in database: {total_records}")
    
    cursor.close()
    conn.close()
    
    print("\n✅ Data insertion complete!")

if __name__ == "__main__":
    main()
