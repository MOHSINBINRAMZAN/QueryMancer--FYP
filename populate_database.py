"""
Database Population Script for QueryMancer
==========================================
This script ensures all database tables have more than 5 rows of sample data.
It reads the schema.json to understand table structures and generates appropriate data.

Author: QueryMancer
Date: 2026-02-07
"""

import pyodbc
import json
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os

# Database connection settings
SERVER = r'MECHREVO-\SQLEXPRESS'
DATABASE = 'CMS'
MIN_ROWS_PER_TABLE = 6  # Ensure at least 6 rows (more than 5)

# Sample data generators
FIRST_NAMES = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 'Robert', 'Lisa', 'William', 'Jennifer', 
               'James', 'Emma', 'Daniel', 'Olivia', 'Matthew', 'Sophia', 'Andrew', 'Isabella', 'Joseph', 'Mia']
LAST_NAMES = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
              'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin']
CITIES = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego',
          'Dallas', 'San Jose', 'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte', 'Seattle', 'Denver',
          'Boston', 'Nashville', 'Baltimore']
COUNTRIES = ['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Spain', 'Italy', 'Netherlands', 'Belgium']
COLORS = ['#FF5733', '#33FF57', '#3357FF', '#FF33F5', '#F5FF33', '#33FFF5', '#FF8C33', '#8C33FF', '#33FF8C', '#FF3333']
STATUSES = ['Active', 'Inactive', 'Pending', 'Approved', 'Rejected', 'In Progress', 'Completed', 'On Hold']
DESCRIPTIONS = [
    'Sample record for testing purposes',
    'Demo data entry for development',
    'Test record with sample values',
    'Placeholder data for demonstration',
    'Example entry for system testing',
    'Development sample record',
    'Quality assurance test data',
    'Integration testing record',
    'User acceptance test data',
    'System validation entry'
]
COMPANIES = ['TechCorp Inc', 'Global Solutions Ltd', 'Innovation Systems', 'Digital Dynamics', 'Smart Services Co',
             'Future Tech LLC', 'Premier Partners', 'Elite Enterprises', 'Strategic Solutions', 'Prime Industries']


def get_connection():
    """Establish database connection using Windows Authentication"""
    conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;'
    return pyodbc.connect(conn_str)


def load_schema() -> Dict[str, Any]:
    """Load the database schema from schema.json"""
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.json')
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_table_row_count(cursor, table_name: str) -> int:
    """Get the current number of rows in a table"""
    try:
        cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
        return cursor.fetchone()[0]
    except Exception as e:
        return -1  # Table doesn't exist or error


def get_max_record_id(cursor, table_name: str) -> int:
    """Get the maximum RECORD_ID in a table"""
    try:
        cursor.execute(f"SELECT ISNULL(MAX(RECORD_ID), 0) FROM [{table_name}]")
        return cursor.fetchone()[0]
    except Exception:
        return 0


def generate_random_date(start_year: int = 2024, end_year: int = 2026) -> str:
    """Generate a random date string"""
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    return random_date.strftime('%Y-%m-%d')


def generate_random_datetime() -> str:
    """Generate a random datetime string"""
    return generate_random_date() + ' ' + f'{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}'


def generate_random_email(first_name: str, last_name: str) -> str:
    """Generate a random email address"""
    domains = ['example.com', 'test.org', 'sample.net', 'demo.io', 'company.com']
    return f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 999)}@{random.choice(domains)}"


def generate_random_phone() -> str:
    """Generate a random phone number"""
    return f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"


def generate_column_value(col_name: str, record_num: int, table_name: str) -> Any:
    """Generate an appropriate value for a column based on its name"""
    col_upper = col_name.upper()
    
    # Primary key
    if col_upper == 'RECORD_ID':
        return record_num
    
    # Skip foreign keys - will be handled separately or set to NULL
    if col_upper.endswith('_ID') and col_upper != 'RECORD_ID':
        return None
    
    # Boolean/bit fields
    if col_upper in ['DELETED', 'ACTIVE', 'APPROVED', 'PROCESSED', 'CONFIRMED']:
        return 0
    if col_upper.startswith('IS_') or col_upper.startswith('HAS_'):
        return random.choice([0, 1])
    
    # Date fields
    if 'DATE' in col_upper:
        if 'DELETED' in col_upper:
            return None
        return generate_random_date()
    
    # Name fields
    if col_upper in ['FIRST_NAME', 'FIRSTNAME']:
        return random.choice(FIRST_NAMES)
    if col_upper in ['LAST_NAME', 'LASTNAME', 'SURNAME']:
        return random.choice(LAST_NAMES)
    if col_upper == 'NAME' or col_upper.endswith('_NAME'):
        if 'ACCOUNT' in col_upper or 'COMPANY' in col_upper:
            return f"{random.choice(COMPANIES)} {record_num}"
        if 'CONTACT' in col_upper:
            return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        return f"{table_name} Item {record_num}"
    
    # Contact information
    if 'EMAIL' in col_upper:
        return generate_random_email(random.choice(FIRST_NAMES), random.choice(LAST_NAMES))
    if 'PHONE' in col_upper or 'TELEPHONE' in col_upper or 'MOBILE' in col_upper or 'FAX' in col_upper:
        return generate_random_phone()
    
    # Address fields
    if col_upper in ['CITY', 'TOWN']:
        return random.choice(CITIES)
    if 'COUNTRY' in col_upper and 'NAME' not in col_upper:
        return None  # FK
    if 'ADDRESS' in col_upper and 'LINE' in col_upper:
        return f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Maple', 'Cedar', 'Pine'])} Street"
    if 'POSTCODE' in col_upper or 'ZIP' in col_upper:
        return f"{random.randint(10000, 99999)}"
    
    # Color
    if col_upper == 'COLOR':
        return random.choice(COLORS)
    
    # Status fields
    if 'STATUS' in col_upper and 'NAME' not in col_upper:
        return None  # Usually FK
    if col_upper.endswith('_STATUS_NAME') or col_upper == 'STATUS_NAME':
        return random.choice(STATUSES)
    
    # Description/text fields
    if col_upper in ['DESCRIPTION', 'SHORT_DESCRIPTION', 'REMARKS', 'NOTE', 'NOTES', 'COMMENT', 'COMMENTS']:
        return f"{random.choice(DESCRIPTIONS)} - Record {record_num}"
    if col_upper == 'TITLE':
        return f"Title for Record {record_num}"
    if col_upper == 'BODY':
        return f"Body content for record {record_num}. This is sample text for testing purposes."
    
    # User tracking fields
    if col_upper in ['CREATED_BY', 'MODIFIED_BY', 'DELETED_BY']:
        if col_upper == 'DELETED_BY':
            return 0
        return 'admin'
    
    # Numeric fields
    if 'AMOUNT' in col_upper or 'PRICE' in col_upper or 'COST' in col_upper or 'VALUE' in col_upper:
        return round(random.uniform(10.0, 10000.0), 2)
    if 'QUANTITY' in col_upper or 'COUNT' in col_upper or 'NUMBER' in col_upper or 'TOTAL' in col_upper:
        return random.randint(1, 100)
    if 'PERCENTAGE' in col_upper or 'RATE' in col_upper:
        return round(random.uniform(0.0, 100.0), 2)
    
    # Website/URL
    if 'WEBSITE' in col_upper or 'URL' in col_upper:
        return f"https://www.example{record_num}.com"
    
    # Default for string columns
    return f"Sample {col_name} {record_num}"


def insert_records_into_table(cursor, conn, table_name: str, columns: List[str], rows_needed: int, start_id: int) -> int:
    """Insert records into a specific table"""
    inserted = 0
    
    # Filter out columns that are likely computed or identity columns (except RECORD_ID which we set)
    # Also filter out columns ending with _NAME that are computed from FKs
    skip_suffixes = ['_NAME']  # These are usually computed columns
    
    valid_columns = []
    for col in columns:
        col_upper = col.upper()
        # Skip columns that look like computed columns from foreign keys
        is_fk_name = any(col_upper.endswith(suffix) and col_upper.replace(suffix, '') in [c.upper() for c in columns] 
                       for suffix in skip_suffixes)
        if not is_fk_name:
            valid_columns.append(col)
    
    for i in range(rows_needed):
        record_id = start_id + i + 1
        
        values = {}
        for col in valid_columns:
            value = generate_column_value(col, record_id, table_name)
            if value is not None:
                values[col] = value
        
        if not values:
            continue
        
        # Build INSERT statement
        cols = list(values.keys())
        placeholders = ', '.join(['?' for _ in cols])
        col_names = ', '.join([f'[{c}]' for c in cols])
        
        sql = f"INSERT INTO [{table_name}] ({col_names}) VALUES ({placeholders})"
        
        try:
            cursor.execute(sql, list(values.values()))
            inserted += 1
        except Exception as e:
            error_str = str(e).lower()
            # Skip duplicate key errors silently
            if 'duplicate' in error_str or 'violation' in error_str:
                continue
            # Skip identity insert errors - try without RECORD_ID
            if 'identity' in error_str:
                try:
                    if 'RECORD_ID' in values:
                        del values['RECORD_ID']
                        cols = list(values.keys())
                        placeholders = ', '.join(['?' for _ in cols])
                        col_names = ', '.join([f'[{c}]' for c in cols])
                        sql = f"INSERT INTO [{table_name}] ({col_names}) VALUES ({placeholders})"
                        cursor.execute(sql, list(values.values()))
                        inserted += 1
                except:
                    pass
            # Skip other errors but continue
            continue
    
    try:
        conn.commit()
    except:
        pass
    
    return inserted


def main():
    """Main function to populate database tables"""
    print("=" * 70)
    print("  DATABASE POPULATION SCRIPT FOR QUERYMANCER")
    print("  Ensuring all tables have more than 5 rows of data")
    print("=" * 70)
    print()
    
    # Load schema
    print("📋 Loading schema.json...")
    schema = load_schema()
    print(f"   Found {len(schema)} tables in schema")
    print()
    
    # Connect to database
    print("🔌 Connecting to database...")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        print(f"   Connected to {SERVER}/{DATABASE}")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return
    print()
    
    # Process each table
    print("📊 Processing tables...")
    print("-" * 70)
    
    tables_processed = 0
    tables_updated = 0
    total_rows_inserted = 0
    tables_skipped = 0
    tables_with_errors = 0
    
    for table_name, table_info in schema.items():
        tables_processed += 1
        columns = table_info.get('columns', [])
        
        if not columns:
            print(f"   ⚠️  {table_name}: No columns defined, skipping")
            tables_skipped += 1
            continue
        
        # Check current row count
        current_count = get_table_row_count(cursor, table_name)
        
        if current_count == -1:
            print(f"   ❌ {table_name}: Table not found or error")
            tables_with_errors += 1
            continue
        
        if current_count >= MIN_ROWS_PER_TABLE:
            print(f"   ✅ {table_name}: Already has {current_count} rows")
            continue
        
        # Calculate rows needed
        rows_needed = MIN_ROWS_PER_TABLE - current_count
        max_id = get_max_record_id(cursor, table_name)
        
        print(f"   🔄 {table_name}: Has {current_count} rows, inserting {rows_needed} more...")
        
        # Insert records
        inserted = insert_records_into_table(cursor, conn, table_name, columns, rows_needed, max_id)
        
        if inserted > 0:
            tables_updated += 1
            total_rows_inserted += inserted
            # Verify new count
            new_count = get_table_row_count(cursor, table_name)
            print(f"      ✅ Inserted {inserted} rows (now has {new_count} rows)")
        else:
            print(f"      ⚠️  Could not insert rows")
    
    print()
    print("-" * 70)
    print("📈 SUMMARY")
    print("-" * 70)
    print(f"   Tables in schema:     {len(schema)}")
    print(f"   Tables processed:     {tables_processed}")
    print(f"   Tables updated:       {tables_updated}")
    print(f"   Tables skipped:       {tables_skipped}")
    print(f"   Tables with errors:   {tables_with_errors}")
    print(f"   Total rows inserted:  {total_rows_inserted}")
    print()
    
    # Close connection
    cursor.close()
    conn.close()
    print("✅ Database population complete!")
    print()


if __name__ == "__main__":
    main()
