import sqlite3

def init_db():
    conn = sqlite3.connect('gesture_auth.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create gesture_samples table
    c.execute('''
        CREATE TABLE IF NOT EXISTS gesture_samples (
            sample_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            feature_data BLOB,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    # Create access_logs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            log_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            username TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence REAL,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    # Create indexes
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_user_id ON gesture_samples (user_id)
    ''')
    
    conn.commit()
    
    # Run migration to add new columns to existing databases
    migrate_database_for_negative_learning(conn)
    
    return conn

def migrate_database_for_negative_learning(conn):
    """Migrate existing database to support negative learning"""
    cursor = conn.cursor()
    
    try:
        print("🔍 Checking database schema for negative learning support...")
        
        # Check if is_negative column exists in gesture_samples
        cursor.execute("PRAGMA table_info(gesture_samples)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'is_negative' not in columns:
            print("➕ Adding is_negative column to gesture_samples table...")
            cursor.execute("ALTER TABLE gesture_samples ADD COLUMN is_negative INTEGER DEFAULT 0")
            print("✅ Added is_negative column")
        else:
            print("✅ is_negative column already exists")
        
        # Check if notes column exists in access_logs
        cursor.execute("PRAGMA table_info(access_logs)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'notes' not in columns:
            print("➕ Adding notes column to access_logs table...")
            cursor.execute("ALTER TABLE access_logs ADD COLUMN notes TEXT")
            print("✅ Added notes column")
        else:
            print("✅ notes column already exists")
        
        # Add additional indexes if they don't exist
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_negative ON gesture_samples (is_negative)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_logs_user_id ON access_logs (user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_logs_timestamp ON access_logs (timestamp)")
        except Exception as e:
            # Indexes might already exist, that's okay
            pass
        
        conn.commit()
        print("🎉 Database migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database migration failed: {e}")
        conn.rollback()
        return False

def get_database_stats(conn):
    """Get statistics about the database for debugging"""
    cursor = conn.cursor()
    
    stats = {}
    
    try:
        # User statistics
        cursor.execute("SELECT COUNT(*) FROM users WHERE username != 'UNAUTHORIZED_SAMPLES'")
        stats['registered_users'] = cursor.fetchone()[0]
        
        # Check if UNAUTHORIZED_SAMPLES user exists
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'UNAUTHORIZED_SAMPLES'")
        stats['has_unauthorized_user'] = cursor.fetchone()[0] > 0
        
        # Gesture samples statistics
        cursor.execute("SELECT COUNT(*) FROM gesture_samples WHERE is_negative = 0 OR is_negative IS NULL")
        stats['positive_samples'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM gesture_samples WHERE is_negative = 1")
        stats['negative_samples'] = cursor.fetchone()[0]
        
        # Access logs statistics
        cursor.execute("SELECT COUNT(*) FROM access_logs")
        stats['total_access_logs'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM access_logs WHERE notes LIKE '%positive%'")
        stats['positive_learning_logs'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM access_logs WHERE notes LIKE '%corrective%'")
        stats['corrective_learning_logs'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM access_logs WHERE notes LIKE '%negative%' OR notes LIKE '%unauthorized%'")
        stats['negative_learning_logs'] = cursor.fetchone()[0]
        
        return stats
        
    except Exception as e:
        print(f"Error getting database stats: {e}")
        return {}

def print_database_info(conn):
    """Print detailed information about the database"""
    print("\n📊 Database Information:")
    print("=" * 50)
    
    # Get and print statistics
    stats = get_database_stats(conn)
    
    print(f"👥 Registered Users: {stats.get('registered_users', 'Unknown')}")
    print(f"🚫 Unauthorized Samples User: {'✅ Exists' if stats.get('has_unauthorized_user') else '❌ Not created yet'}")
    print(f"📊 Positive Samples: {stats.get('positive_samples', 'Unknown')}")
    print(f"🚫 Negative Samples: {stats.get('negative_samples', 'Unknown')}")
    print(f"📝 Total Access Logs: {stats.get('total_access_logs', 'Unknown')}")
    print(f"🟢 Positive Learning Logs: {stats.get('positive_learning_logs', 'Unknown')}")
    print(f"🔴 Corrective Learning Logs: {stats.get('corrective_learning_logs', 'Unknown')}")
    print(f"🚫 Negative Learning Logs: {stats.get('negative_learning_logs', 'Unknown')}")
    
    # Check table schemas
    cursor = conn.cursor()
    
    print(f"\n🗂️  Table Schemas:")
    print("-" * 30)
    
    for table in ['users', 'gesture_samples', 'access_logs']:
        print(f"\n📋 {table.upper()} table:")
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            nullable = "NOT NULL" if col[3] else "NULL"
            default = f"DEFAULT {col[4]}" if col[4] else ""
            print(f"  • {col_name} ({col_type}) {nullable} {default}".strip())

def cleanup_database(conn):
    """Clean up test data and reset database for fresh start"""
    cursor = conn.cursor()
    
    print("🧹 Cleaning up database...")
    
    try:
        # Ask for confirmation
        response = input("This will delete ALL data. Are you sure? (type 'yes' to confirm): ")
        if response.lower() != 'yes':
            print("❌ Cleanup cancelled")
            return False
        
        # Delete all data
        cursor.execute("DELETE FROM access_logs")
        cursor.execute("DELETE FROM gesture_samples")
        cursor.execute("DELETE FROM users")
        
        conn.commit()
        print("✅ Database cleaned successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error cleaning database: {e}")
        conn.rollback()
        return False

def test_database_operations(conn):
    """Test basic database operations"""
    cursor = conn.cursor()
    
    print("🧪 Testing database operations...")
    
    try:
        # Test user insertion
        cursor.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", ("test_user",))
        cursor.execute("SELECT user_id FROM users WHERE username = 'test_user'")
        user_result = cursor.fetchone()
        if user_result:
            user_id = user_result[0]
            print(f"✅ Test user created with ID: {user_id}")
        else:
            print("❌ Failed to create test user")
            return False
        
        # Test positive sample insertion
        test_features = b"test_feature_data"
        cursor.execute(
            "INSERT INTO gesture_samples (user_id, feature_data, is_negative) VALUES (?, ?, ?)",
            (user_id, test_features, 0)
        )
        print("✅ Positive sample insertion test passed")
        
        # Test negative sample insertion
        cursor.execute(
            "INSERT INTO gesture_samples (user_id, feature_data, is_negative) VALUES (?, ?, ?)",
            (user_id, test_features, 1)
        )
        print("✅ Negative sample insertion test passed")
        
        # Test access log insertion with notes
        cursor.execute(
            "INSERT INTO access_logs (user_id, username, confidence, notes) VALUES (?, ?, ?, ?)",
            (user_id, "test_user", 0.95, "Learning sample (positive)")
        )
        print("✅ Access log with notes insertion test passed")
        
        # Clean up test data
        cursor.execute("DELETE FROM access_logs WHERE username = 'test_user'")
        cursor.execute("DELETE FROM gesture_samples WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE username = 'test_user'")
        
        conn.commit()
        print("✅ All database operation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Database operation test failed: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return False

# Command-line interface for database management
if __name__ == "__main__":
    import sys
    
    conn = init_db()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "info":
            print_database_info(conn)
        elif command == "test":
            test_database_operations(conn)
        elif command == "clean":
            cleanup_database(conn)
        elif command == "migrate":
            migrate_database_for_negative_learning(conn)
        else:
            print("Available commands:")
            print("  python database.py info     - Show database information")
            print("  python database.py test     - Test database operations")
            print("  python database.py clean    - Clean all data (DESTRUCTIVE)")
            print("  python database.py migrate  - Run migration for negative learning")
    else:
        print("✅ Database initialized successfully!")
        print_database_info(conn)
    
    conn.close()