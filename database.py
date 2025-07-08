import sqlite3

def init_db():
    conn = sqlite3.connect('gesture_auth.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Positive gesture samples
    c.execute('''
        CREATE TABLE IF NOT EXISTS gesture_samples (
            sample_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            feature_data BLOB,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    # Negative gesture samples (NEW)
    c.execute('''
        CREATE TABLE IF NOT EXISTS negative_samples (
            sample_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            feature_data BLOB,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    # Access logs
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
    
    # Indexes
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_user_id ON gesture_samples (user_id)
    ''')
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_negative_user_id ON negative_samples (user_id)
    ''')
    
    conn.commit()
    return conn