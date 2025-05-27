import sqlite3

def init_db():
    conn = sqlite3.connect('gesture_auth.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS gesture_samples (
            sample_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            feature_data BLOB,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS negative_samples (
            sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            feature_data BLOB NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS access_logs (
        log_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        username TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        confidence REAL,
        success INTEGER DEFAULT 1,
        FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_user_id ON gesture_samples (user_id)
    ''')
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_neg_user_id ON negative_samples (user_id)
    ''')
    conn.commit()
    return conn
