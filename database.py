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
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    conn.commit()
    return conn
