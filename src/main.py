from flask import Flask
import mariadb
import sys
import ids

# Connect to MariaDB Platform
try:
    # https://mariadb-corporation.github.io/mariadb-connector-python/module.html#mariadb.connect
    conn = mariadb.connect(
        user="db_user",
        password="db_user_passwd",
        host="192.0.2.1",
        port=3306,
        database="employees",
        connect_timeout=3,
    )
except mariadb.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

# Get Cursor
cur = conn.cursor()
app = Flask("idsendpoint")

@app.route('/')
def hello():
    return "<p>Hello!</p>"
