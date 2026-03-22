from database import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT id, name, timestamp, event
    FROM attendance_events
    ORDER BY id DESC
""")

rows = cursor.fetchall()
conn.close()

for row in rows:
    print(row)