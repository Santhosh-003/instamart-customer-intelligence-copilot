import sqlite3

connection = sqlite3.connect("data/instamart.db")
cursor = connection.cursor()

cursor.execute("""
    SELECT order_id, status, order_amount, delivery_minutes
    FROM orders
    WHERE status = 'completed'
      AND delivery_minutes > 45
      AND order_amount <= 1000
    LIMIT 5
""")

orders = cursor.fetchall()

print("Orders eligible for a simulated refund:")

for order in orders:
    print(order)

if not orders:
    print("No matching orders found.")

connection.close()