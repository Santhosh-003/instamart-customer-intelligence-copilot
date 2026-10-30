from part_d import get_order

import sqlite3

connection = sqlite3.connect("data/instamart.db")

cursor = connection.cursor()

cursor.execute("SELECT order_id FROM orders LIMIT 1")

row = cursor.fetchone()

connection.close()

if row is None:
    print("No orders found in the database.")
else:
    order_id = row[0]

    print("Testing an existing order:")
    print(get_order(order_id))

    print("\nTesting a nonexistent order:")
    print(get_order("INVALID_ORDER_ID"))