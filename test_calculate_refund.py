from part_d import calculate_refund

print("Test 1: Delivery in 19 minutes")
print(calculate_refund(127.50, 19))

print("\nTest 2: Delivery in exactly 45 minutes")
print(calculate_refund(500, 45))

print("\nTest 3: Delivery in 46 minutes")
print(calculate_refund(500, 46))

print("\nTest 4: Delivery in 60 minutes")
print(calculate_refund(1500, 60))