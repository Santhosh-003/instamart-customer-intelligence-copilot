from part_d import issue_refund, escalate_to_human

print("Test 1: Record an approved refund")
print(issue_refund("TEST-REFUND-001", 500))

print("\nTest 2: Record a refund requiring human approval")
print(escalate_to_human("TEST-REFUND-002", 1500))