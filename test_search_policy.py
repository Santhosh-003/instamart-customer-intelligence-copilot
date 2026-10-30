from part_d import search_policy

result = search_policy("refund policy late delivery")

print("Policy search results:")

for item in result["results"]:
    print("\nSource:", item["source"])
    print("Similarity:", round(item["score"], 4))
    print("Text:", item["text"])