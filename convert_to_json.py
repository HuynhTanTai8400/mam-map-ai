import json
from mock_restaurants import sample_restaurants

# Ghi ra file JSON
with open("restaurant_data.json", "w", encoding="utf-8") as f:
    json.dump(sample_restaurants, f, ensure_ascii=False, indent=4)

print("✅ Đã chuyển thành công sang restaurant_data.json")
