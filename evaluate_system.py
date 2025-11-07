import requests
import pandas as pd
import time

# Test queries
test_queries = [
    "I want a sci-fi movie with time travel and a twist ending",
    "Show me a romantic comedy with a happy ending",
    "Recommend a thriller with unexpected plot twists",
    "I'm looking for an animated movie for children",
    "Find me a war movie based on true events"
]

# API endpoint
url = "http://localhost:8000/recommend"

# Evaluate each query
results = []
for query in test_queries:
    start_time = time.time()
    
    payload = {"text": query}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    
    end_time = time.time()
    response_time = end_time - start_time
    
    if response.status_code == 200:
        result = response.json()
        recommendations = result["recommendations"]
        
        results.append({
            "Query": query,
            "Response": recommendations,
            "Response Time (s)": response_time,
            "Response Length": len(recommendations)
        })
    else:
        results.append({
            "Query": query,
            "Response": f"Error: {response.status_code}",
            "Response Time (s)": response_time,
            "Response Length": 0
        })

# Save results
df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)
print("Evaluation completed. Results saved to evaluation_results.csv")

# Print summary
print("\nSummary:")
print(f"Total queries: {len(test_queries)}")
print(f"Successful responses: {len([r for r in results if 'Error' not in r['Response']])}")
print(f"Average response time: {df['Response Time (s)'].mean():.2f} seconds")
print(f"Average response length: {df['Response Length'].mean():.0f} characters")
