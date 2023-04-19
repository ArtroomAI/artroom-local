import json
import requests
import csv
import time 

items_list = []

for page in range(1,63):
    print(f"Running for page {page}")
    # Make a GET request to the API endpoint and get the JSON response
    response = requests.get(f'https://civitai.com/api/v1/tags?limit=200&page={page}')
    json_response = response.json()
    # Extract the items list from the JSON response
    items_list += json_response['items']
# Create a list to store the tags data
tags_data = []
# Loop through each item in the items list and extract the relevant data
for item in items_list:
    try:
        name = item['name']
        modelCount = item['modelCount']
        link = item['link']
        tags_data.append((name, modelCount, link))
    except:
        pass
# Write the tags data to a text file with one tag per line
with open('tags.txt', 'w', encoding='utf-8') as f:
    for tag in tags_data:
        try:
            f.write(tag[0] + '\n')
        except:
            print(f"Failed on {tag[0]}")

# Write the tags data to a CSV file with columns for name, modelCount, and link
with open('tags.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['name', 'modelCount', 'link'])
    for tag in tags_data:
        try:
            writer.writerow([tag[0], tag[1], tag[2]])   
        except:
            print(f"Failed on {[tag[0], tag[1], tag[2]]}")