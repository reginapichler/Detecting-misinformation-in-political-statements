# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:23:58 2025

@author: repic
"""

import os
import requests
import pandas as pd

#%%
# read data
data_dir = ".\\data\\"
# for the debate
file_path = os.path.join(data_dir, 'debate.xlsx')
debate_data = pd.read_excel(file_path)
# for the speech of Trump
file_path = os.path.join(data_dir, 'speech_trump.xlsx')
trump_data = pd.read_excel(file_path)
# for the speech of Harris
file_path = os.path.join(data_dir, 'speech_harris.xlsx')
harris_data = pd.read_excel(file_path)
#%%



#%%
# add API key, endpoint and prompts
key_dir = ".\\keys\\"
with open(os.path.join(key_dir, 'openai_key.txt'), 'r') as file:
    API_KEY = file.read()

with open(os.path.join(key_dir, 'endpoint_azure.txt'), 'r') as file:
    ENDPOINT = file.read()

system_prompt = "You are a fact checker. Check if the given claim is correct. Answer ONLY 'True', 'False' or 'Explanation needed' and a number how much percent you are sure. Nothing else, NO EXPLANATION!"
# %%

def send_requests(data):
    results = []
    errors = []
    for claim in data['original']:
        person = data.loc[data['original'] == claim, 'person'].iloc[0]
        text = f"'{person}: {claim}'"
        
        # Prepare the request payload for Azure OpenAI
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.95
        }
        
        # Set the headers for the request
        headers = {
            "Content-Type": "application/json",
            "api-key": API_KEY
        }
        # Send the request to the Azure OpenAI API
        try:
            response = requests.post(ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()  # Raises an error if the request was unsuccessful
        except requests.RequestException as e:
            errors.append({
                'person': person,
                'original': claim,
                'error': str(e)
            })
            continue
        
        # Extract the response content
        message_content = response.json()['choices'][0]['message']['content']
        
        # Store the claim and the API's response
        result = {
            'person': person,
            'original': claim,
            'response': message_content
        }
        results.append(result)
        
    print(f"Processed {len(results)} successful requests, {len(errors)} failed requests.")

    return results, errors
#%%
results_debate, errors_debate = send_requests(debate_data)
#%%
# Convert the list of results to a DataFrame
results_debate = pd.DataFrame(results_debate)
# merge with input df
debate_merged = pd.merge(debate_data, results_debate, on=['person', 'original'], how='outer')
# save as excel
debate_merged.to_excel('.\\debate_output_openai.xlsx', index=False)

#%%
# for harris
results_harris, errors_harris = send_requests(harris_data)
#%%
# Convert the list of results to a DataFrame
results_harris = pd.DataFrame(results_harris)
# merge with input df
harris_merged = pd.merge(harris_data, results_harris, on=['person', 'original'], how='outer')
# save as excel
harris_merged.to_excel('.\\harris_output_openai.xlsx', index=False)

#%%
# for trump
results_trump, errors_trump = send_requests(trump_data)
#%%
# Convert the list of results to a DataFrame
results_trump = pd.DataFrame(results_trump)
# merge with input df
trump_merged = pd.merge(trump_data, results_trump, on=['person', 'original'], how='outer')
# Convert confidence to numeric for analysis
#df['confidence'] = pd.to_numeric(df['confidence'])
# save as excel
trump_merged.to_excel('.\\trump_output_openai.xlsx', index=False)
