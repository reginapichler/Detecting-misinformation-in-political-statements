# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:38:30 2024

@author: repic
"""

import anthropic
import os
import pandas as pd
import openpyxl


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

key_dir = ".\\keys\\"
with open(os.path.join(key_dir, 'key_claude.txt'), 'r') as file:
    key = file.read()

client = anthropic.Anthropic(
    api_key = key
)

system_prompt = "You are a fact checker. Check if the given claim is correct. Answer ONLY 'True', 'False' or 'Explanation needed' and a number how much percent you are sure. Nothing else, NO EXPLANATION!"

#%%
# function for sending requests
def send_requests(data):
    results = []
    # Iterate through the claims
    for claim in data['original']:
        person = data.loc[data['original'] == claim, 'person'].iloc[0]
        text = f"'{person}: {claim}'"
        
        message = client.messages.create(
           model="claude-3-5-haiku-20241022",
           max_tokens=100,
           system=system_prompt,
           messages=[
               {"role": "user", "content": text}
           ]
        )
        
        # Store the claim and the API's response
        result = {
            'person': person,
            'original': claim,
            'response': message.content
        }
        results.append(result)
    
    return results
#%%
# for debate data
results_debate = send_requests(debate_data)
#%%
# Convert the list of results to a DataFrame
results_debate = pd.DataFrame(results_debate)
# merge with input df
debate_merged = pd.merge(debate_data, results_debate, on=['person', 'original'], how='outer')
# save as excel
debate_merged.to_excel('.\\debate_output_claude.xlsx', index=False)

#%%
# for data of trumps speech
results_trump = send_requests(trump_data)
#%%
# Convert the list of results to a DataFrame
results_trump = pd.DataFrame(results_trump)
# merge with input df
trump_merged = pd.merge(trump_data, results_trump, on=['person', 'original'], how='outer')
# save as excel
trump_merged.to_excel('.\\trump_output_claude.xlsx', index=False)

#%%
# for data of harris' speech
results_harris = send_requests(harris_data)
#%%
# Convert the list of results to a DataFrame
results_harris = pd.DataFrame(results_harris)
# merge with input df
harris_merged = pd.merge(harris_data, results_harris, on=['person', 'original'], how='outer')
# save as excel
harris_merged.to_excel('.\\harris_output_claude.xlsx', index=False)
