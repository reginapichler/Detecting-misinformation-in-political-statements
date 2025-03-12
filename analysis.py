# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:38:21 2025

@author: repic
"""

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#%%
###### FUNCTION DEFINITION FOR PREPROCESSING ######

# asign given label to the output of the LLM (string: text output of the LLM)
def assign_label(string):
    string = str(string)
    if 'true' in string.lower():
        return 'T'
    elif 'false' in string.lower():
        return 'F'
    elif 'explanation' in string.lower():
        return 'E'
    else:
        return None

# clean the annotated labels (sometimes spaces have been in the label)
def clean_labels(string):
    if 'E ' in string:
        return 'E'
    elif 'F ' in string:
        return 'F'
    elif 'T ' in string:
        return'T'
    else:
        return string
    
# clean the annotated topics (slight differences happend during annotation)
def merge_topic(data_complete):
    data_complete = data_complete.merge(trump_data[['original', 'topic', 'CNN', 'fact check']], on=['original', 'CNN', 'fact check'], how='left', suffixes=('', '_new'))
    data_complete['topic'] = data_complete['topic_new'].fillna(data_complete['topic'])
    data_complete = data_complete.drop(columns=['topic_new'])
    
    # Merge with harris_data
    data_complete = data_complete.merge(harris_data[['original', 'topic', 'CNN', 'fact check']], on=['original', 'CNN', 'fact check'], how='left', suffixes=('', '_new'))
    data_complete['topic'] = data_complete['topic_new'].fillna(data_complete['topic'])
    data_complete = data_complete.drop(columns=['topic_new'])
    
    # Merge with debate_data
    data_complete = data_complete.merge(debate_data[['original', 'topic', 'CNN', 'fact check']], on=['original', 'CNN', 'fact check'], how='left', suffixes=('', '_new'))
    data_complete['topic'] = data_complete['topic_new'].fillna(data_complete['topic'])
    data_complete = data_complete.drop(columns=['topic_new'])
    data_complete = data_complete.drop_duplicates(subset=['original', 'CNN'])
    
    data_complete.loc[data_complete['topic'].isin(['NATO', 'military', 'fact about the US', 'Hurricane Response', 'story in the past']), 'topic'] = 'security'
    data_complete.loc[data_complete['topic'].isin(['abortion', 'Paris Climate Accord']), 'topic'] = 'social'
    data_complete.loc[data_complete['topic'].isin(['statement about a candidate']), 'topic'] = 'candidate'
    data_complete.loc[data_complete['topic'].isin(['economy/foreign policy']), 'topic'] = 'economy'

    
    return data_complete

# load excel data
def load_excel_data(base_path, file_name):
    # Construct the full file path
    file_path = os.path.join(base_path, file_name)
    
    # Load the Excel file into a DataFrame
    data = pd.read_excel(file_path)
    
    return data
#%%
###### READ DATA ######

# define parent directory (wouldn't be necessary in this git project)
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
base_path = ".\\Outputs02_promptohneName"       # insert name of desired output folder
# for the debate
debate_claude = load_excel_data(base_path, 'debate_output_claude.xlsx')
#extract confidence and label from response of LLM
debate_claude['confidence'] = debate_claude['response'].str.extract(r'(\d+)%')
debate_claude['label'] = debate_claude['response'].apply(assign_label)
debate_claude['T/F/E'] = debate_claude['T/F/E'].apply(clean_labels)
#%%
# prepare data of the debate and gpt
debate_gpt = load_excel_data(base_path, 'debate_output_openai.xlsx')
# extract the confidence
debate_gpt['confidence'] = debate_gpt['response'].str.extract(r'(\d+)%')
# apply the functions defined above to extract labels
debate_gpt['label'] = debate_gpt['response'].apply(assign_label)
debate_gpt['T/F/E'] = debate_gpt['T/F/E'].apply(clean_labels)
#%%
# prepare data of harris and claude, steps are similar to the ones for the debate
harris_claude = load_excel_data(base_path, 'harris_output_claude.xlsx')
harris_claude['confidence'] = harris_claude['response'].str.extract(r'(\d+)%')
harris_claude['label'] = harris_claude['response'].apply(assign_label)
harris_claude['T/F/E'] = harris_claude['T/F/E'].apply(clean_labels)
#%%
# prepare data of harris and gpt, steps are similar to the ones for the debate
harris_gpt = load_excel_data(base_path, 'harris_output_openai.xlsx')
harris_gpt['confidence'] = harris_gpt['response'].str.extract(r'(\d+)%')
harris_gpt['label'] = harris_gpt['response'].apply(assign_label)
harris_gpt['T/F/E'] = harris_gpt['T/F/E'].apply(clean_labels)
#%%
# prepare data of trump and claude, steps are similar to the ones for the debate
trump_claude = load_excel_data(base_path, 'trump_output_claude.xlsx')
trump_claude['confidence'] = trump_claude['response'].str.extract(r'(\d+)%')
trump_claude['label'] = trump_claude['response'].apply(assign_label)
trump_claude['T/F/E'] = trump_claude['T/F/E'].apply(clean_labels)
#%%
# prepare data of trump and gpt, steps are similar to the ones for the debate
trump_gpt = load_excel_data(base_path, 'trump_output_openai.xlsx')
trump_gpt['confidence'] = trump_gpt['response'].str.extract(r'(\d+)%')
trump_gpt['label'] = trump_gpt['response'].apply(assign_label)
trump_gpt['T/F/E'] = trump_gpt['T/F/E'].apply(clean_labels)
#%%
###### FILTER DATA ######

# combine all datasets to complete data for gpt and claude
gpt_complete = pd.concat([debate_gpt, harris_gpt, trump_gpt])
claude_complete = pd.concat([debate_claude, harris_claude, trump_claude])

# create subsets of labels for gpt and claude separately
true_gpt = gpt_complete[gpt_complete['T/F/E'] == 'T']
false_gpt = gpt_complete[gpt_complete['T/F/E'] == 'F']
explanation_gpt = gpt_complete[gpt_complete['T/F/E'] == 'E']

true_claude = claude_complete[claude_complete['label'] == 'T']
false_claude = claude_complete[claude_complete['label'] == 'F']
explanation_claude = claude_complete[claude_complete['label'] == 'E']
    

#%%

###### DEFINE TOPICS AND REMOVE DUPLICATES ######

# topics: economy, social, security, candidate
# read orignal data (with updated topis)
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
# fix topics and drop duplicates
gpt_complete = merge_topic(gpt_complete)
claude_complete = merge_topic(claude_complete)

#%%
# filter topics
gpt_social = gpt_complete[gpt_complete['topic'] == 'social']
gpt_economy = gpt_complete[gpt_complete['topic'] == 'economy']
gpt_candidate = gpt_complete[gpt_complete['topic'] == 'candidate']
gpt_security = gpt_complete[gpt_complete['topic'] == 'security']

claude_social = claude_complete[claude_complete['topic'] == 'social']
claude_economy = claude_complete[claude_complete['topic'] == 'economy']
claude_candidate = claude_complete[claude_complete['topic'] == 'candidate']
claude_security = claude_complete[claude_complete['topic'] == 'security']

#%%
# check length
print(len(gpt_complete))
print(len(claude_complete))
#%%
###### FUNCTION DEFINITION FOR ANALYSIS ######

# extract the name of a dataframe
def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

# analyse a dataset: compute classification metrics and save them in a folder
def analyse_dataset(data):
    truth = data['T/F/E']
    llm = data['label']
    # Report with Precision, Recall, F1-Score
    report = (classification_report(truth, llm, labels=['T', 'F', 'E']))
    file_path = '.\\classification_reports\\classification_report_{dataset}.txt'
    path = file_path.format(dataset=get_df_name(data))
    with open(path, 'w') as file:
        file.write(report)
    
# calculate metrics: precision, recall, f1 and accuracy
def calculate_metrics(dataset):
    truth = dataset['T/F/E']
    llm = dataset['label']
    precision = precision_score(truth, llm, average="weighted", zero_division=0)
    recall = recall_score(truth, llm, average="weighted", zero_division=0)
    f1 = f1_score(truth, llm, average="weighted")
    accuracy = accuracy_score(truth, llm)
    out = {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1}
    return(out)

# for test cases (not used for final analysis in the paper): calculate the metrics for binary labels (true label is given to the function, could be T, F or E)
def calculate_metrics_binary(dataset, true_label):
    dataset = dataset.copy()
    dataset['T/F/E'] = np.where(dataset['T/F/E'].astype('string') == true_label, 1, 0)
    dataset['label'] = np.where(dataset['label'].astype('string') == true_label, 1, 0)
    truth = dataset['T/F/E']
    llm = dataset['label']
    precision = precision_score(truth, llm, average="weighted", zero_division=0)
    recall = recall_score(truth, llm, average="weighted", zero_division=0)
    f1 = f1_score(truth, llm, average="weighted")
    accuracy = accuracy_score(truth, llm)
    out = {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1}
    return(out)

#%%
###### CALCULATE METRICS FOR ALL SUBSETS ######

m_debate_gpt = calculate_metrics(debate_gpt)
m_harris_gpt = calculate_metrics(harris_gpt)
m_trump_gpt = calculate_metrics(trump_gpt)
m_complete_gpt = calculate_metrics(gpt_complete)

m_debate_claude = calculate_metrics(debate_claude)
m_harris_claude = calculate_metrics(harris_claude)
m_trump_claude = calculate_metrics(trump_claude)
m_complete_claude = calculate_metrics(claude_complete)

# combine metrics
metrics = {
    'debate_gpt': m_debate_gpt,
    'debate_claude': m_debate_claude,
    'harris_gpt': m_harris_gpt,
    'harris_claude': m_harris_claude,
    'trump_gpt': m_trump_gpt,
    'trump_claude': m_trump_claude,
    'gpt_complete': m_complete_gpt,
    'claude_complete': m_complete_claude
}

# Convert the dictionary to a DataFrame
metrics_df = pd.DataFrame(metrics)

# Display the DataFrame
metrics_df = metrics_df.T  # Transpose for better readability
print(metrics_df)
#%%
###### CALCULATE METRICS AND MEAN CONFIDENCE FOR ALL SUBSETS ######

# define all datasets which should be included
data_list = ['debate_gpt', 'debate_claude', 'harris_gpt', 'harris_claude', 'trump_gpt', 'trump_claude', 'gpt_complete', 'claude_complete', 'gpt_social', 'gpt_economy', 'gpt_security', 'gpt_candidate', 'claude_social', 'claude_economy', 'claude_security', 'claude_candidate']

# compute metrics for the datalist defined above: share of each label, metrics, mean confidence for each label, number of observations for each subset
for dat in data_list:
    df = globals()[dat]

    analyse_dataset(df)
    # compute shares for each label
    t_share_value = (df["T/F/E"] == "T").sum() / len(df['T/F/E'])
    f_share_value = (df["T/F/E"] == "F").sum() / len(df['T/F/E'])
    e_share_value = (df["T/F/E"] == "E").sum() / len(df['T/F/E'])
    
    metrics_df.loc[dat, 'T_share'] = t_share_value
    metrics_df.loc[dat, 'F_share'] = f_share_value
    metrics_df.loc[dat, 'E_share'] = e_share_value
    
    # compute mean confidence per label
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    mean_confidence_per_label = df.groupby('T/F/E')['confidence'].mean()
    # Store the mean confidence for each label in metrics_df
    for label in ['T', 'F', 'E']:
        metrics_df.loc[dat, f'{label}_mean_confidence'] = mean_confidence_per_label.get(label, np.nan)
    
    #compute mean confidence over all labels
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')    
    metrics_df.loc[dat, 'mean_confidence'] = np.mean(df['confidence'])
    metrics_df.loc[dat, 'n_obs'] = len(df)
#%%
# save metrics
metrics_df.to_excel('.\\metrics.xlsx')

#%%
# exclude?
# analysis of difference between labels, only for false and explanation bc too few right answers
metrics_labels = {
    #'true_gpt': calculate_metrics(true_gpt),
    #'true_claude': calculate_metrics(true_claude),
    'false_gpt': calculate_metrics(false_gpt),
    'false_claude': calculate_metrics(false_claude),
    'explanation_gpt': calculate_metrics(explanation_gpt),
    'explanation_claude': calculate_metrics(explanation_claude)
}

# Convert the metrics into a DataFrame
metrics_labels = pd.DataFrame(metrics_labels)
 #%%

# only for testing: check label subsets
a = calculate_metrics_binary(gpt_complete, 'F')
b = calculate_metrics_binary(claude_complete, 'F')
c = calculate_metrics_binary(gpt_complete, 'E')
d = calculate_metrics_binary(claude_complete, 'E')

#%%
###### BAR PLOTS ######
# create bar plots of the weighted F1-score

# only harris and trump
harris_complete_gpt = gpt_complete[gpt_complete['person'] == 'Harris']
trump_complete_gpt = gpt_complete[gpt_complete['person'] == 'Trump']
harris_complete_claude = claude_complete[claude_complete['person'] == 'Harris']
trump_complete_claude = claude_complete[claude_complete['person'] == 'Trump']

# extract the f1-scores
f1_scores = {
    'harris_gpt': calculate_metrics(harris_complete_gpt)['f1'],
    'trump_gpt': calculate_metrics(trump_complete_gpt)['f1'],
    'harris_claude': calculate_metrics(harris_complete_claude)['f1'],
    'trump_claude': calculate_metrics(trump_complete_claude)['f1']
}

# create plot
fig, ax = plt.subplots()
ax.bar(f1_scores.keys(), f1_scores.values(), color=['#FFB84D', '#FF9933', '#E60000', '#B30000'])

# Add descriptions
ax.set_title('Weighted F1-Score Comparison: Harris vs. Trump (GPT vs. Claude)')
ax.set_ylabel('F1-Score')
ax.set_xlabel('Model and speaker')

plt.ylim(0, 1)
plt.yticks(np.arange(0, 1, 0.1))


# set ticks of x axis
ax.set_xticklabels(['GPT: Harris', 'GPT: Trump', 'Claude: Harris', 'Claude: Trump'])

# finetune diagram
plt.tight_layout()

plt.show()

#%%
# create a plot for the comparison of gpt and claude
selected_models = ['gpt_complete', 'claude_complete']
f1_scores = metrics_df.loc[selected_models, 'f1']

# create diagram
fig, ax = plt.subplots()  # Erstelle das Axes-Objekt
ax.bar(f1_scores.index, f1_scores, color=['#FF9933', '#B30000'])

ax.set_title('F1-Score Comparison: GPT vs. Claude')
ax.set_ylabel('F1-Score')
ax.set_xlabel('Model')

ax.set_xticklabels(['GPT', 'Claude'])

plt.tight_layout()

plt.show()
#%% 
####### CONFUSION MATRICES AND METRICS #######

# for debate and claude
truth = debate_claude['T/F/E']
llm = debate_claude['label']
# Report with Precision, Recall, F1-Score
print(classification_report(truth, llm, labels=['T', 'F', 'E'], zero_division=0))

# Plot confusion matrix
cm = confusion_matrix(truth, llm, labels=['T', 'F', 'E'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['True', 'False', 'Explanation needed'], yticklabels=['True', 'False', 'Explanation needed'])
plt.xlabel('Predicted by claude')
plt.ylabel('Ground truth')
plt.title('Confusion Matrix for the debate')
plt.savefig('C:\\Users\\repic\\Documents\\Studium\\Semester5\\LLM_Seminar\\Codework\\plots\\debate_claude.png', dpi=300)
plt.show()
#%%
# for debate and gpt
truth = debate_gpt['T/F/E']
llm = debate_gpt['label']

# Report with Precision, Recall, F1-Score
print(classification_report(truth, llm, labels=['T', 'F', 'E']))

# plot confusion matrix
cm = confusion_matrix(truth, llm, labels=['T', 'F', 'E'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['True', 'False', 'Explanation needed'], yticklabels=['True', 'False', 'Explanation needed'])
plt.xlabel('Predicted by openai')
plt.ylabel('Ground truth')
plt.title('Confusion Matrix for the debate')
plt.savefig('C:\\Users\\repic\\Documents\\Studium\\Semester5\\LLM_Seminar\\Codework\\plots\\debate_openai.png', dpi=300)
plt.show()
#%%
# for harris and claude
truth = harris_claude['T/F/E']
llm = harris_claude['label']

# Report with Precision, Recall, F1-Score
print(classification_report(truth, llm, labels=['T', 'F', 'E']))

# plot confusion matrix
cm = confusion_matrix(truth, llm, labels=['T', 'F', 'E'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['True', 'False', 'Explanation needed'], yticklabels=['True', 'False', 'Explanation needed'])
plt.xlabel('Predicted by claude')
plt.ylabel('Ground truth')
plt.title('Confusion Matrix for the speech of Kamala Harris')
plt.savefig('C:\\Users\\repic\\Documents\\Studium\\Semester5\\LLM_Seminar\\Codework\\plots\\harris_claude.png', dpi=300)
plt.show()
#%%
# for harris and gpt
truth = harris_gpt['T/F/E']
llm = harris_gpt['label']

# Report with Precision, Recall, F1-Score
print(classification_report(truth, llm, labels=['T', 'F', 'E']))

# plot confusion matrix
cm = confusion_matrix(truth, llm, labels=['T', 'F', 'E'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['True', 'False', 'Explanation needed'], yticklabels=['True', 'False', 'Explanation needed'])
plt.xlabel('Predicted by openai')
plt.ylabel('Ground truth')
plt.title('Confusion Matrix for the speech of Kamala Harris')
plt.savefig('C:\\Users\\repic\\Documents\\Studium\\Semester5\\LLM_Seminar\\Codework\\plots\\harris_openai.png', dpi=300)
plt.show()
#%%
# for trump and claude
truth = trump_claude['T/F/E']
llm = trump_claude['label']

# Report with Precision, Recall, F1-Score
print(classification_report(truth, llm, labels=['T', 'F', 'E']))

# plot confusion matrix
cm = confusion_matrix(truth, llm, labels=['T', 'F', 'E'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['True', 'False', 'Explanation needed'], yticklabels=['True', 'False', 'Explanation needed'])
plt.xlabel('Predicted by claude')
plt.ylabel('Ground truth')
plt.title('Confusion Matrix for the speech of Donald Trump')
plt.savefig('C:\\Users\\repic\\Documents\\Studium\\Semester5\\LLM_Seminar\\Codework\\plots\\trump_claude.png', dpi=300)
plt.show()
#%%
# for trump and gpt

truth = trump_gpt['T/F/E']
llm = trump_gpt['label']

# Report with Precision, Recall, F1-Score
print(classification_report(truth, llm, labels=['T', 'F', 'E']))

# plot confusion matrix
cm = confusion_matrix(truth, llm, labels=['T', 'F', 'E'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['True', 'False', 'Explanation needed'], yticklabels=['True', 'False', 'Explanation needed'])
plt.xlabel('Predicted by openai')
plt.ylabel('Ground truth')
plt.title('Confusion Matrix for the speech of Donald Trump')
plt.savefig('C:\\Users\\repic\\Documents\\Studium\\Semester5\\LLM_Seminar\\Codework\\plots\\trump_openai.png', dpi=300)
plt.show()
#%%
# for gpt complete
truth = gpt_complete['T/F/E']
llm = gpt_complete['label']
print('for gpt complete')

# Report with Precision, Recall, F1-Score
print(classification_report(truth, llm, labels=['T', 'F', 'E']))

# plot confusion matrix
cm = confusion_matrix(truth, llm, labels=['T', 'F', 'E'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['True', 'False', 'Explanation needed'], yticklabels=['True', 'False', 'Explanation needed'])
plt.xlabel('Predicted by GPT')
plt.ylabel('Ground truth')
plt.title('Confusion Matrix for all predictions of GPT')
plt.savefig('C:\\Users\\repic\\Documents\\Studium\\Semester5\\LLM_Seminar\\Codework\\plots\\openai_complete.png', dpi=300)
plt.show()

print('For claude complete:')
truth = claude_complete['T/F/E']
llm = claude_complete['label']
# Report with Precision, Recall, F1-Score
print(classification_report(truth, llm, labels=['T', 'F', 'E']))

cm = confusion_matrix(truth, llm, labels=['T', 'F', 'E'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['True', 'False', 'Explanation needed'], yticklabels=['True', 'False', 'Explanation needed'])
plt.xlabel('Predicted by Claude')
plt.ylabel('Ground truth')
plt.title('Confusion Matrix for all predictions of Claude')
plt.savefig('C:\\Users\\repic\\Documents\\Studium\\Semester5\\LLM_Seminar\\Codework\\plots\\claude_complete.png', dpi=300)
plt.show()

#%%
###### ANALYSIS OF TOPICS ######

# define function for analysing a topic of a desired model (the model is also specified in the used dataset)
def analyse_topic(dataset, topic, model):
    data = dataset[dataset['topic'] == topic]
    truth = data['T/F/E']
    llm = data['label']
    # compute metrics
    report = classification_report(truth, llm, labels=['T', 'F', 'E'])
    # plot the confusion matrix
    cm = confusion_matrix(truth, llm, labels=['T', 'F', 'E'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['True', 'False', 'Explanation needed'], yticklabels=['True', 'False', 'Explanation needed'])
    plt.xlabel(f'Predicted by {model}')
    plt.ylabel('Ground truth')
    plt.title(f'Confusion Matrix for the topic {topic}, {model}')
    plt.savefig(f'C:\\Users\\repic\\Documents\\Studium\\Semester5\\LLM_Seminar\\Codework\\plots\\confusion_mat_{model}_{topic}.png', dpi=300)
    plt.show()
    return report
#%%
# analyse the topics for claude and gpt
analyse_topic(claude_complete, 'economy', 'Claude')
analyse_topic(claude_complete, 'social', 'Claude')
analyse_topic(claude_complete, 'security', 'Claude')
analyse_topic(claude_complete, 'candidate', 'Claude')

analyse_topic(gpt_complete, 'economy', 'GPT')
analyse_topic(gpt_complete, 'social', 'GPT')
analyse_topic(gpt_complete, 'security', 'GPT')
analyse_topic(gpt_complete, 'candidate', 'GPT')

#%%
###### PLOT LABEL DISTRIBUTION ######

# define a function for plotting the label distribution
def plot_pie(ax, data, title):
    counts = data['T/F/E'].value_counts()
    color_dict = {'T': '#2ca02c', 'F': '#d62728', 'E': '#1f77b4'}
    colors = [color_dict[label] for label in counts.index]
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors = colors)
    ax.set_title(title)

# create pie chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# label axes
plot_pie(axes[0], gpt_complete, "Overall Distribution")
plot_pie(axes[1], harris_gpt, "For Kamala Harris")
plot_pie(axes[2], trump_gpt, "For Donald Trump")

plt.tight_layout()
plt.show()