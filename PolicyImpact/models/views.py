
from django.shortcuts import render, redirect
from openai import OpenAI
from collections import defaultdict
<<<<<<< HEAD
import requests,re
=======
import re
import torch
import json
import os
import random
from transformers import DistilBertTokenizerFast, DistilBertModel
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor


>>>>>>> jeshwanth

client = OpenAI(
  api_key="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"
)
def generate_suggestions(prompt):
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=[
        {"role": "user", "content": "You are a policy advisor AI assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    max_tokens=500
    )
    return completion.choices[0].message.content

def live_news(request):
    API_KEY = 'f730406a37954c52a9fc6d08e675b9cc'
    selected_topic = request.GET.get('topic', 'government policy')

    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': selected_topic,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 6,
        'apiKey': API_KEY
    }

    response = requests.get(url, params=params)
    articles = response.json().get("articles", [])

    sectors = [
        'government policy', 'agriculture', 'healthcare', 'education', 'economy',
        'technology', 'infrastructure', 'energy', 'defense', 'environment'
    ]

    context = {
        'articles': [
            {
                'title': article['title'],
                'description': article['description'] or 'No description available.',
                'url': article['url'],
                'source': article['source']['name'],
                'date': article['publishedAt'][:10]
            }
            for article in articles
        ],
        'sectors': sectors,
        'selected_topic': selected_topic
    }

    return render(request, 'news.html', context)

# def liven(request):
#     API_KEY = 'f730406a37954c52a9fc6d08e675b9cc'
#     url = 'https://newsapi.org/v2/everything'

#     params = {
#         'q': 'Indian government policy',
#         'language': 'en',
#         'sortBy': 'publishedAt',
#         'apiKey': API_KEY,
#         'pageSize': 20,
#         'domains': 'economictimes.indiatimes.com,businesstoday.in,livemint.com'
#     }

#     response = requests.get(url, params=params)
#     articles = response.json().get("articles", [])

#     news_headlines = [article['title'] for article in articles]
#     return render(request, 'home.html', {'news_headlines': news_headlines})

def home(request):
    render(request, 'home.html')
    API_KEY = 'f730406a37954c52a9fc6d08e675b9cc'
    url = 'https://newsapi.org/v2/everything'

    params = {
        'q': 'Indian government policy',
        'language': 'en',
        'sortBy': 'publishedAt',
        'apiKey': API_KEY,
        'pageSize': 20,
        'domains': 'economictimes.indiatimes.com,businesstoday.in,livemint.com'
    }

    response = requests.get(url, params=params)
    articles = response.json().get("articles", [])

    news_headlines = [article['title'] for article in articles]
    return render(request, 'home.html', {'news_headlines': news_headlines})
    # return render(request, 'home.html')

def chat(request):
    return render(request, 'chat.html')
 


def run_policy_nlp(policy_text):

    # Paths
    BASE_DIR = os.path.join(os.path.dirname(__file__), 'ml_models', 'model_files')

    # Load keywords
    with open(os.path.join(BASE_DIR, 'label_list.json'), 'r') as f:
        keywords = json.load(f)
    num_labels = len(keywords)

    # Define model class
    class MultiLabelClassifier(nn.Module):
        def __init__(self, num_labels):
            super(MultiLabelClassifier, self).__init__()
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        def forward(self, input_ids=None, attention_mask=None):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0]
            logits = self.classifier(pooled_output)
            return logits

    # Load model
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = MultiLabelClassifier(num_labels)
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'policy_classifier_model.pth'), map_location=torch.device('cpu')))
    model.eval()

    # External logic
    dd = ["drop", "decrease", "reduce", "decreasing", "decreased", "reduced", "dropped", "declined", "lowered", "diminished", "minimized", 
          "weakened", "contracted", "rolled back", "reducing", "dropping", "declining", "lowering", "diminishing", "minimizing", "weakening", "contracting", "rolling back"]
    mm = ["cutting", "removing", "removed", "trimming", "cut", "remove", "clear", "clearing", "cleared"]
    mmm = ["trees", "forest", "forests"]
    p = q = ohm = edu = 0

    ushh = policy_text.lower().split()
    for kk in ushh:
        if kk in dd:
            ohm = 1
        if kk == "education":
            edu = 1
        if kk in mm:
            p = 1
        if kk in mmm:
            q = 1
    kk_list = ["Deforestation", "Biodiversity loss", "Land clearing", "Forest removal", "Forest clearance"]

    # Predict
    inputs = tokenizer(policy_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs).squeeze()

    # Threshold logic
    absolute_threshold = 0.19
    relative_margin = 0.15
    max_prob = torch.max(probs).item()

    predicted_labels = [
        keywords[i]
        for i, prob in enumerate(probs)
        if prob.item() >= absolute_threshold or prob.item() >= (max_prob - relative_margin)
    ]
    if not predicted_labels:
        predicted_labels = [keywords[torch.argmax(probs).item()]]

    # Final logic
    final_keywords = []
    for chi in predicted_labels:
        if chi == "Subsidy Removal" and ohm == 1:
            final_keywords.append("Subsidy Removal")
        elif chi == "Education Subsidy" and edu == 1:
            final_keywords.append("Education Subsidy")
        elif chi not in ["Subsidy Removal", "Education Subsidy"]:
            final_keywords.append(chi)
    if p == 1 and q == 1:
        final_keywords.append(random.choice(kk_list))

    return final_keywords





def run_economic_model(sectors):
    BASE_DIR = os.path.join(os.path.dirname(__file__), 'ml_models', 'model_files')
    df = pd.read_csv(os.path.join(BASE_DIR, "dataset2.csv"))

    # Encode
    encoder = LabelEncoder()
    df["Policy_Type_Encoded"] = encoder.fit_transform(df["Policy_Type"])
    X = df[["Policy_Type_Encoded"]]
    y = df[[
        "GDP_Growth (%)",
        "Employment (%)",
        "Inflation (%)",
        "Healthcare_Index",
        "Education_Index",
        "Carbon_Emissions (MT)"
    ]]

    model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    model.fit(X, y)

    # Results
    results = {}

    deforestation_keywords = ["Deforestation", "Biodiversity loss", "Land clearing", "Forest removal", "Forest clearance"]
    for sector in sectors:
        if sector in deforestation_keywords:
            results[sector] = {
                "GDP_Growth (%)": 0.53,
                "Employment (%)": 0.67,
                "Inflation (%)": 1.04,
                "Healthcare_Index": 68.20,
                "Education_Index": 85.90,
                "Carbon_Emissions (MT)": 225.30,
            }
        else:
            encoded = encoder.transform([sector])
            pred = model.predict(pd.DataFrame({"Policy_Type_Encoded": encoded}))[0]
            results[sector] = dict(zip(y.columns, [round(p, 2) for p in pred]))
    return results
    


def parse_structured_response(response_lines):
    structured_data = defaultdict(list)
    current_section = None
    current_point = ""

    for line in response_lines:
        line = line.strip()
        if not line:
            continue
        # Section header like ### Investment Areas for Investors
        if line.startswith("###"):
            current_section = line.replace("###", "").strip()
            structured_data[current_section] = []
        # New main point (numbered)
        elif re.match(r"^\d+\.\s", line):
            if current_point:
                structured_data[current_section].append(current_point.strip())
            current_point = line
        # Bullet point under current main point
        elif line.startswith("-"):
            current_point += " " + line.strip()
        else:
            # Catch any additional content
            current_point += " " + line.strip()

    if current_point:
        structured_data[current_section].append(current_point.strip())

    return dict(structured_data)





def chat_process(request):
    if request.method == 'POST':
        user_input = request.POST['policy_input']
        
        # 1️⃣ NLP-Based Sector Analysis
        sectors = run_policy_nlp(user_input)

        # 2️⃣ Economic Prediction (depends on sectors)
        results = run_economic_model(sectors)

        # # 3️⃣ Investment Forecasting (depends on econ data)
        # investment_data = run_investment_model(economic_data)

        # # 4️⃣ Policy Optimization (depends on previous outputs)
        # optimized_policy = run_optimization_model(user_input, sectors, economic_data)
        output_from_some_process = "Policy on AI regulation in healthcare" 
         # Replace with your dynamic logic
        prompt = f"""
        A new policy affecting {sectors} has been proposed.
        It is predicted to change of GDP_Growth,carbom emmission , employment rate,inflation rate,health
        index , education index as {results}%.
        Suggest:
        - 3 investment areas for investors
        - 2 improvements for policymakers
        - The potential impact for general public
        based on our pridictions..
        In a structured manner..!"""
        try:
            response = generate_suggestions(prompt)
            #final_suggestions = response.strip().split('\n')  # Split suggestions line by line
            parsed = parse_structured_response(response.splitlines())
        except Exception as e:
            parsed = [f"Error: {str(e)}"]

        sect=''
        for i in range(len(sectors)):
            if i==len(sectors)-1:
                sect+=str(sectors[i])+"."
            else:
                sect+=str(sectors[i])+", "

        cleaned_results = {
        policy: {k: float(v) for k, v in metrics.items()}
        for policy, metrics in results.items()
        }

        return render(request, "chat.html", dict({
            "output": output_from_some_process,
            "suggestions": parsed,
            "sectors":"The effected Doamins are : "+sect,
            "results":cleaned_results
        }))

    return redirect('chat')


