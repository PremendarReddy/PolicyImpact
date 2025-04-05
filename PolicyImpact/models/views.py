from django.shortcuts import render, redirect
from openai import OpenAI
from collections import defaultdict
import re

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



def home(request):
    return render(request, 'home.html')

def chat(request):
    return render(request, 'chat.html')
 
def run_policy_nlp(policy_text):
    return ["Healthcare", "Taxation"]

def run_economic_model(sectors):
    return {
        "GDP_growth": "+2.3%",
        "inflation_rate": "-0.5%",
        "employment_rate": "+0.5%",
        "carbon_emission" : "0.8%",
        "Health_Index" : "0.765%",
        "Education_index" : "0.345%",
    }


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
        economic_data = run_economic_model(sectors)

        # # 3️⃣ Investment Forecasting (depends on econ data)
        # investment_data = run_investment_model(economic_data)

        # # 4️⃣ Policy Optimization (depends on previous outputs)
        # optimized_policy = run_optimization_model(user_input, sectors, economic_data)
        final_suggestions = []
        output_from_some_process = "Policy on AI regulation in healthcare"  # Replace with your dynamic logic

        prompt = f"""
        A new policy affecting Agriculture has been proposed.
        It is predicted to change GDP_Growth rate by 0.5%.
        It is predicted to change carbon emmision rate by 0.689%.
        It is predicted to change employment rate by 0.67%.
        It is predicted to change inflation rate by 1.2%%.
        It is predicted to change health index by 8.5%%.
        It is predicted to change education index by 3%.
        Suggest:
        - 3 investment areas for investors
        - 2 improvements for policymakers
        - The potential impact for general public
        In a structured manner..!"""
        try:
            response = generate_suggestions(prompt)
            #final_suggestions = response.strip().split('\n')  # Split suggestions line by line
            parsed = parse_structured_response(response.splitlines())
        except Exception as e:
            parsed = [f"Error: {str(e)}"]

        return render(request, "chat.html", dict({
            "output": output_from_some_process,
            "suggestions": parsed
        }))

    return redirect('chat')



#It is predicted to change employment rate by {carbon_emission:.2f}%.
#It is predicted to change employment rate by {employment_rate:.2f}%.
#It is predicted to change employment rate by {inflation_rate:.2f}%.
#It is predicted to change employment rate by {Health_Index:.2f}%.
#It is predicted to change employment rate by {Education_index:.2f}%.
