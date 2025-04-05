from django.shortcuts import render, redirect
from .models import run_policy_nlp, run_economic_model
# run_investment_model, run_optimization_model

def home(request):
    return render(request, 'home.html')

def chat(request):
    return render(request, 'chat.html',{'responses': []})
 
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

        responses = [
            {"sender": "user", "text": user_input},
            {"sender": "bot", "text": f"🧠 NLP Model ➤ Affected Sectors: {', '.join(sectors)}"},
            {"sender": "bot", "text": f"📊 Economic Impact ➤ {economic_data}"},
            # {"sender": "bot", "text": f"📈 Investment Insights ➤ {investment_data}"},
            # {"sender": "bot", "text": f"🛠️ Optimization Suggestion ➤ {optimized_policy}"},
        ]

        return render(request, 'chat.html', {'responses': responses})

    return redirect('chat')
