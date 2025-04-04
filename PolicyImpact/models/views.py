from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, 'home.html')

def sentimentanalysis(request):
    return render(request, 'sentiment_analysis.html')

def economicimpact(request):
    return render(request, 'economic_impact.html')

def forecasting(request):
    return render(request, 'forecasting.html')

def policyoptimization(request):
    return render(request, 'policy_optimization.html')