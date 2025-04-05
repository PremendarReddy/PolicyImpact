from django.db import models

# Create your models here.
def run_policy_nlp(policy_text):
    # Example output
    return ["Healthcare", "Taxation"]

def run_economic_model(sectors):
    return {
        "GDP Impact": "+2.3%",
        "Inflation Impact": "-0.5%",
        "Jobs": "+1 million"
    }

def run_investment_model(economic_data):
    return {
        "High Growth Sectors": ["Green Energy", "Telemedicine"],
        "Investment Risk": "Low",
        "Recommended Investment Horizon": "3-5 years"
    }

def run_optimization_model(original_policy, sectors, econ_data):
    return "Consider reallocating part of defense budget instead of reducing education funding."
