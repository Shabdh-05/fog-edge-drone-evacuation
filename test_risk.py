from ml_modules.risk_model import RiskScorer

risk_model = RiskScorer()

# Example cluster
cluster_size = 5
hazard_severity = 9   # scale 1–10
distance_to_hazard = 45

result = risk_model.predict_risk(cluster_size, hazard_severity, distance_to_hazard)

print("Risk Level:", result)