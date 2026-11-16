"""Step 1: train the toy triage model on biased synthetic history and export weights as JSON."""
from sklearn.neural_network import MLPClassifier
from common import make_data, export_weights, save_weights, evaluate

X, y_hist, _ = make_data(6000, seed=7)
clf = MLPClassifier(hidden_layer_sizes=(8,), activation="relu",
                    max_iter=800, random_state=0).fit(X, y_hist)

w = export_weights(clf)
save_weights("weights_biased.json", w)

m = evaluate(w)
print("Biased model saved -> weights_biased.json")
print(f"  disparity (mean |delta urgency| when flipping attribute): {m['disparity']}")
print(f"  accuracy vs unbiased clinical need: {m['accuracy_vs_unbiased_need']:.3f}")
