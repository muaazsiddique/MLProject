from flask import Flask, render_template, request, jsonify
from src.pipeline.predict_pipeline import PredictPipeline


app = Flask(__name__)

# Paths to saved model and preprocessor
MODEL_PATH = r"E:\Projects\MLProject\artifacts\best_model.pkl"
PREPROCESSOR_PATH = r"E:\Projects\MLProject\artifacts\preprocessor.pkl"

# Initialize the prediction pipeline
predict_pipeline = PredictPipeline(model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    """Renders the home page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles prediction requests."""
    try:
        # Extract input data from form
        input_data = {
            "gender": [request.form["gender"]],
            "race/ethnicity": [request.form["race_ethnicity"]],
            "parental level of education": [request.form["parental_education"]],
            "lunch": [request.form["lunch"]],
            "test preparation course": [request.form["test_prep"]],
            "reading score": [float(request.form["reading_score"])],
            "writing score": [float(request.form["writing_score"])]
        }

        # Make prediction
        prediction = predict_pipeline.predict(input_data)

        # Render results
        return render_template("result.html", prediction=prediction[0])

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
