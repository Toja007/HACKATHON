import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend like Agg for generating charts in Flask
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)
CORS(app)

# Ensure the static folder is used for serving images and static content
app.config['UPLOAD_FOLDER'] = 'static'

# Load pre-trained model (ensure your stacked model is saved as a .pkl file)
model_path = 'stacked_model.pkl'
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('frontend.html')

# Helper function to calculate average scores
def calculate_averages(df):
    score_columns = [
        'math_score', 'history_score', 'physics_score', 
        'chemistry_score', 'biology_score', 'english_score', 'geography_score'
    ]
    df['average_score'] = df[score_columns].mean(axis=1)
    return df

# Helper function to generate statistics and insights
def generate_statistics(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['average_score'], kde=True, bins=10)
    plt.title("Distribution of Average Scores")
    
    # Save the chart in the static folder
    chart_path = os.path.join(app.config['UPLOAD_FOLDER'], "average_score_chart.png")
    plt.savefig(chart_path)
    plt.close()
    
    stats = df['average_score'].describe()  # Summary statistics for average scores
    return stats, chart_path

# Improvement suggestions based on certain conditions
def suggest_improvements(row):
    suggestions = []
    if row['part_time_job']:
        suggestions.append("Consider reducing part-time work hours to focus more on studies.")
    if row['absence_days'] > 5:
        suggestions.append("Improve attendance.")
    if not row['extracurricular_activities']:
        suggestions.append("Participate in extracurricular activities to develop soft skills.")
    if row['weekly_self_study_hours'] <= 7:
        suggestions.append("Increase self-study hours to at least 10 hours per week.")
    return suggestions

@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        
        # Check if necessary columns are in the CSV file
        required_columns = [
            'id', 'first_name', 'last_name', 'part_time_job', 'absence_days',
            'extracurricular_activities', 'weekly_self_study_hours',
            'math_score', 'history_score', 'physics_score', 'chemistry_score',
            'biology_score', 'english_score', 'geography_score'
        ]
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": "Missing required columns in CSV"}), 400

        # Calculate average score
        df = calculate_averages(df)
        
        # Encode categorical variables and scale numerical features
        le = LabelEncoder()
        df['part_time_job'] = le.fit_transform(df['part_time_job'])
        df['extracurricular_activities'] = le.fit_transform(df['extracurricular_activities'])
        
        # Select the prediction features and scale them
        prediction_features = [
            'part_time_job', 'absence_days', 'extracurricular_activities',
            'weekly_self_study_hours', 'math_score', 'history_score', 
            'physics_score', 'chemistry_score', 'biology_score', 
            'english_score', 'geography_score'
        ]
        scaler = StandardScaler()
        df[prediction_features] = scaler.fit_transform(df[prediction_features])
        
        # Make predictions using the pre-trained model
        df['pass_prediction'] = model.predict(df[prediction_features])
        
        # For students predicted not to pass, suggest improvements
        df['improvement_suggestions'] = df.apply(lambda row: suggest_improvements(row) if row['pass_prediction'] == 0 else '', axis=1)

        # Generate statistics and save chart
        stats, chart_path = generate_statistics(df)

        # Respond with predictions, stats, and chart
        return jsonify({
            "predictions": df[['id', 'first_name', 'last_name', 'pass_prediction', 'improvement_suggestions']].to_dict(orient='records'),
            "statistics": {
                "mean": stats['mean'],
                "median": df['average_score'].median(),
                "stddev": stats['std']
            },
            "chart_url": chart_path
        })

    return jsonify({"error": "File type not supported"}), 400

if __name__ == '__main__':
    app.run(debug=True)
