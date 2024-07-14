from flask import Flask, render_template, request, session,Response
import pandas as pd
import mysql.connector
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure MySQL connection
db_config = {
    'user': 'root',
    'password': 'dbms',
    'host': 'localhost',
    'database': 'health_data'
}

# Load dataset
df = pd.read_csv("heart_attack_prediction_dataset.csv")

# Drop unnecessary columns
df1 = df.drop(['Patient ID', 'Income', 'Country', 'Continent', 'Hemisphere', 'Cholesterol'], axis=1)

# Convert 'Sedentary Hours Per Day' to integer
df1['Sedentary Hours Per Day'] = df1['Sedentary Hours Per Day'].astype(int)

# Encode categorical variables
le = LabelEncoder()
df1['Sex'] = le.fit_transform(df1['Sex'])
df1['Diet'] = le.fit_transform(df1['Diet'])

# Split 'Blood Pressure' into 'BP1' and 'BP2'
def split_blood_pressure(blood_pressure):
    return pd.Series(blood_pressure.split('/', 1))

df1[['BP1', 'BP2']] = df1['Blood Pressure'].apply(split_blood_pressure)
df1 = df1.drop(['Blood Pressure', 'Triglycerides', 'Sedentary Hours Per Day'], axis=1)

# Convert 'BP1' and 'BP2' to numeric
df1['BP1'] = pd.to_numeric(df1['BP1'], errors='coerce')
df1['BP2'] = pd.to_numeric(df1['BP2'], errors='coerce')

# Define weights for features
weights = {
    'Age': 0.1,
    'Sex': 0.05,
    'Heart Rate': 0.15,
    'Diabetes': 0.2,
    'Family History': 0.15,
    'Smoking': 0.25,
    'Obesity': 0.2,
    'Alcohol Consumption': 0.1,
    'Exercise Hours Per Week': 0.1,
    'Diet': 0.15,
    'Previous Heart Problems': 0.3,
    'Medication Use': 0.05,
    'Stress Level': 0.2,
    'BMI': 0.15,
    'Physical Activity Days Per Week': 0.1,
    'Sleep Hours Per Day': 0.1,
    'Heart Attack Risk': 0.4,  # This is a composite measure, not a factor itself
    'BP1': 0.25,  # Systolic Blood Pressure
    'BP2': 0.25   # Diastolic Blood Pressure
}


# Modify weights based on conditions
for index, row in df1.iterrows():
    if row['Age'] >= 45:
        weights['Age'] = 0.2
    if row['Sex'] == 0:
        weights['Sex'] = 0.1
    if row['Heart Rate'] < 60:
        weights['Heart Rate'] = 10 + (row['Heart Rate'] - 1) * 0.02
    elif row['Heart Rate'] > 100:
        weights['Heart Rate'] = 0.2 + (row['Heart Rate'] - 100) * 0.02
    if row['BP1'] > 150:
        weights['BP1'] = 0.2 + (row['BP1'] - 150) * 0.02
    if row['BP2'] > 90:
        weights['BP2'] = 0.2 + (row['BP2'] - 90) * 0.02

# Calculate total weighted sum
total_weighted_sum = df1.apply(lambda row: sum(row[col] * weights[col] for col in df1.columns), axis=1)

# Normalize total weighted sum
max_weighted_sum = total_weighted_sum.max()
min_weighted_sum = total_weighted_sum.min()
df1['percentage'] = ((total_weighted_sum - min_weighted_sum) / (max_weighted_sum - min_weighted_sum)) * 100

# Features and target variable
X = df1.drop(columns=['Heart Attack Risk', 'percentage'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train Random Forest Classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_scaled, df1['Heart Attack Risk'])

# Initialize and train Random Forest Regressor
random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_regressor.fit(X_scaled, df1['percentage'])

# Function to predict manually
def predict_manually(age, sex, heart_rate, diabetes, family_history, smoking, obesity, alcohol_consumption,
                     exercise_hours_per_week, diet, previous_heart_problems, medication_use, stress_level,
                     bmi, physical_activity_days_per_week, sleep_hours_per_day, bp1, bp2):
    input_data = pd.DataFrame([[age, sex, heart_rate, diabetes, family_history, smoking, obesity, alcohol_consumption,
                                exercise_hours_per_week, diet, previous_heart_problems, medication_use, stress_level,
                                bmi, physical_activity_days_per_week, sleep_hours_per_day, bp1, bp2]],
                              columns=X.columns)
    input_scaled = scaler.transform(input_data)
    predicted_heart_attack_risk = random_forest_classifier.predict(input_scaled)[0]
    predicted_percentage = random_forest_regressor.predict(input_scaled)[0]
    return predicted_heart_attack_risk, predicted_percentage

# Function to get user details from the database
def get_user_details(email):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT age, sex, diabetes, famhistory, smoking, obesity, alcohol, exercise_hours, diet, heart_problem, bmi, physical_activity, sleep_hours, blood_pressure_systolic, blood_pressure_diastolic FROM user_details WHERE email = %s", (email,))
    user_details = cursor.fetchone()
    cursor.close()
    conn.close()
    return user_details

@app.route('/')
def index():
    email = session.get('email')
    user_details = None
    if email:
        user_details = get_user_details(email)
    return render_template('index.html', user_details=user_details)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = session.get('email')
        if not email:
            return render_template('index.html', error="Session expired. Please log in again.")

        user_details = get_user_details(email)
        if not user_details:
            return render_template('index.html', error="User details not found in the database.")

        # Extract user data from database
        age = int(user_details['age'])
        sex = int(user_details['sex'])
        diabetes = int(user_details['diabetes'])
        family_history = int(user_details['famhistory'])
        smoking = int(user_details['smoking'])
        obesity = int(user_details['obesity'])
        alcohol_consumption = int(user_details['alcohol'])
        exercise_hours_per_week = float(user_details['exercise_hours'])
        diet = int(user_details['diet'])
        previous_heart_problems = int(user_details['heart_problem'])
        bmi = float(user_details['bmi'])
        physical_activity_days_per_week = int(user_details['physical_activity'])
        sleep_hours_per_day = float(user_details['sleep_hours'])
        bp1 = int(user_details['blood_pressure_systolic'])
        bp2 = int(user_details['blood_pressure_diastolic'])

        # Debugging: Print the form data
        print("Form data received:", request.form)

        try:
            # Extract heart rate, stress level, HRV, and SpO2 from form data
            heart_rate = int(request.form['heart_rate'])
            stress_level = int(request.form['stress_level'])
            hrv = int(request.form['hrv'])
            spo2 = int(request.form['spo2'])
        except KeyError as e:
            return render_template('index.html', error=f"Missing form data: {e}")

        # Predict using user data and input heart rate and stress level
        predicted_heart_attack_risk, predicted_percentage = predict_manually(
            age, sex, heart_rate, diabetes, family_history, smoking, obesity,
            alcohol_consumption, exercise_hours_per_week, diet, previous_heart_problems,
            0, stress_level, bmi, physical_activity_days_per_week, sleep_hours_per_day, bp1, bp2
        )
        messages = []
        # Adjust predicted percentage based on HRV and SpO2
        if hrv < 45 :
            messages.append("Your HRV is too low ")
            predicted_percentage += 25.8
        if hrv > 200:
            messages.append("Your HRV is too high ")
            predicted_percentage += 5.65

        if spo2 < 90:
            messages.append("Your SpO2 level is below normal.")
            predicted_percentage += 4.5
        

        # Generate messages based on user's health data
        
        if age > 50:
            messages.append("Age greater than 50.")
        if heart_rate < 60 or heart_rate > 100:
            messages.append("Your heart rate is not good.")
        if diabetes:
            messages.append("You have diabetes.")
        if family_history:
            messages.append("You have a family history of heart problems.")
        if smoking:
            messages.append("You are a smoker.")
        if obesity:
            messages.append("You are obese.")
        if alcohol_consumption:
            messages.append("You consume alcohol.")
        if previous_heart_problems:
            messages.append("You have had previous heart problems.")
        if stress_level > 6:
            messages.append("Stress Level is High, Take rest or Hangout.")
        if bp1 > 140 or bp2 > 80:
            messages.append("Your blood pressure is high.")
        

        # Add HRV and SpO2 messages
     

        # Determine the user's category
        if predicted_heart_attack_risk == 0 and predicted_percentage < 50:
            category = "Heart attack risk 0, heart attack risk percentage less than 50. You are safe. Please take care of yourself."
        elif predicted_heart_attack_risk == 0 and predicted_percentage >= 50:
            category = "Heart attack risk 0, heart attack risk percentage greater than 50. Please consult the doctor by sharing the report."
        elif predicted_heart_attack_risk == 1 and predicted_percentage < 50:
            category = "Heart attack risk 1, heart attack risk percentage less than 50. Something unpredictable. Please consult a doctor."
        else:
            category = "Heart attack risk 1, heart attack risk percentage greater than 50. Don't be afraid. Just contact the nearest hospital. That's all."

        return render_template('result.html', heart_attack_risk=predicted_heart_attack_risk,
                               percentage=predicted_percentage, messages=messages, category=category)
    else:
        return render_template('index.html')
    
    


@app.route('/download_report', methods=['POST'])
def download_report():
    if request.method == 'POST':
        email = session.get('email')
        if not email:
            return "Session expired. Please log in again."

        user_details = get_user_details(email)
        if not user_details:
            return "User details not found in the database."

        # Retrieve report content from form data
        report_content = request.form['report_content']

        # Create a PDF buffer
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

        # Define the styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Title'],
            fontSize=18,
            spaceAfter=14,
        )

        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['BodyText'],
            fontSize=12,
            leading=14,
            spaceAfter=10,
        )

        # Create the report elements
        elements = []

        # Add title and user details
        elements.append(Paragraph("Heart Attack Risk Report", title_style))
        elements.append(Spacer(1, 12))

        user_info = f"""
        <b>User Details:</b><br/>
        Age: {user_details['age']}<br/>
        Sex: {user_details['sex']}<br/>
        Diabetes: {user_details['diabetes']}<br/>
        Family History: {user_details['famhistory']}<br/>
        Smoking: {user_details['smoking']}<br/>
        Obesity: {user_details['obesity']}<br/>
        Alcohol Consumption: {user_details['alcohol']}<br/>
        Exercise Hours Per Week: {user_details['exercise_hours']}<br/>
        Diet: {user_details['diet']}<br/>
        Previous Heart Problems: {user_details['heart_problem']}<br/>
        BMI: {user_details['bmi']}<br/>
        Physical Activity Days Per Week: {user_details['physical_activity']}<br/>
        Sleep Hours Per Day: {user_details['sleep_hours']}<br/>
        Blood Pressure (Systolic/Diastolic): {user_details['blood_pressure_systolic']}/{user_details['blood_pressure_diastolic']}<br/>
        """

        elements.append(Paragraph(user_info, body_style))
        elements.append(Spacer(1, 12))

        # Add the report content
        elements.append(Paragraph(report_content, body_style))
        elements.append(Spacer(1, 12))

        # Add the image
        image_path = 'static/final_graph.png'
        elements.append(Image(image_path, width=4 * inch, height=4 * inch))
        elements.append(Spacer(1, 12))

        # Build the PDF
        doc.build(elements)

        # Set up response to initiate download
        pdf_buffer.seek(0)
        response = Response(pdf_buffer, mimetype='application/pdf')
        response.headers.set("Content-Disposition", "attachment", filename="heart_attack_report.pdf")

        return response
    else:
        return "Method not allowed."
if __name__ == '__main__':
    app.run(debug=True, port=3005)
