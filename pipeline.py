import configparser
import pandas as pd
from sqlalchemy import create_engine

# Load the CSV data
file_path_scores = r'C:\Users\user\Downloads\subject_scores_table.csv'  # Use raw string or double backslashes
file_path_students = r'C:\Users\user\Downloads\student_table.csv'
file_path_attendance = r'C:\Users\user\Downloads\attendance_table.csv'
file_path_financial = r'C:\Users\user\Downloads\finance_table.csv'  # Corrected the path
file_path_staff = r'C:\Users\user\Downloads\staff_table.csv'  # Ensure the correct file extension

# Read the CSVs into pandas DataFrames
try:
    scores_data = pd.read_csv(file_path_scores)
    students_data = pd.read_csv(file_path_students)
    attendance_data = pd.read_csv(file_path_attendance)
    financial_data = pd.read_csv(file_path_financial)
    staff_data = pd.read_csv(file_path_staff)
except FileNotFoundError as e:
    print(f"Error reading file: {e}")
    raise  # Re-raise the exception after logging it

# Display initial data for each dataset
print("Initial Scores Data:")
print(scores_data.head())
print("\nInitial Students Data:")
print(students_data.head())
print("\nInitial Attendance Data:")
print(attendance_data.head())
print("\nInitial Financial Data:")
print(financial_data.head())
print("\nInitial Staff Data:")
print(staff_data.head())

# Clean column names (strip spaces, convert to lowercase, and replace spaces with underscores)
def clean_columns(df):
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# Apply the cleaning function to all DataFrames
datasets = [scores_data, students_data, attendance_data, financial_data, staff_data]
for dataset in datasets:
    clean_columns(dataset)

# Function to capitalize string columns properly
def proper_case_strings(df, string_columns):
    for col in string_columns:
        df[col] = df[col].str.strip().str.title()  # Proper case and remove extra spaces

# Apply proper case to specific columns in students, staff, and financial data

# For students_data: first_name, last_name, career_aspiration
proper_case_strings(students_data, ['first_name', 'last_name', 'career_aspiration'])

# For staff_data: first_name, last_name, role, subject, gender, marital_status, level_of_education
proper_case_strings(staff_data, ['first_name', 'last_name', 'role', 'subject', 'gender', 'marital_status', 'level_of_education'])

# For financial_data: description, category, remark
proper_case_strings(financial_data, ['description', 'category', 'remark'])

# Validate 'score' columns in scores_data (ensure it’s between 0 and 100)
score_columns = ['math_score', 'history_score', 'physics_score', 
                 'chemistry_score', 'biology_score', 'english_score', 
                 'geography_score']
scores_data[score_columns] = scores_data[score_columns].clip(lower=0, upper=100)



# Load into PostgreSQL Database
config = configparser.ConfigParser()
config.read('config.ini')

username = config['database']['username']
password = config['database']['password']
host = config['database']['host']
port = config['database']['port']
database = config['database']['database']

# Create a PostgreSQL database connection
engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database}')

# Load the transformed DataFrames into the database
scores_data.to_sql('exam_scores', con=engine, if_exists='replace', index=False)
students_data.to_sql('students', con=engine, if_exists='replace', index=False)
attendance_data.to_sql('attendance', con=engine, if_exists='replace', index=False)
financial_data.to_sql('financial', con=engine, if_exists='replace', index=False)
staff_data.to_sql('staff', con=engine, if_exists='replace', index=False)


print("\nData loaded successfully into the SQL database!")

# Step 4: Query the database to verify the data is loaded correctly
with engine.connect() as connection:
    # Verify 'exam_scores' table
    print("\nVerifying data in the 'exam_scores' table:")
    result = connection.execute("SELECT * FROM exam_scores LIMIT 5;")
    for row in result:
        print(row)
    
    # Verify 'students' table
    print("\nVerifying data in the 'students' table:")
    result = connection.execute("SELECT * FROM students LIMIT 5;")
    for row in result:
        print(row)
    
    # Verify 'attendance' table
    print("\nVerifying data in the 'attendance' table:")
    result = connection.execute("SELECT * FROM attendance LIMIT 5;")
    for row in result:
        print(row)
    
    # Verify 'financial' table
    print("\nVerifying data in the 'financial' table:")
    result = connection.execute("SELECT * FROM financial LIMIT 5;")
    for row in result:
        print(row)
    
    # Verify 'staff' table
    print("\nVerifying data in the 'staff' table:")
    result = connection.execute("SELECT * FROM staff LIMIT 5;")
    for row in result:
        print(row)

print("\nAll tables verified successfully!")
