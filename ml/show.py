from flask import Flask, render_template, session
import mysql.connector

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Set a secret key for session management

# MySQL database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'dbms',
    'database': 'health_data',
}

# Function to get MySQL connection
def get_db_connection():
    return mysql.connector.connect(**db_config)

# Route to fetch and display user details
@app.route('/user_details', methods=['GET'])
def user_details():
    # Retrieve email from session
    if 'email' in session:
        user_email = session['email']
        try:
            # Connect to the database
            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Query to fetch user details based on email
            query = "SELECT * FROM user_details WHERE email = %s"
            cursor.execute(query, (user_email,))
            user = cursor.fetchone()  # Assuming only one user with this email
            
            cursor.close()
            connection.close()
            
            if user:
                # Render details in HTML page using a template (assuming show_detail.html)
                return render_template('show_detail.html', user=user)
            else:
                return "User not found"  # Handle if user is not found
            
        except mysql.connector.Error as err:
            print(f"Error fetching user details: {err}")
            return "Error fetching user details"
        
    else:
        return "Email not found in session"  # Handle if email is not found in session

if __name__ == '__main__':
    app.run(debug=True, port=8000)
