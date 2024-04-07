from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
with open('Models/model_file.p', 'rb') as file:
    model_data = pickle.load(file)
    model = model_data['model']
    scaler = model_data['scaler']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve form data from the request
        step = int(request.form['step'])
        age = int(request.form['age'])
        gender = request.form['gender']
        amount = float(request.form['amount'])
        category = request.form['category']
        
        # Convert gender to numeric (0 for Male, 1 for Female)
        gender = 0 if gender == 'M' else 1
        if category == 'Sports':
            user_input = np.array([step,age,gender,amount,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        elif category == 'Media Contents':
            user_input = np.array([step,age,gender,amount,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
        elif category == 'Fashion':
            user_input = np.array([step,age,gender,amount,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
        elif category == 'Food':
            user_input = np.array([step,age,gender,amount,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
        elif category == 'Health':
            user_input = np.array([step,age,gender,amount,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
        elif category == 'Home':
            user_input = np.array([step,age,gender,amount,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
        
        # Scale the input values
        scaled_input = scaler.transform([user_input])
        
        # Make a prediction
        prediction = model.predict(scaled_input)[0]
        
        # Display the appropriate message based on the prediction
        if prediction == 1:
            result_message = "This is a fraudulent transaction!"
            result_color = "red"
        else:
            result_message = "This is a safe transaction!"
            result_color = "green"

        # Render the result on the index.html template
        return render_template('index.html', result_message=result_message, result_color=result_color)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

