import requests
import json

def test_prediction():
    # Define the API endpoint
    url = 'http://127.0.0.1:8000/predict'

    test_comment = {
        'reddit_comment': 'This is a test comment to see if our model works correctly.'
    }
    response = requests.post(url, json=test_comment)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    return response.json()

if __name__ == "__main__":
    prediction = test_prediction()
    print(f"Prediction result: {prediction}")