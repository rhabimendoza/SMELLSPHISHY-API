import os
from converter import *
from flask_cors import CORS
from flask import Flask, request, jsonify

# Create flask app
app = Flask(__name__)
CORS(app)

# Add a default page
@app.route("/")
def index():
    return "<h1>SmellsPhishy API</h1>"

# Create a route for checking
@app.route('/check_url', methods=['POST'])
def check_url():
    
    # Get the url
    data = request.json
    url = data.get('url', '')

    # Check if url input is valid
    if validURL(url) == 3:
        return jsonify({"result": 3})

    # Send the url to function for checking
    result, benign, phishing, features = checkURLInput(url)

    # Return classification with probabilities and features
    return jsonify({
        "result": result,
        "benign": benign,
        "phishing": phishing,
        "features": features
    })

# Run app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)