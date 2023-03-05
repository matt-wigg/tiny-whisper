from flask import Flask, request
import whisper
from tempfile import NamedTemporaryFile
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/test/*": {"origins": "http://localhost:3000"}})

@app.route('/test/<language>/<model_size>', methods=['POST'])
def handler(language, model_size):
    """
    This function handles POST requests to the '/test' endpoint.
    It expects request.files to contain the uploaded files.
    """
    # Sanitize inputs
    language = secure_filename(language)
    model_size = secure_filename(model_size)

    # If the user didn't submit any files, return a 400 (Bad Request) error.
    if not request.files:
        return {'error': 'No files were uploaded'}, 400

    # For each file, let's store the results in a list of dictionaries.
    results = []

    print("model size: " + model_size)
    print("language: " + language)
    # Load the Whisper model based on the given `model_size`:
    model = whisper.load_model(model_size)

    # Loop over every file that the user submitted.
    for filename, handle in request.files.items():
        # Create a temporary file.
        # The location of the temporary file is available in `temp.name`.
        with NamedTemporaryFile() as temp:
            # Write the user's uploaded file to the temporary file.
            # The file will get deleted when it drops out of scope.
            handle.save(temp)
            # Let's get the transcript of the temporary file.
            result = model.transcribe(temp.name, language=language, fp16=False)

            # Now we can store the result object for this file.
            results.append({
                'filename': filename,
                'transcript': result,
            })

    # This will be automatically converted to JSON.
    return {'results': results}

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
