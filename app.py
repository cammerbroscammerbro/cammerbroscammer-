from flask import Flask, request, jsonify, render_template, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import openai  # Replace 'from openai import OpenAI' with this import
import os
import json
import logging
import zipfile
from io import BytesIO
import requests  # Add this import for making HTTP requests

app = Flask(__name__, template_folder='.', static_folder='.')  # Set template folder to current directory
CORS(app)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Store files in memory for simplicity
files = {}

VERCEL_TOKEN = os.getenv("VERCEL_TOKEN")  # Ensure the Vercel token is loaded

# Chatbot memory file
CHATBOT_MEMORY_FILE = "chatbot_memory.json"

# Load chatbot memory from a local file
def load_chatbot_memory():
    if os.path.exists(CHATBOT_MEMORY_FILE):
        with open(CHATBOT_MEMORY_FILE, "r") as file:
            return json.load(file)
    return []

# Save chatbot memory to a local file
def save_chatbot_memory(memory):
    with open(CHATBOT_MEMORY_FILE, "w") as file:
        json.dump(memory, file, indent=2)

# Initialize chatbot memory
chatbot_memory = load_chatbot_memory()

@app.route('/favicon.ico')
def favicon():
    # Serve logo.png as favicon.ico for browser tab icon
    return send_from_directory(
        os.path.join(app.root_path, ''),
        'logo.png',
        mimetype='image/png'
    )

@app.route('/favicon.png')
def favicon_png():
    # Also serve logo.png as favicon.png for browsers that request it
    return send_from_directory(
        os.path.join(app.root_path, ''),
        'logo.png',
        mimetype='image/png'
    )

@app.route('/logo.png')
def logo():
    return send_from_directory(
        os.path.join(app.root_path, ''),
        'logo.png',
        mimetype='image/png'
    )

@app.route('/')
def home():
    return render_template('main.html')

def sanitize_response(response_text):
    """
    Sanitize the response text to ensure it is valid JSON.
    - If the response is already valid JSON, return it as is.
    - Otherwise, attempt to fix common issues like improperly escaped characters.
    """
    try:
        # Check if the response is already valid JSON
        json.loads(response_text)  # This will raise an error if the JSON is invalid
        return response_text
    except json.JSONDecodeError:
        logging.warning("Response is not valid JSON. Attempting to sanitize...")
        try:
            # Fix improperly escaped quotes (e.g., \\" -> ")
            sanitized_text = response_text.replace('\\"', '"')

            # Fix improperly escaped backslashes (e.g., \\ -> \)
            sanitized_text = sanitized_text.replace('\\\\', '\\')

            # Fix trailing backslashes (e.g., +\ -> +)
            sanitized_text = sanitized_text.replace('\\\n', '\n')

            # Attempt to parse the sanitized text to ensure it is valid JSON
            json.loads(sanitized_text)  # This will raise an error if the JSON is still invalid
            return sanitized_text
        except json.JSONDecodeError as e:
            logging.error(f"Sanitization failed: {str(e)}")
            logging.debug(f"Sanitized text causing error: {sanitized_text}")
            return None

def parse_files(response_text):
    try:
        # --- FIX: Use a simple, robust JSON extraction (no (?R) regex) ---
        import re
        # Find the first top-level JSON object (from first { to last })
        first_brace = response_text.find('{')
        last_brace = response_text.rfind('}')
        if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
            logging.error("No JSON object found in the response text.")
            logging.debug(f"Response text received: {response_text}")
            return None

        json_response = response_text[first_brace:last_brace+1]

        # Sanitize the JSON response
        sanitized_response = sanitize_response(json_response)
        if not sanitized_response:
            logging.error("Failed to sanitize the response text.")
            return None

        # Parse the sanitized JSON
        parsed_files = json.loads(sanitized_response)

        # Flatten nested dictionaries into a single-level dictionary
        def flatten_files(files_dict, parent_key=""):
            flat_files = {}
            for k, v in files_dict.items():
                new_key = f"{parent_key}/{k}" if parent_key else k
                if isinstance(v, dict):
                    flat_files.update(flatten_files(v, new_key))
                else:
                    flat_files[new_key] = v
            return flat_files

        flattened_files = flatten_files(parsed_files)

        def convert_to_string(value):
            if isinstance(value, dict):
                return json.dumps(value, indent=2)
            elif isinstance(value, bytes):
                return value.decode('utf-8', errors='replace')
            return str(value)

        valid_files = {}
        for k, v in flattened_files.items():
            if not k.endswith('/') and isinstance(v, str) and v.strip():
                valid_files[k] = convert_to_string(v)
            else:
                logging.warning(f"Skipping invalid or empty entry: {k}")

        if isinstance(valid_files, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in valid_files.items()):
            return valid_files
        else:
            logging.error("Parsed response is not a valid dictionary of files.")
            logging.debug(f"Parsed response structure: {valid_files}")
            return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error: {str(e)}")
        logging.debug(f"Response text causing error: {response_text}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error while parsing response text: {str(e)}")
        logging.debug(f"Response text causing error: {response_text}")
        return None

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        logging.debug(f"Received data: {data}")  # Debugging log
        if not data:
            logging.error("No data received in the request.")
            return jsonify({'error': 'No data received'}), 400

        prompt = data.get('prompt')
        output_type = data.get('type')

        logging.debug(f"Prompt: {prompt}, Type: {output_type}")  # Debugging log

        if not prompt or not output_type:
            logging.error(f"Missing prompt or type. Prompt: {prompt}, Type: {output_type}")
            return jsonify({'error': 'Prompt and type are required'}), 400

        # Force OpenAI to always include a README.md file with explanation, and make it clear it's compulsory
        final_prompt = (
            build_prompt(prompt, output_type.lower())
            + "\n\nIMPORTANT: You must include a README.md file with a clear, human-friendly explanation of the project, its structure, and usage. "
            + "The README.md must be present in the JSON output. Do not skip README.md under any circumstances. "
            + "Do not put any explanation in any other file or outside the README.md."
            + "\n\nReturn ONLY a JSON object, no markdown, no code block, no extra text, no triple backticks, no explanation outside the JSON."
        )

        logging.debug(f"Final prompt sent to OpenAI: {final_prompt}")

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a developer who returns only code files in JSON format with correct filenames and full file content. If you want to provide an explanation or summary, put it in a README.md file."},
                {"role": "user", "content": final_prompt}
            ]
        )

        # Extract the JSON object from the response
        response_text = response['choices'][0]['message']['content']
        logging.debug(f"Raw response from OpenAI: {response_text}")  # Log the raw response for debugging

        # --- IMPROVEMENT: Remove code block markers and markdown, aggressively ---
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        if first_brace != -1 and last_brace != -1:
            cleaned = cleaned[first_brace:last_brace+1]

        global files  # Ensure the global `files` object is updated
        files = parse_files(cleaned)

        if not files:
            logging.error("Failed to parse files from OpenAI response.")
            return jsonify({'error': 'Failed to generate files. The response was incomplete or invalid.'}), 500

        # Extract README/explanation if present
        explanation = None
        readme_keys = [k for k in files if k.lower() in ['readme.md', 'readme.txt', 'readme', 'readme.markdown']]
        if readme_keys:
            explanation = files[readme_keys[0]]
            del files[readme_keys[0]]
        else:
            # Try to extract README from nested keys (sometimes LLMs nest files)
            for k in list(files.keys()):
                if 'readme' in k.lower():
                    explanation = files[k]
                    del files[k]
                    break

        # --- If still no README, try to extract from any .md file ---
        if not explanation:
            for k in list(files.keys()):
                if k.lower().endswith('.md'):
                    explanation = files[k]
                    del files[k]
                    break

        # --- If still no README, try to extract from the original response (fallback) ---
        if not explanation:
            # Try to extract README.md content from the raw response (if LLM failed to put it in JSON)
            import re
            match = re.search(r'"README\.md"\s*:\s*"((?:[^"\\]|\\.)*)"', response_text, re.DOTALL)
            if match:
                explanation = match.group(1).encode('utf-8').decode('unicode_escape')

        logging.debug(f"Generated files: {files.keys()}")
        return jsonify({'files': files, 'explanation': explanation or "No README.md was generated."}), 200

    except Exception as e:
        logging.error(f"Error in /generate: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message')
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Let the AI decide when to generate code or just chat
        chat_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant for software development. "
                        "If the user asks for code, generate the code and explain it. "
                        "If the user is chatting, respond conversationally. "
                        "You can generate code, explain code, or just chat as appropriate."
                    )
                },
                {"role": "user", "content": message}
            ]
        )
        reply = chat_response['choices'][0]['message']['content']
        return jsonify({'message': reply}), 200

    except Exception as e:
        logging.error(f"Error in /chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate-stream', methods=['POST'])
def generate_stream():
    try:
        data = request.get_json()
        if not data:
            logging.error("No data received in the request.")
            return jsonify({'error': 'No data received'}), 400

        prompt = data.get('prompt')
        output_type = data.get('type')

        if not prompt or not output_type:
            logging.error("Missing prompt or type in the request.")
            return jsonify({'error': 'Prompt and type are required'}), 400

        logging.info(f"Received prompt: {prompt}")
        logging.info(f"Received app type: {output_type}")

        final_prompt = build_prompt(prompt, output_type.lower())
        logging.debug(f"Final prompt sent to OpenAI: {final_prompt}")

        def stream_files():
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a developer who returns only code files in JSON format with correct filenames and full file content."},
                        {"role": "user", "content": final_prompt}
                    ],
                    stream=True
                )
                buffer = ""
                for chunk in response:
                    content = chunk['choices'][0]['delta'].get('content', '')
                    buffer += content
                    if buffer.endswith("}"):  # Check if the JSON object is complete
                        files = parse_files(buffer)
                        if files:
                            logging.info("Streaming files to frontend.")
                            yield f"data: {json.dumps(files)}\n\n"
                            buffer = ""  # Reset buffer after sending
            except Exception as e:
                logging.error(f"Error during streaming: {str(e)}")
                yield f"data: [ERROR] {str(e)}\n\n"

        return Response(stream_files(), mimetype='text/event-stream')

    except Exception as e:
        logging.error(f"Error during generation process: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/sync-file', methods=['POST'])
def sync_file():
    try:
        data = request.get_json()
        filename = data.get('filename')
        content = data.get('content')

        if not filename or content is None:
            return jsonify({'error': 'Filename and content are required'}), 400

        files[filename] = content  # Update the file content in memory
        return jsonify({'message': 'File synced successfully'}), 200
    except Exception as e:
        logging.error(f"Error syncing file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/file-updates', methods=['GET'])
def file_updates():
    def stream_files():
        previous_files = {}
        while True:
            if files != previous_files:
                yield f"data: {json.dumps(files)}\n\n"
                previous_files = files.copy()
    return Response(stream_with_context(stream_files()), mimetype='text/event-stream')

@app.route('/export', methods=['POST'])
def export():
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            logging.error("No valid file data received for export.")
            return jsonify({'error': 'No valid file data received'}), 400

        if not data:
            logging.error("No files provided for export.")
            return jsonify({'error': 'No files provided'}), 400

        # Create a zip file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename, content in data.items():
                zip_file.writestr(filename, content)

        zip_buffer.seek(0)

        # Serve the zip file
        return Response(
            zip_buffer,
            mimetype='application/zip',
            headers={
                'Content-Disposition': 'attachment; filename=fronti_project.zip'
            }
        )
    except Exception as e:
        logging.error(f"Error during export: {str(e)}")
        return jsonify({'error': 'Failed to export files', 'details': str(e)}), 500

@app.route('/deploy', methods=['POST'])
def deploy():
    try:
        data = request.get_json()
        logging.debug(f"Received deployment data: {data}")  # Debugging log

        if not data or not isinstance(data, dict):
            logging.error("No valid file data received for deployment.")
            return jsonify({'error': 'No valid file data received'}), 400

        # Prepare files for deployment
        files = data.get('files', {})
        if not files:
            logging.error("No files provided for deployment.")
            return jsonify({'error': 'No files provided'}), 400

        logging.debug(f"Files to deploy: {files.keys()}")  # Debugging log

        # Create the payload for Vercel deployment
        deployment_payload = {
            "name": "fronti-web-app",
            "files": [{"file": filename, "data": content} for filename, content in files.items()],
            "projectSettings": {
                "framework": None  # Set to None for static deployments
            }
        }

        # Send the deployment request to Vercel
        headers = {
            "Authorization": f"Bearer {VERCEL_TOKEN}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            "https://api.vercel.com/v13/deployments",
            headers=headers,
            json=deployment_payload
        )

        logging.debug(f"Vercel API response: {response.text}")  # Debugging log

        if response.status_code != 200:
            logging.error(f"Vercel API error: {response.text}")
            return jsonify({'error': 'Failed to deploy to Vercel', 'details': response.text}), 500

        deployment_data = response.json()
        deployment_url = deployment_data.get("url")
        if not deployment_url:
            logging.error("No deployment URL returned by Vercel.")
            return jsonify({'error': 'Deployment failed. No URL returned.'}), 500

        logging.info(f"Deployment successful: {deployment_url}")
        return jsonify({'message': 'Deployment successful', 'url': f"https://{deployment_url}"})

    except Exception as e:
        logging.error(f"Error during deployment: {str(e)}")
        return jsonify({'error': 'Failed to deploy files', 'details': str(e)}), 500

@app.route('/preview', methods=['POST'])
def preview():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        app_type = data.get('type', '').lower()
        files = data.get('files', {})

        if not app_type or not files:
            return jsonify({'error': 'App type and files are required'}), 400

        # For web apps, just return the files as-is
        if app_type == "web app":
            # Ensure there's at least one HTML file
            html_files = [f for f in files.keys() if f.lower().endswith('.html')]
            if not html_files:
                # Create a minimal index.html if none exists
                files['index.html'] = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Generated App</title>
                </head>
                <body>
                    <h1>No HTML file was generated</h1>
                    <p>This web app doesn't contain an HTML file.</p>
                </body>
                </html>
                """
            return jsonify({'files': files})

        # For mobile/desktop apps, generate a simulated preview
        combined_files = "\n\n".join([f"### {filename}\n{content}" for filename, content in files.items()])
        conversion_prompt = f"""
        Convert the following {app_type} app project files into a single, self-contained HTML file that simulates the app.
        - Include all necessary CSS and JavaScript inline.
        - Replicate the app's behavior, interactions, and appearance as closely as possible.
        - If the app uses platform-specific features, simulate them using web technologies.
        - Ensure the generated HTML is fully functional and visually identical to the original app.

        Return ONLY the complete HTML code wrapped in <html> tags, no explanations:
        {combined_files}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You convert apps to web previews. Return only HTML code."},
                {"role": "user", "content": conversion_prompt}
            ]
        )

        converted_code = response.choices[0].message.content.strip()

        # Clean the response
        if converted_code.startswith("```html"):
            converted_code = converted_code[7:]
        if converted_code.endswith("```"):
            converted_code = converted_code[:-3]
        converted_code = converted_code.strip()

        # Validate we have proper HTML
        if not (converted_code.startswith("<!DOCTYPE html>") or "<html" in converted_code):
            # Create fallback preview
            converted_code = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{app_type} Preview</title>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif; 
                        padding: 20px;
                        background: #f5f5f5;
                    }}
                    .container {{ 
                        max-width: 800px; 
                        margin: 0 auto; 
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>{app_type.capitalize()} Preview</h1>
                    <p>This is a simulated preview of your {app_type}.</p>
                    <h3>Project Files:</h3>
                    <ul>
                        {"".join(f"<li>{f}</li>" for f in files.keys())}
                    </ul>
                </div>
            </body>
            </html>
            """

        return jsonify({'files': {'preview.html': converted_code}})

    except Exception as e:
        logging.error(f"Error in /preview: {str(e)}")
        return jsonify({'error': str(e)}), 500

def build_prompt(user_prompt, type_):
    if type_ == "desktop app":
        return f"""
You are a world-class code generator.

Generate a complete, production-ready desktop app based on this user description: "{user_prompt}"

👉 Use Python with a desktop GUI framework like Tkinter or PyQt. Alternatively, you can use Electron with JavaScript if specified in the user description.

👉 Output all files required to run this app in a single JSON object with the format:
{{ "filename.ext": "file content here", ... }}

🧠 Ensure the code is:
- Fully functional and error-free.
- Well-structured, modular, and follows best practices.
- Includes comments explaining key parts of the code.

📦 Include all necessary files, such as configuration files, dependencies, and assets.

🎯 Return only the JSON object with code files. No explanations, no markdown.
"""

    elif type_ == "mobile app":
        return f"""
You are a world-class code generator.

Generate a complete, production-ready mobile app based on this user description: "{user_prompt}"

👉 Use a mobile app framework like React Native (JavaScript) or Flutter (Dart). Alternatively, use native code (Swift for iOS or Kotlin for Android) if specified in the user description.

👉 Output all files required to run this app in a single JSON object with the format:
{{ "filename.ext": "file content here", ... }}

🧠 Ensure the code is:
- Fully functional and error-free.
- Well-structured, modular, and follows best practices.
- Includes comments explaining key parts of the code.

📦 Include all necessary files, such as configuration files, dependencies, and assets.

🎯 Return only the JSON object with code files. No explanations, no markdown.
"""

    else:  # Default to web app
        return f"""
You are a world-class code generator.

Generate a complete, production-ready {type_} based on this user description: "{user_prompt}"

👉 Output all files required to run this app in a single JSON object with the format:
{{ "filename.ext": "file content here", ... }}

🧠 Ensure the code is:
- Fully functional and error-free.
- Well-structured, modular, and follows best practices.
- Includes comments explaining key parts of the code.

📦 Include all necessary files, such as configuration files, dependencies, and assets.

🎯 Return only the JSON object with code files. No explanations, no markdown.
"""

if __name__ == '__main__':
    app.run(debug=True, port=5000)
