from flask import Flask, request, jsonify, render_template
import openai
import os
import zipfile
import re

app = Flask(__name__)

# Set your OpenAI API Key
openai_client = openai.OpenAI(api_key="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")

# Extract only code from the response
def extract_code(response_text):
    """Filters and extracts only code from OpenAI response."""
    code_blocks = re.findall(r'```[a-zA-Z]*\n(.*?)```', response_text, re.DOTALL)
    return '\n'.join(code_blocks) if code_blocks else response_text

# Generate structured code
def generate_code(prompt, platform, app_type):
    """Uses OpenAI to generate structured code based on user input."""
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": f"You are an expert {platform} {app_type} developer. Generate clean and error-free code without explanations."},
            {"role": "user", "content": prompt} 
        ]
    )
    raw_output = response.choices[0].message.content.strip()
    return extract_code(raw_output)

@app.route('/')
def index():
    """Render the main UI page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_app():
    """Handles user requests to generate frontend, backend, or both."""
    data = request.json
    prompt = data.get("prompt")
    platform = data.get("platform")
    app_type = data.get("app_type")  # frontend, backend, fullstack

    if not prompt or not platform or not app_type:
        return jsonify({"error": "Missing prompt, platform, or app_type."}), 400

    try:
        output_dir = "generated_apps"
        os.makedirs(output_dir, exist_ok=True)
        file_paths = []
        response_data = {}

        if app_type in ["frontend", "fullstack"]:
            frontend_code = generate_code(prompt, platform, "frontend")
            frontend_file = os.path.join(output_dir, f"frontend_{platform}.txt")
            with open(frontend_file, "w") as f:
                f.write(frontend_code)
            file_paths.append(frontend_file)
            response_data["frontend_code"] = frontend_code

        if app_type in ["backend", "fullstack"]:
            backend_code = generate_code(prompt, platform, "backend")
            backend_file = os.path.join(output_dir, f"backend_{platform}.txt")
            with open(backend_file, "w") as f:
                f.write(backend_code)
            file_paths.append(backend_file)
            response_data["backend_code"] = backend_code

        # Zip the files
        zip_path = os.path.join(output_dir, f"app_{platform}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in file_paths:
                zipf.write(file, os.path.basename(file))

        response_data["message"] = "App generated successfully."
        response_data["download_link"] = zip_path
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
