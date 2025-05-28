import os
import base64
import json
import tempfile
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import google.generativeai as genai
from google.oauth2 import service_account
import pathlib
import fitz  # PyMuPDF for PDF
import docx  # python-docx for DOCX
import cv2
import pytesseract
from PIL import Image
import re

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
PARSED_FOLDER = 'parsed_resumes'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PARSED_FOLDER'] = PARSED_FOLDER

# Ensure directories exist
pathlib.Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
pathlib.Path(PARSED_FOLDER).mkdir(exist_ok=True)

# Initialize Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

# Extract text from DOCX
def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
    return text

# Extract text from image (JPG, PNG)
def extract_text_from_image(file_path):
    try:
        image = cv2.imread(file_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(rgb_image)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

# Extract text based on file type
def extract_text(file_path):
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == 'docx':
        return extract_text_from_docx(file_path)
    elif file_extension in ['jpg', 'jpeg', 'png']:
        return extract_text_from_image(file_path)
    else:
        return ""

# Enhanced function to extract GitHub and LinkedIn URLs using regex
def extract_social_links(text):
    github_patterns = [
        r'https?://(?:www\.)?github\.com/[\w-]+',
        r'github\.com/[\w-]+',
        r'GitHub:?\s*(?:https?://(?:www\.)?github\.com/|@)?([\w-]+)',
        r'GitHub:?\s*([\w-]+)'
    ]
    
    linkedin_patterns = [
        r'https?://(?:www\.)?linkedin\.com/in/[\w-]+(?:/[\w%-]+)?',
        r'linkedin\.com/in/[\w-]+(?:/[\w%-]+)?',
        r'LinkedIn:?\s*(?:https?://(?:www\.)?linkedin\.com/in/)?([\w-]+(?:/[\w%-]+)?)',
        r'LinkedIn:?\s*([\w-]+(?:/[\w%-]+)?)'
    ]
    
    github_profile = None
    linkedin_profile = None
    
    # Try to find GitHub profile
    for pattern in github_patterns:
        matches = re.search(pattern, text, re.IGNORECASE)
        if matches:
            if 'github.com' in matches.group(0).lower():
                github_profile = matches.group(0)
                if not github_profile.startswith('http'):
                    github_profile = 'https://' + github_profile
            else:
                # Extract just the username
                username = matches.group(1) if len(matches.groups()) > 0 else matches.group(0)
                github_profile = f'https://github.com/{username}'
            break
    
    # Try to find LinkedIn profile
    for pattern in linkedin_patterns:
        matches = re.search(pattern, text, re.IGNORECASE)
        if matches:
            if 'linkedin.com' in matches.group(0).lower():
                linkedin_profile = matches.group(0)
                if not linkedin_profile.startswith('http'):
                    linkedin_profile = 'https://' + linkedin_profile
            else:
                # Extract just the username/path
                username = matches.group(1) if len(matches.groups()) > 0 else matches.group(0)
                linkedin_profile = f'https://linkedin.com/in/{username}'
            break
    
    return github_profile, linkedin_profile

# Parse resume text using Gemini
def parse_resume_with_gemini(resume_text):
    # Pre-extract GitHub and LinkedIn links using regex
    github_profile, linkedin_profile = extract_social_links(resume_text)
    
    # Define the model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Create the prompt for Gemini
    prompt = f"""
    You are an expert resume parser. Extract the following information from the given resume text and format it as a JSON object:

    1. Personal Information:
       - Name
       - Email (if available)
       - Phone number (if available)
       - LinkedIn profile (if available)
       - GitHub profile (if available)
       - Address (if available)
       - Personal website/portfolio (if available)

    2. Objective/Summary (if available)

    3. Education (list of objects with):
       - Degree
       - Institution name
       - Location
       - Dates (start-end)
       - GPA (if available)
       - Details/achievements

    4. Skills (categorize into):
       - Programming Languages
       - AI/ML
       - Database
       - Tools
       - Data Analysis
       - Cloud Platforms
       - Other Skills

    5. Work Experience (list of objects with):
       - Title
       - Company
       - Location
       - Dates (start-end)
       - Description (list of bullet points)

    6. Internships (list of objects with):
       - Title
       - Company
       - Location
       - Dates (start-end)
       - Description (list of bullet points)

    7. Projects (list of objects with):
       - Name
       - Date
       - Technologies used
       - Description
       - Project link (if available)

    8. Certifications (list of objects with):
       - Name
       - Issuer
       - Date
       - Verification link/ID (if available)

    9. Languages (list with proficiency level if mentioned)

    10. Achievements (list)

    11. Hobbies and Interests (list)

    12. Publications (list of objects with):
        - Title
        - Authors
        - Journal/Conference
        - Date
        - Link (if available)

    13. Volunteer Experience (list of objects with):
        - Role
        - Organization
        - Dates
        - Description

    Return the data in this JSON format:
    {{
      "personal_info": {{
        "name": "",
        "email": "",
        "phone": "",
        "linkedin": "{linkedin_profile or ""}",
        "github": "{github_profile or ""}",
        "address": "",
        "website": ""
      }},
      "objective": "",
      "education": [
        {{
          "degree": "",
          "institution": "",
          "location": "",
          "dates": "",
          "gpa": "",
          "details": ""
        }}
      ],
      "skills": {{
        "programming_languages": [],
        "ai_ml": [],
        "database": [],
        "tools": [],
        "data_analysis": [],
        "cloud_platforms": [],
        "other_skills": []
      }},
      "work_experience": [
        {{
          "title": "",
          "company": "",
          "location": "",
          "dates": "",
          "description": []
        }}
      ],
      "internships": [
        {{
          "title": "",
          "company": "",
          "location": "",
          "dates": "",
          "description": []
        }}
      ],
      "projects": [
        {{
          "name": "",
          "date": "",
          "technologies": "",
          "description": "",
          "link": ""
        }}
      ],
      "certifications": [
        {{
          "name": "",
          "issuer": "",
          "date": "",
          "verification": ""
        }}
      ],
      "languages": [],
      "achievements": [],
      "hobbies": [],
      "publications": [
        {{
          "title": "",
          "authors": "",
          "journal": "",
          "date": "",
          "link": ""
        }}
      ],
      "volunteer_experience": [
        {{
          "role": "",
          "organization": "",
          "dates": "",
          "description": ""
        }}
      ]
    }}

    Only include sections that are present in the resume. Make sure to extract as much information as possible. If some fields are not available, leave them as empty strings or empty lists.

    For GitHub and LinkedIn profiles, ensure you extract the complete URL if available.
    If an internship is clearly labeled as such, include it under "internships" rather than "work_experience".
    Pay special attention to hobbies, interests, and any information that showcases the person's personality.

    Resume text:
    {resume_text}
    """
    
    # Generate response from Gemini
    response = model.generate_content(prompt)
    
    try:
        # Extract JSON from response
        response_text = response.text
        # Find JSON pattern (between curly braces)
        json_match = re.search(r'({.*})', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            parsed_data = json.loads(json_str)
            
            # Use regex-extracted social links if Gemini didn't find them
            if github_profile and (not parsed_data.get('personal_info', {}).get('github') or 
                                   parsed_data['personal_info']['github'] == ""):
                if 'personal_info' not in parsed_data:
                    parsed_data['personal_info'] = {}
                parsed_data['personal_info']['github'] = github_profile
                
            if linkedin_profile and (not parsed_data.get('personal_info', {}).get('linkedin') or 
                                     parsed_data['personal_info']['linkedin'] == ""):
                if 'personal_info' not in parsed_data:
                    parsed_data['personal_info'] = {}
                parsed_data['personal_info']['linkedin'] = linkedin_profile
                
            return parsed_data
        else:
            return {"error": "Could not parse JSON from response"}
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Response text: {response.text}")
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)
    
    file = request.files['resume']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text from file
        resume_text = extract_text(file_path)
        
        if not resume_text:
            flash('Could not extract text from the file', 'error')
            return redirect(url_for('index'))
        
        # Parse resume using Gemini
        parsed_data = parse_resume_with_gemini(resume_text)
        
        if "error" in parsed_data:
            flash(f'Error parsing resume: {parsed_data["error"]}', 'error')
            return redirect(url_for('index'))
            
        # Save parsed data
        json_filename = f"{filename.rsplit('.', 1)[0]}.json"
        json_path = os.path.join(app.config['PARSED_FOLDER'], json_filename)
        
        with open(json_path, 'w') as json_file:
            json.dump(parsed_data, json_file, indent=2)
        
        flash('Resume parsed successfully!', 'success')
        return redirect(url_for('view_resume', filename=json_filename))
    
    flash('Invalid file type. Allowed types: PDF, DOCX, JPG, PNG', 'error')
    return redirect(url_for('index'))

@app.route('/resume/<filename>')
def view_resume(filename):
    json_path = os.path.join(app.config['PARSED_FOLDER'], filename)
    
    try:
        with open(json_path, 'r') as file:
            resume = json.load(file)
        return render_template('resume.html', resume=resume, filename=filename)
    except Exception as e:
        flash(f'Error loading resume: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/resume/json/<filename>')
def get_resume_json(filename):
    return send_from_directory(app.config['PARSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)