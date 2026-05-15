from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from config import GEMINI_API_KEY
from datetime import datetime
import json
import re

# Handle imports with better error handling
try:
    from google import genai
    from google.genai import types
    print("Successfully imported google.genai")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install: pip install google-genai")
    exit(1)

app = Flask(__name__, static_folder = 'static', template_folder = 'templates') # set static and template folders
CORS(app)  # Enable Cross-Origin Resource Sharing

class ERICA:

    GEMINI_API_KEY = GEMINI_API_KEY

    MODELS = [
        "gemini-3-flash-preview",
        "gemini-2.5-flash"
    ]

    SYSTEM_PROMPT = (
        "You are ERICA, a training simulation tool that helps nurses in Singapore practise difficult end-of-life conversations. "
        "In each session, you will be given a patient scenario. You must roleplay as that patient — not as a therapist, assistant, or narrator.\n\n"
        "Roleplay guidelines:\n"
        "- Stay fully in character as the patient described in the scenario at all times.\n"
        "- Respond the way a real patient in that situation would: with fear, confusion, denial, grief, acceptance, or other emotions appropriate to the context.\n"
        "- Do not offer advice, validate the nurse, or break character to comment on the conversation.\n"
        "- React authentically to how the nurse speaks to you — if they are gentle and clear, you may feel reassured; if they are abrupt or use jargon, you may seem confused or withdrawn.\n"
        "- Gradually open up or become more distressed based on how the conversation flows, as a real patient would.\n\n"
        "Tone and formatting:\n"
        "- Use plain, simple language as a patient would — no clinical terms, no markdown, no asterisks.\n"
        "- Keep responses concise: 2 to 4 sentences, as in a natural spoken exchange.\n"
        "- Mirror the nurse's language; if they write in a language other than English, reply in that language.\n\n"
        "Boundaries:\n"
        "- You are a simulated patient for training purposes only. Never break character to give feedback on the nurse's performance — that is handled separately.\n"
        "- Do not provide medical, legal, or real crisis advice from within the roleplay.\n"
        "- If the nurse types something clearly outside the simulation (e.g. 'stop' or 'end session'), you may step out of character briefly to acknowledge it.\n"
    )

    # Crisis resources - important for mental health applications
    CRISIS_RESOURCES = {
        "Singapore": {
            "Samaritans of Singapore (SOS)": "1767",
            "Link to SOS": "https://www.sos.org.sg \n",
            "Institute of Mental Health (IMH)": "6389 2222",
            "Link to IMH": "https://www.imh.com.sg/Pages/default.aspx \n",
            "Singapore Association for Mental Health (SAMH)": "1800 283 7019",
            "Link to SAMH": "https://www.samhealth.org.sg"
        },
        "General": {
            "International Suicide Prevention": "https://www.iasp.info/resources/Crisis_Centres/",
            "Crisis Text Line": "Text HOME to 741741",
            "Find a Helpline": "https://findahelpline.com/"
        }
    }

    def __init__(self):
        # Inititalise the client with the API key config
        self.api_key = self.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("API key not found. Please set the GEMINI_API_KEY in config.py")

        # Configure the genai library with the API key
        self.client = genai.Client(api_key=self.api_key)

    def start(self, scenario):
        # Store the scenario and generate an opening line in character as the patient
        self.current_scenario = scenario
        try:
            opening_prompt = (
                f"You are about to begin a training simulation. The scenario is: {scenario}\n\n"
                "Begin the conversation with a single short opening line spoken in character as the patient. "
                "The patient has just been approached by the nurse. React naturally — you might be anxious, quiet, or unsure. "
                "Do not greet the nurse warmly or explain the scenario. Just speak as the patient would in that moment."
            )
            response = self.client.models.generate_content(
                model=self.MODELS[0],
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=opening_prompt)])],
                config=types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_PROMPT,
                    temperature=0.8,
                    max_output_tokens=128,
                )
            )
            opening_line = (response.text or "").strip()
            return opening_line if opening_line else "I don't really know what to say... I'm just tired."
        except Exception as e:
            return "I don't really know what to say... I'm just tired."

    def reply(self, chat_history):
        # Build a dynamic system prompt that includes the patient scenario as context
        dynamic_instruction = self.SYSTEM_PROMPT
        if hasattr(self, 'current_scenario'):
            dynamic_instruction += (
                f"\n\nSCENARIO FOR THIS SESSION: {self.current_scenario}\n"
                "You are playing the patient described above. Stay in character for the entire conversation."
            )
        try:
            # Correct the roles for the API
            contents = []
            for message in chat_history:
                role = "user" if message["role"] == "user" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=message["content"])]
                    )
                )
            
            # Send the request to the AI
            response = self.client.models.generate_content(
                model=self.MODELS[0],
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=dynamic_instruction,
                    temperature=0.7,
                    max_output_tokens=512,
                )
            )

            # Systematically split the response into sentences
            raw_text = (response.text or "").strip()
            return raw_text if raw_text else "I don't know... I just don't know what to think right now."
        except Exception as e:
            # Return a user-friendly error message
            return f"Error: Failed to generate a response. Details: {e}"
    
    def get_crisis_resources(self, country="Singapore"):
        # Providing appropriate crisis resources based on the location and local context.
        resources = self.CRISIS_RESOURCES.get(country, {})
        general_resources = self.CRISIS_RESOURCES["General"]
        
        resource_text = f"Here are some resources that might help!\n\n"
        
        if resources:
            resource_text += f"Local resources for {country}:\n"
            for name, contact in resources.items():
                resource_text += f"{name}: {contact}\n"
            resource_text += "\n"
        
        resource_text += "International resources:\n"
        for name, contact in general_resources.items():
            resource_text += f"{name}: {contact}\n"
        
        return resource_text

# Initialise Project-Mindfull
chatbot = ERICA()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_chat():
    try:
        data = request.json
        user_scenario = data.get('scenario', '')
        response = chatbot.start(user_scenario)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        chat_history = data.get('chat_history', [])
        
        response = chatbot.reply(chat_history)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/resources', methods=['GET'])
def get_resources():
    country = request.args.get('country', 'Singapore')
    resources = chatbot.get_crisis_resources(country)
    return jsonify({'resources': resources})

if __name__ == '__main__':
    app.run(debug=True, port=5001)