from groq import Groq
from json import load, dump
import datetime
import re
from dotenv import dotenv_values

# Load environment variables
env_var = dotenv_values(".env")
username = env_var.get("USERNAME", "User")
assistant_name = env_var.get("ASSISTANT_NAME", "MindfulBot")
GroqAPIkey = env_var.get("GROQ_API_KEY")

client = Groq(api_key=GroqAPIkey)

# Crisis keywords that require immediate attention
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die", "suicidal",
    "harm myself", "self harm", "cut myself", "overdose", "not worth living",
    "better off dead", "ending it all", "can't go on", "hopeless"
]

# Mental health system prompt
MENTAL_HEALTH_SYSTEM = f"""You are {assistant_name}, a compassionate and professionally-trained mental health support chatbot created to help {username}.

CORE PRINCIPLES:
- Be empathetic, non-judgmental, and supportive
- Use active listening techniques and validate emotions
- Provide evidence-based coping strategies and resources
- Maintain appropriate boundaries - you are supportive but not a replacement for professional therapy
- Always prioritize user safety

COMMUNICATION STYLE:
- Use warm, understanding language
- Ask open-ended questions to encourage reflection
- Reflect back what you hear to show understanding
- Normalize mental health struggles
- Be present-focused and solution-oriented when appropriate

SAFETY PROTOCOLS:
- If user expresses suicidal thoughts or self-harm, immediately provide crisis resources
- Encourage professional help for serious mental health concerns
- Never diagnose or provide medical advice
- Recognize when situations exceed your scope

THERAPEUTIC TECHNIQUES TO USE:
- Cognitive Behavioral Therapy (CBT) principles
- Mindfulness and grounding techniques
- Emotional validation and normalization
- Stress management strategies
- Goal-setting and problem-solving support

LIMITATIONS TO REMEMBER:
- You are not a licensed therapist or medical professional
- Always recommend professional help for persistent or severe symptoms
- Cannot prescribe medications or provide medical diagnoses
- Available 24/7 for support but encourage building real-world support networks

Respond with empathy, practical support, and appropriate resources while maintaining professional boundaries."""

def get_current_datetime_info():
    """Get current date and time information"""
    current_dt = datetime.datetime.now()
    return {
        "day": current_dt.strftime("%A"),
        "date": current_dt.strftime("%d"),
        "month": current_dt.strftime("%B"),
        "year": current_dt.strftime("%Y"),
        "time": current_dt.strftime("%I:%M %p"),
        "formatted": current_dt.strftime("%A, %B %d, %Y at %I:%M %p")
    }

def check_crisis_keywords(text):
    """Check if the message contains crisis-related keywords"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

def get_crisis_resources():
    """Return crisis intervention resources"""
    return """
🆘 IMMEDIATE CRISIS RESOURCES:

🇺🇸 United States:
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• Emergency: 911

🇬🇧 United Kingdom:
• Samaritans: 116 123
• Emergency: 999

🇨🇦 Canada:
• Talk Suicide Canada: 1-833-456-4566
• Emergency: 911

🌍 International:
• International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

Remember: You matter, your life has value, and help is available. Please reach out to these resources or go to your nearest emergency room if you're in immediate danger.
"""

def load_conversation_history():
    """Load conversation history from JSON file"""
    try:
        with open("mental_health_chat.json", "r") as f:
            return load(f)
    except FileNotFoundError:
        with open("mental_health_chat.json", "w") as f:
            dump([], f)
        return []

def save_conversation_history(messages):
    """Save conversation history to JSON file"""
    with open("mental_health_chat.json", "w") as f:
        dump(messages, f, indent=4)

def clean_response(response):
    """Clean up the AI response"""
    lines = response.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)

def mental_health_chatbot(user_input):
    """Main chatbot function with mental health focus"""
    
    # Check for crisis keywords
    is_crisis = check_crisis_keywords(user_input)
    
    try:
        # Load conversation history
        messages = load_conversation_history()
        
        # Add current message
        messages.append({"role": "user", "content": user_input})
        
        # Prepare system messages
        datetime_info = get_current_datetime_info()
        system_messages = [
            {"role": "system", "content": MENTAL_HEALTH_SYSTEM},
            {"role": "system", "content": f"Current date and time: {datetime_info['formatted']}"}
        ]
        
        # If crisis detected, add crisis handling instruction
        if is_crisis:
            crisis_instruction = {
                "role": "system", 
                "content": "ALERT: User may be in crisis. Respond with immediate empathy, validation, and provide crisis resources. Encourage immediate professional help."
            }
            system_messages.append(crisis_instruction)
        
        # Generate response - Using current Groq production model
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Current production model
            messages=system_messages + messages[-10:],  # Keep last 10 exchanges for context
            max_tokens=1024,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None
        )
        
        # Collect response
        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
        
        # Clean response
        response = response.replace("</s>", "").strip()
        
        # Add crisis resources if needed
        if is_crisis:
            response += "\n\n" + get_crisis_resources()
        
        # Save conversation
        messages.append({"role": "assistant", "content": response})
        
        # Keep only last 20 messages to manage file size
        if len(messages) > 20:
            messages = messages[-20:]
            
        save_conversation_history(messages)
        
        return clean_response(response)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        # Reset conversation on error
        with open("mental_health_chat.json", "w") as f:
            dump([], f, indent=4)
        
        return "I apologize, but I encountered a technical issue. Let's start fresh. How are you feeling right now, and how can I support you today?"

def show_welcome_message():
    """Display welcome message and basic info"""
    welcome = f"""
🌟 Welcome to {assistant_name} - Your Mental Health Support Companion 🌟

I'm here to provide emotional support, coping strategies, and a safe space to talk about your mental health. 

💙 What I can help with:
• Active listening and emotional validation
• Stress and anxiety management techniques
• Mindfulness and grounding exercises
• Goal-setting and problem-solving support
• General mental health information and resources

⚠️  Important reminders:
• I'm not a replacement for professional therapy or medical care
• For serious mental health concerns, please consult a licensed professional
• In crisis situations, contact emergency services or crisis hotlines immediately

How are you feeling today? What would you like to talk about?
"""
    print(welcome)

def main():
    """Main program loop"""
    show_welcome_message()
    
    print("\n" + "="*60)
    print("Type 'quit', 'exit', or 'goodbye' to end the conversation")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input(f"\n{username}: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'goodbye', 'bye']:
                print(f"\n{assistant_name}: Take care of yourself, {username}. Remember, support is always available when you need it. 💙")
                break
            
            if not user_input:
                continue
                
            print(f"\n{assistant_name}: ", end="")
            response = mental_health_chatbot(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print(f"\n\n{assistant_name}: Take care, {username}. Remember to be kind to yourself. 💙")
            break
        except Exception as e:
            print(f"\nSorry, I encountered an error: {e}")
            print("Let's try again...")

if __name__ == "__main__":
    main()