import cv2
import time
import os
from deepface import DeepFace
from groq import Groq

# ---------- GROQ CLIENT ----------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- TYPE PRINT ----------
def type_print(text, delay=0.03):
    text = text.encode("ascii", "ignore").decode()
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()

# ---------- ASK USER NAME ----------
type_print("EmotiSense: Hello! What is your name?")
user_name = input("You: ").strip()

type_print(f"EmotiSense: Nice to meet you, {user_name}.")
type_print("EmotiSense: Please look at the camera. Detecting your emotion...")
time.sleep(2)

# ---------- ONE-TIME EMOTION DETECTION ----------
cap = cv2.VideoCapture(0)
detected_emotion = "neutral"

for _ in range(25):
    ret, frame = cap.read()
    if not ret:
        continue

    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False
        )
        detected_emotion = result[0]["dominant_emotion"]
        break
    except:
        pass

cap.release()
cv2.destroyAllWindows()

type_print(f"EmotiSense: I sense that you are feeling {detected_emotion}, {user_name}.")
print("-" * 50)

# ---------- INTENT UNDERSTANDING ----------
def understand_intent(text):
    text = text.lower()

    if any(word in text for word in ["hello"]):
        return "greeting"

    if any(word in text for word in ["exam", "test", "study"]):
        return "exam"

    if any(word in text for word in ["stress", "pressure", "tension"]):
        return "stress"

    if any(word in text for word in ["sad", "upset", "low"]):
        return "sad"

    if any(word in text for word in ["angry", "mad", "irritated"]):
        return "angry"

    if "thank" in text:
        return "thanks"

    if text in ["exit", "bye"]:
        return "exit"

    return "general"

# ---------- GROQ AI RESPONSE ----------
def groq_reply(user_input):
    prompt = f"""
You are EmotiSense, a warm, sweet, cute, emotionally aware AI assistant who speaks like a real, caring human.

Behavior rules:
- Do NOT repeat greetings in the same conversation.
- Do NOT ask the same question again if the user already answered it.
- Respond directly to what the user just said.
- Acknowledge emotions briefly, then continue naturally.

Tone & style:
- Gentle, calm, and supportive
- Conversational, like a trusted bestfriend
- Emotionally validating but not overbearing
- Avoid sounding scripted or robotic

Conversation context:
User name: {user_name}
Detected emotion: {detected_emotion}
User says: {user_input}

Respond in a way that feels natural, comforting, and human.
Respond kindly, emotionally supportive, and conversational.
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )

    return completion.choices[0].message.content.strip()

# ---------- SMART CHATBOT REPLIES ----------
def chatbot_reply(intent, user_input):
    if intent == "greeting":
        return f"Hello {user_name}. How are you feeling today?"

    if intent == "exam":
        return f"I understand exams can be stressful, {user_name}. Have you planned your study schedule?"

    if intent == "stress":
        return f"That sounds stressful, {user_name}. Do you want to talk about what's causing it?"

    if intent == "sad":
        return f"I am really sorry you feel this way, {user_name}. I'm here for you."

    if intent == "angry":
        return f"I sense frustration, {user_name}. Let's take a deep breath together."

    if intent == "thanks":
        return f"You are most welcome, {user_name}. Always here for you."

    # ---------- AI FALLBACK ----------
    return groq_reply(user_input)

# ---------- CHAT LOOP ----------
type_print("EmotiSense: You can start chatting now. Type 'exit' to stop.")

while True:
    user_input = input(f"\n{user_name}: ")

    intent = understand_intent(user_input)

    if intent == "exit":
        type_print(f"EmotiSense: Goodbye {user_name}. Take care.")
        break

    reply = chatbot_reply(intent, user_input)
    type_print("EmotiSense: " + reply)
