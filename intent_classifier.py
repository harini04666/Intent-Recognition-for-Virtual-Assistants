from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Updated dataset
data = [
    {"intent": "greeting", "text": "Hello"},
    {"intent": "greeting", "text": "Hi"},
    {"intent": "greeting", "text": "Hii"},
    {"intent": "greeting", "text": "Hey"},
    {"intent": "greeting", "text": "Hey there"},
    {"intent": "greeting", "text": "Hi there"},
    {"intent": "greeting", "text": "Good morning"},

    {"intent": "goodbye", "text": "Goodbye"},
    {"intent": "goodbye", "text": "Bye"},
    {"intent": "goodbye", "text": "See you later"},
    {"intent": "goodbye", "text": "See ya"},
    {"intent": "goodbye", "text": "Catch you later"},

    {"intent": "weather", "text": "What's the weather like?"},
    {"intent": "weather", "text": "Is it raining today?"},
    {"intent": "weather", "text": "Tell me the weather forecast"},
    {"intent": "weather", "text": "Will it rain today?"},
    {"intent": "weather", "text": "Is it sunny outside?"},

    {"intent": "time", "text": "What time is it?"},
    {"intent": "time", "text": "Can you tell me the time?"},
    {"intent": "time", "text": "What’s the current time?"},
    {"intent": "time", "text": "Time please"},
    {"intent": "time", "text": "Tell me the time"}
]

# Prepare the dataset
texts = [item['text'] for item in data]
labels = [item['intent'] for item in data]

# Build the model (no train/test split)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(texts, labels)

# Predict function
def predict_intent(text):
    return model.predict([text])[0]

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Goodbye!")
        break
    intent = predict_intent(user_input)
    print(f"Intent recognized: {intent}")
