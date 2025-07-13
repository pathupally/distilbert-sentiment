# reproduce.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

def load_model_for_inference(model_id, lora_adapter_path, device):
    """Loads the base model and applies the LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(model, lora_adapter_path).to(device)
    model.eval()
    return tokenizer, model

def classify_sentiment(text, tokenizer, model, device):
    """Classifies the sentiment of a given text."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model_id = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
    lora_path = './model_output'  # Path where your LoRA adapter is saved

    tokenizer, model = load_model_for_inference(model_id, lora_path, device)

    # Example help desk ticket
    ticket_text = "I've been waiting for a response for three days. Your service is incredibly slow."
    sentiment = classify_sentiment(ticket_text, tokenizer, model, device)
    
    print(f"Help Desk Ticket: '{ticket_text}'")
    print(f"Predicted Sentiment: {sentiment.upper()}")