import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


def load_model_for_inference(model_id, lora_adapter_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model = PeftModel.from_pretrained(model, lora_adapter_path).to(device)
    model.eval()
    return tokenizer, model


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model_id = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
lora_path = './models'

tokenizer, model = load_model_for_inference(model_id, lora_path, device)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    labels = ["NEGATIVE", "POSITIVE"] 
    
    return {labels[i]: float(probabilities[i]) for i in range(len(labels))}

# Define the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, label="Enter Help Desk Ticket Text"),
    outputs=gr.Label(num_top_classes=2),
    title="Customer Support Ticket Sentiment Analysis",
    description="This app classifies the sentiment of a help desk ticket as POSITIVE or NEGATIVE in real-time."
)

if __name__ == "__main__":
    iface.launch()