
# Add comprehensive error handling
import logging
import sys
from functools import wraps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def handle_errors(func):
    """Decorator for error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper


# app_cpu.py - Force CPU mode to avoid CUDA issues
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import gradio as gr
import torch

# Force CPU mode
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

base_model_id = "google/gemma-2-2b"
adapter_id = "KlemGunn0519/Mighty_Mountain_Ski_Resort"

print("🔧 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print("🧠 Loading base model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="cpu",  # Force CPU
    torch_dtype=torch.float32,  # Use float32 for CPU
    low_cpu_mem_usage=True
)

print("🔁 Loading fine-tuned LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_id)

def respond(message, history):
    try:
        prompt = f"### Instruction\n{message}\n\n### Response"
        
        inputs = tokenizer(prompt, return_tensors="pt")  # No .to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            top_k=50
        )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Response" in full_response:
            answer = full_response.split("### Response")[-1].strip()
        else:
            answer = full_response.strip()
        
        return answer if answer else "I couldn't generate a response. Please try again."
        
    except Exception as e:
        return f"Error: {str(e)}"

print("🚀 Launching Ski Resort Assistant (CPU Mode)...")

gr.ChatInterface(
    fn=respond,
    title="🏔️ Mighty Mountain Ski Resort Assistant (CPU Mode)",
    description="Running on CPU for stability - ask about lift tickets, trails, and policies!",
    examples=[
        "What are daily lift ticket prices?",
        "What time do lifts open?", 
        "Do you rent snowboards?",
        "Is the Black Diamond trail open today?",
        "What are your mask policies?"
    ]
).launch(server_name="0.0.0.0", server_port=7860)
