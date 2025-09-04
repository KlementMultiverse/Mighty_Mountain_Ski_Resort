# Mighty_Mountain_Ski_Resort
AI assistant for ski resort customer servic
⛷️ Mighty Mountain Ski Resort Assistant

A fine-tuned version of Google Gemma-2-2b that answers customer questions about lift tickets, trail conditions, hours, and policies at Mighty Mountain Ski Resort.
🤖 Model Details

    Base model: google/gemma-2-2b
    Fine-tuned with: LoRA + QLoRA (PEFT)
    Training data: 831 instruction-response pairs
    Framework: Hugging Face Transformers
    Use case: Customer service automation

💬 Example Usage

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Load base model
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

# Load your fine-tuned adapter
model = PeftModel.from_pretrained(model, "KlemGunn0519/Mighty_Mountain_Ski_Resort")

# Run inference
prompt = "### Instruction\\nWhat are daily lift ticket prices?\\n\\n### Response"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))



You should see: 

    Daily lift tickets range from $89–129 depending on day of the week... 
     

📦 Files 

    adapter_model.safetensors: Trained LoRA weights
    adapter_config.json: PEFT configuration
    Tokenizer files for compatibility
     
