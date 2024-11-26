# %%
import torch
import torch.nn.functional as F
import transformer_lens
from transformer_lens import HookedTransformer
import os
import sys
# %%
os.chdir('/root/advint')
# Add the new working directory to sys.path
sys.path.append(os.getcwd())
# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# Load the adversarially trained model
adversarial_model = HookedTransformer.from_pretrained("tiny-stories-33M")
adversarial_model.load_state_dict(torch.load("saved_models/adversarially_trained_model.pth"))
adversarial_model.to(device)
print("Adversarial model loaded.")
# %%
# Load the unaugmented model
unaugmented_model = HookedTransformer.from_pretrained("tiny-stories-33M")
unaugmented_model.to(device)
print("Unaugmented model loaded.")

# Set both models to evaluation mode
adversarial_model.eval()
unaugmented_model.eval()

# Define prompts for text generation
prompts = [
    "Once upon a time,",
    "The quick brown fox",
    "In a galaxy far, far away,",
    "The stock market today",
    "Artificial intelligence is",
]
# %%
# Function to generate text using a given model and prompt
def generate_text(model, prompt, max_new_tokens=50, temperature=1.0):
    # Tokenize the prompt
    input_ids = model.to_tokens(prompt).to(device)
    generated = input_ids

    # Generate tokens iteratively
    for _ in range(max_new_tokens):
        outputs = model(generated)
        next_token_logits = outputs[:, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

    # Decode the generated tokens to text
    generated_text = model.to_string(generated[0])
    return generated_text
# %%
# Generate and compare texts from both models
for prompt in prompts:
    print(f"Prompt: {prompt}\n")

    print("Adversarial Model Output:")
    adversarial_output = generate_text(adversarial_model, prompt)
    print(adversarial_output)

    print("\nUnaugmented Model Output:")
    unaugmented_output = generate_text(unaugmented_model, prompt)
    print(unaugmented_output)

    print("\n" + "-" * 80 + "\n")

# %%
# Chat loop for interacting with the adversarial model
print("You can now chat with the adversarially trained model. Type 'exit' to quit.\n")

while True:
    # Get user input
    user_input = input("You: ")
    
    # Exit condition
    if user_input.lower() == 'exit':
        print("Exiting chat. Goodbye!")
        break

    # Generate response using the adversarial model
    response = generate_text(adversarial_model, user_input, max_new_tokens=50, temperature=1.0)
    print(f"Model: {response}\n")
# %%
