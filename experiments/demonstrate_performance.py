"""
python -m experiments.demonstrate_performance
"""

# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import torch.nn.functional as F
import transformer_lens
from transformer_lens import HookedTransformer

import sys
# %%
#os.chdir('/root/advint')
# Add the new working directory to sys.path
#sys.path.append(os.getcwd())
# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# Load the adversarially trained model
adversarial_model = HookedTransformer.from_pretrained("tiny-stories-33M")
adversarial_model.load_state_dict(torch.load("models/lm_adv.pth"))
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
# Generate and compare texts from both models
for prompt in prompts:
    print(f"Prompt: {prompt}\n")

    print("Adversarial Model Output:")
    adversarial_output = adversarial_model.generate(prompt, max_new_tokens=50, temperature=1.0)
    print(adversarial_output)

    print("\nUnaugmented Model Output:")
    unaugmented_output = unaugmented_model.generate(prompt, max_new_tokens=50, temperature=1.0)
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
    response = adversarial_model.generate(user_input, max_new_tokens=50, temperature=1.0)
    print(f"Model: {response}\n")
# %%
