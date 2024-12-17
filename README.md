# Advint: Adversarial Interpretability with Sparse Autoencoders

Advint tests if transformers can restructure their internal representations to become harder (or easier) to interpret. Using sparse autoencoders (SAEs), this framework adversarially trains language models (LMs) to reduce interpretability while retaining performance. The project explores if models can “hide” information from SAEs, impacting their effectiveness in interpretability tasks. This method scales from small models to larger ones like the 33M Tiny Stories model, with ongoing work to test across even bigger models. Advint aims to inform AI alignment and transparency by revealing if models can control how understandable their internals are.


## Getting Started

Follow these steps to set up and run the code on a new machine or cluster.

### Prerequisites

- Python 3.8 or higher
- `git` for cloning the repository
- GPU support (recommended for faster training)

### Setup and Run

```bash
# 1. Clone the Repository
git clone https://github.com/HannesThurnherr/advint.git
cd advint

# 2. Set Up a Virtual Environment
python -m venv myenv
source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Run the Code
python adversarial_training.py!

```
After this you will have an adversarially trained model in the "saved_models" folder and both SAEs (for the original and the adversarially trained model) in the "saved_SAEs" folder. From here you can run any of the experiments in the experiments folder.