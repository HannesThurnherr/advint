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

# 4. (Optional) Train the advesarial model (SLOW! requires GPU)
python adversarial_training.py
# Or pull the pre-trained model from huggingface
python pull_from_hf.py 

```
After this you will have an adversarially trained model in the "saved_models" folder and both SAEs (for the original and the adversarially trained model) in the "saved_SAEs" folder. From here you can run any of the experiments in the experiments folder.


## Experiments

Each experiment can be run by running the script in the experiments folder. Any data generated will be saved in the `experiments/out` folder.
Run the corresponding plot script to visualize the data, which will save the plots in the `experiments/img` folder.

```
python -m experiments.steering_vector # saves data in experiments/out/steering_vector.json
python -m experiments.plot_steering_vector # saves plots in experiments/img/steering_vector.svg

python -m experiments.ce_loss # saves data in experiments/out/ce_loss.csv
python -m experiments.plot_ce_loss # saves plots in experiments/img/ce_loss.svg
```