# Advint

This repository contains code for training and evaluating a sparse autoencoder model on tokenized text data. The model architecture, training scripts, and dataset handling are designed to be lightweight and portable for easy setup on different environments, including clusters.


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
python main.py!
```
