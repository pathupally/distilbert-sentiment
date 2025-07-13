# Sentiment Analysis: Customer Support Ticket Classifier

This project provides a sentiment analysis app for classifying help desk ticket text as POSITIVE or NEGATIVE using a fine-tuned DistilBERT model with LoRA adapters. The app is built with Gradio for an interactive web interface.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Dataset Download](#dataset-download)
- [Local Setup & Usage](#local-setup--usage)
- [Docker Setup & Usage](#docker-setup--usage)
- [Training the Model](#training-the-model)
- [Running the App](#running-the-app)
- [Reproducing Results](#reproducing-results)
- [Project Structure](#project-structure)

---

## Features

- Fine-tuned DistilBERT for sentiment analysis
- LoRA adapter for efficient training
- Gradio web app for real-time inference
- Docker support for easy deployment

---

## Prerequisites

- Python 3.8+
- [Kaggle API](https://github.com/Kaggle/kaggle-api) (for dataset download)
- (Optional) Docker

---

## Dataset Download

1. **Install Kaggle API** (if not already):

   ```bash
   pip install kaggle
   ```

2. **Get your Kaggle API key:**
   - Go to your [Kaggle account settings](https://www.kaggle.com/account)
   - Click "Create New API Token" (downloads `kaggle.json`)
   - Place `kaggle.json` in `~/.kaggle/`
3. **Download the IMDB dataset:**

   ```bash
   cd data
   kaggle datasets download lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
   unzip imdb-dataset-of-50k-movie-reviews.zip -d imdb_50k
   rm imdb-dataset-of-50k-movie-reviews.zip
   cd ..
   ```

   The main CSV should be at `data/imdb_50k/IMDB_Dataset.csv` or move it to `data/IMDB_Dataset.csv` as needed.

---

## Local Setup & Usage

1. **Clone the repository and install dependencies:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **(Optional) Download the dataset** (see above).
3. **Train the model:**

   ```bash
   python train.py --batch_size 32 --num_epochs 3
   ```

   - The trained adapter and config will be saved in the `models/` directory.
4. **Launch the Gradio app:**

   ```bash
   python app.py
   ```

   - The app will be available at the local URL printed in the terminal.

---

## Docker Setup & Usage

1. **Build the Docker image:**

   ```bash
   docker build -t sentiment-app -f Docker/Dockerfile .
   ```

2. **(Optional) Download the dataset** (see above) and ensure it is present in the `data/` directory before building the image, or mount it as a volume.
3. **Train the model (inside Docker):**
   - You can run training as a one-off command:

     ```bash
     docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models sentiment-app python train.py --batch_size 32 --num_epochs 3
     ```

   - This will save the trained model to your local `models/` directory.
4. **Run the app (inside Docker):**

   ```bash
   docker run --rm -p 7860:7860 -v $(pwd)/models:/app/models sentiment-app
   ```

   - The app will be available at [http://localhost:7860](http://localhost:7860)

---

## Training the Model

- The training script uses the IMDB dataset and saves LoRA adapter weights to `models/`.
- You can adjust batch size and epochs:

  ```bash
  python train.py --batch_size 16 --num_epochs 5
  ```

---

## Running the App

- After training, run `python app.py` (locally) or use Docker as above.
- The app uses Gradio and will print a local URL for access.

---

## Reproducing Results

- To run a quick inference from the command line, use `reproduce.py`:

  ```bash
  python reproduce.py
  ```

- Edit `reproduce.py` to point to your adapter/model directory if needed.

---

## Project Structure

```
.
├── app.py              # Gradio web app
├── train.py            # Model training script
├── reproduce.py        # Script for quick inference
├── requirements.txt    # Python dependencies
├── data/               # Dataset directory
├── models/             # Saved adapters and config
├── utils/              # Utility modules
├── Docker/
│   └── Dockerfile      # Docker build file
└── README.md           # This file
```

---

## Notes

- Ensure the dataset is present before training or running the app.
- For GPU/Apple Silicon acceleration, the code will use `mps` if available, otherwise CPU.
- For any issues, check dependencies in `requirements.txt` and ensure your Python version matches the one in the Dockerfile or GitHub Actions.
