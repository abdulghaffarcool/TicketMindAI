# TicketMind AI — Intelligent Support Ticket Analysis

**An Applied AI portfolio project for automated support-ticket intent classification using PyTorch, FastAPI, and a web interface.**

TicketMind AI analyzes a customer support message, predicts its intent, estimates the model's confidence, and recommends whether the ticket can be accepted automatically or should receive manual review.

## Features

* **AI-powered classification:** Predicts one of 77 customer-support intent categories.
* **PyTorch model:** Uses trainable word embeddings, average pooling, and a neural-network classifier.
* **Machine learning baseline:** Compares the neural model with a TF-IDF + Logistic Regression classifier built with scikit-learn.
* **Confidence-based routing:** Assigns a provisional review status based on prediction confidence.
* **REST API:** Exposes prediction and health-check endpoints through FastAPI.
* **Interactive web interface:** Submit support messages and view predicted intent, confidence, and review status.
* **Exploratory data analysis:** Investigates dataset distribution, class balance, and text characteristics.
* **Reusable inference pipeline:** Loads the pretrained model and vocabulary without retraining.
* **API documentation:** Provides interactive Swagger documentation through FastAPI.

## Technology Stack

| Category         | Technologies               |
| ---------------- | -------------------------- |
| Language         | Python                     |
| Machine learning | PyTorch, scikit-learn      |
| Data processing  | pandas, NumPy              |
| Visualization    | Matplotlib, Seaborn        |
| API              | FastAPI, Uvicorn, Pydantic |
| Frontend         | HTML, CSS, JavaScript      |
| Development      | Jupyter Notebook, Git      |
| Dataset          | Banking77 by PolyAI        |

## Architecture

```text
                 Customer Support Message
                            |
                            v
                  Web Interface
                HTML / CSS / JavaScript
                            |
                      HTTP POST
                            |
                            v
                    FastAPI Backend
                       /predict
                            |
                            v
                   Text Preprocessing
                            |
                            v
                 PyTorch Ticket Classifier
                  +-------------------+
                  | Word Embeddings   |
                  | Average Pooling   |
                  | Neural Classifier |
                  +-------------------+
                            |
                            v
                Predicted Intent + Confidence
                            |
                            v
                    Review Status
                            |
                            v
                   Web Interface Result
```

The model and its vocabulary are stored locally in the repository. The application does not need to download a pretrained model or train one at startup.

## Machine Learning Approach

### Dataset

The project uses the **Banking77** dataset, a collection of customer banking queries labeled with their intended support categories.

| Dataset property       |                 Value |
| ---------------------- | --------------------: |
| Training examples      |                10,003 |
| Official test examples |                 3,080 |
| Intent categories      |                    77 |
| Input                  | Customer message text |
| Target                 |       Intent category |

Dataset source: [PolyAI Banking77 dataset](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking77).

The dataset is intended for banking-related intent classification. Performance on it should not be assumed to represent performance on every customer-support domain.

### Models evaluated

**1. TF-IDF + Logistic Regression**

A scikit-learn baseline converts text into TF-IDF features and uses Logistic Regression to predict the intent.

**2. PyTorch neural classifier**

The neural model uses:

* A vocabulary built from the training data.
* Tokenization and fixed-length sequence encoding.
* Trainable word embeddings.
* Masked average pooling to create a sentence representation.
* A feed-forward neural network with ReLU activation and dropout.
* Cross-entropy loss and the Adam optimizer.

The model is designed to run on a CPU, making local experimentation straightforward.

## Evaluation Results

The recorded experiments produced the following approximate results:

| Model / evaluation                              |          Accuracy |
| ----------------------------------------------- | ----------------: |
| TF-IDF + Logistic Regression, official test set |            85.78% |
| PyTorch model, validation set                   |            85.11% |
| PyTorch model, official test set                | Approximately 86% |

The recorded PyTorch evaluation also showed macro F1 and weighted F1 scores of approximately 0.86 on the official test set.

These results are from the project's current experiments, not a guarantee of production performance. The official test set was used during model comparison, so it should not be treated as a completely untouched final benchmark.

### Notebooks

The notebooks document the exploration and development process:

* `notebooks/01_eda.ipynb` — Exploratory data analysis.
* `notebooks/02_preprocessing.ipynb` — Text preprocessing, TF-IDF baseline, and evaluation.
* `notebooks/03_pytorch.ipynb` — Neural classifier development and evaluation.

## Confidence-Based Review Routing

The API returns a predicted intent, confidence score, and a review status.

| Confidence       | Status                  | Meaning                           |
| ---------------- | ----------------------- | --------------------------------- |
| 85% or higher    | `auto_accepted`         | Eligible for automatic acceptance |
| 60% to below 85% | `needs_review`          | Send for human review             |
| Below 60%        | `manual_classification` | Require manual classification     |

These thresholds are **provisional, not calibrated probabilities or validated business rules**. A high confidence score does not guarantee a correct prediction. Before real-world deployment, thresholds should be evaluated against representative data and the costs of incorrect classifications.

## Project Structure

```text
ticketmindai/
├── api/
│   └── main.py
├── data/
│   ├── raw/
│   │   └── banking77/
│   │       ├── train.csv
│   │       └── test.csv
│   └── processed/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/
│   ├── ticket_classifier.pt
│   ├── word_to_id.json
│   └── id_to_label.json
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_pytorch.ipynb
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── preprocessing/
│   ├── inference.py
│   └── training/
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

* Python 3.11 recommended.
* Git.
* A terminal such as PowerShell on Windows.
* A web browser.

The instructions below assume Windows PowerShell.

### 1. Clone the repository

Replace the placeholder with the actual URL after publishing the repository to GitHub.

```powershell
git clone https://github.com/abdulghaffarcool/TicketMindAI.git
cd TicketMindAI
```

Alternatively, download the repository ZIP and extract it.

### 2. Create a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can instead run the environment's Python executable directly or follow the Python documentation for virtual-environment activation.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project is designed for CPU inference. If PyTorch installation fails or you specifically need a CPU-only build, use the appropriate command from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

### 4. Start the API

From the repository root, with the virtual environment activated:

```powershell
python -m uvicorn api.main:app --reload
```

The API will be available at:

* API base: `http://127.0.0.1:8000`
* Health check: `http://127.0.0.1:8000/health`
* Interactive API documentation: `http://127.0.0.1:8000/docs`

Keep this terminal running while testing the application.

### 5. Start the frontend

Open a **second PowerShell terminal** and run:

```powershell
cd D:\projects\ticketmindai
.\.venv\Scripts\Activate.ps1
python -m http.server 5500 --directory frontend
```

Open `http://127.0.0.1:5500` in your browser.

You can now submit a customer message and inspect the predicted intent, confidence score, and review status.

> If you cloned the repository to a different folder, update the `cd` path accordingly.

## API Usage

### Health check

```http
GET /health
```

Use this endpoint to check the API's health response.

### Predict ticket intent

```http
POST /predict
Content-Type: application/json
```

Example request:

```json
{
  "text": "I cannot access my banking account"
}
```

The response includes the predicted intent, confidence score, and review status. Exact predictions depend on the message and the model's learned patterns.

You can inspect the live request and response schema at `http://127.0.0.1:8000/docs`.

## Limitations and Future Improvements

Current limitations include:

* Banking77 is banking-specific; other support domains require appropriate training and evaluation data.
* Similar intents can be difficult to distinguish from short or ambiguous messages.
* Confidence scores are not calibrated probabilities.
* The model can produce confidently incorrect predictions.
* The current application does not implement a production ticketing-system integration or persistent ticket storage.

Potential improvements:

* Evaluate confidence calibration and select review thresholds using validation data.
* Add representative examples for difficult or underrepresented intents.
* Compare the current model with transformer-based text classifiers.
* Add automated tests for preprocessing, inference, and API endpoints.
* Add structured logging, monitoring, and model-version tracking.
* Evaluate performance on domain-specific and out-of-distribution messages.
* Package and deploy the application using Docker and a CI/CD pipeline.

## Responsible Use

TicketMind AI is a portfolio and learning project. Its predictions should be reviewed before being used for consequential customer-service decisions. The confidence-based routing logic is illustrative and should not replace human oversight or proper production validation.

## Acknowledgements

* **Banking77 dataset:** PolyAI — [dataset repository](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking77).
* **PyTorch:** Neural-network implementation.
* **scikit-learn:** Baseline model and machine-learning utilities.
* **FastAPI:** REST API framework.

Please consult the dataset source for its original license and usage terms before redistributing or using the dataset commercially.

## Author

Developed as an Applied AI portfolio project demonstrating text classification, model evaluation, Python development, API integration, and frontend-to-model deployment.

If you find this project useful, feel free to star the repository or share feedback.
