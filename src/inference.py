
import json
from pathlib import Path

import torch

from src.models.ticket_classifier import TicketClassifier
from src.preprocessing.text_preprocessor import preprocess_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


class TicketIntentPredictor:

    def __init__(self):
        with open(
            MODEL_DIR / "word_to_id.json",
            encoding="utf-8"
        ) as f:
            self.word_to_id = json.load(f)

        with open(
            MODEL_DIR / "id_to_label.json",
            encoding="utf-8"
        ) as f:
            self.id_to_label = {
                int(key): value
                for key, value in json.load(f).items()
            }

        checkpoint = torch.load(
            MODEL_DIR / "ticket_classifier.pt",
            map_location="cpu",
            weights_only=True
        )

        self.max_length = checkpoint["max_length"]
        self.pad_id = checkpoint["pad_id"]

        self.model = TicketClassifier(
            vocab_size=checkpoint["vocab_size"],
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_classes=checkpoint["num_classes"],
            pad_id=self.pad_id
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.model.eval()

    def predict(self, text: str) -> dict:
        sequence = preprocess_text(
            text,
            self.word_to_id,
            self.max_length,
            self.pad_id
        )

        input_tensor = torch.tensor(
            [sequence],
            dtype=torch.long
        )

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)

            predicted_id = logits.argmax(dim=1).item()
            confidence = probabilities[0, predicted_id].item()

        return {
            "intent": self.id_to_label[predicted_id],
            "confidence": round(confidence, 4)
        }