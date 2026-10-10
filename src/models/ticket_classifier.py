
import torch
import torch.nn as nn


class TicketClassifier(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        pad_id: int
    ):
        super().__init__()

        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        embeddings = self.embedding(x)

        # Exclude padding tokens from the average
        mask = (x != self.pad_id).unsqueeze(-1)
        masked_embeddings = embeddings * mask

        token_counts = mask.sum(dim=1).clamp(min=1)

        pooled = (
            masked_embeddings.sum(dim=1) / token_counts
        )

        return self.classifier(pooled)