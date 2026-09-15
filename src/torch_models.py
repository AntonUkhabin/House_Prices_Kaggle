import torch
import torch.nn as nn


def calculate_embedding_dim(cardinality: int) -> int:
    '''Calculate a compact embedding dimension from feature cardinality.'''

    return min(8, max(2, round(cardinality ** 0.5)))


class TabularDNN(nn.Module):
    '''Predict house prices from numerical features and categorical embeddings.'''

    def __init__(self, numerical_feature_count: int, categorical_cardinalities: list[int]):
        super().__init__()

        if numerical_feature_count < 1:
            raise ValueError('numerical_feature_count must be positive.')

        if any(cardinality < 1 for cardinality in categorical_cardinalities):
            raise ValueError('All categorical cardinalities must be positive.')

        self.numerical_feature_count = numerical_feature_count
        self.categorical_cardinalities = list(categorical_cardinalities)
        self.embedding_dims = [calculate_embedding_dim(cardinality) for cardinality in categorical_cardinalities]

        # padding_idx=0 оставляет embedding для missing/unseen category нулевым и не обновляет его.
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim, padding_idx=0)
            for cardinality, embedding_dim in zip(self.categorical_cardinalities, self.embedding_dims)
        ])

        input_size = numerical_feature_count + sum(self.embedding_dims)

        self.mlp = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
        )

    def forward(self, numerical_features, categorical_features):
        '''Return standardized log-price predictions with shape batch_size.'''

        embedded_features = [
            embedding(categorical_features[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]

        # Все embeddings объединяются с числовыми признаками в один dense-вектор.
        combined_features = torch.cat([numerical_features, *embedded_features], dim=1)
        return self.mlp(combined_features).squeeze(1)


def build_torch_model(model_name: str, numerical_feature_count: int, categorical_cardinalities: list[int]) -> nn.Module:
    '''Build a PyTorch model by name.'''

    if model_name == 'dnn':
        return TabularDNN(numerical_feature_count, categorical_cardinalities)

    raise ValueError(f'Unknown PyTorch model: {model_name}')