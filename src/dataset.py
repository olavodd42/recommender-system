import torch
from torch.utils.data import Dataset
import pandas as pd

# --- 1. Dataset Class ---
class RecSysDataset(Dataset):
    """
    Dataset para Recomendação usando IDs e Features Numéricas.
    Espera um arquivo Parquet com as colunas:
        - user_index: int -> índice numérico do usuário
        - item_index: int -> índice numérico do item
        - avg_spend: float -> gasto médio do usuário
        - purchase_count: float -> número de compras do usuário
        - popularity_score: float -> score de popularidade do item
        - avg_price: float -> preço médio do item

    Args:
        - parquet_file (str): Caminho para o arquivo Parquet com os dados.
    """

    def __init__(self, parquet_file):
        self.data = pd.read_parquet(parquet_file)
        
        # Features Numéricas normalizadas (Gambi de engenheiro: dividir pelo max)
        self.users = torch.LongTensor(self.data['user_index'].values)
        self.items = torch.LongTensor(self.data['item_index'].values)
        
        # Features extras (Contexto)
        self.user_features = torch.FloatTensor(self.data[['avg_spend', 'purchase_count']].values)
        self.item_features = torch.FloatTensor(self.data[['popularity_score', 'avg_price']].values)
        
    def __len__(self):
        """Retorna o tamanho do dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retorna um item do dataset no formato:
            {
                'user_id': int,
                'item_id': int,
                'user_feats': Tensor[float],
                'item_feats': Tensor[float]
            }
        """

        return {
            'user_id': self.users[idx],
            'item_id': self.items[idx],
            'user_feats': self.user_features[idx],
            'item_feats': self.item_features[idx]
        }
