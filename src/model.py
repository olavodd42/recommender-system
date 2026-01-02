import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# --- 2. A Arquitetura Two-Tower ---
class TwoTowerModel(pl.LightningModule):
    """
    Modelo Two-Tower para Recomendação com IDs e Features Numéricas.
    Cada torre (usuário e item) possui:
        - Embedding para ID
        - MLP que combina Embedding + Features Numéricas
    Args:
        - num_users (int): Número total de usuários.
        - num_items (int): Número total de itens.
        - embedding_dim (int): Dimensão dos embeddings.
    """
    
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()
        self.save_hyperparameters()
        
        # --- Torre do Usuário ---
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # Rede Neural que combina ID + Features Numéricas
        self.user_mlp = nn.Sequential(
            nn.Linear(embedding_dim + 2, 64), # +2 pois temos 2 features numéricas de user
            nn.ReLU(),
            nn.Linear(64, 32) # Saída final: vetor de tamanho 32
        )
        
        # --- Torre do Item ---
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        # Rede Neural que combina ID + Features Numéricas
        self.item_mlp = nn.Sequential(
            nn.Linear(embedding_dim + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
    
    def forward(self, batch):
        """
        Embeddings -> Concatenate -> MLP -> Normalize
        1. Gerar Embeddings de ID
        2. Concatenar com features numéricas
        3. Passar pelos MLPs
        4. Normalizar vetores (para usar Cosine Similarity)
        """
        u_emb = self.user_embedding(batch['user_id'])
        i_emb = self.item_embedding(batch['item_id'])
        
        # 2. Concatenar com features numéricas
        u_input = torch.cat([u_emb, batch['user_feats']], dim=1)
        i_input = torch.cat([i_emb, batch['item_feats']], dim=1)
        
        # 3. Passar pelos MLPs
        user_vector = self.user_mlp(u_input)
        item_vector = self.item_mlp(i_input)
        
        # 4. Normalizar vetores (para usar Cosine Similarity)
        user_vector = F.normalize(user_vector, p=2, dim=1)
        item_vector = F.normalize(item_vector, p=2, dim=1)
        
        return user_vector, item_vector

    def training_step(self, batch, batch_idx):
        """
        Treinamento com In-Batch Negatives Loss.
        
        """
        user_vector, item_vector = self(batch)
        
        # --- In-Batch Negatives Loss (O Segredo do Retrieval) ---
        # Em vez de criar negativos manualmente, usamos os outros itens do batch como negativos.
        # Se o batch tem tamanho 128, para cada usuário temos 1 positivo e 127 negativos.
        
        # Matriz de similaridade (Batch x Batch)
        # U x I
        scores = torch.matmul(user_vector, item_vector.T)
        
        # O objetivo é que a diagonal principal (user i com item i) tenha score alto
        labels = torch.arange(scores.size(0), device=self.device)
        
        loss = F.cross_entropy(scores * 10, labels) # *10 é a "temperatura"
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)