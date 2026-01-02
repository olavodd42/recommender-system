import torch
import numpy as np
import json
import joblib  # Para carregar scalers do Scikit-Learn (se usado no treino)
from qdrant_client import QdrantClient
from typing import List, Dict, Any, Tuple
from model import TwoTowerModel

class RetrievalService:
    """
    Serviço de Recuperação de Itens usando Two-Tower Model e Qdrant Vector DB.
    Args:
        - model_path (str): Caminho para o checkpoint do modelo treinado.
        - artifacts_path (str): Caminho para os artefatos de pré-processamento (mappings, scalers).
        - qdrant_host (str): Host do servidor Qdrant.
        - collection_name (str): Nome da coleção no Qdrant onde os vetores de itens estão armazenados.
    """
    def __init__(
        self, 
        model_path: str, 
        artifacts_path: str, # Caminho onde salvamos os mappings do treino
        qdrant_host: str = "localhost", 
        collection_name: str = "items"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚡ Inicializando Retrieval Service em {self.device}...")
        
        # 1. Carregar Modelo
        self._load_model(model_path)
        
        # 2. Carregar Artefatos de Pré-processamento
        # Isso garante que User "A" seja sempre ID 42
        self._load_artifacts(artifacts_path)
        
        # 3. Conexão Vector DB
        self.client = QdrantClient(host=qdrant_host, port=6333)
        self.collection = collection_name

    def _load_model(self, path: str):
        """Carrega os pesos da User Tower."""
        # Carrega o modelo usando o método do PyTorch Lightning
        # Ele recupera automaticamente os hiperparâmetros (num_users, num_items) salvos no checkpoint
        self.model = TwoTowerModel.load_from_checkpoint(path)
        self.model.to(self.device)
        self.model.eval()
        print(f"   └── Modelo carregado de {path}")

    def _load_artifacts(self, path: str):
        """
        Carrega dicionários e scalers salvos durante o treino.
        Exemplo: user_to_idx.json, age_scaler.bin
        """
        try:
            with open(f"{path}/user_to_idx.json", 'r') as f:
                self.user_mapping = json.load(f)
            
            # Exemplo de metadados estatísticos para normalização
            # Em prod, isso viria da Feature Store 
            self.stats = {"age_mean": 30.5, "age_std": 12.0}
            
            print(f"   └── Artefatos carregados (Total usuários conhecidos: {len(self.user_mapping)})")
        except FileNotFoundError:
            print("⚠️ AVISO: Artefatos não encontrados. Usando mocks para teste.")
            self.user_mapping = {}
            self.stats = {"age_mean": 0, "age_std": 1}

    def preprocess(self, raw_features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Converte dados crus (JSON) em Tensores prontos para o modelo.
        """
        user_id = str(raw_features.get("user_id"))
        age = raw_features.get("age", 0)

        # 1. Tratamento de ID (Categorical Encoding)
        # Se usuário não existe (Cold Start), usa índice 0 ou token <UNK> 
        user_idx = self.user_mapping.get(user_id, 0) 
        
        # 2. Tratamento Numérico (Normalization)
        # (X - Mean) / Std
        age_norm = (age - self.stats["age_mean"]) / (self.stats["age_std"] + 1e-6)

        # 3. Criação de Tensores (Batch dimension = 1)
        return {
            "user_idx": torch.tensor([user_idx], dtype=torch.long).to(self.device),
            "age": torch.tensor([age_norm], dtype=torch.float32).to(self.device)
        }

    def get_user_embedding(self, raw_features: Dict[str, Any]) -> np.ndarray:
        """
        Gera o embedding do usuário a partir de features crus.
        Pipeline: Raw Data -> Tensor -> Embedding
        """
        # Preprocessamento REAL
        tensors = self.preprocess(raw_features)
        
        with torch.no_grad():
            # 1. Gera o embedding de ID
            u_emb = self.model.user_embedding(tensors["user_idx"])
            
            # 2. Prepara features numéricas
            # O modelo espera 2 features (definido em model.py), mas aqui temos apenas 'age'.
            # Ajustamos a dimensão para (Batch, 1) e fazemos padding para (Batch, 2)
            user_feats = tensors["age"].unsqueeze(1)
            if user_feats.shape[1] < 2:
                padding = torch.zeros((user_feats.shape[0], 2 - user_feats.shape[1]), device=self.device)
                user_feats = torch.cat([user_feats, padding], dim=1)

            # 3. Concatena e passa pelo MLP
            u_input = torch.cat([u_emb, user_feats], dim=1)
            embedding = self.model.user_mlp(u_input)
            
            # 4. Normaliza (L2)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
            # # Mock para o exemplo funcionar sem o modelo treinado
            # embedding = torch.randn(1, 32).to(self.device) 
            
        return embedding.cpu().numpy()[0]

    def retrieve_candidates(self, user_id: str, context: Dict, k: int = 100) -> List[Dict]:
        """
        Pipeline completo: Raw Data -> Tensor -> Embedding -> Vector Search
        Args:
            - user_id (str): ID do usuário para quem buscar recomendações.
            - context (Dict): Dicionário com features adicionais do usuário (e.g., idade).
            - k (int): Número de candidatos a recuperar.
        Returns:
            - List[Dict]: Lista de itens recomendados com seus scores.
        """
        # Feature Collection (Simulando uma chamada à Feature Store [cite: 24])
        # Na prática: feature_store.get_user_features(user_id)
        user_features = {
            "user_id": user_id,
            "age": context.get("age", 25) # Default ou vindo do request
        }
        
        # 1. Gerar Embedding
        query_vector = self.get_user_embedding(user_features)
        
        # 2. Busca ANN (Qdrant)
        # O método .search() foi removido nas versões recentes do qdrant-client (v1.10+).
        # Substituído por .query_points() que é mais genérico.
        search_result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=k,
            # score_threshold=0.5 # Opcional: filtrar itens muito distantes
        ).points
        
        # 3. Formatar Output
        return [
            {"item_id": hit.payload.get("item_id"), "score": hit.score} 
            for hit in search_result
            if hit.payload is not None
        ]