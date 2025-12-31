import pandas as pd
from pathlib import Path

# Caminho onde as features foram salvas
FEATURE_DATA_PATH = Path("./feature_repo/data")

def fix_parquet(file_name):
    path = FEATURE_DATA_PATH / file_name
    print(f"🔧 Reparando tipos em: {file_name}...")
    
    # 1. Carregar com Pandas (que converte LargeString para Object/String padrão em RAM)
    df = pd.read_parquet(path)
    
    # 2. Forçar conversão explícita para string Python padrão nas colunas de texto
    # Isso remove os metadados 'LargeString' do Polars/Arrow
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in str_cols:
        df[col] = df[col].astype(str)
        
    # 3. Salvar novamente sobrescrevendo o arquivo
    # engine='pyarrow' sem especificar versão tende a usar o formato mais compatível
    df.to_parquet(path, engine='pyarrow', index=False)
    print(f"✅ {file_name} corrigido e salvo!")

if __name__ == "__main__":
    # Corrigir User Features
    if (FEATURE_DATA_PATH / "user_features.parquet").exists():
        fix_parquet("user_features.parquet")
    else:
        print("❌ user_features.parquet não encontrado.")

    # Corrigir Item Features
    if (FEATURE_DATA_PATH / "item_features.parquet").exists():
        fix_parquet("item_features.parquet")
    else:
        print("❌ item_features.parquet não encontrado.")