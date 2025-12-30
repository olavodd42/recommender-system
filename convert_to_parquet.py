import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

DATA_PATH = Path("./data")
CSV_FILE = DATA_PATH / "transactions_train.csv"
PARQUET_FILE = DATA_PATH / "transactions_train.parquet"

# Definição de Tipos (Crucial para reduzir overhead)
dtype_dict = {
    'article_id': 'str', 
    'customer_id': 'str',
    'price': 'float32',
    'sales_channel_id': 'int8'
}

print(f"🚀 Iniciando conversão STREAMING: {CSV_FILE} -> {PARQUET_FILE}")

chunksize = 500_000  # Reduzi um pouco o chunk para ser ainda mais seguro
parquet_writer = None

# Lendo e escrevendo simultaneamente
with pd.read_csv(CSV_FILE, chunksize=chunksize, dtype=dtype_dict) as reader:
    for i, chunk in enumerate(tqdm(reader, desc="Escrevendo Chunks")):
        
        # 1. Processamento Leve
        chunk['t_dat'] = pd.to_datetime(chunk['t_dat'])
        
        # 2. Conversão para Tabela Arrow (Formato binário em memória)
        table = pa.Table.from_pandas(chunk)
        
        # 3. Inicializa o Writer no primeiro chunk (usando o schema do primeiro pedaço)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(PARQUET_FILE, table.schema, compression='snappy')
        
        # 4. Escreve o chunk no disco e libera a memória imediatamente
        parquet_writer.write_table(table)

# Fechar o arquivo para gravar o rodapé do Parquet
if parquet_writer:
    parquet_writer.close()

print(f"✅ Sucesso! Arquivo gerado em {PARQUET_FILE}")