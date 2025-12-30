from datetime import timedelta
from feast import Entity, Field, FeatureView, FileSource
from feast.types import Float32, Int64, String

# 1. Definir as Entidades (As chaves primárias do nosso mundo)
customer = Entity(name="customer", join_keys=["customer_id"])
article = Entity(name="article", join_keys=["article_id"])

# 2. Fonte de Dados (Aponta para os Parquets gerados no ETL)
user_stats_source = FileSource(
    name="user_stats_source",
    path="/feature_repo/data/user_features.parquet", # Path absoluto dentro do container ou relativo
    timestamp_field="event_timestamp",
)

item_stats_source = FileSource(
    name="item_stats_source",
    path="/feature_repo/data/item_features.parquet",
    timestamp_field="event_timestamp",
)

# 3. Feature Views (Grupos lógicos de features)
user_stats_fv = FeatureView(
    name="user_stats",
    entities=[customer],
    ttl=timedelta(days=3650), # As features são válidas por quanto tempo?
    schema=[
        Field(name="avg_spend", dtype=Float32),
        Field(name="purchase_count", dtype=Int64),
        Field(name="total_spend", dtype=Float32),
    ],
    source=user_stats_source,
)

item_stats_fv = FeatureView(
    name="item_stats",
    entities=[article],
    ttl=timedelta(days=30), # Popularidade muda rápido, TTL menor
    schema=[
        Field(name="popularity_score", dtype=Int64),
        Field(name="avg_price", dtype=Float32),
    ],
    source=item_stats_source,
)