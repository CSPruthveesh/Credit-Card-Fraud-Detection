from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    ValueType,
)
from feast.types import Float32, Int64

# Define the primary key entity for transactions with mandatory value_type specified
transaction_entity = Entity(
    name="transaction_id",
    join_keys=["tx_id"],
    value_type=ValueType.INT64,
    description="Unique transaction identifier"
)

# Define the source file pointing to local Parquet data source
offline_source = FileSource(
    name="transactions_source",
    path="d:/COLLEGE PREP/Self Projects/data/processed_transactions.parquet",
    timestamp_field="event_timestamp",  # Updated for Feast 0.31+ compatibility
    created_timestamp_column="created_timestamp",
)

# Define the Feature View containing our raw and engineered features
transaction_feature_view = FeatureView(
    name="transaction_features",
    entities=[transaction_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="Time_Delta", dtype=Float32),
        Field(name="Last_5_Tx_Time_Span", dtype=Float32),
        Field(name="Rolling_Mean_Amount_5", dtype=Float32),
        Field(name="Amount_scaled", dtype=Float32),
        Field(name="Time_scaled", dtype=Float32),
    ],
    online=True,
    source=offline_source,
    tags={"team": "fraud_analytics", "tier": "tier-1"},
)
