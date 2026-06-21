# Amazon Shopping Recommendation Engine

This document adapts the generic recommendation-system design from `RECOMMENDATION.md` into an Amazon-style shopping recommendation engine using standard AI and ML infrastructure.

## 1. Business Goal

The system should recommend relevant products to shoppers across Amazon-like surfaces:

- Homepage recommendations
- Product detail page recommendations
- Cart cross-sell and upsell
- Search and category personalization
- Email, push, and re-engagement campaigns

The primary objective is:

```text
Maximize long-term purchase value and customer satisfaction
```

This means the system should optimize not only clicks, but also add-to-cart rate, purchase conversion, revenue, repeat purchases, low return rate, and customer trust.

## 2. Functional Requirements

- Recommend personalized products for each user.
- Support item-to-item recommendations such as "similar products".
- Support basket recommendations such as "frequently bought together".
- Adapt to real-time session behavior.
- Handle new users and new products.
- Support business rules such as inventory, delivery availability, pricing, promotions, and marketplace policy.
- Support A/B testing and experimentation.

## 3. Non-Functional Requirements

- Low latency recommendation serving, ideally under 100 ms for most requests.
- High availability across shopping surfaces.
- Horizontal scalability for millions of users and products.
- Fresh recommendations based on recent user behavior and catalog changes.
- Reliable monitoring for model quality, business impact, and infrastructure health.
- Training-serving consistency through shared feature definitions.

## 4. High-Level Architecture

```text
User Events / Orders / Catalog / Inventory
                ↓
       Data and Feature Pipelines
                ↓
           Feature Store
                ↓
      Offline Training Pipelines
                ↓
          Model Registry
                ↓
Online Recommendation Service
                ↓
Candidate Generation
                ↓
Ranking
                ↓
Re-Ranking
                ↓
Final Product Recommendations
```

The online recommendation path follows a multi-stage recommendation architecture:

```text
Candidate Generation → Ranking → Re-Ranking → Final Recommendations
```

## 4.1 Task and Technology Map

This section separates the tools used in the local learning project from the tools normally used in a production Amazon-scale recommendation platform.

### Local MVP Tools Used in This Repository


| Task                                 | Technology Used                          | Why It Is Used                                                                 |
| ------------------------------------ | ---------------------------------------- | ------------------------------------------------------------------------------ |
| Download small Amazon dataset        | `datasets`                               | Streams a small Amazon Reviews sample from Hugging Face.                       |
| Store local sample data              | `CSV`, `data/amazon_recommendation/`     | Keeps `events.csv` and `products.csv` simple and easy to inspect.              |
| Basic MVP recommendations            | Python standard library                  | Builds item-to-item collaborative filtering without heavy dependencies.        |
| User-product interaction modeling    | Item-to-item collaborative filtering     | Finds products that co-occur in user histories.                                |
| Two-tower retrieval model            | `PyTorch`                                | Builds separate user and product neural-network towers.                        |
| User embeddings                      | `torch.nn.Embedding`                     | Converts user IDs into dense vectors.                                          |
| Product embeddings                   | `torch.nn.Embedding`                     | Converts product IDs into dense vectors.                                       |
| No-training architecture demo        | Deterministic random PyTorch weights     | Shows the two-tower and FAISS flow without training.                           |
| No-training useful retrieval         | `scikit-learn` TF-IDF                    | Builds lightweight product/user profile text embeddings without PyTorch.       |
| Optional heavy text retrieval        | `sentence-transformers/all-MiniLM-L6-v2` | Uses a pretrained transformer checkpoint when the machine has enough disk.     |
| Product vector retrieval             | `FAISS` / `faiss-cpu`                    | Searches nearest product embeddings quickly.                                   |
| Recommendation API                   | `FastAPI`                                | Exposes `/recommend` and `/health` endpoints.                                  |
| Local API server                     | `Uvicorn`                                | Runs the FastAPI app locally.                                                  |
| Request/response validation          | `Pydantic`                               | Defines the API request schema.                                                |
| Numeric arrays for FAISS             | `NumPy`                                  | Converts text embeddings into FAISS-compatible arrays.                         |
| Production-style validation hook     | `Great Expectations`                     | Provides a place for schema and data quality checks before building the index. |
| Production-style experiment metadata | `MLflow`                                 | Logs index build metadata and artifact information.                            |
| Optional model training              | `PyTorch`, pairwise ranking loss         | Learns user/product embeddings from positive and negative examples.            |
| Model checkpointing                  | `torch.save`                             | Saves two-tower weights and ID mappings locally.                               |
| Local generated artifacts            | `.gitignore`, `data/`, `models/`         | Prevents datasets and model checkpoints from being committed.                  |


### Production Tools by Task


| Task                     | Standard Tools                                                                 | Purpose                                                                                                                |
| ------------------------ | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Event collection         | `Kafka`, `Kinesis`, `Pulsar`                                                   | Collects clicks, views, add-to-cart events, purchases, search queries, and impressions in near real time.              |
| Data lake storage        | `S3`, `GCS`, `ADLS`                                                            | Stores raw logs, catalog snapshots, training datasets, model artifacts, and replayable historical data.                |
| Data warehouse           | `Snowflake`, `BigQuery`, `Redshift`                                            | Supports analytics, BI dashboards, metric computation, and SQL-based feature exploration.                              |
| Batch processing         | `Spark`, `Databricks`                                                          | Builds large-scale training datasets, aggregates user history, computes item popularity, and creates offline features. |
| Stream processing        | `Flink`, `Spark Structured Streaming`, `Kafka Streams`                         | Computes real-time session features such as recent clicks, active cart, and current category intent.                   |
| Workflow orchestration   | `Airflow`, `Dagster`, `Prefect`                                                | Schedules dataset creation, feature generation, model training, validation, and deployment jobs.                       |
| Feature store            | `Feast`, `Tecton`, `Vertex AI Feature Store`                                   | Reuses the same feature definitions for training and serving to avoid training-serving skew.                           |
| Offline feature storage  | `S3`, `GCS`, `ADLS`, `Snowflake`, `BigQuery`, `Redshift`                       | Stores historical features used for model training and backfills.                                                      |
| Online feature storage   | `Redis`, `DynamoDB`, `Cassandra`, `Feast Online Store`, `Tecton Online Store`  | Serves low-latency user/session/product features during recommendation requests.                                       |
| Retrieval model training | `PyTorch`, `TensorFlow`, `JAX`                                                 | Trains two-tower, DSSM, or other embedding-based candidate generation models.                                          |
| Ranking model training   | `LightGBM`, `XGBoost`, `DeepFM`, `Wide & Deep`, `DCN`, `PyTorch`, `TensorFlow` | Scores retrieved candidates using richer user, item, context, and business features.                                   |
| Vector search            | `FAISS`, `ScaNN`, `Milvus`, `Pinecone`, `Weaviate`, `OpenSearch k-NN`          | Retrieves nearest product embeddings for a user or session embedding.                                                  |
| Experiment tracking      | `MLflow`, `Weights & Biases`                                                   | Tracks model versions, parameters, metrics, artifacts, and training runs.                                              |
| Model registry           | `MLflow Model Registry`, `SageMaker Model Registry`                            | Stores approved model versions and supports promotion from staging to production.                                      |
| Model serving            | `Triton`, `TorchServe`, `TensorFlow Serving`, `KServe`, `SageMaker Endpoint`   | Serves retrieval and ranking models behind scalable APIs.                                                              |
| Recommendation API       | `FastAPI`, `gRPC`, `Java/Spring`, `Go`                                         | Exposes recommendation endpoints to web, mobile, search, cart, and email systems.                                      |
| API hosting              | `Kubernetes`, `EKS`, `GKE`, `AKS`, `ECS`                                       | Runs serving services with autoscaling, rolling deploys, and high availability.                                        |
| Cache                    | `Redis`, `Memcached`                                                           | Caches popular recommendations, user embeddings, product embeddings, and fallback lists.                               |
| Data validation          | `Great Expectations`, `TensorFlow Data Validation`                             | Checks schema, nulls, ranges, data drift, and feature quality before training or serving.                              |
| Model monitoring         | `Prometheus`, `Grafana`, `Datadog`, `CloudWatch`                               | Tracks latency, QPS, error rate, model drift, feature freshness, and business metrics.                                 |
| A/B testing              | `Statsig`, `Optimizely`, internal experimentation platform                     | Measures online impact of recommendation changes against control traffic.                                              |


### Where These Fit in the Architecture

```text
Kafka / Kinesis
      ↓
S3 / GCS / ADLS Data Lake
      ↓
Spark / Databricks / Flink
      ↓
Feast / Tecton / Vertex AI Feature Store
      ↓
PyTorch Two-Tower Retrieval + LightGBM Ranking
      ↓
MLflow / W&B Tracking + MLflow or SageMaker Model Registry
      ↓
FAISS / Milvus / Pinecone Vector Retrieval
      ↓
FastAPI / gRPC Serving on Kubernetes
      ↓
Redis Cache + Re-Ranking Rules
      ↓
Customer Recommendations
```

### Pretrained Checkpoint Note

A generic downloaded two-tower checkpoint usually cannot be used directly with this local dataset. Two-tower models contain embedding tables tied to the exact `user_id` and `product_id` mapping used during training. If a checkpoint was trained on different users and products, its embedding rows do not match our local `events.csv`.

For learning the ID-based two-tower architecture without training, use:

```bash
python -m src.amazon_two_tower_recommendation --demo-no-train --top-k 10
```

For a more useful no-training production-style setup in this constrained environment, use lightweight text embeddings plus FAISS:

```text
Default encoder: scikit-learn TF-IDF
Use case: product text embeddings and user/session profile embeddings
Training required locally: no
Retrieval engine: FAISS
Serving layer: FastAPI
```

The optional heavy encoder is:

```text
Checkpoint: sentence-transformers/all-MiniLM-L6-v2
Install file: requirements-production-recommendation-heavy.txt
Use only when the machine has enough disk for PyTorch/SentenceTransformers.
```

This is implemented in:

```text
src/amazon_pretrained_production_retrieval.py
```

Install production dependencies:

```bash
python -m pip install -r requirements-production-recommendation.txt
```

Download the Amazon sample data:

```bash
python -m src.amazon_recommendation_mvp --download --limit 5000
```

Build the production-style retrieval artifacts without training:

```bash
python -m src.amazon_pretrained_production_retrieval --build-index --log-mlflow
```

What this command does:

- Validates the local `events.csv` and `products.csv` schema.
- Builds product text features.
- Encodes products into lightweight TF-IDF vectors by default.
- Builds a FAISS product index.
- Saves artifacts under `models/amazon_pretrained_retrieval/`.
- Optionally logs build metadata to `MLflow`.

If you have enough disk and want the heavier SentenceTransformers path:

```bash
python -m pip install -r requirements-production-recommendation-heavy.txt
python -m src.amazon_pretrained_production_retrieval \
  --build-index \
  --encoder sentence-transformers/all-MiniLM-L6-v2 \
  --log-mlflow
```

Run one local recommendation by user ID:

```bash
python -m src.amazon_pretrained_production_retrieval --recommend --user-id <user_id> --top-k 10
```

Run one local recommendation by shopping intent:

```bash
python -m src.amazon_pretrained_production_retrieval --recommend --query-text "hair care products for dry hair" --top-k 10
```

Start the API:

```bash
uvicorn src.amazon_pretrained_production_retrieval:create_app --factory --reload
```

Call the API:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query_text": "skin care moisturizer", "top_k": 10}'
```

For meaningful recommendations on this dataset, train a small local checkpoint:

```bash
python -m src.amazon_two_tower_recommendation --train --epochs 5 --top-k 10
```

## 5. Data Sources

## 4.2 Implemented Production-Style Stack in This Repo

The local implementation now wires the production concepts into concrete files:


| Production Task     | Technology                                                          | Function in This System                                                                                                       | Repo Location                                                                                        |
| ------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Data lake           | Local CSV now, maps to `S3` / `GCS` / `ADLS`                        | Stores raw events, product catalog data, and replayable source files for feature generation and index builds.                 | `data/amazon_recommendation/`, `configs/production_recommendation_stack.yaml`                        |
| Warehouse           | Local files now, maps to `Snowflake` / `BigQuery` / `Redshift`      | Represents the analytics layer where teams would query events, products, user behavior, and business metrics.                 | `configs/production_recommendation_stack.yaml`                                                       |
| Processing          | Python local now, maps to `Spark` / `Databricks` / `Flink`          | Transforms raw events into product/user features such as interaction counts, average ratings, and recent user history.        | `src/amazon_feature_store.py`, `configs/production_recommendation_stack.yaml`                        |
| Orchestration       | `Airflow`, `Dagster`, `Prefect` templates                           | Schedules the pipeline steps: data download, feature materialization, FAISS index build, and MLflow logging.                  | `orchestration/`                                                                                     |
| Feature store       | Local feature store artifacts plus Feast definitions                | Stores reusable product/user features so retrieval and serving use the same feature values.                                   | `src/amazon_feature_store.py`, `feature_store/recommendation_feature_repo/`                          |
| Retrieval           | `FAISS`                                                             | Stores product vectors and retrieves nearest products for a user profile or query embedding.                                  | `src/amazon_pretrained_production_retrieval.py`, `models/amazon_pretrained_retrieval/products.faiss` |
| Serving             | `FastAPI`, `Uvicorn`                                                | Exposes recommendation APIs through `/health` and `/recommend` endpoints.                                                     | `src/amazon_pretrained_production_retrieval.py`                                                      |
| Experiment tracking | `MLflow`                                                            | Logs retrieval index build metadata such as encoder type, product count, and embedding dimension.                             | `log_build_to_mlflow()` in `src/amazon_pretrained_production_retrieval.py`                           |
| Model registry      | `MLflow Model Registry` / `SageMaker Model Registry` mapping        | Defines where approved retrieval/ranking model versions would be promoted and loaded from in production.                      | `configs/production_recommendation_stack.yaml`                                                       |
| Data validation     | Built-in checks, `Great Expectations` suite, `TFDV` schema template | Validates required columns, non-empty data, event schema, and expected value constraints before building features or indexes. | `validation/`                                                                                        |


Run the implemented local production path in the `rag` conda environment:

```bash
conda run -n rag python -m src.amazon_recommendation_mvp --download --limit 5000
conda run -n rag python -m src.amazon_feature_store --materialize
conda run -n rag python -m src.amazon_pretrained_production_retrieval --build-index --log-mlflow
conda run -n rag python -m src.amazon_pretrained_production_retrieval --recommend --user-id <user_id> --top-k 10
```

Start the API:

```bash
conda run -n rag uvicorn src.amazon_pretrained_production_retrieval:create_app --factory --reload
```

Important note: cloud systems such as `S3`, `GCS`, `ADLS`, `Snowflake`, `BigQuery`, `Redshift`, `Tecton`, `Vertex AI Feature Store`, and `SageMaker Model Registry` require external accounts and credentials. This repo includes local runnable equivalents plus config/templates showing where those services plug in.

### User Behavior Data

- Product views
- Clicks
- Add-to-cart events
- Purchases
- Wishlist actions
- Ratings and reviews
- Returns and refunds
- Search queries
- Category browsing behavior

### Product Data

- Title
- Description
- Category
- Brand
- Price
- Ratings
- Images
- Product attributes
- Inventory status
- Delivery promise
- Promotion status

### Context Data

- Device
- Location
- Time of day
- Day of week
- Session activity
- Current search query
- Referral source

### Business Data

- Margin
- Promotion eligibility
- Seller quality
- Inventory availability
- Delivery speed
- Sponsored placement metadata

## 6. Candidate Generation

Candidate generation reduces the product universe from millions of products to a smaller set of likely relevant candidates.

Example:

```text
100 million products
        ↓
1,000 candidate products
```

Useful candidate sources:

- Collaborative filtering: products liked or bought by similar users.
- Matrix factorization: user and item embeddings trained from historical interactions.
- Two-tower retrieval model: one tower embeds users or sessions, and another embeds products.
- Item-to-item similarity: products similar by metadata, image, text, or user behavior.
- Frequently bought together: co-occurrence from shopping baskets.
- Trending products: popular products by category, location, and time.
- Recently viewed continuation: products related to the current session.
- Cold-start candidates: content-based products for new users or new items.

Recommended tools:

- Training: `PyTorch`, `TensorFlow`, `JAX`
- Embedding search: `FAISS`, `ScaNN`, `Milvus`, `Pinecone`, `Weaviate`, `OpenSearch k-NN`
- Streaming: `Kafka`, `Kinesis`, `Pulsar`
- Batch processing: `Spark`, `Databricks`, `Flink`

## 7. Ranking

The ranking model scores each candidate using richer user, product, interaction, and context features.

Example:

```text
1,000 candidates
        ↓
Top 50 ranked products
```

Useful prediction targets:

- Probability of click
- Probability of add-to-cart
- Probability of purchase
- Expected revenue
- Expected long-term value
- Return or refund risk

Common ranking models:

- `LightGBM`
- `XGBoost`
- `DeepFM`
- `Wide & Deep`
- `Deep & Cross Network`
- Transformer-based session models

Useful ranking features:

- User-category affinity
- User-brand affinity
- User price sensitivity
- Product popularity
- Product rating
- Product freshness
- Product margin
- Product availability
- Delivery speed
- Current session intent
- Query-product relevance

## 8. Re-Ranking

The re-ranking layer turns model scores into a final shopping experience.

Goals:

- Improve diversity across categories, brands, and price ranges.
- Remove unavailable or low-quality products.
- Avoid near-duplicate recommendations.
- Respect user preferences and marketplace policies.
- Balance exploitation with exploration.
- Promote fresh products in controlled ways.
- Apply business constraints such as inventory, delivery eligibility, and promotions.
- Keep sponsored recommendations transparent and policy-compliant.

Example:

```text
Top 50 ranked products
        ↓
Top 20 final recommendations
```

## 9. Online Serving Flow

```text
User opens shopping page
        ↓
Fetch online and offline features
        ↓
Generate user or session embedding
        ↓
Retrieve candidates using ANN search and business candidate sources
        ↓
Merge and deduplicate candidates
        ↓
Score candidates with ranking model
        ↓
Apply re-ranking and business rules
        ↓
Return final recommendations
```

Recommended serving tools:

- API layer: `FastAPI`, `gRPC`, `Java/Spring`, `Go`
- Model serving: `Triton`, `TorchServe`, `TensorFlow Serving`, `KServe`, `SageMaker`
- Cache: `Redis`, `Memcached`
- Infrastructure: `Kubernetes`, `EKS`, `GKE`, `AKS`
- Load balancing: `Envoy`, `NGINX`, cloud load balancers

## 10. Offline Training Pipeline

```text
Raw event logs
      ↓
Sessionization and label generation
      ↓
Feature engineering
      ↓
Training dataset creation
      ↓
Train retrieval model
      ↓
Train ranking model
      ↓
Offline evaluation
      ↓
Model registry
      ↓
Shadow, canary, or A/B deployment
```

Recommended tools:

- Data lake: `S3`, `GCS`, `ADLS`
- Warehouse: `Snowflake`, `BigQuery`, `Redshift`
- Processing: `Spark`, `Databricks`, `Flink`
- Orchestration: `Airflow`, `Dagster`, `Prefect`
- Feature store: `Feast`, `Tecton`, `Vertex AI Feature Store`
- Experiment tracking: `MLflow`, `Weights & Biases`
- Model registry: `MLflow Model Registry`, `SageMaker Model Registry`
- Data validation: `Great Expectations`, `TensorFlow Data Validation`

## 11. Feature Store

A feature store helps prevent training-serving skew.

Example problem:

```text
Training calculates 30-day user-category affinity one way.
Production calculates it another way.
```

This mismatch causes model quality problems. A feature store provides shared feature definitions for both training and online serving.

Feature types:

- Offline features: historical purchase counts, 30-day category affinity, monthly spend, return rate.
- Online features: current session clicks, recent searches, active cart, last viewed products.

## 12. Cold Start

### New User

When a user has little or no history:

- Recommend popular products by location and category.
- Use onboarding preferences.
- Use current session clicks and searches.
- Use contextual signals such as device, time, and location.
- Use trending products and editorial collections.

### New Product

When a product has no interaction history:

- Use title, description, category, brand, and attributes.
- Generate text and image embeddings.
- Place it near similar products in the vector index.
- Use controlled exploration to gather early feedback.
- Use category and seller priors until enough interactions arrive.

## 13. Monitoring and Metrics

### Offline ML Metrics

- `Recall@K`
- `Precision@K`
- `NDCG@K`
- `MAP`
- Calibration of purchase probability

### Online Business Metrics

- Click-through rate
- Add-to-cart rate
- Conversion rate
- Revenue per session
- Average order value
- Repeat purchase rate
- Return or refund rate
- Long-term customer value

### Infrastructure Metrics

- p50, p95, and p99 latency
- QPS
- Error rate
- Cache hit rate
- Model serving latency
- ANN retrieval latency
- ANN recall quality

## 14. Scalability

Important scaling strategies:

- Horizontally scale retrieval and ranking services.
- Cache user embeddings, item embeddings, and popular recommendation results.
- Use batch updates for expensive long-term features.
- Use streaming updates for session-level features.
- Partition ANN indexes by marketplace, language, category, or geography.
- Use fallback recommendations when a model or feature service is degraded.

## 15. Trade-Offs

### Retrieval Quality vs Latency

Retrieving more candidates improves recall, but increases latency and ranking cost.

### Large Model vs Small Model

Large neural models may improve accuracy, but cost more to train and serve. Tree-based rankers are often a strong starting point.

### Real-Time Features vs Batch Features

Real-time features improve freshness, but require more complex infrastructure.

### Revenue vs Customer Trust

Recommendations should not over-optimize short-term revenue at the cost of poor quality, excessive sponsored placements, or products likely to be returned.

## 16. Recommended MVP

A practical first version:

```text
Batch features
      +
Item-to-item similarity
      +
Frequently bought together
      +
LightGBM ranking model
      +
Redis cache
      +
FAISS vector index
```

Suggested MVP architecture:

```text
Kafka or Kinesis
      ↓
S3 data lake
      ↓
Spark and Airflow
      ↓
Feast feature store
      ↓
PyTorch two-tower retrieval model
      ↓
FAISS vector index
      ↓
LightGBM ranking model
      ↓
FastAPI recommendation service
      ↓
Redis cache and re-ranking rules
```

This MVP is standard, scalable, and practical. It can later evolve into a real-time system with session embeddings, online feature updates, deeper ranking models, and continuous A/B experimentation.

## 17. Small Dataset to Build and Run

For a small local prototype, start with a public retail recommendation dataset instead of trying to reproduce Amazon-scale data.

### Best Small Dataset Options

1. Amazon Reviews Dataset, small category subset
  Use one small category such as `Beauty`, `Video Games`, `Musical Instruments`, or `Office Products`.
   Useful fields:
  - `reviewerID` as `user_id`
  - `asin` as `product_id`
  - `overall` as rating
  - `unixReviewTime` as timestamp
  - product metadata such as title, category, brand, and price when available
   This is the closest dataset for an Amazon-style recommender.
2. RetailRocket Recommender System Dataset
  Good for event-based recommendation because it includes views, add-to-cart events, and transactions.
   Useful fields:
  - `visitorid` as `user_id`
  - `itemid` as `product_id`
  - `event` as interaction type
  - `timestamp`
   This is better if the goal is to model shopping sessions.
3. Instacart Market Basket Dataset
  Good for basket recommendations such as "frequently bought together".
   Useful fields:
  - `user_id`
  - `order_id`
  - `product_id`
  - `department`
  - `aisle`
   This is useful for cart cross-sell and bundle recommendations.

### Recommended First Dataset

For the first version, use:

```text
Amazon Reviews 2023 from Hugging Face
Dataset repository: McAuley-Lab/Amazon-Reviews-2023
Small file: raw/review_categories/All_Beauty.jsonl
Rows: 10,000 to 100,000 interactions
Products: 1,000 to 10,000
Users: 1,000 to 20,000
```

This is small enough to run locally and still realistic enough for collaborative filtering, item similarity, and ranking experiments.

Install the recommendation dependency:

```bash
python -m pip install -r requirements-recommendation.txt
```

Download a small streamed sample and run the first recommender:

```bash
python -m src.amazon_recommendation_mvp --download --limit 5000
```

The script prints a few sample `user_id` values from the downloaded data. Use one of those IDs for a personalized run.

The script writes:

```text
data/amazon_recommendation/events.csv
data/amazon_recommendation/products.csv
```

Then run recommendations for any downloaded user ID:

```bash
python -m src.amazon_recommendation_mvp --user-id <user_id> --top-k 10
```

If you only want to test the code path without downloading data:

```bash
python -m src.amazon_recommendation_mvp --sample --user-id u1
```

The first MVP uses item-to-item collaborative filtering. It finds products that often appear in the same users' histories, then recommends similar products the target user has not already interacted with.

### Minimal Local Schema

Create three simple files.

`users.csv`

```text
user_id,location,age_bucket
u1,IN,25-34
u2,US,35-44
u3,UK,18-24
```

`products.csv`

```text
product_id,title,category,brand,price,rating
p1,Wireless Mouse,Electronics,Logi,19.99,4.5
p2,USB Keyboard,Electronics,KeyPro,29.99,4.3
p3,Running Shoes,Sports,RunFast,59.99,4.6
```

`events.csv`

```text
user_id,product_id,event_type,timestamp
u1,p1,view,2026-01-01T10:00:00Z
u1,p1,add_to_cart,2026-01-01T10:02:00Z
u1,p1,purchase,2026-01-01T10:05:00Z
u2,p2,view,2026-01-02T09:00:00Z
u3,p3,purchase,2026-01-03T11:00:00Z
```

### MVP Modeling Plan

Start with simple models before deep learning:

1. Popularity baseline: recommend popular products by category.
2. Item-to-item similarity: recommend products often viewed or bought by the same users.
3. Matrix factorization: learn user and product embeddings from purchases or ratings.
4. LightGBM ranker: rank candidates using product, user, and interaction features.
5. FAISS retrieval: store product embeddings and retrieve similar products quickly.

### Simple Labels

For ranking, convert events into labels:

```text
view = 1
add_to_cart = 2
purchase = 3
return/refund = negative signal
```

For a first binary classifier:

```text
purchase = 1
view without purchase = 0
```

### Local Prototype Stack

Use this lightweight stack:

```text
Python
Pandas
Scikit-learn
LightGBM
Implicit or Surprise
FAISS
FastAPI
SQLite or DuckDB
```

This is enough to build a working local recommendation service without Kafka, Spark, Kubernetes, or a full feature store.

# Two-Tower MVP With Suggested Technologies

The file `src/amazon_two_tower_recommendation.py` implements a separate two-tower retrieval model using the exact standard tools suggested for this stage:

- `PyTorch` for the user tower and product tower neural networks.
- `FAISS` for approximate nearest-neighbor style product retrieval.
- `FastAPI` for a simple recommendation API.
- `Uvicorn` for running the API locally.

Install dependencies:

```bash
python -m pip install -r requirements-recommendation.txt
```

Make sure the Amazon sample data exists:

```bash
python -m src.amazon_recommendation_mvp --download --limit 5000
```

Run the architecture without training:

```bash
python -m src.amazon_two_tower_recommendation --demo-no-train --top-k 10
```

This creates a deterministic untrained PyTorch two-tower model, saves a checkpoint under `models/amazon_two_tower/`, builds a FAISS product index, and prints recommendations. The recommendations are not meaningful yet because the weights are random, but the command demonstrates the architecture and technology flow.

Generic downloaded two-tower weights are usually not useful for this project because user and product embedding tables depend on the exact dataset and ID mappings. A public checkpoint trained on different user IDs and product IDs would not align with our local `events.csv` and `products.csv`.

After the no-training demo, start the local API:

```bash
uvicorn src.amazon_two_tower_recommendation:create_app --factory --reload
```

Train the two-tower model and print recommendations:

```bash
python -m src.amazon_two_tower_recommendation --train --epochs 5 --top-k 10
```

Train for a specific user ID from the downloaded data:

```bash
python -m src.amazon_two_tower_recommendation --train --user-id <user_id> --epochs 5 --top-k 10
```

After training, start the same local API:

```bash
uvicorn src.amazon_two_tower_recommendation:create_app --factory --reload
```

Call the API:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "<user_id>", "top_k": 10}'
```

This implementation is a retrieval model, not a final production ranker. In a production Amazon-style stack, the two-tower model would generate candidates, then a ranking model such as `LightGBM`, `XGBoost`, `DeepFM`, or `Wide & Deep` would score those candidates with richer features.

# 🧠 How to Normalize a Co-occurrence Matrix and Compute Similarity

Let's use a small example.

---

# Step 1: Co-occurrence Matrix

Suppose we have 4 words:

```text
cat
dog
car
truck
```

Co-occurrence matrix:


| Word  | pet | animal | vehicle | road |
| ----- | --- | ------ | ------- | ---- |
| cat   | 10  | 8      | 0       | 0    |
| dog   | 9   | 10     | 0       | 0    |
| car   | 0   | 0      | 8       | 10   |
| truck | 0   | 0      | 10      | 9    |


---

# Word Vectors

```text
cat   = [10, 8, 0, 0]
dog   = [9, 10, 0, 0]
car   = [0, 0, 8, 10]
truck = [0, 0, 10, 9]
```

---

# Why Normalize?

Without normalization:

```text
cat = [100, 80, 0, 0]
dog = [10, 8, 0, 0]
```

represent exactly the same semantic relationship.

But raw counts differ greatly.

We care about:

```text
Direction
```

not:

```text
Magnitude
```

---

# Step 2: L2 Normalization

## Formula

```text
                v
Normalized = ───────
              ||v||
```

where

```text
||v|| = √(x₁² + x₂² + ... + xₙ²)
```

---

## Normalize "cat"

Vector:

```text
cat = [10, 8, 0, 0]
```

Norm:

```text
√(10² + 8²)

= √(100 + 64)

= √164

≈ 12.81
```

Normalized:

```text
cat

[
10/12.81,
8/12.81,
0,
0
]

≈

[0.780, 0.625, 0, 0]
```

---

## Normalize "dog"

Vector:

```text
dog = [9, 10, 0, 0]
```

Norm:

```text
√(9² + 10²)

= √181

≈ 13.45
```

Normalized:

```text
dog

[
9/13.45,
10/13.45,
0,
0
]

≈

[0.669, 0.743, 0, 0]
```

---

# Step 3: Compute Cosine Similarity

## Formula

```text
                    A · B
Cosine Similarity = ─────────
                   ||A|| ||B||
```

where:

```text
A · B

=

A₁B₁ + A₂B₂ + ... + AₙBₙ
```

---

## Since Vectors Are Normalized

After L2 normalization:

```text
||A|| = 1
||B|| = 1
```

Therefore:

```text
Cosine Similarity

=

A · B
```

which becomes a simple dot product.

---

## Similarity(cat, dog)

```text
cat = [0.780, 0.625, 0, 0]

dog = [0.669, 0.743, 0, 0]
```

Dot product:

```text
(0.780 × 0.669)

+

(0.625 × 0.743)

=

0.522

+

0.464

=

0.986
```

Result:

```text
Similarity(cat,dog) ≈ 0.986
```

Very similar.

---

## Similarity(cat, car)

```text
cat = [0.780,0.625,0,0]

car = [0,0,0.625,0.780]
```

Dot product:

```text
0
```

Result:

```text
Similarity(cat,car)=0
```

Not related.

---

# Cosine Similarity Range

```text
+1  → Identical direction
 0  → Unrelated
-1  → Opposite direction
```

For word embeddings:

```text
0.8 - 1.0 → Very similar
0.5 - 0.8 → Related
0.0 - 0.5 → Weakly related
```

---

# Better Normalization: PPMI

Raw counts are often biased by frequent words.

Example:

```text
the
is
of
```

appear everywhere.

Instead of raw counts, NLP often uses:

```text
PPMI
(Positive Pointwise Mutual Information)
```

## Formula

```text
PPMI(word, context)

=

max(

log(

P(word, context)
────────────────────────
P(word) × P(context)

),

0

)
```

---

## Why PPMI Helps

Suppose:

```text
the
```

appears near every word.

Raw counts:

```text
the → 100000
cat → 50
dog → 60
```

The word:

```text
the
```

dominates the co-occurrence matrix.

PPMI reduces the impact of such extremely common words and highlights informative associations.

---

# Classical NLP Pipeline

```text
Text Corpus
      ↓
Co-occurrence Matrix
      ↓
PPMI
      ↓
SVD
      ↓
Dense Word Vectors
      ↓
Cosine Similarity
```

This was the foundation of many embedding techniques before Word2Vec.

---

# Relationship to Word2Vec

Co-occurrence Matrix:

```text
Stores counts explicitly
```

Example:

```text
cat → pet = 50
cat → animal = 40
```

Word2Vec:

```text
Learns these relationships implicitly
```

and directly produces dense vectors.

---

# Interview Answer

A co-occurrence matrix represents each word as a vector of context-word counts. To compare words, each vector is first normalized, typically using L2 normalization, which removes the effect of magnitude differences. Cosine similarity is then computed between the normalized vectors to measure how similar their context distributions are. Words appearing in similar contexts produce similar vectors and therefore have high cosine similarity scores. In classical NLP systems, PPMI normalization and SVD were often applied before computing cosine similarity to obtain more meaningful semantic representations.

# 🎧 Spotify Recommendation System — Which Data Structure Gives Constant Time Access?

This is actually a tricky interview question.

The answer depends on **what operation** needs constant-time access.

---

# 🎯 Requirement 1: Get User Profile Quickly

Example:

```text
User ID = 12345
```

Need:

```text
Liked Songs
Recently Played
Favorite Genres
```

```text

```

Use:

```python
dict[user_id] -> UserProfile
```

Complexity:

```text
Lookup = O(1)
```

---

# 🎯 Requirement 2: Get Song Metadata Quickly

Example:

```text
Song ID = 5678
```

Need:

```text
Artist
Genre
Duration
Popularity
```

Use:

```python
dict[song_id] -> Song
```

Complexity:

```text
O(1)
```

---

# 🎯 Requirement 3: Recommend Similar Songs

Example:

```text
Current Song:
Shape of You
```

Need:

```text
Top Similar Songs
```

A simple hash map won't work.

---

Use:

```python
song_id -> list of similar songs
```

Example:

```python
similar_songs = {
    "song_1": ["song_2", "song_3", "song_4"]
}
```

Complexity:

```text
Lookup = O(1)
```

This is essentially a:

```text
Graph (Adjacency List)
```

---

# 🎯 Requirement 4: Top Trending Songs

Need:

```text
Top K Songs
```

Use:

```text
Max Heap
Priority Queue
```

Complexity:

```text
Insert = O(log n)

Get Top Song = O(1)

Remove Top Song = O(log n)
```

---

# 🎯 Requirement 5: Similarity Search Using Embeddings

Modern Spotify-like systems use:

```text
Song Embeddings
User Embeddings
```

Example:

```text
768-dimensional vectors
```

Need:

```text
Nearest Neighbour Search
```

Use:

```text
HNSW
FAISS
ScaNN
Annoy
```

Complexity:

```text
Approximate O(log n)
```

Not O(1).

---

# 🚨 What Interviewers Usually Expect

If they specifically ask:

```text
"Which data structure provides constant-time access?"
```

Answer:

```text
Hash Map (Dictionary)
```

Example:

```python
song_catalog[song_id]
user_profiles[user_id]
```

Average Complexity:

```text
Insert = O(1)

Lookup = O(1)

Delete = O(1)
```

---

# 🎧 Spotify Design Answer

A practical Spotify recommendation system would use multiple data structures:


| Component            | Data Structure         | Complexity            |
| -------------------- | ---------------------- | --------------------- |
| User Lookup          | Hash Map               | O(1)                  |
| Song Lookup          | Hash Map               | O(1)                  |
| User History         | Hash Map + List        | O(1) lookup           |
| Similar Songs        | Graph / Adjacency List | O(1) neighbour access |
| Trending Songs       | Heap                   | O(log n) update       |
| Embedding Search     | HNSW / FAISS           | ~O(log n)             |
| Recommendation Cache | Redis Hash Map         | O(1)                  |


---

# 🎤 Strong Interview Answer

```text
If the requirement is constant-time retrieval of users, songs, or precomputed recommendations, I would use a Hash Map because it provides O(1) average lookup time.

For similarity-based recommendations, modern systems typically store embeddings in an ANN index such as HNSW or FAISS, which provides near-logarithmic retrieval rather than true O(1).

In production, Spotify would likely combine Hash Maps for metadata access and HNSW/FAISS for recommendation retrieval.
```

---

# 🔥 Follow-Up Answer (Senior-Level)

```text
True recommendation generation cannot be O(1) because similarity search requires comparing vectors. To achieve near O(1) serving latency, recommendations are often precomputed offline and stored in a key-value store:

user_id -> recommended_song_ids

This allows recommendation retrieval in O(1) at serving time while the expensive ranking computations happen offline.
```

