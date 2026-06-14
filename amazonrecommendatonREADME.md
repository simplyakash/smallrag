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

## 5. Data Sources

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

