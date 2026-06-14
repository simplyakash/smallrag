
# ML System Design: Recommendation System

This is one of the most common Senior ML Engineer interview questions.

Interviewer:

> Design a recommendation system for Netflix / Amazon / YouTube / Spotify.

---

# Step 1: Clarify Requirements

Never jump to the architecture.

Ask:

## Functional Requirements

```text
Recommend items to users
Personalized recommendations
Real-time updates?
Cold start support?
```

Example:

```text
Recommend movies to users
```

---

## Non-Functional Requirements

```text
Latency < 100 ms
100M Users
10M Movies
High Availability
```

---

# Step 2: Understand Business Goal

The recommendation objective matters.

Examples:

### Netflix

```text
Maximize Watch Time
```

### Amazon

```text
Maximize Purchases
```

### YouTube

```text
Maximize Engagement
```

### Spotify

```text
Increase Listening Time
```

---

# Step 3: High-Level Architecture

A modern recommendation system is usually:

```text
Candidate Generation
         ↓
Ranking
         ↓
Re-ranking
         ↓
Final Recommendations
```

This is called a:

```text
Two-Stage Architecture
```

or

```text
Multi-Stage Recommendation System
```

---

# Why Not Score Every Item?

Suppose:

```text
100 Million Users
10 Million Products
```

For every request:

```text
10 Million Predictions
```

Impossible.

Need:

```text
Candidate Generation
```

to reduce search space.

---

# Stage 1: Candidate Generation

Goal:

```text
10 Million Items
       ↓
1000 Candidates
```

Fast but approximate.

---

# Inputs

User Features:

```text
Age
Location
Interests
Watch History
Purchase History
```

---

# Item Features

```text
Category
Brand
Price
Genre
Tags
```

---

# Common Approaches

## Collaborative Filtering

Idea:

```text
Users Similar To You
Liked These Items
```

Example:

```text
User A → Movie X
User B → Movie X

User A → Movie Y

Recommend Movie Y to User B
```

---

## Matrix Factorization

Represent:

```text
Users → Embeddings
Items → Embeddings
```

Similarity:

```text
Dot Product
```

---

## Deep Retrieval Models

Examples:

```text
Two-Tower Model
DSSM
YouTube DNN
```

Architecture:

```text
User Tower
      ↓
User Embedding

Item Tower
      ↓
Item Embedding
```

Retrieve:

```text
Nearest Neighbors
```

---

# Candidate Generation Architecture

```text
User
  ↓
User Embedding
  ↓
ANN Search
  ↓
Top 1000 Candidates
```

---

# Why ANN?

Suppose:

```text
10 Million Items
```

Brute Force:

```text
Compare Against All
```

Too expensive.

Use:

```text
Approximate Nearest Neighbor Search
```

Examples:

```text
FAISS
ScaNN
HNSW
```

---

# Stage 2: Ranking

Now we have:

```text
1000 Candidates
```

Need:

```text
Top 20 Recommendations
```

Ranking model is slower but more accurate.

---

# Features Used

## User Features

```text
Age
Gender
Country
Preferences
```

---

## Item Features

```text
Genre
Category
Popularity
```

---

## Interaction Features

```text
Clicks
Purchases
Watch Time
```

---

## Context Features

```text
Time
Device
Location
```

---

# Ranking Models

Examples:

```text
XGBoost
LightGBM
DeepFM
Wide & Deep
DCN
Transformers
```

---

# Output

```text
Candidate 1 → Score 0.95
Candidate 2 → Score 0.89
Candidate 3 → Score 0.81
```

Sort by score.

---

# Re-Ranking Layer

Most candidates forget this.

---

Goal:

Improve:

```text
Diversity
Freshness
Business Rules
```

---

# Example

Without reranking:

```text
Movie A
Movie B
Movie C
Movie D

(All Action Movies)
```

Bad experience.

---

After reranking:

```text
Action
Comedy
Drama
Documentary
```

Better diversity.

---

# Final Pipeline

```text
User
  ↓
Candidate Generation
  ↓
1000 Candidates
  ↓
Ranking Model
  ↓
Top 50
  ↓
Re-Ranking
  ↓
Top 20 Recommendations
```

---

# Offline Training Pipeline

```text
Logs
  ↓
Feature Engineering
  ↓
Training Dataset
  ↓
Model Training
  ↓
Validation
  ↓
Model Registry
```

---

# Feature Store

Very important interview topic.

Problem:

Training:

```text
Average Watch Time
```

Production:

```text
Different Calculation
```

This causes:

```text
Training-Serving Skew
```

---

# Solution

Feature Store

Examples:

```text
Feast
Tecton
Vertex Feature Store
```

Architecture:

```text
Feature Pipeline
        ↓
Feature Store
        ↓
Training + Serving
```

Same features everywhere.

---

# Online vs Offline Features

## Offline Features

Examples:

```text
30-Day Watch Time
Monthly Spend
```

Updated:

```text
Hourly / Daily
```

---

## Online Features

Examples:

```text
Current Session
Recent Clicks
```

Updated:

```text
Milliseconds
```

---

# Real-Time Recommendation Flow

```text
User Opens App
      ↓
Fetch Features
      ↓
Candidate Retrieval
      ↓
Ranking
      ↓
Re-Ranking
      ↓
Recommendations
```

Target:

```text
< 100 ms
```

---

# Cold Start Problem

Interviewer almost always asks this.

---

# User Cold Start

New user.

No history.

Solutions:

```text
Popular Items
Onboarding Questions
Demographic Signals
```

---

# Item Cold Start

New item.

No interactions.

Solutions:

```text
Content Features
Metadata
Embeddings
```

---

# Scalability Discussion

Interviewers love:

```text
What happens at 100x traffic?
```

---

# Horizontal Scaling

```text
Load Balancer
      ↓
Retrieval Servers
Ranking Servers
```

---

# Caching

Cache:

```text
Popular Recommendations
User Embeddings
Item Embeddings
```

---

# Asynchronous Updates

Don't recompute:

```text
10 Million User Embeddings
```

for every request.

Instead:

```text
Batch Updates
```

---

# Monitoring

Most candidates forget.

---

# Infrastructure Metrics

```text
Latency
CPU
GPU
Memory
QPS
Error Rate
```

---

# Model Metrics

Offline:

```text
Precision@K
Recall@K
NDCG
MAP
```

---

# Business Metrics

Most important.

```text
CTR
Watch Time
Revenue
Conversion Rate
Retention
```

---

# Common Trade-Offs

## Retrieval Quality vs Latency

More candidates:

```text
Higher Recall
```

but:

```text
More Latency
```

---

## Large Model vs Small Model

Large:

```text
Better Accuracy
```

but:

```text
Higher Cost
```

---

## Real-Time Features vs Batch Features

Real-Time:

```text
Fresh
```

but:

```text
Expensive
```

---

# Typical Interview Diagram

```text
User Activity Logs
          ↓
Feature Pipeline
          ↓
Feature Store
          ↓
────────────────────────────

User Request
      ↓
Candidate Generation
      ↓
ANN Search
      ↓
Top 1000 Items
      ↓
Ranking Model
      ↓
Top 50 Items
      ↓
Re-Ranker
      ↓
Top 20 Recommendations
      ↓
User
```

---

# Senior ML Engineer Interview Answer

A modern recommendation system typically uses a multi-stage architecture. Candidate generation retrieves a small set of potentially relevant items using techniques such as collaborative filtering, two-tower models, and ANN search. A ranking model then scores candidates using user, item, interaction, and contextual features. A re-ranking stage improves diversity, freshness, and business objectives. The system relies on feature stores to avoid training-serving skew, supports both batch and real-time features, and is monitored using infrastructure, ML, and business metrics. Key design considerations include scalability, latency, cold-start handling, and retrieval-quality versus cost trade-offs.
