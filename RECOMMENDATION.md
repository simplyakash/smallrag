
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


# Inverse Document Frequency (IDF)

IDF is a measure of **how important or rare a word is across all documents**.

The intuition is:

```text
Common words
    ↓
Less useful

Rare words
    ↓
More useful
```

---

# Why Do We Need IDF?

Suppose we have 3 documents:

```text
Doc1: "The cat sat on the mat"
Doc2: "The dog sat on the floor"
Doc3: "The cat chased the dog"
```

The word:

```text
"the"
```

appears everywhere.

The word:

```text
"mat"
```

appears only once.

Clearly:

```text
"mat"
```

helps identify a document much better than:

```text
"the"
```

IDF captures this idea.

---

# Formula

```text
IDF(word) = log(N / DF)
```

where:

```text
N  = Total number of documents
DF = Number of documents containing the word
```

---

# Example

Suppose:

```text
Total Documents (N) = 1000
```

Word:

```text
"laptop"
```

appears in:

```text
DF = 10 documents
```

Then:

```text
IDF = log(1000 / 10)
     = log(100)
     ≈ 2
```

High IDF.

Meaning:

```text
Rare word
```

---

# Common Word Example

Word:

```text
"the"
```

appears in:

```text
DF = 900
```

Then:

```text
IDF = log(1000 / 900)
     ≈ 0.046
```

Very low.

Meaning:

```text
Not useful for ranking
```

---

# Visualization

```text
Word         DF      IDF

the          900     0.046
and          850     0.07
laptop        10     2.0
macbook        2     2.7
```

As DF increases:

```text
IDF decreases
```

As DF decreases:

```text
IDF increases
```

---

# What is DF?

DF means:

```text
Document Frequency
```

Number of documents containing the term.

Example:

```text
Doc1: Apple Laptop
Doc2: Dell Laptop
Doc3: Gaming Mouse
Doc4: HP Laptop
```

For word:

```text
Laptop
```

DF:

```text
3
```

because it appears in:

```text
Doc1
Doc2
Doc4
```

---

# TF-IDF

IDF is usually combined with TF.

### TF (Term Frequency)

How often the word appears in a document.

### IDF

How rare the word is across all documents.

Combined:

```text
TF-IDF = TF × IDF
```

---

# Example

Query:

```text
Gaming Laptop
```

Documents:

```text
Doc1:
Gaming Gaming Gaming Laptop

Doc2:
Laptop

Doc3:
Gaming Laptop RTX 5090
```

Scores depend on:

```text
TF
+
IDF
```

Rare and important terms get larger scores.

---

# Why BM25 Uses IDF

BM25 improves TF-IDF but still relies heavily on IDF.

```text
Rare terms
     ↓
Higher weight

Common terms
     ↓
Lower weight
```

Query:

```text
"wireless gaming mouse"
```

Words like:

```text
gaming
mouse
wireless
```

receive higher importance than:

```text
the
is
for
```

---

# Interview Answer

IDF (Inverse Document Frequency) measures how rare or informative a term is across a collection of documents. It is calculated as:

```text
IDF = log(N / DF)
```

where N is the total number of documents and DF is the number of documents containing the term. Rare words receive a high IDF score, while common words receive a low IDF score. IDF is commonly used in TF-IDF and BM25 ranking algorithms.

# Term Frequency (TF)

TF measures **how often a word appears in a document**.

The intuition is:

```text
More occurrences in a document
           ↓
More important to that document
```

---

# Formula

Simplest version:

```text
TF(term) = Number of times term appears in document
```

---

# Example

Document:

```text
"gaming laptop gaming mouse gaming keyboard"
```

Count occurrences:

```text
gaming   = 3
laptop   = 1
mouse    = 1
keyboard = 1
```

Therefore:

```text
TF(gaming) = 3
TF(laptop) = 1
```

The document is probably more about:

```text
gaming
```

than:

```text
laptop
```

---

# Normalized TF

Sometimes documents have different lengths.

So we normalize TF:

```text
TF(term)
=
(Term Count)
/
(Total Number of Words)
```

---

# Example

Document:

```text
"gaming laptop gaming mouse gaming keyboard"
```

Total words:

```text
6
```

Counts:

```text
gaming = 3
```

Normalized TF:

```text
TF(gaming)
=
3/6
=
0.5
```

---

# Why TF Alone Is Not Enough

Suppose every document contains:

```text
the
the
the
the
the
```

TF would be very high.

But:

```text
"the"
```

is not useful for search.

That's why we combine TF with IDF.

---

# TF + IDF

Consider:

```text
Doc1:
gaming gaming gaming laptop

Doc2:
the the the laptop
```

TF Scores:

```text
TF(gaming) = 3
TF(the)    = 3
```

Both are equal.

But:

```text
gaming
```

is much more informative than:

```text
the
```

IDF fixes this.

---

# TF-IDF

Formula:

```text
TF-IDF = TF × IDF
```

Example:

```text
TF(gaming) = 3
IDF(gaming) = 2
```

Score:

```text
3 × 2 = 6
```

---

Common word:

```text
TF(the) = 3
IDF(the) = 0.05
```

Score:

```text
3 × 0.05 = 0.15
```

Result:

```text
gaming > the
```

which is what we want.

---

# Search Example

Query:

```text
gaming laptop
```

Documents:

```text
Doc1:
gaming gaming gaming laptop

Doc2:
laptop stand for desk

Doc3:
gaming mouse keyboard
```

TF:

```text
Doc1:
gaming = 3
laptop = 1

Doc2:
gaming = 0
laptop = 1

Doc3:
gaming = 1
laptop = 0
```

Doc1 gets the highest score because query terms appear more frequently.

---

# BM25 vs TF

BM25 improves TF because raw TF can be misleading.

Example:

```text
gaming repeated 100 times
```

should not make a document 100× better.

BM25 uses:

```text
Saturated TF
```

meaning:

```text
TF helps
      ↓
but with diminishing returns
```

---

# Interview Answer

Term Frequency (TF) measures how often a term appears in a document. A higher TF indicates that the term is more important to that document. TF is usually combined with Inverse Document Frequency (IDF) to form TF-IDF, which balances term importance within a document against how common the term is across all documents.