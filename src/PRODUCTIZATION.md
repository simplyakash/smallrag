# AI Productization & Deploying AI Solutions in Production

## What is AI Productization?

AI Productization means:

```text
Research Model
      ↓
Production System
      ↓
Business Product
```

A model alone is not a product.

Example:

```text
Jupyter Notebook
     ↓
95% Accuracy
```

❌ Not a product

A production AI system includes:

```text
Users
  ↓
API
  ↓
AI Service
  ↓
Response
```

along with:

```text
Monitoring
Security
Versioning
Scaling
Feedback Loops
Retraining
```

---

# End-to-End AI Lifecycle

```text
Problem Definition
        ↓
Data Collection
        ↓
Training
        ↓
Evaluation
        ↓
Model Registry
        ↓
Containerization
        ↓
Deployment
        ↓
Monitoring
        ↓
Feedback Collection
        ↓
Retraining
```

---

# 1. Model Training

Example:

```text
Fraud Detection Model
```

or

```text
RAG Chatbot
```

Train using:

```text
Historical Data
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Validation
```

---

# 2. Model Registry

Purpose:

```text
Versioning
Rollback
Governance
Auditability
```

Tools:

```text
MLflow
SageMaker Registry
Vertex AI Registry
```

Example:

```text
fraud_model_v1
fraud_model_v2
fraud_model_v3
```

Benefits:

```text
Track Experiments
Store Metrics
Rollback Safely
```

---

# 3. Containerization

Package:

```text
Model
Dependencies
Inference Code
```

into:

```text
Docker Image
```

Example:

```docker
FROM python:3.11

COPY model.pkl .
COPY app.py .

RUN pip install -r requirements.txt
```

Benefit:

```text
Works Consistently Everywhere
```

---

# 4. Deployment Types

## Batch Inference

Example:

```text
Generate Product Recommendations
Every Night
```

Architecture:

```text
Data Warehouse
      ↓
Batch Job
      ↓
Predictions
      ↓
Database
```

Pros:

```text
Cheap
Simple
```

Cons:

```text
Not Real-Time
```

---

## Real-Time Inference

Example:

```text
Fraud Detection
Chatbots
Search Ranking
```

Architecture:

```text
User Request
      ↓
API
      ↓
Model Service
      ↓
Prediction
```

Pros:

```text
Low Latency
Fresh Predictions
```

Cons:

```text
Higher Infrastructure Cost
```

---

## Streaming Inference

Example:

```text
Ad Click Prediction
Fraud Detection
```

Architecture:

```text
Kafka
   ↓
Feature Pipeline
   ↓
Prediction Service
```

Pros:

```text
Near Real-Time
```

Cons:

```text
More Complex
```

---

# Model Serving

Architecture:

```text
Client
   ↓
Load Balancer
   ↓
API Gateway
   ↓
Inference Service
   ↓
Model
```

Common Tools:

```text
FastAPI
Triton Inference Server
TorchServe
BentoML
```

---

# LLM Serving

Architecture:

```text
User Query
      ↓
API Gateway
      ↓
vLLM
      ↓
GPU
      ↓
Response
```

---

## Why vLLM?

Key Features:

```text
Paged Attention
Continuous Batching
Efficient KV Cache Management
```

Benefits:

```text
Higher Throughput
Lower Cost
Better GPU Utilization
```

---

# Scalability

Interview Question:

```text
Traffic Increases 100x
What Do You Do?
```

---

## Horizontal Scaling

Instead of:

```text
1 Bigger Server
```

Use:

```text
Load Balancer
      ↓
Server 1
Server 2
Server 3
Server N
```

Benefits:

```text
Scalable
Fault Tolerant
```

---

## Caching

Cache:

```text
Popular Queries
Embeddings
Retrieval Results
LLM Responses
```

Benefits:

```text
Lower Latency
Reduced Cost
```

---

# Monitoring

Most candidates stop at deployment.

Senior engineers discuss monitoring.

---

## Infrastructure Monitoring

Track:

```text
Latency
Throughput
CPU Usage
GPU Usage
Memory Usage
Error Rate
```

---

## Model Monitoring

Track:

```text
Prediction Distribution
Feature Distribution
Accuracy
Precision
Recall
```

---

# Data Drift

Training Data:

```text
Age: 20-40
```

Production Data:

```text
Age: 50-80
```

Input distribution changed.

```text
Data Drift
```
```
Track using PSI (Population Stability Index),Kolmogorov-Smirnov Test,Jensen-Shannon Distance, Feature Statistics like Mean
Median
Variance
Min
Max
Percentiles
```

---

# Concept Drift

Training:

```text
Feature → Label Relationship
```

changes over time.

Example:

```text
Customer Behavior Changes
Fraud Patterns Change
```

Result:

```text
Model Accuracy Drops
```
```
The relationship between inputs and outputs changes.
```
| Property          | Data Drift           | Concept Drift             |
| ----------------- | -------------------- | ------------------------- |
| What changes?     | Input distribution   | Input-output relationship |
| Mathematical Form | P(X) changes         | P(Y|X) changes            |
| Labels needed?    | No                   | Usually yes               |
| Easier to detect? | Yes                  | Harder                    |
| Detection         | PSI, KS, JS Distance | Accuracy,ErrorRate, DDM, ADWIN      |
| Example           | Users become older   | User behavior changes     |

| Tool                 | Purpose                            |
| -------------------- | ---------------------------------- |
| Evidently AI         | Drift detection dashboards         |
| WhyLabs              | Data & concept drift monitoring    |
| Arize AI             | Model monitoring                   |
| MLflow               | Metrics tracking                   |
| Prometheus + Grafana | Production alerting and dashboards |

---

# LLM-Specific Monitoring

Track:

```text
Hallucination Rate:
Hallucination Rate=
Total Answers/Unsuppported Answers ) 
Answer Relevance:LLM Evaluation
Mainly LLM as a judge to rate relevance and brevity of responses
Groundedness:
Citation Accuracy
```

---

# Retraining Pipeline

```text
Production Data
       ↓
Validation
       ↓
Retraining
       ↓
Evaluation
       ↓
Canary Deployment
       ↓
Production
```

Purpose:

```text
Keep Model Fresh
Handle Drift
Improve Accuracy
```

---

# Deployment Strategies

## Blue-Green Deployment

```text
Blue  = Current Version
Green = New Version
```

Switch traffic after validation.

Benefits:

```text
Easy Rollback
Low Risk
```

---

## Canary Deployment

```text
95% Traffic → Old Model
 5% Traffic → New Model
```

If successful:

```text
10%
25%
50%
100%
```

Benefits:

```text
Reduced Risk
Real Traffic Validation
```

---

## A/B Testing

```text
Model A
    vs
Model B
```

Compare:

```text
CTR
Revenue
Retention
Conversion Rate
```

---

# Production RAG Architecture

```text
Documents
      ↓
Chunking
      ↓
Embeddings
      ↓
Vector Database
      ↓
Retriever
      ↓
Reranker
      ↓
LLM
      ↓
Answer
```

Production Concerns:

```text
Latency
Cost
Security
Freshness
Access Control
Hallucinations
```

---

# Important Trade-Offs

## Large Model vs Small Model

### Large Model

Pros:

```text
Higher Quality
```

Cons:

```text
Higher Cost
Higher Latency
```

---

### Small Model

Pros:

```text
Fast
Cheap
```

Cons:

```text
Lower Accuracy
```

---

## Real-Time Features vs Batch Features

### Real-Time

Pros:

```text
Fresh Information
```

Cons:

```text
Complex Infrastructure
```

---

### Batch

Pros:

```text
Simple
Cheap
```

Cons:

```text
Stale Information
```

---

# Senior ML Engineer Interview Answer

## Key Considerations When Deploying AI Solutions

```text
1. Version models using a model registry.

2. Containerize using Docker.

3. Deploy using scalable serving infrastructure.

4. Choose batch, real-time, or streaming inference.

5. Monitor latency, throughput, errors, and model quality.

6. Detect data drift and concept drift.

7. Use canary or blue-green deployments.

8. Build automated retraining pipelines.

9. Optimize cost using caching and batching.

10. Ensure security, observability, and reliability.
```

---

# One-Line Interview Summary

```text
AI Productization is the process of transforming a trained model into a scalable, reliable, monitored, secure, and maintainable production system that continuously delivers business value.
```



# Model Deployment Concepts & Serving Strategies

This topic is extremely common in **Senior ML Engineer** interviews because it evaluates whether you can take:

```text
Trained Model
      ↓
Production Service
      ↓
Business Value
```

---

# What is Model Deployment?

Model deployment is the process of making a trained model available for real-world predictions.

```text
Training
    ↓
Model Artifact
    ↓
Deployment
    ↓
Inference Service
    ↓
Predictions
```

Example:

```text
Fraud Model
     ↓
REST API
     ↓
Banking Application
```

---

# Deployment Pipeline

```text
Training
    ↓
Validation
    ↓
Model Registry
    ↓
Docker Image
    ↓
CI/CD
    ↓
Production Deployment
    ↓
Monitoring
```

---

# Model Artifact

After training:

```python
model.pkl
```

or

```python
model.pt
```

or

```python
model.onnx
```

These artifacts are stored in:

```text
MLflow
S3
Model Registry
```

---

# Model Serving

Model serving means:

```text
Receive Request
       ↓
Load Model
       ↓
Generate Prediction
       ↓
Return Response
```

Example:

```text
User
  ↓
API
  ↓
Model
  ↓
Prediction
```

---

# Types of Serving Strategies

There are 3 major serving strategies.

```text
Batch Inference
Real-Time Inference
Streaming Inference
```

---

# 1. Batch Inference

Predictions generated periodically.

Example:

```text
Netflix Recommendations
```

Architecture:

```text
Data Warehouse
       ↓
Spark Job
       ↓
Model
       ↓
Predictions
       ↓
Database
```

---

## Example

Generate recommendations for:

```text
50 Million Users
```

every night.

```text
12 AM
   ↓
Batch Job
   ↓
Store Recommendations
```

---

## Advantages

```text
Cheap
Simple
Scalable
```

---

## Disadvantages

```text
Not Real-Time
Predictions Can Become Stale
```

---

# 2. Real-Time (Online) Inference

Predictions generated immediately.

Architecture:

```text
User Request
      ↓
API
      ↓
Model Server
      ↓
Prediction
```

---

## Example

```text
Fraud Detection
ChatGPT
Search Ranking
```

---

## Flow

```text
Credit Card Transaction
          ↓
Fraud Model
          ↓
Approve / Reject
```

Latency requirement:

```text
< 100 ms
```

---

## Advantages

```text
Fresh Predictions
Better User Experience
```

---

## Disadvantages

```text
Expensive
Complex Infrastructure
```

---

# 3. Streaming Inference

Data arrives continuously.

Architecture:

```text
Kafka
   ↓
Feature Pipeline
   ↓
Model
   ↓
Prediction
```

---

## Example

```text
Fraud Detection
Ad Click Prediction
IoT Monitoring
```

---

## Advantages

```text
Near Real-Time
Handles Continuous Events
```

---

## Disadvantages

```text
Operational Complexity
```

---

# Serving Architecture

Typical architecture:

```text
Client
   ↓
Load Balancer
   ↓
API Gateway
   ↓
Model Server
   ↓
Prediction
```

---

# Model Serving Frameworks

## FastAPI

Most common.

```python
@app.post("/predict")
```

Pros:

```text
Simple
Fast
Easy Integration
```

---

## Triton Inference Server

NVIDIA's serving framework.

Pros:

```text
Dynamic Batching
Multi-Model Serving
GPU Optimization
```

---

## TorchServe

For PyTorch models.

Pros:

```text
Simple Deployment
```

---

## BentoML

Designed for ML deployment.

Pros:

```text
Packaging
Versioning
Deployment
```

---

# LLM Serving

Architecture:

```text
User Query
      ↓
API Gateway
      ↓
vLLM
      ↓
GPU Cluster
      ↓
Response
```

---

# Why vLLM?

Interview favorite.

Features:

```text
Paged Attention
Continuous Batching
Efficient KV Cache
```

Benefits:

```text
Higher Throughput
Lower Cost
```

---

# Batching

Without batching:

```text
Request 1
Request 2
Request 3
```

processed separately.

---

With batching:

```text
Request 1
Request 2
Request 3
      ↓
Single GPU Batch
```

Benefits:

```text
Better GPU Utilization
Higher Throughput
```

---

# Dynamic Batching

Wait briefly:

```text
5-10 ms
```

Collect requests.

Process together.

```text
Request A
Request B
Request C
      ↓
Batch
      ↓
GPU
```

Common in:

```text
Triton
vLLM
```

---

# Horizontal vs Vertical Scaling

## Vertical Scaling

```text
1 Server
   ↓
Bigger Server
```

Pros:

```text
Simple
```

Cons:

```text
Expensive
Limited
```

---

## Horizontal Scaling

```text
Load Balancer
      ↓
Server 1
Server 2
Server 3
Server N
```

Pros:

```text
Highly Scalable
Fault Tolerant
```

Cons:

```text
Operational Complexity
```

---

# Deployment Strategies

---

# Blue-Green Deployment

```text
Blue  = Current Model
Green = New Model
```

```text
100% Traffic
      ↓
Blue

Validate Green

Switch Traffic
```

---

## Benefits

```text
Easy Rollback
Minimal Downtime
```

---

# Canary Deployment

```text
95% → Old Model
 5% → New Model
```

If successful:

```text
10%
25%
50%
100%
```

---

## Benefits

```text
Reduced Risk
Real User Validation
```

---

# Shadow Deployment

Very common.

```text
Production Traffic
         ↓
Old Model
         ↓
Response

Production Traffic
         ↓
New Model
         ↓
Predictions Logged
```

Users only see:

```text
Old Model Results
```

Purpose:

```text
Evaluate New Model Safely
```

---

# A/B Testing

```text
Users
  ↓
50% → Model A

50% → Model B
```

Compare:

```text
CTR
Revenue
Retention
Conversion
```

---

# Model Monitoring

---

## Infrastructure Metrics

Track:

```text
Latency
CPU
GPU
Memory
QPS
Error Rate
```

---

## Model Metrics

Track:

```text
Accuracy
Precision
Recall
F1
```

---

# Data Drift

Training:

```text
Age: 20-40
```

Production:

```text
Age: 50-80
```

Input distribution changed.

---

# Concept Drift

Relationship changes.

Example:

```text
Fraud Patterns Change
```

Model becomes outdated.

---

# Cost Optimization

Interviewers love this.

---

## Caching

Cache:

```text
Predictions
Embeddings
LLM Responses
```

---

## Quantization

Convert:

```text
FP32
  ↓
INT8
  ↓
INT4
```

Benefits:

```text
Lower Memory
Lower Latency
```

---

## Distillation

```text
Large Model
      ↓
Small Model
```

Benefits:

```text
Cheaper Inference
```

---

# Interview Answer

## What are the major model serving strategies?

```text
1. Batch Inference
   - Periodic predictions
   - Cheap and scalable

2. Real-Time Inference
   - On-demand predictions
   - Low latency applications

3. Streaming Inference
   - Continuous event processing
   - Near real-time predictions
```

---

# Senior ML Engineer Answer

When deploying AI models to production, I first determine whether the use case requires batch, online, or streaming inference. I package the model using Docker, store versions in a model registry, and deploy through scalable serving infrastructure such as FastAPI, Triton, or vLLM. I use load balancing, batching, caching, and horizontal scaling to handle traffic growth. For safe rollouts, I use canary, blue-green, or shadow deployments. Finally, I continuously monitor infrastructure metrics, model performance, data drift, and business KPIs while maintaining a retraining pipeline.


# AI/ML System Design, Scalability & Architectural Trade-offs

This is usually the most important section in a Senior ML Engineer interview.

Interviewers are NOT evaluating whether you know a model.

They are evaluating whether you can build:

```text
Reliable
Scalable
Cost Efficient
Production-Ready
AI Systems
```

---

# System Design Framework

For ANY design question:

```text
Requirements
      ↓
Scale Estimation
      ↓
High-Level Architecture
      ↓
Data Flow
      ↓
Model Design
      ↓
Serving Strategy
      ↓
Scalability
      ↓
Monitoring
      ↓
Trade-offs
```

Never jump directly to the model.

---

# Example System Design Questions

```text
Design ChatGPT
Design a RAG System
Design Fraud Detection
Design a Recommendation System
Design Customer Support Agent
Design Document Search
```

The process remains similar.

---

# Step 1: Requirements Gathering

Ask:

## Functional Requirements

```text
What should the system do?
```

Example:

```text
Users upload documents
Users ask questions
System returns answers
```

---

## Non-Functional Requirements

```text
Latency?
QPS?
Availability?
Cost Constraints?
```

Example:

```text
Latency < 2 sec
1000 QPS
99.9% Availability
```

---

# Step 2: Scale Estimation

Interviewers love this.

Example:

```text
100K Users
```

Assume:

```text
10 Queries/User/Day
```

Then:

```text
1 Million Queries/Day
```

Average:

```text
~12 QPS
```

Peak:

```text
100-200 QPS
```

This helps decide:

```text
Number of Servers
GPU Count
Database Size
```

---

# Step 3: High-Level Architecture

Example: Enterprise RAG

```text
Documents
      ↓
Chunking
      ↓
Embedding Model
      ↓
Vector DB

────────────────────────

User Query
      ↓
Embedding
      ↓
Retriever
      ↓
Reranker
      ↓
LLM
      ↓
Response
```

---

# Scalability

Interviewers always ask:

```text
What if traffic grows 100x?
```

---

# Horizontal Scaling

Preferred approach.

```text
Load Balancer
      ↓
Server 1
Server 2
Server 3
Server N
```

Pros:

```text
Fault Tolerant
Scalable
```

Cons:

```text
More Operational Complexity
```

---

# Vertical Scaling

```text
Small Server
      ↓
Large Server
```

Pros:

```text
Simple
```

Cons:

```text
Expensive
Eventually Hits Limits
```

---

# Caching

One of the most important scalability techniques.

Cache:

```text
Popular Queries
Embeddings
LLM Responses
```

Example:

```text
"What is leave policy?"
```

asked 10,000 times.

No need to call LLM every time.

---

# Asynchronous Processing

Bad:

```text
User
 ↓
Wait 5 Minutes
```

Good:

```text
User
 ↓
Queue
 ↓
Worker
 ↓
Notification
```

Examples:

```text
Document Ingestion
Video Processing
Large Batch Jobs
```

---

# Architectural Trade-offs

This is where senior candidates stand out.

Interviewers LOVE trade-offs.

---

# Trade-off 1

## Large Model vs Small Model

### GPT-4 / 70B

Pros:

```text
Better Quality
```

Cons:

```text
Higher Cost
Higher Latency
```

---

### 8B Model

Pros:

```text
Fast
Cheap
```

Cons:

```text
Lower Accuracy
```

---

# Trade-off 2

## Real-Time vs Batch Inference

### Real-Time

```text
User
 ↓
Prediction
```

Pros:

```text
Fresh Predictions
```

Cons:

```text
Expensive Infrastructure
```

---

### Batch

```text
Nightly Job
      ↓
Predictions Stored
```

Pros:

```text
Cheap
Scalable
```

Cons:

```text
Stale Predictions
```

---

# Trade-off 3

## Retrieval Depth

### Top-10 Retrieval

Pros:

```text
Fast
```

Cons:

```text
Lower Recall
```

---

### Top-100 Retrieval

Pros:

```text
Higher Recall
```

Cons:

```text
More Latency
More Tokens
```

---

# Trade-off 4

## Reranker

Without reranker:

```text
Retriever
     ↓
LLM
```

Fast.

---

With reranker:

```text
Retriever
     ↓
Cross Encoder
     ↓
LLM
```

Pros:

```text
Better Relevance
```

Cons:

```text
Additional Latency
```

---

# Trade-off 5

## Self-Hosted vs API Models

### OpenAI API

Pros:

```text
Easy
High Quality
```

Cons:

```text
Vendor Dependency
Data Privacy Concerns
```

---

### Self-Hosted Llama

Pros:

```text
Data Control
Lower Long-Term Cost
```

Cons:

```text
GPU Infrastructure Required
```

---

# Reliability

Interviewers expect this discussion.

---

## Redundancy

```text
Primary Service
        ↓
Backup Service
```

---

## Failover

If:

```text
Model Service Fails
```

Switch to:

```text
Backup Model
```

---

## Graceful Degradation

Instead of:

```text
System Down
```

Use:

```text
Keyword Search
Cached Results
Smaller Model
```

---

# Monitoring

Most candidates forget this.

---

# Infrastructure Metrics

Track:

```text
Latency
CPU
GPU
Memory
QPS
Error Rate
```

---

# ML Metrics

Track:

```text
Accuracy
Precision
Recall
F1
```

---

# RAG Metrics

Track:

```text
Recall@K
MRR
NDCG
```

---

# LLM Metrics

Track:

```text
Groundedness
Hallucination Rate
Answer Relevance
Citation Accuracy
```

---

# Data Drift

Training:

```text
Customer Age
20-40
```

Production:

```text
Customer Age
50-80
```

Input distribution changed.

---

# Concept Drift

Relationship changes.

Example:

```text
Fraud Patterns Evolve
```

Model performance drops.

---

# Cost Optimization

Senior-level discussion.

---

## Quantization

```text
FP32
 ↓
INT8
 ↓
INT4
```

Benefits:

```text
Lower Memory
Lower Cost
Lower Latency
```

---

## Distillation

```text
70B Model
      ↓
8B Model
```

Benefits:

```text
Cheaper Inference
```

---

## Caching

```text
Frequently Asked Questions
Popular Responses
Embeddings
```

reduces LLM calls.

---

# AI System Design Checklist

Whenever asked to design any AI system, discuss:

```text
✓ Functional Requirements

✓ Non-Functional Requirements

✓ Scale Estimation

✓ Architecture

✓ Data Flow

✓ Model Selection

✓ Serving Strategy

✓ Scalability

✓ Reliability

✓ Monitoring

✓ Security

✓ Cost Optimization

✓ Trade-offs
```

---

# Senior ML Engineer Interview Answer

AI/ML system design focuses on building scalable and reliable production systems around machine learning models. I start by understanding requirements, estimating scale, and designing the high-level architecture. I then discuss model serving, data pipelines, scalability strategies such as horizontal scaling and caching, monitoring, reliability mechanisms, and cost optimization. Most importantly, I evaluate architectural trade-offs such as model quality versus latency, real-time versus batch inference, retrieval quality versus speed, and self-hosted versus managed solutions.

# QPS (Queries Per Second)

QPS measures:

> How many requests a system can handle every second.

Formula:

```text
QPS = Total Requests / Total Time (seconds)
```

Example:

```text
1000 requests
in
10 seconds
```

QPS:

```text
1000 / 10 = 100 QPS
```

Meaning:

```text
The system processes 100 requests every second.
```

---

# Why is QPS Important?

It helps determine:

```text
How many servers are needed
How many GPUs are needed
System scalability
Infrastructure cost
```

Example:

```text
Chatbot receives

10,000 requests/sec
```

A single server may not handle it.

Need:

```text
Load Balancer
      ↓
Multiple Inference Servers
```

---

# Interview Example

Suppose:

```text
1 Million requests/day
```

Convert to QPS:

```text
1,000,000 / 86,400

≈ 11.57 QPS
```

But traffic isn't uniform.

Peak traffic may be:

```text
5x to 10x higher
```

So design for:

```text
100 QPS
```

instead of:

```text
12 QPS
```

---

# Error Rate

Error Rate measures:

> What percentage of requests failed.

Formula:

```text
Error Rate =
(Failed Requests / Total Requests) × 100
```

---

# Example

Suppose:

```text
1000 requests
```

and

```text
20 failed
```

Then:

```text
Error Rate

= (20 / 1000) × 100

= 2%
```

---

# Why is Error Rate Important?

High error rate means:

```text
Bad User Experience
Revenue Loss
System Instability
```

---

# Common Errors in AI Systems

## API Errors

```text
HTTP 500
HTTP 503
Timeouts
```

---

## Model Errors

```text
Model not loaded
GPU out of memory
Inference failure
```

---

## LLM Errors

```text
Rate limits
Provider outage
Token limit exceeded
```

---

# Example Dashboard

Monitor:

```text
QPS          = 500
Latency      = 200 ms
Error Rate   = 0.5%
CPU          = 70%
GPU          = 80%
```

Healthy system.

---

# Real Interview Follow-up

### Q: Traffic suddenly doubles. What metric will increase first?

Possible answers:

```text
QPS
CPU Usage
GPU Usage
Latency
```

Eventually:

```text
Error Rate
```

also rises because servers become overloaded.

---

# Relationship Between QPS, Latency, and Error Rate

```text
Traffic Increase
       ↓
Higher QPS
       ↓
CPU/GPU Saturation
       ↓
Higher Latency
       ↓
Timeouts
       ↓
Higher Error Rate
```

Example:

```text
100 QPS
Latency = 100 ms
Error Rate = 0.1%
```

After traffic spike:

```text
1000 QPS
Latency = 3 sec
Error Rate = 10%
```

---

# Senior ML Engineer Answer

### QPS

```text
Queries Per Second (QPS) measures the number of requests handled by a system every second and is used to estimate scalability, capacity planning, and infrastructure requirements.
```

### Error Rate

```text
Error Rate is the percentage of requests that fail due to system, infrastructure, or model-related issues. It is a critical reliability metric and is usually monitored alongside latency, throughput, CPU/GPU utilization, and availability.
```
