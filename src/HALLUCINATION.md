# 🧠 Tracking Hallucination Rate, Answer Relevance, Groundedness & Citation Accuracy in RAG Systems

These are among the most important **LLM Production Metrics** used to monitor the quality of RAG and Agentic AI systems.

Traditional ML metrics such as:

```text
Accuracy
Precision
Recall
F1 Score
```

are often insufficient for evaluating LLM applications.

Instead, production systems monitor:

```text
✅ Hallucination Rate
✅ Answer Relevance
✅ Groundedness
✅ Citation Accuracy
```

---

# 🎯 1. Hallucination Rate

## What is it?

Measures how often the model generates information that is **not supported** by the retrieved context.

### Example

#### Retrieved Context

```text
Paris is the capital of France.
```

#### Generated Answer

```text
Paris is the capital of France.
Its population is 50 million.
```

The second statement is not present in the retrieved context.

➡️ Hallucination detected.

---

## How to Measure

Use an **LLM-as-a-Judge**.

### Evaluation Prompt

```text
Context:
{retrieved_documents}

Answer:
{generated_answer}

Determine whether every factual claim
in the answer is supported by the context.

Return:
SUPPORTED or UNSUPPORTED
```

---

## Formula

```text
Hallucination Rate =
Hallucinated Answers / Total Answers
```

### Example

```text
Total Answers = 1000
Hallucinated  = 150
```

Result:

```text
Hallucination Rate = 15%
```

---

# 🎯 2. Answer Relevance

## What is it?

Measures whether the answer actually addresses the user's question.

### Example

#### User Query

```text
What is self-attention in transformers?
```

#### Model Answer

```text
Transformers were introduced in 2017.
```

The answer is factually correct but does not answer the question.

➡️ Low relevance.

---

## Method 1: LLM Judge

```text
Question:
{query}

Answer:
{answer}

Rate answer relevance from 1-5.
```

---

## Method 2: Embedding Similarity

Compute:

```text
Embedding(Query)
Embedding(Answer)
```

Then calculate:

```text
Cosine Similarity
```

Higher similarity indicates higher relevance.

---

# 🎯 3. Groundedness

## What is it?

Measures how much of the answer is supported by retrieved documents.

Groundedness is one of the most important RAG metrics.

### Example

#### Retrieved Context

```text
Amazon was founded by Jeff Bezos in 1994.
```

#### Generated Answer

```text
Amazon was founded by Jeff Bezos in 1994.
```

All claims are supported.

```text
Groundedness = 100%
```

---

### Another Example

#### Retrieved Context

```text
Amazon was founded by Jeff Bezos in 1994.
```

#### Generated Answer

```text
Amazon was founded by Jeff Bezos in 1994.

Amazon became profitable in 1995.
```

Only the first statement is supported.

```text
Groundedness = 50%
```

---

## Measuring Groundedness

Prompt:

```text
For every claim in the answer:

1. Supported
2. Unsupported

Return percentage supported.
```

---

## Formula

```text
Groundedness =
Supported Claims / Total Claims
```

### Example

```text
Supported Claims = 8
Total Claims     = 10
```

Result:

```text
Groundedness = 80%
```

---

# 🎯 4. Citation Accuracy

## What is it?

Measures whether the cited document actually supports the statement being referenced.

### Example

#### Answer

```text
Transformers were introduced in 2017 [Doc 1]
```

Check:

```text
Does Doc 1 contain this fact?
```

If yes:

```text
Citation Correct
```

---

### Bad Example

#### Answer

```text
Transformers were introduced in 2017 [Doc 5]
```

But:

```text
Doc 5 discusses CNNs.
```

➡️ Incorrect citation.

---

## Citation Verification Pipeline

```text
Claim
  ↓
Citation ID
  ↓
Referenced Document
  ↓
LLM Judge
  ↓
Supported?
```

---

## Formula

```text
Citation Accuracy =
Correct Citations / Total Citations
```

### Example

```text
Correct Citations = 92
Total Citations   = 100
```

Result:

```text
Citation Accuracy = 92%
```

---

# 🏗️ Production Monitoring Architecture

```text
User Query
    │
    ▼
Retriever
    │
    ▼
Retrieved Documents
    │
    ▼
LLM Generation
    │
    ├────────────► Relevance Evaluator
    │
    ├────────────► Groundedness Evaluator
    │
    ├────────────► Hallucination Detector
    │
    └────────────► Citation Validator
                         │
                         ▼
                  Metrics Database
                         │
                         ▼
                 Grafana Dashboard
```

---

# 📊 Metrics Dashboard Example

| Metric              | Target    |
| ------------------- | --------- |
| Hallucination Rate  | < 5%      |
| Groundedness        | > 90%     |
| Answer Relevance    | > 4.5 / 5 |
| Citation Accuracy   | > 95%     |
| Retrieval Precision | > 90%     |
| Retrieval Recall    | > 90%     |

---

# 🛠️ Popular Evaluation Frameworks

| Framework     | Purpose                                    |
| ------------- | ------------------------------------------ |
| Ragas         | Faithfulness, Relevance, Context Precision |
| DeepEval      | Hallucination, Answer Relevance            |
| TruLens       | Groundedness, Relevance                    |
| LangSmith     | Tracing + Evaluation                       |
| Arize Phoenix | RAG Monitoring                             |
| OpenAI Evals  | Automated Evaluation                       |

---

# 🎤 Interview Answer

**In production RAG systems, Hallucination Rate measures how often generated content is unsupported by retrieved evidence. Answer Relevance measures whether the response addresses the user's query. Groundedness measures the percentage of claims supported by retrieved documents, while Citation Accuracy verifies that cited sources actually contain the referenced information. These metrics are typically computed using LLM-as-a-Judge approaches or frameworks such as Ragas, DeepEval, TruLens, LangSmith, and Arize Phoenix, and are continuously monitored through observability dashboards and alerting systems.**

# 🧠 Can Hallucination Be Tracked at Inference Time?

## Short Answer

✅ Yes, but not perfectly.

Unlike traditional ML systems, hallucination cannot be directly observed during inference because the model does not know whether its answer is true.

Instead, production systems estimate hallucination risk using various techniques.

---

# Why It Is Difficult

Suppose the user asks:

```text
Who won the Nobel Prize in Physics in 2025?
```

The LLM responds:

```text
John Smith won the Nobel Prize in Physics in 2025.
```

At inference time, the model has no built-in fact checker.

The model only predicts the next token:

```text
P(next_token | previous_tokens)
```

Therefore:

```text
LLM Confidence ≠ Factual Correctness
```

A highly confident answer can still be wrong.

---

# Real-Time Hallucination Detection Approaches

## Approach 1: Groundedness Verification (Most Common in RAG)

### Step 1

Retrieve documents:

```text
Doc A
Doc B
Doc C
```

### Step 2

Generate answer:

```text
Answer:
Amazon was founded in 1994.
```

### Step 3

Run a verifier:

```text
Does the retrieved context support this claim?
```

### Step 4

Assign groundedness score:

```text
Groundedness = 95%
```

If score falls below threshold:

```text
Groundedness < 70%
```

Trigger:

```text
⚠️ Potential Hallucination
```

---

## Production Architecture

```text
User Query
      │
      ▼
Retriever
      │
      ▼
Retrieved Context
      │
      ▼
LLM Answer
      │
      ▼
Groundedness Checker
      │
      ▼
Hallucination Risk Score
```

---

# Approach 2: Claim-by-Claim Verification

Extract claims:

```text
Amazon was founded in 1994.
Jeff Bezos was CEO until 2020.
```

Then verify each claim.

```text
Claim 1 → Supported
Claim 2 → Supported
Claim 3 → Unsupported
```

Result:

```text
Hallucination Risk = 1 / 3
```

---

# Approach 3: Secondary LLM Judge

Use another model.

Prompt:

```text
Question:
{query}

Retrieved Context:
{context}

Generated Answer:
{answer}

Determine whether every factual claim
is supported by the provided context.
```

Output:

```text
Supported: 8
Unsupported: 2

Hallucination Risk = 20%
```

---

# Approach 4: Citation Validation

Generated answer:

```text
Transformers were introduced in 2017 [Doc 3]
```

Verifier checks:

```text
Does Doc 3 contain this fact?
```

If not:

```text
Potential Hallucination
```

---

# Approach 5: Retrieval Coverage Check

Measure how much of the answer is covered by retrieved chunks.

Example:

```text
Answer Tokens = 100
Supported Tokens = 85
```

Result:

```text
Coverage = 85%
```

Low coverage often correlates with hallucination.

---

# Approach 6: Self-Consistency

Ask the model multiple times.

```text
Answer 1
Answer 2
Answer 3
```

If answers differ significantly:

```text
High Uncertainty
```

Potential hallucination.

---

# Approach 7: Confidence Scoring

Some systems monitor:

```text
Token Probabilities
Log Probabilities
Entropy
```

Example:

```text
Average Token Probability = 0.92
```

However:

```text
High Confidence ≠ Correct Answer
```

Therefore confidence alone is not reliable.

---

# What Production Systems Actually Use

Most enterprise RAG systems combine:

```text
1. Retrieval Coverage
2. Groundedness Scoring
3. Citation Validation
4. LLM-as-a-Judge
```

to compute:

```text
Hallucination Risk Score
```

Example:

```text
Groundedness      = 92%
Citation Accuracy = 96%
Coverage          = 89%

Final Risk Score  = Low
```

---

# Real-Time Mitigation

If hallucination risk is high:

```text
Risk > Threshold
```

the system can:

### Option 1

Regenerate answer.

```text
Retrieve Again
      ↓
Generate Again
```

### Option 2

Return:

```text
I could not find sufficient evidence
to answer confidently.
```

### Option 3

Show citations only.

### Option 4

Escalate to human review.

---

# Interview Answer

Yes, hallucination can be tracked during inference, but it cannot be measured directly because the model does not know whether its answer is factually correct. Production systems estimate hallucination risk using groundedness checks, claim verification, citation validation, retrieval coverage analysis, and LLM-as-a-Judge evaluations. In RAG systems, the most common approach is to verify whether generated claims are supported by retrieved documents and compute a groundedness or hallucination-risk score in real time.
