# Evaluating Large Language Models for Automated Exploratory Data Analysis
### A Statistical and Applied Science Perspective

This project evaluates whether Large Language Models (LLMs) can replicate
statistically grounded human Exploratory Data Analysis (EDA).

The work compares **human statistical reasoning** against **LLM-generated insights**
using semantic similarity and coverage metrics, following an applied science approach.

---

## Motivation

Exploratory Data Analysis (EDA) is a critical first step in data-driven decision making,
but it is often manual, time-consuming, and dependent on analyst expertise.

With the rise of Large Language Models, an important question arises:

> **Can LLMs meaningfully assist or replicate human EDA reasoning?**

This project answers that question through **controlled experiments and quantitative evaluation**.

---

## High-Level Idea

The project follows a simple but rigorous pipeline:

1. Perform **human-style statistical EDA**
2. Generate a **structured EDA context** programmatically
3. Ask an **LLM to generate insights** based on that context
4. **Evaluate alignment** between human and LLM insights

This allows us to measure how close LLM reasoning is to statistically grounded analysis.

---

## Project Structure
app/
├── eda.py # Programmatic EDA and context generation
├── llm_insights.py # LLM-based insight generation (Ollama)
├── stats_human_baseline.py # Human statistical EDA baseline
├── main.py # Gradio-based EDA automation demo

experiments/
├── llm_vs_human_sales.ipynb # Core applied science evaluation experiment

data/
├── sales_data.csv 
├── titanic_dataset_final.csv
test datsets 

reqiirements.txt   

---

## Human Statistical Baseline

The human baseline is implemented using classical statistical techniques:

- Descriptive statistics (mean, median, standard deviation)
- Missing value analysis
- Categorical feature summaries
- Correlation analysis

This represents a **rigorous and interpretable statistical EDA baseline**.

---

## LLM-Based EDA System

The automated EDA system works as follows:

1. Numerical summaries and correlations are computed programmatically
2. Results are converted into a **structured EDA context**
3. The context is passed to an LLM (via Ollama)
4. The LLM generates analytical insights grounded in the provided statistics

This design minimizes hallucination by restricting the LLM to computed data only.

---

## Evaluation Methodology

To evaluate how closely LLM insights match human reasoning, the project uses:

- **Semantic Similarity**  
  Measured using sentence embeddings and cosine similarity

- **Insight Coverage**  
  Measures overlap between analytical concepts identified by humans and the LLM

These metrics provide a **quantitative comparison** between human and LLM-generated analysis.

---

## Core Experiment

**Main evaluation notebook:**

experiments/llm_vs_human_sales.ipynb

This notebook:
1. Computes human statistical EDA
2. Generates LLM-based insights
3. Evaluates similarity and coverage metrics
4. Interprets the results

---

## How to Run

### Setup
```bash
pip install -r requirements.txt
ollama run mistral (ctrl+z to abort mistral)
python -m app.main
jupyter notebook
experiments/llm_vs_human_sales.ipynb

