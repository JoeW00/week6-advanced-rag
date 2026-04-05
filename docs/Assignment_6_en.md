# Week 6

## Learning Objectives

- **(Open WebUI) "View" Data:** Understand how data quality affects model performance. Learn to identify high-quality datasets from platforms like Hugging Face and design your own datasets based on specific needs.
- **(Hugging Face) "Use" Data:** Learn how to convert datasets into external knowledge bases and use RAG to allow the model to answer based on real data.
- **"Expand" & "Calibrate" Data:** Understand the role of synthetic data in cybersecurity scenarios with limited samples. Learn how to generate additional data based on your existing dataset and review AI-generated outputs to ensure logical consistency and high-quality results.

---

## TASK 1 (Achieving): Dataset Collection and Preparation

### Assignment Content

1. **Topic Data Collection**
   - Search for datasets on the same topic on platforms like Hugging Face and examine their data structure and format. Which fields do professional datasets include that you didn't think of? How might these fields help the model's reasoning?
   - Select a cybersecurity-related topic (it can be the same as last week), design the schema of your dataset based on this topic, and explain how the schema helps the model understand the data.
   - Based on your designed fields, collect, clean, and organize an initial version of your dataset.

2. **Knowledge Base Application**
   - Using Open WebUI, convert your dataset into structured Markdown or labeled paragraphs and add them to the knowledge base.
   - Compare the model's responses to the same question **before and after connecting the knowledge base**, observing whether the model can answer accurately based on real data.

### Screenshots to Include

- Field/schema design explanation
- Compared datasets
- Open WebUI knowledge base settings
- Before-and-after comparison of responses

---

## TASK 2 (Exceeding): RAG Workflow Implementation

### Assignment Content

1. Continuing the topic from Task 1 on Hugging Face, implement the RAG workflow through code, understanding the underlying logic and steps (including data reading & chunking, vectorization, retrieval, generation, etc.)

   Reference link: [Hugging Face Advanced RAG Cookbook](https://huggingface.co/learn/cookbook/zh-CN/advanced_rag)

### Screenshots to Include

- RAG implementation code
- Program execution results

---

## TASK 3 (Outstanding): Synthetic Data

### Assignment Content

1. Explain the role and value of synthetic data in the cybersecurity field.
2. Using the dataset schema designed in Task 1, write prompts to guide an LLM to generate 10–20 synthetic records, or use a non-LLM algorithm to generate synthetic data. Explain your design rationale.
3. **Quality calibration and analysis:** From the generated data, select **1 record that is most reasonable** and **1 record that is unreasonable** (AI-generated hallucination or algorithmic flaw), and explain how you would address the issue.

### Screenshots to Include

- Synthetic data generation prompts or algorithm screenshots
- Synthetic data results
- Quality calibration and comparison analysis
