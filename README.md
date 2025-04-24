# LLMs-and-text-simplification

This project explores how large language models (LLMs) simplify complex paragraphs into more elementary versions, focusing on the types of simplification operations they perform.

## 🔍 Project Overview

I designed **five distinct prompts** to instruct different LLMs to simplify 800 advanced-to-elementary paragraph pairs. The goal was to evaluate how each model understands and applies **text simplification operations**, such as lexical simplification, sentence splitting, content removal, and more.

The models evaluated:

- **ChatGPT**
- **LLaMA**
- **Gemini**

Each prompt was run on each LLM across the full dataset of paragraph pairs.

## 📊 Analysis

After collecting the LLM outputs, I analyzed the results by:

- **Comparing model behavior across prompts**  
- **Comparing models against each other**  
- **Comparing all LLM outputs to a human-annotated reference file** that labels the simplification operations used

The analysis includes various visualizations to highlight trends in:

- Operation frequency
- Prompt effectiveness
- Model alignment with human annotations

## 📁 Repository Structure

- `LLM_answers /` : Each different prompt. 
- `analysis/`: Scripts and notebooks for visualization and analysis.

## 🎯 Objective

The ultimate aim is to **evaluate the ability of LLMs to mimic human-like simplification strategies** and determine:

- Which prompts lead to the most accurate or diverse operations
- Which models align best with human-annotated simplification patterns
- Insights into how different LLMs approach simplification as a task


