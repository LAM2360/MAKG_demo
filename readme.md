# MAKG Fault Diagnosis Example

## Overview

This repository provides an example implementation DEMO of the MAKG (Multi-Agent and Synergistic Knowledge Graph) fault diagnosis method. The MAKG approach leverages advanced knowledge graph techniques, embedding models, and retrieval-augmented generation (RAG) to enhance fault diagnosis in mechanical systems. The example demonstrates how to use this method for diagnosing faults in industrial machines by integrating structured knowledge and unstructured data sources.

## Features

- **Knowledge Graph-Based Diagnosis**: The method utilizes a knowledge graph that encodes domain-specific fault patterns, symptoms, and corrective actions.
- **Retrieval-Augmented Generation (RAG)**: Combines knowledge retrieval and generative models for more accurate and context-aware fault diagnosis.
- **Fault Detection and Resolution**: Provides an automated approach to detect and suggest resolutions for mechanical faults based on the available data.
- **Multi-Agent Collaboration**: The system integrates multiple agents for tasks like question rewriting, information extraction, and self-reflection, ensuring precise fault diagnosis.

## Requirements

To run the example code, you need the following software and libraries:

- Python 3.11+
- Conda (for environment management)
- langchain,langchain-community,torch,openai,lightrag-hku are required for pip.
- It is recommended to use gpt-4o for a better experience. If hardware resources are limited, it is recommended to use a quantitative version of LLM, such as ollama, anything LLM.

### Installation

1. Clone the repository:

