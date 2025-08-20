# LexiOps: Legal Query Agent
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

LexiOps is a modular legal agent designed to process legal documents, enabling efficient querying and analysis. The current implementation focuses on the EU AI Act but is structured to support other legal domains, such as data laws in Switzerland and the EU, with appropriate evaluation sets to ensure consistent model performance.

## Key Features

### 1. Legal Document Processing
- **ETL Pipeline**: Extracts, transforms, and loads legal PDFs.
- **Chunking and Embedding**: Breaks down documents into manageable chunks, embeds them, and indexes them for efficient retrieval.
- **LangGraph Agent**: Leverages LangGraph to answer user queries based on indexed legal documents.

### 2. Modular Design
- Easily extendable to new tools and configurations.
- Supports quick integration of new LLMs and parameters, triggering new versions for traceability.

### 3. Evaluation and Versioning
- **MLflow Integration**:
  - Ready for evaluation and training workflows.
  - Model versioning tracks parameters and data for full reproducibility.
  - Evaluation set up using LLM-as-a-Judge agents.
- **Data Versioning**:
  - Data snapshots are versioned and will later be managed by DVC.

### 4. Deployment and Tracing
- **LangSmith and LangFuse**: Prepared for deployed model tracing.

## Current Focus
- Processing and querying legal documents related to the EU AI Act.
- Building a foundation for future extensions to other legal domains.

## Future Plans & Improvements
- Implement LoRA for fine-tuning and training managed by MLFlow.
- Integrate DVC for advanced data versioning.
- Enhanced data parsing and smarter chunking tailored for legal documents.
- Development of a user-friendly interface.
- Completion of MLflow integration for seamless evaluation, training, and version tracking.

## Work in Progress
While this project is still under development, a strong foundation has been established:
- Modular structure for easy extensibility.
- Comprehensive versioning for models and data.
- Evaluation framework to ensure consistent performance.

## Getting Started
1. Clone the repository.
2. Install dependencies listed in `requirements.txt`.
3. Run the ETL pipeline to process legal PDFs.
4. Use the LangGraph agent to query processed documents.

### ETL Pipeline
To process documents through the ETL pipeline:
1. Place the desired documents in the `data/raw` directory.
2. Run the ETL job with the following command:
```bash
python -m etl.run
```

## Running the Project

### Chat Functionality
To run the chat agent, execute the following command from the root directory:
```bash
python -m agent.client
```

### Evaluation
To evaluate the model, use:
```bash
python -m agent.score
```

### Configuration
Before running the project, ensure the following:
- Create accounts on Databricks for MLflow, LangSmith, and LangGraph.
- Configure all necessary credentials and settings in the `.env` file.
