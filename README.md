# Lexiops Legal RAG System

A comprehensive legal document Retrieval-Augmented Generation (RAG) system with enterprise-grade MLflow version tracking for complete model lifecycle management.

## Overview

Lexiops is a production-ready legal RAG application that implements systematic version control for all GenAI components using MLflow's latest capabilities. The system provides automated dependency tracking, content-based versioning, and comprehensive model registry management for embeddings, RAG configurations, prompts, and LLM models.

## Architecture

### Core Components

**Embedding Model (`lexiops-embeddings`)**
- Manages sentence transformer configurations
- Tracks model parameters and versions
- Foundation for all downstream components

**RAG Model (`lexiops-rag`)**
- Combines retrieval and pipeline configurations
- Includes reranker settings and chunk parameters
- Depends on embedding model versions

**Prompt Templates**
- System and user prompts stored in MLflow GenAI registry
- Content-based versioning prevents duplicate registrations
- Supports legal domain-specific templates

**LLM Model (`lexiops-llm`)**
- Orchestrates Azure OpenAI configurations
- Tracks dependencies on RAG, embedding, and prompt versions
- Maintains complete system state history

### Version Tracking System

The system implements a cascading dependency model where changes propagate automatically:

```
Embedding Model (v1) → RAG Model (v1) → LLM Model (v1)
                                    ↗
                    Prompts (v1) ──→
```

Each component registers in MLflow with:
- Content-based hash identification
- Dependency version tracking
- Git commit correlation
- Automatic reuse detection

## Installation

### Prerequisites

- Python 3.8+
- MLflow 2.9+
- Azure OpenAI access (for LLM components)
- Git repository for version tracking

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd lexiops
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure system settings**
   Edit `configs/system.yaml` with your specific configurations:
   ```yaml
   components:
     embeddings:
       model: "sentence-transformers/all-mpnet-base-v2"
     llm:
       provider: "azure_openai"
       model: "gpt-4o"
       deployment: "your-deployment-name"
       api_version: "2024-06-01"
     retrieval:
       reranker_enabled: true
       reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
     pipeline:
       chunk_size: 1000
       overlap: 150
       top_k: 7
   ```

4. **Configure prompts**
   Customize `configs/prompts.yaml` for your legal domain:
   ```yaml
   prompts:
     legal_v1.0.0:
       system: |
         You are a concise legal assistant.
         Answer strictly from the context. If unknown, say so.
         Cite sources like [Article X] at the end of relevant sentences.
       user_template: |
         Question: {{question}}
         Context: {{context}}
         Instructions:
         - Use only the context above for your response.
         - Include [Article/Section] citations.
   ```

## Usage

### Basic Version Initialization

```python
from app.version_tracking import initialize_system_version

# Register all components and create system version
version = initialize_system_version()
print(f"System version: {version}")
```

### Advanced Version Management

```python
from app.version_tracking import get_version_manager

# Get version manager instance
manager = get_version_manager()

# Create new system version
version = manager.create_system_version()

# Get current git commit
commit = manager.get_git_commit()

# Promote version to production
manager.set_production_version(version)
```

### Prompt Management

```python
from app.version_tracking import LexiopsPromptManager

prompt_manager = LexiopsPromptManager()

# Get system prompt
system_prompt = prompt_manager.get_prompt_by_name("lexiops_legal_system_prompt")

# Get user prompt template
user_prompt = prompt_manager.get_prompt_by_name("lexiops_legal_user_prompt")
```

## MLflow Integration

### Model Registry

The system automatically registers models in MLflow with comprehensive metadata:

**Embedding Models**
- Model architecture and parameters
- Content hash for duplicate detection
- Git commit tracking

**RAG Models**
- Retrieval configuration
- Pipeline parameters (chunk size, overlap, top_k)
- Embedding model dependencies

**LLM Models**
- Provider and model specifications
- Complete dependency tree
- System version correlation

### Tracking and Monitoring

**Automatic Dependency Tracking**
- Changes in embedding models trigger RAG model updates
- RAG model changes propagate to LLM models
- Prompt modifications create new LLM versions

**Content-Based Versioning**
- SHA256 hashing prevents duplicate registrations
- Only actual configuration changes create new versions
- Intelligent caching reduces unnecessary operations

**Git Integration**
- Automatic commit hash correlation
- Version names include timestamp and git reference
- Complete traceability from code to model versions

## Version Management

### Automatic Version Reuse

The system implements intelligent version reuse:

```python
# First run - creates v1 for all components
version_1 = initialize_system_version()

# Second run - reuses all v1 versions (no changes)
version_2 = initialize_system_version()

assert version_1 == version_2  # Same system version
```

### Change Detection

Only actual configuration changes trigger new versions:

- **Configuration Changes**: New embedding model, different LLM parameters
- **Prompt Updates**: Modified system or user prompts
- **Pipeline Modifications**: Updated chunk size, top_k, or reranker settings

### Dependency Propagation

When upstream components change, downstream components automatically update:

1. Embedding model change → New RAG model version
2. RAG model change → New LLM model version
3. Prompt change → New LLM model version
4. Multiple changes → Coordinated version updates

## Configuration Management

### System Configuration (`configs/system.yaml`)

Central configuration file containing:
- Embedding model specifications
- LLM provider settings
- Retrieval and reranker configurations
- Pipeline parameters

### Prompt Configuration (`configs/prompts.yaml`)

Versioned prompt templates:
- System prompts for AI behavior
- User prompt templates with placeholders
- Domain-specific legal instructions

### Environment Variables

Required environment variables:
```bash
# Azure OpenAI (if using)
AZURE_OPENAI_ENDPOINT="your-endpoint"
AZURE_OPENAI_API_KEY="your-api-key"

# MLflow (optional)
MLFLOW_TRACKING_URI="file:./mlruns"  # Default: local storage
```

## Monitoring and Observability

### MLflow UI

Access the MLflow UI for comprehensive tracking:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Navigate to `http://localhost:5000` to view:
- Model registry with all versions
- Experiment runs and parameters
- Model lineage and dependencies
- Performance metrics and logs

### Version History

Track system evolution through:
- Git commit correlation
- Timestamp-based versioning
- Dependency change logs
- Configuration diff tracking

### Model Lifecycle

Monitor model stages:
- Development versions
- Staging validation
- Production deployment
- Archived versions

## Best Practices

### Configuration Changes

1. **Test Locally**: Validate changes before committing
2. **Incremental Updates**: Make small, traceable modifications
3. **Version Documentation**: Use meaningful commit messages
4. **Dependency Awareness**: Understand cascading effects

### Production Deployment

1. **Version Pinning**: Use specific model versions in production
2. **Rollback Strategy**: Maintain previous version accessibility
3. **Monitoring**: Track model performance and behavior
4. **Gradual Rollouts**: Deploy changes incrementally

### Development Workflow

1. **Feature Branches**: Develop changes in isolation
2. **Version Testing**: Validate new versions thoroughly
3. **Merge Strategy**: Coordinate team changes effectively
4. **Documentation**: Update configurations and prompts

## Troubleshooting

### Common Issues

**Duplicate Model Versions**
- Check for configuration drift
- Verify hash calculation consistency
- Review caching behavior

**Missing Dependencies**
- Ensure proper registration order
- Validate component configurations
- Check MLflow connectivity

**Version Mismatch**
- Clear local cache if needed
- Verify git commit tracking
- Review system configuration

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Initialize with debug output
version = initialize_system_version()
```

### MLflow Database

For persistent tracking, configure external database:

```python
import mlflow
mlflow.set_tracking_uri("postgresql://user:pass@host:port/db")
```

## Contributing

### Development Setup

1. Fork the repository
2. Create feature branch
3. Install development dependencies
4. Run tests and validation
5. Submit pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints throughout
- Document public methods
- Include comprehensive tests

### Version Control

- Use conventional commit messages
- Tag releases appropriately
- Maintain changelog
- Document breaking changes

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For questions, issues, or contributions:
- Create GitHub issues for bugs
- Submit feature requests
- Review documentation
- Join community discussions

## Version History

- **v1.0.0**: Initial release with MLflow integration
- **v1.1.0**: Enhanced dependency tracking
- **v1.2.0**: Content-based versioning
- **v1.3.0**: Zero-waste version control

## Related Documentation

- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow GenAI](https://mlflow.org/docs/latest/genai/index.html)
- [Azure OpenAI Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/)
- [Sentence Transformers](https://www.sbert.net/)
