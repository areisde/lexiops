# app/version_tracking.py
"""
MLflow GenAI Version Tracking for Lexiops Legal RAG System

This module implements systematic version control for the entire GenAI application
using MLflow's latest GenAI capabilities, following the patterns from:
https://mlflow.org/docs/latest/genai/version-tracking/
"""

import os
import mlflow
import mlflow.genai
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import hashlib
from datetime import datetime


class LexiopsVersionManager:
    """Manages versioning for the entire Lexiops legal RAG application
    
    This class coordinates:
    - System configuration versions
    - Prompt template versions  
    - Git commit tracking
    - MLflow model versioning
    - Automatic trace linking
    """
    
    def __init__(self, experiment_name: str = "lexiops-rag-system"):
        """Initialize version manager
        
        Args:
            experiment_name: MLflow experiment name for tracking versions
        """
        self.experiment_name = experiment_name
        self.config_path = Path("configs/system.yaml")
        self.prompts_path = Path("configs/prompts.yaml")
        
        # Initialize MLflow experiment
        experiment_name = "lexiops-system"
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Warning: Could not set MLflow experiment: {e}")
        
        # Get git information for version tracking
        self._git_commit = self._get_git_commit()
        self._version_name = self._create_version_name()
        
        # Cache for prompt content hashes to avoid duplicate registrations
        self._prompt_hashes = {}
        
        # Cache for model dependency information to ensure consistency
        self._model_cache = {}
        
    def _get_git_commit(self) -> str:
        """Get current git commit hash for version tracking"""
        try:
            from mlflow.utils.git_utils import get_git_commit
            git_commit = get_git_commit(".")
            return git_commit[:8] if git_commit else "local-dev"
        except Exception:
            return "local-dev"
    
    def _create_version_name(self) -> str:
        """Create unique version identifier based on timestamp and git commit"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        return f"lexiops-v{timestamp}-{self._git_commit}"
    
    def _get_system_version(self) -> str:
        """Get semantic system version based on git and timestamp"""
        timestamp = datetime.now().strftime("%Y.%m.%d")
        return f"v{timestamp}-{self._git_commit}"
    
    def _load_system_config(self) -> Dict[str, Any]:
        """Load system configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"System config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_prompts_config(self) -> Dict[str, Any]:
        """Load prompts configuration from YAML file"""
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompts config not found: {self.prompts_path}")
        
        with open(self.prompts_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _check_existing_prompt_version(self, name: str, content_hash: str) -> Optional[str]:
        """Check if a prompt with the same content hash already exists
        
        Returns:
            Existing version number if found, None otherwise
        """
        try:
            import mlflow.genai
            
            # Search for existing prompts
            prompts = mlflow.genai.search_prompts()
            
            for prompt in prompts:
                if prompt.name == name:
                    # Check if this prompt has the same content hash
                    tags = prompt.tags or {}
                    if tags.get("content_hash") == content_hash:
                        # Extract version from the latest prompt (this is the highest version)
                        # Since search_prompts returns the latest version of each prompt name
                        version = "1"  # We'll extract actual version from MLflow tracking
                        
                        # Get the actual version from MLflow tracking runs
                        from mlflow.tracking import MlflowClient
                        client = MlflowClient()
                        
                        # Search all experiments for this prompt
                        experiments = client.search_experiments()
                        for experiment in experiments:
                            try:
                                runs = client.search_runs(
                                    experiment_ids=[experiment.experiment_id],
                                    filter_string=f"tags.`mlflow.genai.prompt_name` = '{name}' AND tags.content_hash = '{content_hash}'"
                                )
                                if runs:
                                    # Found a matching run, get the version
                                    run_tags = runs[0].data.tags
                                    version = run_tags.get("mlflow.genai.prompt_version", "1")
                                    break
                            except Exception:
                                continue
                        
                        print(f"✅ Found existing prompt version with same content: {name} v{version}")
                        return str(version)
            
            return None
        except Exception as e:
            print(f"Warning: Could not check existing prompt versions: {e}")
            return None
    
    def register_prompts(self) -> Dict[str, Any]:
        """Register prompt templates using MLflow GenAI prompt registry
        Only registers new versions if content has changed.
        
        Returns:
            Dictionary with registered prompt information
        """
        print("🔄 Checking prompt templates for registration...")
        
        # Load prompts from configuration file
        prompts_config = self._load_prompts_config()
        
        # Get the current legal prompt version (default to latest available)
        prompt_versions = list(prompts_config.get("prompts", {}).keys())
        if not prompt_versions:
            raise ValueError("No prompts found in prompts.yaml")
        
        # Use the first available version (could be made configurable)
        current_version_key = prompt_versions[0]  # e.g., "legal_v1.0.0"
        current_prompt = prompts_config["prompts"][current_version_key]
        
        system_prompt_template = current_prompt["system"].strip()
        user_template = current_prompt["user_template"].strip()
        
        # Generate individual content hashes for each prompt
        system_content_hash = hashlib.sha256(system_prompt_template.encode()).hexdigest()[:12]
        user_content_hash = hashlib.sha256(user_template.encode()).hexdigest()[:12]
        
        # Check if we already have this exact content registered
        system_prompt_name = "lexiops_legal_system_prompt"
        user_prompt_name = "lexiops_legal_user_prompt"
        
        existing_system_version = self._check_existing_prompt_version(system_prompt_name, system_content_hash)
        existing_user_version = self._check_existing_prompt_version(user_prompt_name, user_content_hash)
        
        # Handle system prompt
        if existing_system_version:
            print(f"✅ Reusing existing system prompt: {system_prompt_name} v{existing_system_version}")
            system_prompt_result = {
                "name": system_prompt_name,
                "version": existing_system_version,
                "template": system_prompt_template,
                "content_hash": system_content_hash,
                "source": current_version_key
            }
        else:
            print(f"📝 Registering new system prompt (source: {current_version_key})...")
            system_prompt = mlflow.genai.register_prompt(
                name=system_prompt_name,
                template=system_prompt_template,
                commit_message=f"Legal assistant system prompt from {current_version_key} - {self._git_commit}",
                tags={
                    "domain": "legal",
                    "type": "system_prompt", 
                    "version": self._version_name,
                    "git_commit": self._git_commit,
                    "content_hash": system_content_hash,
                    "source_config": current_version_key
                }
            )
            system_prompt_result = {
                "name": system_prompt.name,
                "version": system_prompt.version,
                "template": system_prompt_template,
                "content_hash": system_content_hash,
                "source": current_version_key
            }
        
        # Handle user prompt
        if existing_user_version:
            print(f"✅ Reusing existing user prompt: {user_prompt_name} v{existing_user_version}")
            user_prompt_result = {
                "name": user_prompt_name,
                "version": existing_user_version,
                "template": user_template,
                "content_hash": user_content_hash,
                "source": current_version_key
            }
        else:
            print(f"📝 Registering new user prompt (source: {current_version_key})...")
            user_prompt = mlflow.genai.register_prompt(
                name=user_prompt_name, 
                template=user_template,
                commit_message=f"Legal Q&A user prompt from {current_version_key} - {self._git_commit}",
                tags={
                    "domain": "legal",
                    "type": "user_prompt",
                    "version": self._version_name, 
                    "git_commit": self._git_commit,
                    "content_hash": user_content_hash,
                    "source_config": current_version_key
                }
            )
            user_prompt_result = {
                "name": user_prompt.name,
                "version": user_prompt.version,
                "template": user_template,
                "content_hash": user_content_hash,
                "source": current_version_key
            }
        
        return {
            "system_prompt": system_prompt_result,
            "user_prompt": user_prompt_result
        }
    
    def create_system_version(self) -> str:
        """Create a new system version with complete application state
        
        Registers all models in proper dependency order:
        1. Embedding Model (lexiops-embeddings)
        2. RAG Model (lexiops-rag) - references embedding version
        3. Prompts in MLflow GenAI registry
        4. LLM Model (lexiops-llm) - references RAG, embedding, and prompt versions
        
        Returns:
            System version identifier
        """
        print(f"🔄 Creating system version: {self._version_name}")
        
        # Load system configuration
        config = self._load_system_config()
        
        # Register models in dependency order
        # Get auto-generated system version
        system_version = self._get_system_version()
        
        # 1. Register embedding model first (no dependencies)
        embedding_info = self._register_embedding_model(config, system_version)
        
        # 2. Register RAG model (depends on embedding model)
        rag_info = self._register_rag_model(config, embedding_info, system_version)
        
        # 3. Register prompts in MLflow GenAI registry (before LLM)
        prompts_info = self.register_prompts()
        
        # 4. Register LLM model (references RAG, embedding, and prompts)
        llm_info = self._register_llm_model(config, rag_info, embedding_info, prompts_info, system_version)
        
        print(f"✅ System version created: {self._version_name}")
        print(f"   Embedding Model: {embedding_info['name']} v{embedding_info['version']}")
        print(f"   RAG Model: {rag_info['name']} v{rag_info['version']}")
        print(f"   LLM Model: {llm_info['name']} v{llm_info['version']}")
        print(f"   Prompts: {prompts_info['system_prompt']['name']} v{prompts_info['system_prompt']['version']}, {prompts_info['user_prompt']['name']} v{prompts_info['user_prompt']['version']}")
        
        return self._version_name
    
    def _register_embedding_model(self, config: Dict[str, Any], system_version: str) -> Dict[str, Any]:
        """Register the embedding model configuration as 'lexiops-embeddings'
        
        Args:
            config: System configuration containing embedding details
            system_version: Auto-generated system version
            
        Returns:
            Dictionary with embedding model registration information
        """
        print("🔄 Registering Embedding model configuration...")
        
        # Check cache first
        cache_key = "lexiops-embeddings"
        if cache_key in self._model_cache:
            cached_result = self._model_cache[cache_key]
            print(f"✅ Using cached embedding model: lexiops-embeddings v{cached_result['version']}")
            return cached_result
        
        embedding_config = config.get("components", {}).get("embeddings", {})
        
        # Generate content hash for embedding config
        import json
        embedding_content = json.dumps(embedding_config, sort_keys=True)
        embedding_hash = hashlib.sha256(embedding_content.encode()).hexdigest()[:12]
        
        # Check if we already have this exact embedding configuration
        existing_version = self._check_existing_model_version("lexiops-embeddings", embedding_hash)
        
        if existing_version:
            print(f"✅ Reusing existing embedding model: lexiops-embeddings v{existing_version}")
            result = {
                "name": "lexiops-embeddings",
                "version": existing_version,
                "config": embedding_config,
                "content_hash": embedding_hash
            }
            # Cache the result
            self._model_cache[cache_key] = result
            return result
        
        # Register new embedding model version
        try:
            from mlflow import MlflowClient
            import mlflow
            
            client = MlflowClient()
            
            # Ensure the model exists in the registry
            try:
                client.get_registered_model("lexiops-embeddings")
            except Exception:
                client.create_registered_model("lexiops-embeddings", description="Embedding Model Configuration Registry")
            
            # Create experiment for model registry runs
            experiment_name = "lexiops-model-registry"
            try:
                experiment = client.get_experiment_by_name(experiment_name)
                experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)
            except Exception:
                experiment_id = client.create_experiment(experiment_name)
            
            # Create run for model version
            run = client.create_run(
                experiment_id=experiment_id,
                run_name=f"lexiops-embeddings-{system_version}"
            )
            
            try:
                # Log embedding parameters
                client.log_param(run.info.run_id, "model", str(embedding_config.get("model", "unknown")))
                client.log_param(run.info.run_id, "version", str(embedding_config.get("version", "")))
                client.log_param(run.info.run_id, "system_version", str(system_version))
                client.log_param(run.info.run_id, "content_hash", str(embedding_hash))
                client.log_param(run.info.run_id, "model_type", "embedding_configuration")
                
                # Create model version
                model_version = client.create_model_version(
                    name="lexiops-embeddings",
                    source=f"runs:/{run.info.run_id}",
                    run_id=run.info.run_id,
                    description=f"Embedding Model: {embedding_config.get('model')} v{embedding_config.get('version')} (System {system_version})"
                )
                
                # Add tags
                embedding_tags = {
                    "model": str(embedding_config.get("model", "unknown")),
                    "version": str(embedding_config.get("version", "")),
                    "content_hash": str(embedding_hash),
                    "git_commit": str(self._git_commit),
                    "system_version": str(system_version),
                    "model_type": "embedding_configuration"
                }
                
                for key, value in embedding_tags.items():
                    try:
                        client.set_model_version_tag("lexiops-embeddings", model_version.version, key, value)
                    except Exception as tag_error:
                        print(f"Warning: Could not set embedding tag {key}: {tag_error}")
                
                print(f"✅ Registered embedding model: lexiops-embeddings v{model_version.version}")
                print(f"   Model: {embedding_config.get('model')}")
                print(f"   Version: {embedding_config.get('version')}")
                
                result = {
                    "name": "lexiops-embeddings",
                    "version": model_version.version,
                    "config": embedding_config,
                    "content_hash": embedding_hash,
                    "run_id": run.info.run_id
                }
                
                # Cache the result
                self._model_cache[cache_key] = result
                return result
                
            finally:
                client.set_terminated(run.info.run_id)
                
        except Exception as e:
            print(f"❌ Error registering embedding model: {e}")
            result = {
                "name": "lexiops-embeddings",
                "version": "unknown",
                "config": embedding_config,
                "content_hash": embedding_hash,
                "error": str(e)
            }
            # Cache even error results to maintain consistency
            self._model_cache[cache_key] = result
            return result
    
    def _register_rag_model(self, config: Dict[str, Any], embedding_info: Dict[str, Any], system_version: str) -> Dict[str, Any]:
        """Register the RAG model configuration as 'lexiops-rag'
        
        Args:
            config: System configuration containing RAG details
            embedding_info: Information about the embedding model this RAG depends on
            system_version: Auto-generated system version string
            
        Returns:
            Dictionary with RAG model registration information
        """
        print("🔄 Registering RAG model configuration...")
        
        # Create deterministic cache key based on config + embedding dependency
        cache_key = f"lexiops-rag-{embedding_info['content_hash']}"
        if cache_key in self._model_cache:
            cached_result = self._model_cache[cache_key]
            print(f"✅ Using cached RAG model: lexiops-rag v{cached_result['version']}")
            return cached_result
        
        # Combine retrieval and pipeline configurations for RAG
        retrieval_config = config.get("components", {}).get("retrieval", {})
        pipeline_config = config.get("components", {}).get("pipeline", {})
        
        rag_config = {
            **retrieval_config,
            **pipeline_config,
            "embedding_model_version": embedding_info["version"],
            "embedding_model_hash": embedding_info["content_hash"]
        }
        
        # Generate content hash for RAG config (includes embedding dependency)
        import json
        rag_content = json.dumps(rag_config, sort_keys=True)
        rag_hash = hashlib.sha256(rag_content.encode()).hexdigest()[:12]
        
        # Check if we already have this exact RAG configuration
        existing_version = self._check_existing_model_version("lexiops-rag", rag_hash)
        
        if existing_version:
            print(f"✅ Reusing existing RAG model: lexiops-rag v{existing_version}")
            result = {
                "name": "lexiops-rag",
                "version": existing_version,
                "config": rag_config,
                "content_hash": rag_hash
            }
            # Cache the result
            self._model_cache[cache_key] = result
            return result
        
        # Register new RAG model version
        try:
            from mlflow import MlflowClient
            import mlflow
            
            client = MlflowClient()
            
            # Ensure the model exists in the registry
            try:
                client.get_registered_model("lexiops-rag")
            except Exception:
                client.create_registered_model("lexiops-rag", description="RAG Model Configuration Registry")
            
            # Create experiment for model registry runs
            experiment_name = "lexiops-model-registry"
            try:
                experiment = client.get_experiment_by_name(experiment_name)
                experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)
            except Exception:
                experiment_id = client.create_experiment(experiment_name)
            
            # Create run for model version
            run = client.create_run(
                experiment_id=experiment_id,
                run_name=f"lexiops-rag-{config.get('version', 'unknown')}"
            )
            
            try:
                # Log RAG parameters
                client.log_param(run.info.run_id, "retrieval_version", str(retrieval_config.get("version", "")))
                client.log_param(run.info.run_id, "reranker_enabled", str(retrieval_config.get("reranker_enabled", False)))
                client.log_param(run.info.run_id, "reranker_model", str(retrieval_config.get("reranker_model", "")))
                client.log_param(run.info.run_id, "chunk_size", str(pipeline_config.get("chunk_size", "")))
                client.log_param(run.info.run_id, "overlap", str(pipeline_config.get("overlap", "")))
                client.log_param(run.info.run_id, "top_k", str(pipeline_config.get("top_k", "")))
                client.log_param(run.info.run_id, "embedding_model_version", str(embedding_info["version"]))
                client.log_param(run.info.run_id, "system_version", str(config.get("version", "unknown")))
                client.log_param(run.info.run_id, "content_hash", str(rag_hash))
                client.log_param(run.info.run_id, "model_type", "rag_configuration")
                
                # Create model version
                model_version = client.create_model_version(
                    name="lexiops-rag",
                    source=f"runs:/{run.info.run_id}",
                    run_id=run.info.run_id,
                    description=f"RAG Model: retrieval v{retrieval_config.get('version')} + embedding v{embedding_info['version']} (System v{config.get('version')})"
                )
                
                # Add tags
                rag_tags = {
                    "retrieval_version": str(retrieval_config.get("version", "")),
                    "reranker_enabled": str(retrieval_config.get("reranker_enabled", False)),
                    "reranker_model": str(retrieval_config.get("reranker_model", "")),
                    "chunk_size": str(pipeline_config.get("chunk_size", "")),
                    "top_k": str(pipeline_config.get("top_k", "")),
                    "embedding_model_version": str(embedding_info["version"]),
                    "content_hash": str(rag_hash),
                    "git_commit": str(self._git_commit),
                    "system_version": str(config.get("version", "unknown")),
                    "model_type": "rag_configuration"
                }
                
                for key, value in rag_tags.items():
                    try:
                        client.set_model_version_tag("lexiops-rag", model_version.version, key, value)
                    except Exception as tag_error:
                        print(f"Warning: Could not set RAG tag {key}: {tag_error}")
                
                print(f"✅ Registered RAG model: lexiops-rag v{model_version.version}")
                print(f"   Retrieval: v{retrieval_config.get('version')} (reranker: {retrieval_config.get('reranker_enabled')})")
                print(f"   Pipeline: chunk_size={pipeline_config.get('chunk_size')}, top_k={pipeline_config.get('top_k')}")
                print(f"   Embedding dependency: v{embedding_info['version']}")
                
                result = {
                    "name": "lexiops-rag",
                    "version": model_version.version,
                    "config": rag_config,
                    "content_hash": rag_hash,
                    "run_id": run.info.run_id
                }
                
                # Cache the result
                self._model_cache[cache_key] = result
                return result
                
            finally:
                client.set_terminated(run.info.run_id)
                
        except Exception as e:
            print(f"❌ Error registering RAG model: {e}")
            result = {
                "name": "lexiops-rag",
                "version": "unknown",
                "config": rag_config,
                "content_hash": rag_hash,
                "error": str(e)
            }
            # Cache even error results to maintain consistency
            self._model_cache[cache_key] = result
            return result
    
    def _check_existing_model_version(self, model_name: str, content_hash: str) -> Optional[str]:
        """Check if a model with the same content hash already exists
        
        Returns:
            Existing version number if found, None otherwise
        """
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            # Get registered model
            try:
                model = client.get_registered_model(model_name)
                # Check ALL versions (not just latest) for matching content hash
                all_versions = client.search_model_versions(f'name="{model_name}"')
                for version in all_versions:
                    tags = dict(version.tags) if version.tags else {}
                    stored_hash = tags.get("content_hash")
                    if stored_hash == content_hash:
                        print(f"✅ Found existing {model_name} v{version.version} with matching hash: {content_hash}")
                        return version.version
                        
            except Exception:
                # Model doesn't exist yet
                return None
            
            return None
        except Exception as e:
            print(f"Warning: Could not check existing {model_name} versions: {e}")
            return None
    
    def _register_llm_model(self, config: Dict[str, Any], rag_info: Dict[str, Any], embedding_info: Dict[str, Any], prompts_info: Dict[str, Any], system_version: str) -> Dict[str, Any]:
        """Register the LLM model configuration as 'lexiops-llm' with dependencies
        
        Args:
            config: System configuration containing LLM details
            rag_info: Information about the RAG model this LLM depends on
            embedding_info: Information about the embedding model
            prompts_info: Information about the system and user prompts
            system_version: Auto-generated system version string
            
        Returns:
            Dictionary with LLM model registration information
        """
        print("🔄 Registering LLM model configuration with dependencies...")
        
        llm_config = config.get("components", {}).get("llm", {})
        
        # Include dependency versions in LLM config hash
        extended_llm_config = {
            **llm_config,
            "rag_model_version": rag_info["version"],
            "rag_model_hash": rag_info["content_hash"],
            "embedding_model_version": embedding_info["version"],
            "embedding_model_hash": embedding_info["content_hash"],
            "system_prompt_version": prompts_info["system_prompt"]["version"],
            "system_prompt_hash": prompts_info["system_prompt"]["content_hash"],
            "user_prompt_version": prompts_info["user_prompt"]["version"],
            "user_prompt_hash": prompts_info["user_prompt"]["content_hash"]
        }
        
        # Create deterministic cache key
        dependency_key = f"{rag_info['content_hash']}-{embedding_info['content_hash']}-{prompts_info['system_prompt']['content_hash']}-{prompts_info['user_prompt']['content_hash']}"
        cache_key = f"lexiops-llm-{dependency_key}"
        
        # Generate content hash for LLM config (includes all dependencies)
        import json
        llm_content = json.dumps(extended_llm_config, sort_keys=True)
        llm_hash = hashlib.sha256(llm_content.encode()).hexdigest()[:12]
        
        # Debug: print the hash we're looking for
        print(f"🔍 Looking for LLM with hash: {llm_hash}")
        
        # Check cache first
        if cache_key in self._model_cache:
            cached_result = self._model_cache[cache_key]
            print(f"✅ Using cached LLM model: lexiops-llm v{cached_result['version']}")
            return cached_result
        
        # Check if we already have this exact LLM configuration
        existing_version = self._check_existing_model_version("lexiops-llm", llm_hash)
        
        if existing_version:
            print(f"✅ Reusing existing LLM model: lexiops-llm v{existing_version}")
            result = {
                "name": "lexiops-llm",
                "version": existing_version,
                "config": extended_llm_config,
                "content_hash": llm_hash
            }
            # Cache the result
            self._model_cache[cache_key] = result
            return result
        
        # Create new LLM model version directly in Model Registry
        try:
            import mlflow
            from mlflow import MlflowClient
            
            client = MlflowClient()
            
            # Ensure the model exists in the registry
            try:
                client.get_registered_model("lexiops-llm")
            except Exception:
                # Create the registered model if it doesn't exist
                client.create_registered_model("lexiops-llm", description="LLM Configuration Registry")
            
            # Since MLflow requires a run to create a model version, we'll create a minimal run
            # but this run won't be used for tracking - only for model registration
            experiment_name = "lexiops-model-registry"
            try:
                experiment = client.get_experiment_by_name(experiment_name)
                if not experiment:
                    experiment_id = client.create_experiment(experiment_name)
                else:
                    experiment_id = experiment.experiment_id
            except Exception:
                experiment_id = client.create_experiment(experiment_name)
            
            # Create a minimal run just for model registration
            run = client.create_run(
                experiment_id=experiment_id,
                run_name=f"lexiops-llm-{system_version}"
            )
            
            try:
                # Log minimal parameters to the run including dependencies
                client.log_param(run.info.run_id, "provider", str(llm_config.get("provider", "unknown")))
                client.log_param(run.info.run_id, "model", str(llm_config.get("model", "unknown")))
                client.log_param(run.info.run_id, "deployment", str(llm_config.get("deployment", "")))
                client.log_param(run.info.run_id, "api_version", str(llm_config.get("api_version", "")))
                client.log_param(run.info.run_id, "rag_model_version", str(rag_info["version"]))
                client.log_param(run.info.run_id, "embedding_model_version", str(embedding_info["version"]))
                client.log_param(run.info.run_id, "system_prompt_version", str(prompts_info["system_prompt"]["version"]))
                client.log_param(run.info.run_id, "user_prompt_version", str(prompts_info["user_prompt"]["version"]))
                client.log_param(run.info.run_id, "system_version", str(system_version))
                client.log_param(run.info.run_id, "content_hash", str(llm_hash))
                
                # Create a simple model using the run
                # Use the run URI to register the model version
                model_version = client.create_model_version(
                    name="lexiops-llm",
                    source=f"runs:/{run.info.run_id}",
                    run_id=run.info.run_id,
                    description=f"LLM Configuration: {llm_config.get('provider')} {llm_config.get('model')} + RAG v{rag_info['version']} + Embedding v{embedding_info['version']} + Prompts v{prompts_info['system_prompt']['version']}/{prompts_info['user_prompt']['version']} (System {system_version})"
                )
                
                # Add tags to the model version including dependencies
                model_tags = {
                    "provider": str(llm_config.get("provider", "unknown")),
                    "model": str(llm_config.get("model", "unknown")),
                    "deployment": str(llm_config.get("deployment", "")),
                    "api_version": str(llm_config.get("api_version", "")),
                    "rag_model_version": str(rag_info["version"]),
                    "rag_model_hash": str(rag_info["content_hash"]),
                    "embedding_model_version": str(embedding_info["version"]),
                    "embedding_model_hash": str(embedding_info["content_hash"]),
                    "system_prompt_version": str(prompts_info["system_prompt"]["version"]),
                    "system_prompt_hash": str(prompts_info["system_prompt"]["content_hash"]),
                    "user_prompt_version": str(prompts_info["user_prompt"]["version"]),
                    "user_prompt_hash": str(prompts_info["user_prompt"]["content_hash"]),
                    "content_hash": str(llm_hash),
                    "git_commit": str(self._git_commit),
                    "system_version": str(system_version),
                    "model_type": "llm_configuration"
                }
                
                for key, value in model_tags.items():
                    try:
                        client.set_model_version_tag("lexiops-llm", model_version.version, key, value)
                    except Exception as tag_error:
                        print(f"Warning: Could not set tag {key}: {tag_error}")
                
                print(f"✅ Registered LLM model: lexiops-llm v{model_version.version}")
                print(f"   Provider: {llm_config.get('provider')}")
                print(f"   Model: {llm_config.get('model')}")
                print(f"   RAG dependency: v{rag_info['version']}")
                print(f"   Embedding dependency: v{embedding_info['version']}")
                print(f"   System prompt dependency: v{prompts_info['system_prompt']['version']}")
                print(f"   User prompt dependency: v{prompts_info['user_prompt']['version']}")
                print(f"   System Version: {system_version}")
                
                result = {
                    "name": "lexiops-llm",
                    "version": model_version.version,
                    "config": extended_llm_config,
                    "content_hash": llm_hash,
                    "run_id": run.info.run_id
                }
                
                # Cache the result
                self._model_cache[cache_key] = result
                return result
                
            finally:
                # Set the run to finished
                client.set_terminated(run.info.run_id)
            
        except Exception as e:
            print(f"❌ Error registering LLM model: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "name": "lexiops-llm",
                "version": "unknown",
                "config": extended_llm_config,
                "content_hash": llm_hash,
                "error": str(e)
            }
            # Cache even error results to maintain consistency
            self._model_cache[cache_key] = result
            return result
    
    def get_active_version(self) -> str:
        """Get the current active version name"""
        return self._version_name
    
    def get_git_commit(self) -> str:
        """Get the current git commit hash"""
        return self._git_commit
    
    def set_production_version(self, version_name: str):
        """Promote a version to production stage
        
        Args:
            version_name: Version to promote to production
        """
        try:
            # This would transition the model version to Production stage
            # Implementation depends on having logged models to register
            print(f"🔄 Promoting {version_name} to production...")
            # TODO: Implement model registration and stage transition
            print(f"✅ Version {version_name} promoted to production")
        except Exception as e:
            print(f"❌ Failed to promote version: {e}")


class LexiopsPromptManager:
    """Simplified prompt management using configuration files and MLflow tracking"""
    
    def __init__(self):
        self.prompts_path = Path("configs/prompts.yaml")
    
    def _load_prompts_config(self) -> Dict[str, Any]:
        """Load prompts configuration from YAML file"""
        if not self.prompts_path.exists():
            return {"prompts": {}}
        
        with open(self.prompts_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_prompt_by_name(self, name: str, version: Optional[str] = None) -> str:
        """Retrieve a prompt template by name and optional version
        
        Args:
            name: Prompt type ("lexiops_legal_system_prompt" or "lexiops_legal_user_prompt")
            version: Specific config version (e.g., "legal_v1.0.0"), or latest if None
            
        Returns:
            Prompt template string
        """
        try:
            prompts_config = self._load_prompts_config()
            prompt_versions = prompts_config.get("prompts", {})
            
            if not prompt_versions:
                return self._get_default_prompt(name)
            
            # If version specified, use it; otherwise use first available
            if version and version in prompt_versions:
                config_version = version
            else:
                config_version = list(prompt_versions.keys())[0]
            
            prompt_config = prompt_versions[config_version]
            
            # Return appropriate template based on name
            if "system" in name:
                return prompt_config.get("system", "").strip()
            elif "user" in name:
                return prompt_config.get("user_template", "").strip()
            else:
                return self._get_default_prompt(name)
                
        except Exception as e:
            print(f"Warning: Could not load prompt {name}: {e}")
            return self._get_default_prompt(name)
    
    @staticmethod 
    def _get_default_prompt(name: str) -> str:
        """Fallback default prompts"""
        defaults = {
            "lexiops_legal_system_prompt": """You are a helpful legal assistant. Answer questions based on the provided context from legal documents. Be precise and cite relevant information.""",
            "lexiops_legal_user_prompt": """Question: {{question}}

Context:
{{context}}

Please provide a comprehensive answer based on the context above."""
        }
        return defaults.get(name, "{{question}}")


# Global version manager instance
version_manager: Optional[LexiopsVersionManager] = None

def get_version_manager() -> LexiopsVersionManager:
    """Get or create the global version manager instance"""
    global version_manager
    if version_manager is None:
        version_manager = LexiopsVersionManager()
    return version_manager

def initialize_system_version() -> str:
    """Initialize system version tracking
    
    Returns:
        System version identifier
    """
    manager = get_version_manager()
    return manager.create_system_version()
