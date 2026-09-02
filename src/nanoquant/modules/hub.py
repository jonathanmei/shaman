# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Hugging Face Hub integration for NanoQuant models.

Provides PyTorchModelHubMixin-based classes for native `from_pretrained()` and `push_to_hub()` support.

Usage:
    # Direct usage
    from ..modules.hub import NanoQuantModel
    model = NanoQuantModel.from_pretrained("username/nanoquant-llama-7b-1bpw")
    model.push_to_hub("username/my-nanoquant-model")

    # Via AutoModelForCausalLM (requires proper config.json with auto_map)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("username/nanoquant-llama-7b-1bpw")
"""

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

import torch
import torch.nn as nn
from huggingface_hub import (PyTorchModelHubMixin, create_repo, snapshot_download)

from ..core.pipeline import run_quantization_pipeline
from ..utils.load_utils import (get_compressed_state_dict, load_compressed_model, load_model)
from ..utils.utils import has_mid_scale


@dataclass
class NanoQuantConfigDataclass:
    """Quantization configuration for NanoQuant models."""
    # model id
    model_id: str = "meta-llama/Llama-2-7b-hf"
    # quant precision
    bits: float = 1.0
    # calib
    seed: int = 0
    num_calib_samples: int = 128
    calib_dataset: str = "wikitext2"
    calib_shrinkage: float = 0.4
    calib_strategy: str = "online"
    # curvature estimate: "diag" (legacy per-feature second moments) or "kron" (nearest Kronecker product)
    curvature: str = "diag"
    kron_nkp_iters: int = 3
    kron_stats_device: str = "cpu"
    kron_eigh_dtype: str = "float64"
    # >0: accumulate the dense factors on the GPU for groups of layers fitting this budget (one pass per group)
    kron_gpu_budget_gb: float = 0.0
    seqlen: int = 2048
    device_map: str = "cpu"
    # stage-level artifact cache / resume ("" disables)
    cache_dir: str = "cache"
    checkpoint_every_blocks: int = 1
    # tune_nonfact
    tune_nonfact: bool = True
    nonfact_lr: float = 1e-4
    nonfact_batch_size: int = 4
    nonfact_epochs: int = 8
    # fact (admm)
    admm_type: str = "nanoquant"
    admm_outer_iters: int = 400
    admm_inner_iters: int = 5
    admm_reg: float = 3e-2
    admm_penalty_scheduler: str = "linear"
    admm_print_steps: bool = False
    admm_mid_scale: bool = False
    # tune_fact
    tune_fact: bool = True
    fact_binary_lr: float = 1e-5
    fact_scale_lr: float = 1e-5
    fact_bias_lr: float = 1e-5
    fact_batch_size: int = 1
    fact_epochs: int = 8
    # tune_model
    tune_model: bool = True
    model_kd_lr: float = 1e-5
    model_kd_batch_size: int = 1
    model_kd_epochs: int = 8
    # teacher logits for KD: "ram" (legacy host cache), "disk" (memmap in cache_dir), "online" (recompute)
    model_kd_teacher: str = "ram"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NanoQuantConfigDataclass":
        # Only use keys that are valid fields in the dataclass
        valid_keys = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


class NanoQuantModel(nn.Module, PyTorchModelHubMixin):
    """
    NanoQuant model wrapper with native HuggingFace Hub support.

    This class wraps a quantized model and provides native `from_pretrained()` and
    `push_to_hub()` methods. It also supports `AutoModelForCausalLM.from_pretrained()`
    when the model has proper `auto_map` configuration in `config.json`.

    Usage:
        >>> from ..modules.hub import NanoQuantModel
        >>>
        >>> # Load from Hub
        >>> model = NanoQuantModel.from_pretrained(
        ...     "username/nanoquant-llama-7b-1bpw",
        ...     dtype=torch.bfloat16
        ... )
        >>>
        >>> # Or via AutoModel (requires auto_map in config.json)
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "username/nanoquant-llama-7b-1bpw"
        ... )
        >>>
        >>> # Push to Hub
        >>> model.push_to_hub("username/my-nanoquant-model")
    """
    def __init__(
        self,
        model: torch.nn.Module,
        config: NanoQuantConfigDataclass,
        base_model_id: Optional[str] = None,
    ):
        """
        Initialize the NanoQuant wrapper.

        Args:
            model: The underlying quantized PyTorch model
            config: NanoQuant quantization configuration
            base_model_id: Original model ID (e.g., "meta-llama/Llama-2-7b")
        """
        super().__init__()
        self.model = model
        self._nanoquant_config = config
        self._base_model_id = base_model_id

    @property
    def config(self):
        """Return the underlying model's config."""
        return self.model.config

    @property
    def nanoquant_config(self) -> NanoQuantConfigDataclass:
        """Return the NanoQuant quantization config."""
        return self._nanoquant_config

    def _save_pretrained(self, save_directory: Path, **kwargs):
        """
        Save the model to a local directory.

        Saves:
            - nanoquant_config.json: Quantization parameters
            - config.json: Model config with auto_map for AutoModel support
            - model weights: Model weights using existing packing logic
            - tokenizer files (if present)
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save NanoQuant quantization config
        quant_config_path = save_directory / "nanoquant_config.json"
        with open(quant_config_path, "w") as f:
            json.dump(self._nanoquant_config.to_dict(), f, indent=2)

        # Save base model info
        if self._base_model_id:
            base_config_path = save_directory / "base_model.json"
            with open(base_config_path, "w") as f:
                json.dump({"model_id": self._base_model_id}, f, indent=2)

        # Update and save model config with auto_map
        model_config = dict(self.model.config)
        model_config["auto_map"] = {"AutoModelForCausalLM": "src.modules.hub.NanoQuantModel"}

        # Add quantization params to config for auto-detection
        for key, value in self._nanoquant_config.to_dict().items():
            if not hasattr(self.model.config, key):
                setattr(self.model.config, key, value)

        config_path = save_directory / "config.json"
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)

        # Save weights using existing logic
        try:
            state_dict = get_compressed_state_dict(self.model)

            # Try to save as safetensors first (preferred format)
            try:
                from safetensors.torch import save_file

                # Filter out non-tensor values for safetensors
                tensor_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
                save_file(tensor_dict, str(save_directory / "model.safetensors"))
            except ImportError:
                # Fallback to torch.save
                torch.save(state_dict, save_directory / "model_state.pt")
        except Exception:
            # Fallback to regular state_dict
            state_dict = self.model.state_dict()
            torch.save(state_dict, save_directory / "model_state.pt")

        # Create/Update sharded index if needed
        index_file = save_directory / "model.safetensors.index.json"
        if not index_file.exists():
            try:
                from safetensors.torch import save_file  # noqa: F401

                # Create a minimal index file for single-shard model
                weight_map = {key: "model.safetensors" for key in self.model.state_dict()}
                with open(index_file, "w") as f:
                    json.dump(
                        {
                            "metadata": {
                                "total_size": sum(p.numel() * p.element_size() for p in self.model.parameters())
                            },
                            "weight_map": weight_map
                        }, f, indent=2)
            except ImportError:
                pass  # If safetensors is not available, skip index creation

        # Copy tokenizer files if they exist in the current directory
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json"]
        for tf in tokenizer_files:
            if os.path.exists(tf):
                shutil.copy(tf, save_directory / tf)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        **kwargs,
    ) -> "NanoQuantModel":
        """
        Load model from local path or HuggingFace Hub.
        """
        # 1. Separate quantization-related parameters from kwargs
        # (to prevent conflicts with HuggingFace Hub download logic)
        nanoquant_kwargs = {k: v for k, v in kwargs.items() if k in NanoQuantConfigDataclass.__dataclass_fields__}
        for k in list(nanoquant_kwargs.keys()):
            kwargs.pop(k)

        # Download from Hub if this looks like a Hub model ID
        local_path = pretrained_model_name_or_path
        if not os.path.isdir(pretrained_model_name_or_path):
            try:
                # Use snapshot_download to pull the entire repository securely into cache
                local_dir = snapshot_download(
                    repo_id=pretrained_model_name_or_path,
                    cache_dir=kwargs.get("cache_dir"),
                    token=kwargs.get("token"),
                )
                local_path = local_dir
            except Exception as e:
                raise ValueError(f"Failed to download model from HuggingFace Hub: {e}")

        # 2. First, attempt to load nanoquant_config.json (for newer models)
        quant_config_path = Path(local_path) / "nanoquant_config.json"
        if quant_config_path.exists():
            with open(quant_config_path) as f:
                config_dict = json.load(f)
            config = NanoQuantConfigDataclass.from_dict(config_dict)
        else:
            # Create a default config
            config = NanoQuantConfigDataclass()

        # 3. Attempt to load from config.json (Fallback for older models)
        model_config_path = Path(local_path) / "config.json"
        if model_config_path.exists():
            with open(model_config_path) as f:
                model_config = json.load(f)
            # Apply fallback if the key exists in model_config and hasn't been explicitly set in config
            for key in NanoQuantConfigDataclass.__dataclass_fields__:
                if key in model_config and (not hasattr(config, key)
                                            or getattr(config, key) == getattr(NanoQuantConfigDataclass(), key, None)):
                    setattr(config, key, model_config[key])

        # 4. Final overwrite with kwargs provided directly by the user (highest priority)
        for k, v in nanoquant_kwargs.items():
            setattr(config, k, v)

        # Load model using existing infrastructure
        # For quantized models, we need to load using the compressed model loader
        if os.path.exists(os.path.join(local_path, "model_state.pt")) or os.path.exists(
                os.path.join(local_path, "model.safetensors")):
            # This appears to be a quantized model, load accordingly
            model = load_compressed_model(model_name_or_path=config.model_id, checkpoint_path=local_path,
                                          seqlen=config.seqlen, device=device_map,
                                          has_mid_scale=has_mid_scale(config.to_dict()), dtype=dtype)
        else:
            # This is likely a base model, load normally
            model = load_model(config.model_id, config.seqlen, device_map=device_map)
            model = model.to(dtype)

        # Load base model info (Optional)
        base_model_id = None
        base_config_path = Path(local_path) / "base_model.json"
        if base_config_path.exists():
            with open(base_config_path) as f:
                base_info = json.load(f)
            base_model_id = base_info.get("model_id")
        elif hasattr(config, "model_id"):
            base_model_id = config.model_id

        return cls(model, config, base_model_id=base_model_id)

    @classmethod
    def quantize_model(cls, model_id: str, quant_config: NanoQuantConfigDataclass) -> torch.nn.Module:
        """
        Quantize a model using the NanoQuant pipeline (calibration, block reconstruction, model-level KD),
        with stage-level caching and resume governed by ``quant_config.cache_dir``.

        Args:
            model_id: The model identifier (e.g., "meta-llama/Llama-2-7b-hf")
            quant_config: Quantization configuration parameters

        Returns:
            Quantized PyTorch model
        """
        return run_quantization_pipeline(model_id, quant_config.to_dict())

    @classmethod
    def from_pretrained_quantize(
        cls,
        model_id: str,
        qmodel_path: Optional[str] = None,
        quant_config: Optional[NanoQuantConfigDataclass] = None,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "cuda",
        **kwargs,
    ) -> "NanoQuantModel":
        """
        Load quantized checkpoint if exists, otherwise quantize the model.

        This method provides the same functionality as AutoNQModel.from_pretrained,
        allowing seamless loading of pre-quantized models or on-the-fly quantization.

        Args:
            model_id: Model identifier (e.g., "meta-llama/Llama-2-7b-hf")
            qmodel_path: Path to save/load quantized model checkpoint
            quant_config: Quantization configuration (uses default if None)
            dtype: Torch dtype for model loading
            device_map: Device map for model placement
            **kwargs: Additional arguments for from_pretrained

        Returns:
            NanoQuantModel instance
        """
        # Create default config if not provided
        if quant_config is None:
            quant_config = NanoQuantConfigDataclass()
            # Set model_id from argument
            quant_config.model_id = model_id

        # Check if qmodel_path exists and points to a valid checkpoint
        if qmodel_path and os.path.isfile(qmodel_path):
            logger.info(f"Loading existing checkpoint from {qmodel_path}")
            model = cls._load_checkpoint(qmodel_path, quant_config, device_map, dtype)
            # Note: If loading from checkpoint, tuning is skipped as the model is already quantized
            # To apply tuning, delete the checkpoint file and re-run
            return cls(model, quant_config, base_model_id=model_id)

        # Quantize the model (includes tuning if tune_model=True)
        logger.info(f"Quantizing model: {model_id}")
        model = cls.quantize_model(model_id, quant_config)

        # Save model if path is provided
        if qmodel_path:
            cls._save_checkpoint(model, qmodel_path)
            logger.info(f"Saved quantized model to {qmodel_path}")

        # Create wrapper instance
        return cls(model, quant_config, base_model_id=model_id)

    @classmethod
    def _load_checkpoint(
        cls,
        qmodel_path: str,
        quant_config: NanoQuantConfigDataclass,
        device_map: str,
        dtype: torch.dtype,
    ) -> torch.nn.Module:
        """
        Load a quantized model from checkpoint file.

        Args:
            qmodel_path: Path to the checkpoint file
            quant_config: Quantization configuration
            device_map: Device map for model placement
            dtype: Torch dtype

        Returns:
            The underlying quantized PyTorch model (the caller wraps it in ``NanoQuantModel``).
        """
        quant_dict = quant_config.to_dict()
        return load_compressed_model(model_name_or_path=quant_dict['model_id'], checkpoint_path=qmodel_path,
                                     seqlen=quant_dict['seqlen'], has_mid_scale=has_mid_scale(quant_dict),
                                     device=device_map, dtype=dtype)

    @classmethod
    def _save_checkpoint(cls, model: torch.nn.Module, qmodel_path: str):
        """
        Save quantized model to checkpoint file.

        Args:
            model: The quantized PyTorch model
            qmodel_path: Path where the checkpoint should be saved
        """
        output_path = Path(qmodel_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        compressed_state_dict = get_compressed_state_dict(model)
        torch.save(compressed_state_dict, qmodel_path)

    def _generate_readme(self, repo_id: str) -> str:
        """Generate README.md content with model metadata."""
        model_name = repo_id.split("/")[-1]

        sections = [
            # YAML frontmatter
            "---\n"
            "license: other\n"
            "tags:\n"
            "- quantization\n"
            "- nanoquant\n"
            f"- {self.config.model_type}\n"
            "---\n",

            # Title
            f"# {model_name}\n\n"
            "A NanoQuant quantized model.\n",

            # Quantization config
            "## Quantization Config\n\n" + "\n".join([
                f"- **Model ID**: {self._nanoquant_config.model_id}",
                f"- **Bits**: {self._nanoquant_config.bits}",
                f"- **Sequence Length**: {self._nanoquant_config.seqlen}",
                f"- **Calibration Dataset**: {self._nanoquant_config.calib_dataset}",
                f"- **Calibration Strategy**: {self._nanoquant_config.calib_strategy}",
                f"- **Curvature**: {self._nanoquant_config.curvature}",
                f"- **ADMM Type**: {self._nanoquant_config.admm_type}",
                f"- **ADMM Reg**: {self._nanoquant_config.admm_reg}",
                f"- **Middle Scale**: {has_mid_scale(self._nanoquant_config.to_dict())}",
            ]) + "\n",

            # Usage
            "## Usage\n\n"
            "Load the model using `transformers.AutoModelForCausalLM`:\n\n"
            f"```python\n"
            f"from transformers import AutoModelForCausalLM\n\n"
            f"model = AutoModelForCausalLM.from_pretrained(\"{repo_id}\")\n"
            f"```\n\n"
            "Or with explicit `NanoQuantModel`:\n\n"
            f"```python\n"
            f"from ..modules.hub import NanoQuantModel\n\n"
            f"model = NanoQuantModel.from_pretrained(\"{repo_id}\")\n"
            f"```\n",

            # Original model
            "## Original Model\n\n" +
            (f"Base model: [{self._base_model_id}](https://huggingface.co/{self._base_model_id})\n"
             if self._base_model_id else "- Not specified\n"),

            # Model details
            "## Model Details\n\n" + "\n".join([
                f"- **Model Type**: {self.config.model_type}",
                f"- **Hidden Size**: {getattr(self.config, 'hidden_size', 'N/A')}",
                f"- **Num Attention Heads**: {getattr(self.config, 'num_attention_heads', 'N/A')}",
                f"- **Num Hidden Layers**: {getattr(self.config, 'num_hidden_layers', 'N/A')}",
            ]),

            # Citation
            "## Citation\n\n"
            "If you use this model, please cite the original NanoQuant paper.\n",
        ]

        return "\n\n".join(sections)

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: bool = True,
        commit_message: str = "Push NanoQuant quantized model to Hub",
        private: bool = False,
        token: Optional[str] = None,
        local_dir: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Push the model to HuggingFace Hub.

        Automatically saves quantization config and model weights, then pushes
        to the Hub with appropriate metadata.

        Args:
            repo_id: Repository ID (e.g., "username/my-nanoquant-model")
            use_temp_dir: Use temporary directory for preparation
            commit_message: Git commit message for the push
            private: Create repository as private
            token: HuggingFace API token (uses cached token if None)
            local_dir: Optional alternative directory for local save before push

        Returns:
            URL of the pushed repository

        Example:
            >>> model.push_to_hub("my-nanoquant-7b-1bpw", private=True)
        """
        # Create repo (may error if already exists, that's fine)
        try:
            create_repo(repo_id, exist_ok=True, private=private, token=token)
        except Exception:
            pass  # Repo may already exist

        # Prepare save directory
        if local_dir:
            save_dir = Path(local_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = Path(tempfile.mkdtemp())

        try:
            # Save all files
            self._save_pretrained(save_dir)

            # Add README with model metadata
            readme_content = self._generate_readme(repo_id)

            with open(save_dir / "README.md", "w") as f:
                f.write(readme_content)

            # Upload to Hub
            api_url = super().push_to_hub(
                repo_id=repo_id,
                commit_message=commit_message,
                token=token,
                use_temp_dir=use_temp_dir,
                **kwargs,
            )
            return api_url

        finally:
            # Cleanup temp directory if we created it
            if not local_dir and save_dir.exists():
                shutil.rmtree(save_dir, ignore_errors=True)

    def to(self, *args, **kwargs):
        """Delegate to() call to underlying model."""
        self.model = self.model.to(*args, **kwargs)
        return self

    def cuda(self, **kwargs):
        """Delegate cuda() call to underlying model."""
        self.model = self.model.cuda(**kwargs)
        return self

    def cpu(self):
        """Delegate cpu() call to underlying model."""
        self.model = self.model.cpu()
        return self

    def state_dict(self, *args, **kwargs):
        """Get state dict from underlying model."""
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load state dict into underlying model."""
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)

    def parameters(self, recurse=True):
        """Iterate over model parameters."""
        return self.model.parameters()

    def named_parameters(self, recurse=True):
        """Iterate over named model parameters."""
        return self.model.named_parameters(recurse)

    def modules(self):
        """Iterate over submodules."""
        return self.model.modules()

    def children(self):
        """Iterate over child modules."""
        return self.model.children()
