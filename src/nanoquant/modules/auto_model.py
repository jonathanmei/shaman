# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch

from ..core.pipeline import run_quantization_pipeline
from ..utils.load_utils import get_compressed_state_dict, load_compressed_model
from ..utils.utils import has_mid_scale


class AutoNQModel():
    def __init__(self):
        self.model = None
        self.quant_config = None

    @classmethod
    def from_pretrained(cls, model_id: str, qmodel_path: str, dtype: torch.dtype = torch.bfloat16,
                        device_map: str = "cuda", quant_config: dict = {}):
        """
        Load quantized checkpoint if exists,
        otherwise quantize the model.
        """
        instance = cls()

        # check if qmodel_path exists
        if qmodel_path:
            if os.path.isfile(qmodel_path):
                model = instance.load_model(model_id, qmodel_path, quant_config, device_map, dtype)
                return model

        # quantize model
        model = instance.quantize_model(model_id, quant_config)
        # save model
        if qmodel_path:
            instance.save_model(model, qmodel_path)
        # return quantized model
        return model

    def quantize_model(self, model_id, quant_config):
        """
        Quantize model (calibration, block reconstruction, model-level KD) via the shared pipeline.
        """
        return run_quantization_pipeline(model_id, quant_config)

    def load_model(self, model_id, qmodel_path, quant_config, device_map, dtype):
        """
        Load quantized model.
        """
        return load_compressed_model(model_name_or_path=model_id, checkpoint_path=qmodel_path,
                                     seqlen=quant_config['seqlen'], has_mid_scale=has_mid_scale(quant_config),
                                     device=device_map, dtype=dtype)

    def save_model(self, model, qmodel_path):
        """
        Save quantized model.
        """
        output_path = Path(qmodel_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        compressed_state_dict = get_compressed_state_dict(model)
        torch.save(compressed_state_dict, qmodel_path)
