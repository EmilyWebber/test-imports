# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Iterable, List, Mapping, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mistral_common.protocol.instruct.messages import ImageChunk
from PIL import Image
from transformers import PixtralVisionConfig
from transformers.models.pixtral.image_processing_pixtral import (
    _num_image_tokens as _get_pixtral_hf_num_image_tokens)
from transformers.models.pixtral.modeling_pixtral import (
    PixtralRotaryEmbedding, apply_rotary_pos_emb, position_ids_in_meshgrid)

from vllm.attention import AttentionMetadata
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.inputs import INPUT_REGISTRY,InputContext

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm.multimodal.utils import cached_get_tokenizer
from vllm.sequence import IntermediateTensors, SequenceData

try:
    from xformers import ops as xops
    USE_XFORMERS_OPS = True
except ImportError:
    USE_XFORMERS_OPS = False

##### imports below this line are proposed as a candiate to replace the original methods. they are not confirmed and need to be tested

# from vllm.config import VllmConfig 
from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MultimodalVisionNeuronConfig)


# from vllm.inputs import DecoderOnlyInputs, DummyData, token_inputs
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

# from vllm.model_executor.layers.sampler import get_sampler
# we just call Sampler() in our CausalLM init ln 67 vllm/ .. /neuronx_distributed.py


# from vllm.multimodal import MultiModalKwargs
# we don't have an explicit multimodalkwargs class, we pass everything into the configs

# from .interfaces import SupportsMultiModal, SupportsPP
# these two classes seem to be required in the mainline vllm, but not in the Neuron fork

# from .utils import init_vllm_registered_model
# we can initialize it using the fork

# from .utils import maybe_prefix
# this is used to initialize the LM through vllm, we can use our syntax for that

# from .vision import VisionEncoderInfo
# we can handle metadata for the vision encoder through NeuronConfig etc

###### imports below this line I do not see an easy way to replace ######## 
from .vision import resolve_visual_encoder_outputs 
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.multimodal.inputs import NestedTensors, PlaceholderRange
from vllm.multimodal.utils import consecutive_placeholder_ranges
from .utils import merge_multimodal_embeddings

