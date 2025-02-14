# Structure

1. `upstreaming-vllm` - This is a customization of the Neuron vLLM fork that includes the necessary logic and API support there to host Pixtral. The main path in this directory to look at is `upstreaming-to-villm/vllm/model_executor/modal_loader/neuronx_distributed.py`. 
2. `neuronx-distributed-inference` - This is a fork of NxD Inference to support development on pixtral. The Pixtral modelling code is availble here: `neuronx-distributed-inference/src/neurox_distributed_inference/models/pixtral`. Currently it uses a `relaxed-atten` fork.
3. `import_checker` - This is a scratchpad space to check the imports from `pixtral.py` and suggest paths to keep or replace them.

# Key development steps
1. Add `modeling_mistral.py` to enable support for `MistralForCausalLM`. The language model checkpoint to use is mistralai/Mistral-Large-Instruct-2407 
2. Add `modeling_pixtral.py` which defines the vision encoder and incorporates it with the language model. 