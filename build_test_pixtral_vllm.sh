#!/bin/bash

'''
This script builds and installs local forks of two libraries, NxD Inference and Neuron vLLM. Then it tries to invoke vLLM using the offline Python invocation.

It is intended to be run after activating a Python virtual environment with the Neuron compiler and SDK depdencies pre-installed.
'''

pip uninstall neuronx-distributed-inference -y

pip uninstall vllm -y

# forces an install from the local fork with Pixtral code
pip install /home/ubuntu/testral/neuronx-distributed-inference

# forces an install from the local fork with Pixtral pointing to NxD I from the local vllm fork
cd upstreaming-to-vllm & git checkout -b v0.6.x-neuron

pip install /home/ubuntu/testral/upstreaming-to-vllm

python run_vllm.py