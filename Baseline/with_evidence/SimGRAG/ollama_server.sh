#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export OLLAMA_MODELS=/usr/share/ollama/.ollama/models
export OLLAMA_HOST=http://127.0.0.1:11451

ollama serve