AdMIRe: Multimodal Idiom Detection
This project explores two distinct Artificial Intelligence approaches to the AdMIRe task: identifying which of several candidate images best represents the figurative meaning of a specific idiom in context.

The project compares a Generative Vision-Language Model (VLM) approach against a custom Two-Tower Discriminative Model.

Dataset: AdMIRe (Subtask A)
Models: Qwen2-VL, ResNet50, DistilBERT

📂 Project Structure
admire_dataset.py

A universal, modular PyTorch Dataset class.

Handles robust file finding (recursive search), text loading, and preprocessing.

Modes:

mode='qwen': Returns file paths and text prompts (for VLM inference).

mode='fusion': Returns pre-processed image tensors and tokenized text (for Fusion training).

VLM_Implementation.ipynb

Implementation of Qwen2-VL-2B-Instruct using zero-shot inference.

Experiments with low-resolution vs. high-resolution inputs and various prompting strategies (stacked vs. interleaved images).

Fusion_Implementation.ipynb

A custom Two-Tower Network built from scratch using PyTorch.

Combines a Vision Encoder (ResNet50) and a Text Encoder (DistilBERT) to learn semantic similarity.

🚀 Approaches & Findings
1. The Generative Approach (VLM)
We attempted to solve the task using Qwen2-VL-2B-Instruct in a zero-shot setting. The model was provided with the idiom context and 5 candidate images, then prompted to output the index of the correct image.

Outcome: Unsuccessful.

Observed Behavior: The model exhibited severe Mode Collapse, consistently predicting "3" (the middle option) for nearly all inputs.

Analysis:

Positional Bias: Due to the small model size (2B parameters), the model likely struggled to maintain the "reasoning span" required to compare 5 distinct images simultaneously against a complex metaphor. When confused, smaller VLMs often default to the "safe" middle position.

Resolution Trade-offs: At low resolutions (256px), the model was effectively "blind" to the subtle details required for idiom detection. At native resolutions, the cognitive load of processing 5 high-res images appeared to overwhelm the model's context window handling.

2. The Discriminative Approach (Fusion Model)
We built a dedicated Two-Tower Architecture to treat this as a matching problem rather than a generation problem.

Architecture:

Vision Tower: ResNet50 (Pre-trained on ImageNet), with the classification head removed.

Text Tower: DistilBERT (base-uncased).

Fusion Layer: Both towers project to a shared 256-dimensional embedding space. A Dot Product calculates the similarity score between the text vector and each of the 5 image vectors.

Outcome: Highly Successful.

Performance: Achieved an overall accuracy of 95.71% on the validation set.

Result Overview:

Model,Approach,Result,Key Observation

Qwen2-VL-2B,Zero-Shot VLM,Failed,"Suffered from positional bias (predicted ""3"" repeatedly)."
Fusion Net,Two-Tower Embedding,95.71% Accuracy,Successfully learned semantic alignment between figurative language and visual imagery.
