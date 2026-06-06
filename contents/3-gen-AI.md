# Generative AI

- Generative AI refers to a class of artificial intelligence models capable of generating new content such as text, images, audio, code, and more. These models learn patterns from existing data and use this knowledge to create coherent and often novel outputs.
  - Text generation (ChatGPT)
  - Image generation (DALL·E, Midjourney)
  - Code generation (Cursor, GitHub Copilot)
  - Video generation (Veo 3 from google)

## Natural Language Processing (NLP)

- Natural Language Processing (NLP) is a subfield of AI and computer science focused on enabling computers to understand, interpret, and generate human language.

### NLP Model Evolution (Shortly)

1. **Rule-Based NLP**

- Uses manually defined linguistic rules.
- Example: Detecting questions by looking for patterns like "Can you", "Will you".

2. **Statistical NLP**

- Learns from data using probabilistic models.
- Example: Part-of-speech tagging using word frequency/context.

3. **Neural NLP (Deep Learning)**

- Uses neural networks (e.g., Transformers).
- Example: Language models like BERT and GPT for translation, QA, summarization.
- **How it works**:

  - Text is first converted into **vector embeddings** — arrays of numbers that represent the meaning of words or sentences.
    - These vectors are **high-dimensional** (often hundreds or thousands of dimensions), not just 2D or 3D.
  - These embeddings capture semantic relationships (e.g., "king" - "man" + "woman" ≈ "queen").
  - The neural network processes these vectors using **matrix operations** and **attention mechanisms** to analyze patterns and context.
  - Final predictions (next word, answer, classification) are based on statistical computations over these numerical vectors.
  - We can use this not only for texts, but also for images, audios, videos, more...

4. **Large Language Models (LLMs)**

- LLMs are an advanced form of Neural NLP using very large Transformer models trained on **massive datasets**.
- Examples: GPT-4, Claude, Gemini, Mistral, LLaMA.
- Capable of **few-shot**, **zero-shot**, and **multi-modal** learning (text, image, code, etc.).
- Represent the current state of the art in generative and understanding capabilities.

## Large Language Model (LLM)

### Masked

- Guess the missing word regardless of its position in a sentence, based on context:
  - Water \_\_\_\_ at 0 degrees Celsius.
  - Water freezes at 0 degrees Celsius.

### Autoregressive (the most well-known models today)

- Predict the next word, based on context:
  - Water freezes at 0 degrees \_\_\_\_.
  - Water freezes at 0 degrees Celsius.

### What is the difference between a Language Model (LM) and a Large Language Model (LLM)?

- Scale: LLMs are much larger in terms of model parameters and training data.
  - LLMs improve over time as new models are trained with more data and better hardware.
- Architecture: LLMs use advanced Transformer architectures.
- Capabilities: LLMs can perform few-shot and zero-shot learning and handle more complex tasks.
  - Few-shot learning: The model can understand and perform a new task after seeing only a few examples.
  - Zero-shot learning: The model can perform a new task without seeing any examples, just from the task description.
  - Handle more complex tasks: LLMs can generate, understand, and reason about complicated content across many domains.

### 1. N-grams

- Estimate a word's probability based on the preceding n-1 words.
  - Example: \_My favorite sport is \_\_\_\_\_
    - n=2 (bigram) → 2-1 = 1 (focus on "is")
    - n=3 (trigram) → 3-1 = 2 (focus on "sport is")
- **Cons**: Limited context, only considers a fixed window of previous words and cannot capture long-range dependencies.

### 2. Recurrent Neural Networks (RNNs)

- Process sequences step-by-step, retaining information from previous steps to consider word order and context.
- Suitable for modeling sequences of variable length.
- **Cons:** Struggle with long-term dependencies due to vanishing or exploding gradients.
  - It makes the model forget information from earlier steps, so it struggles to learn long-term dependencies

### 3. Long Short Term Memory (LSTMs)

- Introduce a gated architecture to control which information to keep or forget, improving memory over longer sequences.
- **Cons:** High computational cost and slower training speed. Difficult to scale for very large datasets.

### 4. Transformers & Attention Mechanism (Attention is All You Need - 2017)

- Use self-attention to weigh the importance of each word or part of sequence (token) relative to others in the sequence when generating output.
  - Achieved by assigning an **attention score** to each word.
- **Pros:** More parallelizable, less computationally intensive than RNNs/LSTMs, and better at capturing long-range dependencies and context.

## Techniques for AI LLM Optimization

### Prompt Engineering

- Prompt engineering is the process of designing and refining input prompts to guide a language model to produce desired and accurate outputs.
- **How it works:**
  1. Write a clear and specific prompt.
  2. Optionally include examples or context.
  3. Adjust wording or structure to influence the model’s response.
  4. Iterate to improve output quality and relevance.
- Doesn’t change the model’s weights or structure.

### Retrieval-Augmented Generation (RAG)

- A method that combines a retrieval system with a generative model. It retrieves relevant documents from an external knowledge base at inference time and uses them to generate more accurate and informed responses without changing the model’s internal weights.
- **How it works:**
  1. User provides a prompt.
  2. Relevant documents are retrieved using vector similarity.
  3. The prompt + documents are passed to a language model.
  4. The model generates a response using both the prompt and retrieved content.
- Doesn’t change the model’s weights. It **uses an external data source** during inference.

### Fine-Tuning

- Fine-tuning is the process of taking a pre-trained model and training it further on a specific smaller dataset to adapt it to a particular task or domain.
- It requires less data and computation than training from scratch because the model already learned general patterns.
- Fine-tuning adjusts the model’s weights to improve performance on the target task while retaining general knowledge.
- **How it works:**
  1. Start with a pre-trained model.
  2. Use a labeled, task-specific dataset.
  3. Train the model to update its weights.
  4. Result: the model performs better on the specific task.
- Updates weight values, **the model changes internally**, but its size usually stays the same.

- **Update weight values of the model means:** Changing the internal numbers (called weights or parameters) that the model uses to make predictions. Each connection between neurons in a network has a weight, and adjusting these weights allows the model to specialize in specific patterns or tasks.

## References

- https://www.ibm.com/think/topics/large-language-models
- https://www.ibm.com/think/topics/natural-language-processing
- https://deepai.org/machine-learning-glossary-and-terms/n-grams
- https://www.ibm.com/think/topics/fine-tuning
- https://huggingface.co/docs/transformers/training
