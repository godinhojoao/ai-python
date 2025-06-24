# Introductory Summary of AI Concepts

## What is Artificial Intelligence (AI)?

- The ability of machines to perform tasks that typically require human intelligence.
- These machines do not have actual human intelligence or consciousness. Instead, they simulate intelligent behavior using mathematical and statistical techniques.
- AI doesn't understand the world like humans. It learns from large amounts of data and adjusts its internal parameters (like weights in neural networks) to improve performance on specific tasks.
- AI systems work by representing data through vectors, matrices, and other structures, and then applying algorithms to find patterns, correlations, or actions.
- For example:
  - **Vectors** represent features of one data point.
    - In AI, a vector can represent the characteristics of an email, such as the number of links, word count, and presence of certain keywords, used to detect spam.
  - **Matrices** hold multiple data points together.
    - A dataset of thousands of emails, where each row is an email represented by its feature vector, is stored as a matrix for training a spam detection model.
  - **Linear algebra** is used to process and transform these vectors and matrices.
  - **Calculus** helps adjust the model by calculating how changes in parameters affect performance.
  - **Probability** models uncertainty in predictions.
  - **Optimization algorithms** (like gradient descent) improve the AI model by minimizing prediction errors during training.

## AI vs Machine Learning vs Deep Learning vs Data Science

- **Artificial Intelligence**: A broad field that includes many subfields such as machine learning, computer vision, natural language processing, knowledge representation, reasoning, deep learning, and more.
- **Machine Learning**:
  - A major subfield of AI that focuses on creating algorithms that allow systems to automatically learn patterns from data and make decisions or predictions based on it.
  - Instead of being explicitly programmed with fixed rules, the system improves its performance over time as it processes more data.
  - **Example**:
    - A spam filter learns to detect spam emails by analyzing thousands of examples, rather than following a hardcoded list of spam words.
    - The system adjusts its internal parameters (like weights) to minimize errors, a process known as training.
- **Deep learning**
  - is a subfield of **machine learning** that uses neural networks with many layers (called deep neural networks) to automatically learn complex patterns from large amounts of data.
  - **LLMs are based on deep learning**.
- **Data Science** combines techniques from AI, statistics, and domain knowledge to extract insights and knowledge from data.
  - It includes data collection, cleaning, visualization, statistical analysis, and building predictive models.
  - Unlike AI, which focuses on creating intelligent systems, data science focuses more broadly on understanding data and supporting decision-making using math and statistics.

## Weak vs Strong AI

- **Weak AI**: AI systems focused on a specific task, like recommending movies. Also known as "narrow AI."
- **Strong AI (AGI)**: AI that would think and understand like a human, capable of handling a wide range of tasks. It does not exist yet.

## What is Large Language Model (LLM)?

- A type of deep learning model trained on massive amounts of text data to understand and generate human-like language.
  - It uses neural networks, especially transformers, to predict and generate text.
  - LLMs can answer questions, write essays, translate languages, summarize text, and more.
  - Examples: ChatGPT, GPT-4, Claude, Gemini, LLaMA.
- LLMs are part of deep learning, which is under machine learning, and are often used in AI applications involving natural language.

## Data (the most important thing to have great AIs)

### Structured vs Unstructured Data

- **Structured data**: Can be organized into rows and columns, like Excel (.xlsx) files or databases.
- **Unstructured data**: Can't be neatly organized into rows/columns — includes images, audio, video, etc. (makes up 80–90% of all data).
  - Hard to analyze before, but now easier thanks to AI.
  - Example: the MNIST database (handwritten digits).
    - Each image is a grid of 0s and 1s, patterns help identify digits like “3”.
    - Same applies to audio and video, which are also converted to 0s and 1s for analysis.

### Labelled vs unlabelled data

- **Labelled data**: Data with known outputs, like images tagged as "dog" or "not dog", or comments labeled "positive", "neutral", "harmful".
  - More reliable and performs well in real-world use (better model accuracy).
  - Expensive and time-consuming to label.
- **Unlabelled data**: Data without any tags or categories.
  - Quicker to collect
  - Cheaper, but needs special techniques (like clustering or self-supervised learning).

### Metadata

- Data that describes other data, helping to organize and understand it. Used for both structured and unstructured data.
- Especially important for unstructured data (images, videos, audio), where manual labeling is hard or incomplete.
  - Example: An image labeled with size, format, creation date, asset type, author, etc.
- Useful for:
  - Searching and filtering
  - Preprocessing and cleaning
  - Tracking model versions, inputs, outputs
  - Ensuring reproducibility
- **Important**: The rapid digitalization of our lives is what made the big advances of AI possible. Metadata plays a crucial role by making it easier to clean unstructured data for training.

## Machine learning

- A field of AI where computers learn patterns from data to make decisions or predictions without being explicitly programmed.
- Imagine that you have a spreadsheet with many data about real estate transactions of past years, and you want to predict the prices for the next years. This could be made with machine learning.

### Main types of machine learning

- **Supervised**: Train model with labeled data (known outputs) to guide learning.
  - **Cons**: Labeling data is time-consuming and costly.
  - Example: Email spam detection, emails labeled as "spam" or "not spam".
- **Unsupervised**: Learn patterns from unlabeled data, finding structure or groups (clustering).
  - Example: Customer clustering, grouping customers by behavior without predefined labels.
- **Reinforcement learning**: Model learns by trial and error, receiving rewards or penalties based on actions to maximize a goal.
  - Example: A game-playing AI learning to win chess by playing many games and improving over time.

### Deep Learning

- A specialized branch of machine learning that uses **neural networks**, inspired by how the human brain processes information.
- Neural networks are built with layers of **neurons** (nodes) that transform input data step by step to extract patterns and make decisions.

#### Neural Network Structure

- **Layers**: Each layer in a neural network has a specific role in transforming the data step by step.
  `Input layer → Hidden layers → Output layer`
  - **Input layer**: Receives the raw data.
  - **Hidden layers**: Learn increasingly abstract patterns.
  - **Output layer**: Produces the final prediction or classification.
- **Neurons**: Basic units that receive inputs, multiply each by a **weight** (a value that represents the importance of that input), sum them up with a **bias**, apply an **activation function**, and pass the result to the next layer.
  - Weights are real numbers, and are learned during training to adjust influence over outputs.
  - `Positive >0`: encourages the output
  - `Negative <0`: suppresses the output
  - `0`: no influence

#### Example

- Suppose you input an image of the handwritten digit **"3"** (28×28 pixels):

  - The input layer has **784 neurons** (one per pixel).
  - These are converted into a vector and fed into the network.
  - The network adjusts its internal connections (weights) to detect patterns like curves or edges across layers.
  - Deeper layers might recognize complex features like the overall shape of the number.

- **Network width**: The number of neurons in a single layer (e.g., 784 in the input layer).
- **Network depth**: The total number of layers (the more layers, the "deeper" the network).

## AI Fields

- **Computer vision**: Uses machine learning and neural networks to teach computers to extract meaningful information from digital images and videos.

  - **Usecases**: Self-driving cars, medical imaging analysis, security, and more.
  - **Convolutional Neural Networks (CNNs)**: Well-suited for high-dimensional image data. They detect and organize visual patterns by importance and spatial relationships.
  - **Generative Adversarial Networks (GANs)**: Generate realistic images by training two networks in competition.
  - **U-Net**: Neural network with a “U”-shaped architecture for precise image segmentation, capturing detailed and contextual info efficiently, ideal for medical images.

- **Traditional ML Usecases**: Detect fraud in financial institutions, predict customer behavior, price changes, demand, and more.

- **Generative AI**: Creates **new data** based on learned patterns from training data.

  - **Usecases**: Creates or works with text, images, videos, code, design, 3D, and more.
  - **Large Language Models (LLMs)**: Neural networks trained on vast text volumes, learning word relationships and predicting next words.
  - **Diffusion Models**: AI tools to generate images and videos.
  - **Generative Adversarial Networks (GANs)**: Two algorithms compete: one generates content, the other judges its quality.
  - **Hybrid Models**: LLMs combined with GANs for powerful content generation.

- **Robotics**: Combines AI fields in physical machines mimicking human abilities in real-world environments.
  - **Usecases**: Medical surgeries, manufacturing, security, and more.

## References:

- https://aws.amazon.com/what-is/artificial-intelligence/
- https://aws.amazon.com/what-is/large-language-model/
- https://aws.amazon.com/what-is/machine-learning/
- https://www.geeksforgeeks.org/machine-learning/what-is-the-difference-between-labeled-and-unlabeled-data/
- https://www.salesforce.com/blog/what-is-metadata/
- https://www.ibm.com/think/topics/machine-learning-types
- https://www.ibm.com/think/topics/computer-vision
- https://www.geeksforgeeks.org/computer-vision/top-computer-vision-models/
