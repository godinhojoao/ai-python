# Tech stack for AI

- More than only using no code platforms you need to know how to code and manage other tools to tailor AI behavior for your specific case.

## Tech stack

- **Anaconda**: A Python distribution that simplifies package management and deployment for data science and machine learning.
- **Jupyter**: An interactive web-based environment for creating and sharing live code, visualizations, and documentation.
- **Python**: Leading programming language for data science and AI.
  - **Numpy**: Multi-dimensional arrays, matrices, and math functions.
  - **pandas**: Data preprocessing.
  - **Matplotlib**: Data visualization.
  - **scikit-learn**: machine learning library in Python built on numpi, scipy, and matpotlib.
  - **TensorFlow**: TensorFlow is an open-source machine learning and deep learning framework to build and train neural networks.
  - **PyTorch**: PyTorch is an open-source deep learning framework that provides dynamic computation graphs, making it flexible and easy to use for building and training neural networks.
- **APIs**: You need to know how to work with APIs and at least HTTP and API architectures such as Rest APIs, OpenAPI specification, etc.
- **Vector databases**: SQL databases are good to work with rows and columns, but they are not good for working with unstructured data.
  - To manage unstructured data we use **vector embeddings** inside a **vector database**.
  - We store arrays of numbers clustered together based on their similarity.
    - Example: grouping videos based on its genre, vlogs, game videos, tutorials, and more.
    - Videos with similar content would be positioned closer together in the vector space.
  - Vector databases work with **Big data** and we need to query large amounts of data; to deal with it vector databases use ML techniques and indexes.
  - Famous databases: Pinecone, Weaviate, Milvus, Chroma, Elasticsearch.
- **Hugging Face**: Advocate for Open source AI, the GitHub of machine learning and AI. Freely available pre-trained models, fine-tune ML models, and more.
- **LangChain**: Tool to develop AI-powered apps, open source orchestration environment available in Python and JavaScript. Allows to build an app and use any foundation model.
  - LangChain is an open-source framework designed to simplify the development of applications powered by large language models.
  - It connects external data sources with LLMs.
  - Offers a collection of commonly used programming components and patterns to simplify integration between an app and AI, and also to avoid bugs since these pre-built modules are very tested and reliable. (Pre-built components)
- **AI evaluation tools and techniques**: One of the hardest tasks is to evaluate your model. If we skip evaluation phase we will never be sure if it's working as it should.
  - Automated metrics (like BLEU, ROUGE for text), human evaluation, and AI-as-judge approaches.
  - AI-as-judge uses AI models to review and score responses based on quality and relevance.

## References

- https://swagger.io/specification/
- https://huggingface.co/
- https://www.cloudflare.com/learning/ai/what-is-vector-database/
- https://www.langchain.com/
