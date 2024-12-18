# [What Does Vector Mean?](https://www.couchbase.com/blog/what-is-vector-search/)
A vector is a data structure that holds an array of numbers. In our case, this refers to vectors that store a digital summary of the data set they were applied to. It can be thought of as a fingerprint or a summary but formally is called an embedding. Here’s a quick example of how one might look:

```javascript
"blue t-shirts": [-0.02511234  0.05473123 -0.01234567 ...  0.00456789  0.03345678 -0.00789012]
```

# What Is Vector Search ?
Vector search is a technique that uses machine learning to find similar items in large data sets by representing them as vectors. Vectors are numerical representations of words, images, documents, and other content.

# [Vector Embedding](https://www.couchbase.com/blog/what-are-vector-embeddings/)
Vector embeddings are a critical component in machine learning that convert “high-dimensional” information, such as text or images, into a structured vector space

vector embeddings are like translating information we understand into something a computer understands. Imagine you’re trying to explain the concept of “Valentine’s Day” to a computer. Since computers don’t understand concepts like holidays, romance, and the cultural context the way we do, we have to translate them into something they DO understand: numbers. That’s what vector embeddings do. They represent words, pictures, or any kind of data in a list of numbers that represent what those words or images are all about.

For example, with words, if “cat” and “kitten” are similar, when processed through a (large) language model, their number lists (i.e., vectors) will be pretty close together. It’s not just about words, though. You can do the same thing with photos or other types of media. So, if you have a bunch of pictures of pets, vector embeddings help a computer see which ones are similar, even if it doesn’t “know” what a cat is.

Let’s say we’re turning the words “Valentine’s Day” into a vector. The string “Valentine’s Day” would be given to some model, typically an LLM (large language model), which would produce an array of numbers to be stored alongside the words.
```javascript
{
  "word": "Valentine's Day",
  "vector": [0.12, 0.75, -0.33, 0.85, 0.21, ...etc...]
}
```

# How to Create Vector Embeddings
Generally speaking, there are four steps:

1. **Choose Your Vector Embedding Model:** Decide on the type of model based on your needs. Word2Vec, GloVe, and FastText are popular for word embeddings, while BERT and GPT-4 are used for sentence and document embeddings, etc.
2. **Prepare Your Data:** Clean and preprocess your data. For text, this can include tokenization, removing “stopwords,” and possibly lemmatization (reducing words to their base form). For images, this might include resizing, normalizing pixel values, etc.
3. **Train or Use Pre-trained Models:** You can train your model on your dataset or use a pre-trained model. Training from scratch requires a significant amount of data, time, and computational resources. Pre-trained models are a quick way to get started and can be fine-tuned (or augmented) with your specific dataset.
4. **Generate Embeddings:** Once your model is ready, feed your data through it (via SDK, REST, etc.) to generate embeddings. Each item will be transformed into a vector that represents its semantic meaning. Typically, the embeddings are stored in a database, sometimes right alongside the original data.

# [What is a Large Language Model (LLM)?](https://www.couchbase.com/blog/large-language-models-explained/)
What is a Large Language Model (LLM)?
A large language model (LLM) is an artificial intelligence (AI) algorithm trained on large amounts of text data to create natural language outputs. These models have become increasingly popular because they can generate text that sounds just as legitimate as a human would write.

# How Do Large Language Models Work and How Are They Trained?
Large language models are powerful tools that have transformed natural language processing, enabling computers to generate human-like text and provide valuable responses. Let’s explore the key aspects of how these models operate:

1. **Pre-training:** Language models are initially pre-trained on a massive amount of text data from the internet. During pre-training, the model learns to predict the next word in a sentence by analyzing the context of surrounding words. This process helps the model learn grammar, facts, and some level of reasoning.
2. **Fine-tuning:** After pre-training, the model is fine-tuned on more specific tasks using task-specific datasets. Fine-tuning involves further training the model on a narrower dataset, which can be tailored to tasks like question answering, translation, summarization, and sentiment analysis. This step helps the model specialize in the desired task and improves performance.
3. **Attention Mechanism:** The key component of large language models is the attention mechanism within the transformer architecture. Attention allows the model to understand the relative importance of each word in a sentence when generating or predicting words. It helps the model capture long-range dependencies and context while processing text.
4. **Inference:** Once trained, the model can be used for inference. Given a prompt or input text, the model generates a response by predicting the most probable words based on the learned patterns and context from its training.

# [Token Documentation](https://gptforwork.com/guides/openai-gpt3-tokens)
Tokens can be thought of as pieces of words. Before the API processes the request, the input is broken down into tokens. These tokens are not cut up exactly where the words start or end - tokens can include trailing spaces and even sub-words.

- [GPT Tokenizer Playground](https://gptforwork.com/tools/tokenizer)

# Vector Database
A vector database is a type of database optimized to store and query data represented as vectors, which are mathematical arrays of numbers. It is commonly used for tasks like similarity search, where data such as text, images, or audio is converted into numerical vectors, allowing for fast and efficient retrieval based on their proximity or similarity in a multi-dimensional space.

[![Watch the video](https://img.youtube.com/vi/dN0lsF2cvm4/maxresdefault.jpg)](https://www.youtube.com/watch?v=dN0lsF2cvm4)

## Vector indexing
Vector indexing is the process of organizing and storing vectors in a way that makes it fast and efficient to search for similar vectors. It involves creating a data structure that allows quick retrieval of vectors that are close to a given query vector, often based on similarity or distance measures like cosine similarity or Euclidean distance.

- [ Watch this ](https://www.youtube.com/watch?v=dN0lsF2cvm4)
- Common examples
  1. Pinecone
  2. Pg vector
  3. chroma db etc.

 

