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
