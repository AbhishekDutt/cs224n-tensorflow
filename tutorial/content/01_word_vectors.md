---
title: "Word Vectors"
metaTitle: "Lecture 1 - Word Vectors"
metaDescription: "All about word vectors"
---

NLP Tasks:
Following are few of the common NLP tasks:
1
2
3
4


# Word Vectors: 
Word vectors are a mathematical representation of written words. <br/>
They are needed as we can't just feed english words into mathematical models directly. <br/>
Essentially word vectors are just a mapping from a word to a vector (hence the name Word Vector). <br/>

Many different ways of creating these Word Vector mappings have been developed. <br/>
As an example, here's what word vectors for the same word "hotel" from two different methods may look like: <br/>
hotel -> [0 0 0 0 0 0 1 0 0 0 0] (10, 1) <br/>
hotel -> [2.32 0.123 -0.645 6.231 8.211] (5, 1) <br/>

Few of the popular word vectors are:
1. Count based Word Vectors:
    1. One hot vector
    2. Word document matrix
    3. Window based co-occurence matrix
    4. TF-IDF
2. Iteration based Word embeddings:
    1. Word2Vec
    2. GLoVe

> PS: When Word Vectors are generated by iteration based methods, they are also called Word embeddings.

Although newer and better methods (e.g. FastText, BERT, GPT-3) are being published regularly, Word2Vec is a great way to learn about the Word embeddings and the workflow of training NLP models.

---

Assuming you already have some vague idea what is a word vector, if you dont this is a word vector:

<TODO: Display a word vector>

Word2Vec is the method by which we get array of numbers corresponding to a list of words.
i.e. Every word in a fixed vocablulary is represnedted by a vector.

A list of such word vectors is called a word embedding, coz these numbers embed the meaning of the word in the vector.


### Word2vec paper suggested 2 methods:
- Continuous Bag Of Words: 
  - Go through each poition 't' in the text, which has a center word 'c' and context ("outside") words "o"
  - Use the similarity of the word vectors for c and o to calculate the probability of o given c (or vice versa in case of Skip gram)
  - Keep adjusting the word vectors to maximize this probability
- Skip-Gram method


History:
- Word2vec (Mikolov et al. 2013) is a framework for learning word vectors. [This is the link to the original paper, go ahead take a look]