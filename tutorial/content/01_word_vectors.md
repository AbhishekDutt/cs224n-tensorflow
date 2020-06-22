---
title: "1. Word Vectors"
metaTitle: "Lecture 1 - Word Vectors"
metaDescription: "All about word vectors"
---
### Aim of CS224n course:
1. An understanding of effective modern methods for deep learning
    - Basics first, then key methods used in NLP: Recurrent networds, attention, etc.
2. A big picture understanding of human languagegs and the difficulties in understadning and producting them
3. An understanding of and ability to build systems (in PyTorch) for some of the major problems in NLP:
    - Word meaning, dependency parsing, machine translation, question answering

### Aims of this tutorial:
1. Be fast
2. Dont waste time
3. Cover only things that are important from a preffesional point of view and not really needed for academic stuff
4. Dont go deep into academic theory, but go deep if needed
5. Use pictures.
6. As less cognitive load as possible
    - Use pictures if needed
    - Use words if needed

I am a visual learner and find it much easier to learn by seeing things,
than by imagining mathematical symbols in my mind.
Follow a mind map approach. Ask and answer What, Why, How.
Go easy on the math, focus on the practical parts.
Theory without code is not very helpful.

### Here's what we will learn today:
- Word2Vec introduction
- Word2Vec objective function gradients
- Optimizaion basics
- Looking at word vectors

***

"You shall know a word by the company it keeps"
-- J. R. Firth (1957:11)

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