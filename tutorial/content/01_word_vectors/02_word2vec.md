---
title: "Word2Vec"
metaTitle: "Lecture 1 - Word Vectors"
metaDescription: "All about word vectors"
---
_(a.k.a. Frequency based Word Vecators)_

For this section, we assume our courpus is just 3 short documents: <br/>
1. I go there, I stay there.
2. I stay there.
3. Don't go there.

These 3 documents are first split into individual words (or tokens) by a tokenizing algorithms to create a vocabulary.
> PS: The terms 'words' and 'tokens' are sometimes used interchangeably, although many tokens such as "n't" are not words.

Our vocubalary $V$ is a dictionary which maps tokens to their index:

$$
V=\begin{cases}i&:1\\go&:2\\there&:3\\,&:4\\stay&:5\\.&:6\\do&:7\\n't&:8\end{cases}
$$

Also, note that vocabulary size $\vert{V}\vert = 8$.

---
# 1. One hot vector
Simplest of all Word vectors. <br/>
Just set $1$ at the word's index and $0$'s for all other positions. <br/>
E.g. Word vectors for $i$, $there$, $.$ and $n't$ will be:
$$
i = \begin{bmatrix}1\\0\\0\\0\\0\\0\\0\\0\\\end{bmatrix}
there = \begin{bmatrix}0\\0\\1\\0\\0\\0\\0\\0\\\end{bmatrix}
. = \begin{bmatrix}0\\0\\0\\0\\0\\1\\0\\0\\\end{bmatrix}
n't = \begin{bmatrix}0\\0\\0\\0\\0\\0\\0\\1\\\end{bmatrix}
$$

---
# 2. Word document matrix
First we create a word-doucment matrix in which: <br/>
- Row $i$ correspond to token indexes.
- Column $j$ corresponds to the documents 

Matrix is populated one column at a time, by counting the number of times a token $i$ appears in the document $j$. 
Each row in this matrix is the word vector for corresponding token. <br/>

For our 8 token, 3 document corups, the matrix $W$:<br/>
$$
W = \begin{bmatrix}
    2 & 1 & 0\\
    1 & 0 & 1\\
    2 & 1 & 1\\
    1 & 0 & 0\\
    1 & 1 & 0\\
    1 & 1 & 0\\
    0 & 0 & 1\\
    0 & 0 & 1\\
    \end{bmatrix}
$$
And the word vectors for $i$, $there$, $.$ and $n't$ (with indexes 1, 3, 6 and 8) are the rows 1, 3, 6 and 8 of $W$: <br/>
$$
i = \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix}
there = \begin{bmatrix} 2 \\ 1 \\ 1 \end{bmatrix}
. = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}
n't = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$

---
# 3. Window based co-occurence matrix
We count the number of times two words occur together (within a maximum distance called the window size).
For e.g. window size 2 of $\text{I}$ includes 2 words before and after $\text{I}$:

$$
\text{I go }\underbrace{\text{there ,}}_{\text{Window}}\text{ I }\underbrace{\text{stay there}}_{\text{Window}}\text{ .}
$$

Then for all documents, we count the number of times the center word and window word occurs together and tabulate it in a matrix.
This will always be a symmetric matrix.

Recall our example corpus: <br/>
1. I go there , I stay there .<br/>
2. I stay there . <br/>
3. Do n't go there . <br/>

We create a matrix by counting the number of times center token $i$ (row) and other token $j$ (column) occurs in window.
E.g. using window size 2, the word $\text{there}$ occurs 3 times in $\text{I}$'s window (and vice-versa).
$$
W = \begin{bmatrix}
          & i & go & there & , & stay & . & do & n't \\
    i     & 0 & 1  & 4     & 1 & 2    & 0 & 0  & 0   \\
    go    & 1 & 0  & 2     & 1 & 0    & 1 & 1  & 1   \\
    there & 4 & 2  & 0     & 1 & 2    & 3 & 0  & 1   \\
    ,     & 1 & 1  & 1     & 0 & 1    & 0 & 0  & 0   \\
    stay  & 2 & 0  & 2     & 1 & 0    & 2 & 0  & 0   \\
    .     & 0 & 1  & 3     & 0 & 2    & 0 & 0  & 0   \\
    do    & 0 & 1  & 0     & 0 & 0    & 0 & 0  & 1   \\
    n't   & 0 & 1  & 1     & 0 & 0    & 0 & 1  & 0   \\
    \end{bmatrix}
$$
Either the row or the column can taken as word vetctor for corresponding token id. <br/>
e.g. For $i$, $there$, $.$ and $n't$:
$$
i = \begin{bmatrix}0\\1\\4\\1\\2\\0\\0\\0\\\end{bmatrix}
there = \begin{bmatrix}4\\2\\0\\1\\2\\3\\0\\1\\\end{bmatrix}
. = \begin{bmatrix}0\\1\\3\\0\\2\\0\\0\\0\\\end{bmatrix}
n't = \begin{bmatrix}0\\1\\1\\0\\0\\0\\1\\0\\\end{bmatrix}
$$

---
# 4. TF-IDF

<TODO: FINISH TF-IDF> <br/>
<TODO: Give disadvantages of each> <br/>
<TODO: Add notes about SVD> <br/>

