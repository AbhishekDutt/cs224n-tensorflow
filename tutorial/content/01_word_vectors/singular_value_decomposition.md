---
title: "Singular Value Decomposition"
metaTitle: "This is the title tag of this page"
metaDescription: "This is the meta description"
---

_(a.k.a. Latent Sematic Analysis or Latent Semantic Indexing in NLP context)_ .<br/>

## Overview
**SVD** states that any matrix can be decomposed into a product of three other matrices. <br/>

This decomposition can also be used to represent a matrix in a lower dimension.<br/>
When SVD is used for dimensionality reduction like this, its called **Truncated SVD**.<br/>
$$
M_\text{(n, d)} \longrightarrow \fbox {\text{Truncated SVD}} \longrightarrow M'_\text{(n, k)}
\\
\text{where: } k << d
$$
Truncated SVD is usually used to reduce the dimension of word embedding matrices such as **co-occurance** or **TF-IDF** matrices discussed earlier.

---

## Code
Before going into the math, Scikit can be used to apply Truncated SVD as follows: <br/>
```
from sklearn.decomposition import TruncatedSVD

k = 2
svd = TruncatedSVD(n_components=k)
svd.fit(M)
M_reduced = svd.transform(M)
```
Here **M** is matrix with $d$ columns, and **M\_reduced** is a matrix with $k$ columns, where k < n. <br/>
_Ref Sklearn [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) for more details._

---

## Manual calculation
Although its unlikely anyone will need to calulate SVD manually, but its still instructive to look at the procedure.

Let's try to reduce the dimension of a word-document matrix[1] described earlier.<br/>
If there were $d$ documents that were tokenized into $n$ unique tokens, the word-document matrix $M$ will be of size $(n, d)$. <br/>
_Remeber each $i^\text{th}$ row of $M$ gives the word vector of $i^\text{th}$ token._ <br/>
$$
M = \underbrace{\begin{bmatrix}
    m_{1, 1} & m_{1, 2} & \cdots & m_{1, d}\\
    m_{2, 1} & m_{2, 2} & \cdots & m_{2, d}\\
    \vdots   & \vdots   & \ddots & \vdots  \\
    m_{n, 1} & m_{n, 2} & \cdots & m_{n, d}\\
    \end{bmatrix}
}_{(n \times d)}
$$

However since the number of documents $d$ is usually very large, matrix $M$ and its word vectors can also be huge.<br/>

Let's select a number $k < d$. Now our aim is to find a matrix $M'$ of size $\text{(n, k)}$.<br/>

SVD says we can decompose any matrix as a product of 3 matrices ($U, S, V^T$):
$$
A_{n \times d} = U_{n \times p} \text{ } S_{p \times p} \text{ } V^T_{p \times d}
$$
Then we truncate $S$ to size $k$ and find $A'$.

Overall it consists of following 5 steps:
1. Find $U$: <br/>
    i. Find the eigenvectors of $A.A^T$. <br/>
    ii. Make these eignevectors orthonormal. <br/>
    iii. Columns of $U$ are these orthonormal eignevectors of $A.A^T$. <br/>
2. Find $V$:  <br/>
    i. Find the eigenvectors of $A^T.A$. <br/>
    ii. Make these eignevectors orthonormal. <br/>
    iii. Columns of $V$ are these orthonormal eignevectors of $A^T.A$. <br/>
3. Find $S$:  <br/>
    i. Find the eigenvalues of $U$ _or_ $V$. <br/>
    ii. $S$ is a diagonal matrix of of square roots of eignevalues in descending order.  <br/>
4. Truncate $S$ to $k$, call it $S'$ <br/>
5. Finally, the required truncated matrix $A' = U.S'$ <br/>

(PS: Both $U$ and $V$ are orthonormal matrices, this means they are composed of vectors which are linearly independent to each other. 
This also means $U.U^T = I$ and $V.V^T = I$. 
$U$ and $V$ will always have same non-zero eigenvalues. )

### Eigenvectors and eigenvalues:
For any **square matrix** $A$, an eigenvector is a non-zero vector that satisfies the equation:
$$
A.\vec x=\lambda \vec x
$$
$\vec x$ is the eigenvector and $\lambda$ is a scalar called eigenvalue. <br/>
(PS: in our case, for $U$: $A = A.A^T$ and for $V$: $A = A^T.A$)

First we solve for eigenvalues $\lambda$.
$$
\begin{aligned}
A.\vec x & = \lambda \vec x \\
\implies (A - \lambda I).\vec x & = 0 \\
\implies A - \lambda I & = 0
\end{aligned}
$$

Then for each non-zero eigenvalue $\lambda_1, \lambda_2, \cdots$, eigenvectors are calulated by solving the following for $\vec x$:
$$
(A - \lambda_1 I).\vec x = 0
$$

### Orthonormal matrices:
Once we have all the eigenvectors for $A.A^T$, First we construct U as columns of eigenvectors.
Now we have to make this matrix U orthonormal. This is done by **Gram-Schmidt Orthonormalization Process**.


Ref:
https://davetang.org/file/Singular_Value_Decomposition_Tutorial.pdf
https://towardsdatascience.com/singular-value-decomposition-example-in-python-dab2507d85a0
https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781783989485/1/ch01lvl1sec21/using-truncated-svd-to-reduce-dimensionality


![Picture of an SVD](../images/svd.png)
