---
title: "03. Word Window Classification, Neural Networks, and Matrix Calculus"
metaTitle: "This is the title tag of this page"
metaDescription: "This is the meta description"
---

# Classification:
Generally heres what a 

```
xᵢ = ⎡ ⎤
     ⎢ ⎥
     ⎣ ⎦
```

$a^2 + b^2 = c^2$

$$
f(x) = \int_{-\infty}^\infty\hat f(\xi)\,e^{2 \pi i \xi x}\,d\xi
$$

$$
\frac{1}{\Bigl(\sqrt{\phi \sqrt{5}}-\phi\Bigr) e^{\frac25 \pi}} = 1+\frac{e^{-2\pi}} {1+\frac{e^{-4\pi}} {1+\frac{e^{-6\pi}} {1+\frac{e^{-8\pi}} {1+\cdots} } } }
$$

$$
1 +  \frac{q^2}{(1-q)}+\frac{q^6}{(1-q)(1-q^2)}+\cdots = \prod_{j=0}^{\infty}\frac{1}{(1-q^{5j+2})(1-q^{5j+3})}, \quad\quad \text{for }\lvert q\rvert<1.
$$

$$ f(x) = \int_{-\infty}^\infty\hat f(\xi)\,e^{2 \pi i \xi x}\,d\xi $$



Aim of this section:
1. Softmax function (and not the softmax classifier)
2. Cross entropy Loss

# Softmax and cross entropy
The famous 2 class binary classifier is represented like this:
logit(p)=ln(p/1-p)=w1x1+w2x2+w3x3+c
<-- ???

This section is an excuse to explain softmax, cross entropy loss:
For this discussion we are trying to classify d dim input vector into C classes.

```
  𝑥                                       𝑦̂
⎡ x1 ⎤                                  ⎡ y1 ⎤
⎢ x2 ⎥       ┌──────────────────┐       ⎢ y2 ⎥
⎢ x3 ⎥ ‒‒‒‒≻ │ Classifier Model │ ‒‒‒‒≻ ⎢ y3 ⎥
⎢ ·  ⎥       └──────────────────┘       ⎢ ·  ⎥
⎢ ·  ⎥                                  ⎢ ·  ⎥
⎣ xd ⎦                                  ⎣ yc ⎦

```
As a black box, a classifier model takes an :
- input vector **𝑥ᵢ** (lets say of dimesnion d) 
- and gives an output vector 𝑦̂ᵢ (of dimension equalling number of classes).
- Here each row in the 𝑦̂ gives a probability of 𝑥 belonging to that class 1, 2, ..., c etc. 
- We simply take x to be belonging to the class which has the highest probablity (in above example it x_i would belong to class 2).

A softmax classifier (called so becauses it uses softmax function), 𝑦̂ by a simple matrix multiplication.
```
  𝑥                                                 𝑦̂
⎡ 𝑥₁ ⎤
⎢ 𝑥₂ ⎥                                            ⎡ 𝑦̂₁ ⎤
⎢ 𝑥₃ ⎥ ‒‒‒‒≻ Wx ‒‒‒‒≻ exp() ‒‒‒‒≻ Softmax() ‒‒‒‒≻ ⎢ 𝑦̂₂ ⎥
⎢ ⋮  ⎥                                            ⎢ ⋮   ⎢
⎣ 𝑥d ⎦                                            ⎣ 𝑦̂c ⎦

```

Weight matrix W has dimensions (C,d) i.e. rows = no. of classes and columns = dimension of input vector
Values of matrix W are determined by training the model on input x_i's for whome we know the correct class y_i's, and compare the model's output yHat_i with y_i's and adjusting values of the matrix W.

So at the start of the training we don't know the matrix W so I have arbitrarily chosen it to be all 0s (in practice there are more intelligent ways to initalize, such as Xaver's initalization for e.g.)

### Step 1:
for (training or prediction both) we calculate the dot product W.x. This will give a vector dim (C, 1). <br/>
If 𝑓ᵧ is the yth element of W.x then: <br/>
```
𝑓ᵧ ═ 𝑊ᵧ·𝑥 ═ ∑𝑊ᵧᵢ·𝑥ᵢ

Where 𝑊ᵧ = yth row of W
and 𝑊ᵧᵢ = (y,i) element of W
```

Each 𝑓ᵧ gives a score how much the model thinks x belongs to the yth class.

```
      ⎡ w11 w12 · w1d ⎤ ⎡ x1 ⎤   ⎡ w11·x1＋w12·x2＋⋯＋w1d·xd ⎤
      ⎢ w21 w22 · w2d ⎥ ⎢ x2 ⎥   ⎢ w21·x1＋w22·x2＋⋯＋w2d·xd ⎥
W·x ═ ⎢  ·   ·  ·  ·  ⎥ ⎢ x3 ⎥   ⎢    ·       ·         ·   ⎥
      ⎢ wy1 wy2 · wyd ⎥·⎢ ·  ⎥ ═ ⎢    ·       ·         ·   ⎥
      ⎢  ·   ·  ·  ·  ⎥ ⎢ ·  ⎥   ⎢ wy1·x1＋wy2·x2＋⋯＋wyd·xd ⎥
      ⎣ wc1 wc2 · wcd ⎦ ⎢ ·  ⎥   ⎢    ·       ·         ·   ⎥
                        ⎣ xd ⎦   ⎣ wc1·x1＋wc2·x2＋⋯＋wcd·xd ⎦

      ⎡ Σ w1j·xi ⎤   ⎡ f1 ⎤
      ⎢ Σ w2j·xi ⎥   ⎢ f2 ⎥
      ⎢      ·   ⎥   ⎢ ·  ⎥
    ═ ⎢      ·   ⎥ ═ ⎢ ·  ⎥ 
      ⎢ Σ wyj·xi ⎥   ⎢ fy ⎥
      ⎢      ·   ⎥   ⎢ ·  ⎥
      ⎣ Σ wcj·xi ⎦   ⎣ fc ⎦
```

### Step 2:
Value in each row 1, 2, 3 gives a score which tells how strongly the classifier thinks the input vector x_i belongs to class 1, 2, 3. In this example classifier thinks x_i belongs to class 1, 2, 3 in those order.
But these scores are not probablilites (e.g. they may be greater than 1 or less than zero. Also all 3 dont add up to 1 (coz x_i must belong to one of the 3 classes)). Now to convert these scores into probabiliites we apply a softmax function.

Softmax operator takes a vector of numbers and normalizes them into a probabalities that add up to one.
e.g. Lets say we built classification model (e.g. the logistic one described above) which outputs a score f_y for our input vector xi. 
Each row represents the probabablity of the input x_i belonging to class 0 or 1 or 2.

After softmax operation yHat_i = softmax(f_y) = exp(f_y) / Sigma exp(f_c).
```
Softmax(fy) = exp(fy)/ Σ exp(fc)
```
Applying element wise softmax to the vector W.x we get probababilites of x belonging to each class.
e.g. probability of x belonging to class y: <br/>
```
p(y|x) = softmax(fy) = exp(fy)/ Σ exp(fc) = exp(∑𝑊ᵧ·𝑥)/ Σ exp(∑𝑊꜀·𝑥)

Applied to whole W.x:

    ⎡ softmax(f1) ⎤   ⎡ p(Y=1|x) ⎤
    ⎢ softmax(f2) ⎥   ⎢ p(Y=2|x) ⎥
𝑦̂ ═ ⎢      .      ⎥ ═ ⎢ .        ⎥
    ⎢      .      ⎥   ⎢ .        ⎥
    ⎣ softmax(fc) ⎦   ⎣ p(Y=c|x) ⎦
```


Visually if f = W_y.x = [ f_1, f_2, ..., f_y, ...]^T 
yHat_i = softmax(f)
yHat_i = [exp(f_1) / Sigma exp(f_c), exp(f_2) / Sigma exp(f_c), ..., exp(f_y) / Sigma exp(f_c), ...]^T
yHat_i = 1/(Sigma exp(f_c)) * [exp(f_1), exp(f_1), ..., exp(f_1), ..] ^T
Ok so we have probablities of x_i belonging to each class. 
Now a good classifer model we should give highest probablity to the correct class that x_i belongs to.
And also do so for any input x that we give it in future. (or atleast as try to be as good as it can)

### Step 3:
Now how good or bad our model performs depends on the weight matrix. 
We need to set the values in the weight matrix such that it the model gives highest probababilites to the correct class for each input x.
FOr this we need training data, i.e. x's for which we know the correct class y.
So we can compare the model's probabilites with correct class and adjust weight matrix accordingly.
This is done by loss function. One of them is Cross entropy loss.
For one input x, it is defined as H(p, q) { - SUM true probability * log(predicted probability) } (Summation over class 1..c)
Predicted probability is yHat we get from the model. 
True probability is a one hot vector p=[0,0,0..,1,..0]  (since we already know the correct class for each training input x.)
Therefore H(p,q) = -log(q) {Here q is the prob corresponding to the correct class of x}
This H(p, q) is for one example x. To calculate loss for whole training set we average H(p, q) for all x's belonging to training set.
J(0) = -1/N(SUM -log(q)) = -1/N(SUM -log(exp(fy)/ Σ exp(fc)))


A small example of the same with numbers.
Lets say we have a 3 dimension input we want to classify into 3 classes (i.e. d=3, C=3):
x_i = [1, 2, 3] 

W=[[1 , 2, 3][1, 2, 3][1, 2, 3]] 
f = W_y.x = [1, 2, 3] = [f_1, f_2, f_3] = [Sigma W_1i * xi, Sigma W_2i * xi, Sigma W_3i * xi] = [W_1.x, W_2.x, W_3.x] 
Value in each row 1, 2, 3 gives a score which tells how strongly the classifier thinks the input vector x_i belongs to class 1, 2, 3. In this example classifier thinks x_i belongs to class 1, 2, 3 in those order.
But these scores are not probablilites (e.g. they may be greater than 1 or less than zero. Also all 3 dont add up to 1 (coz x_i must belong to one of the 3 classes)). Now to convert these scores into probabiliites we apply a softmax function.

yHat_i = Softmax(f) = [p(y=1|x_i), p(y=2|x_i), p(y=3|x_i)] = 


(We dont know the values w_ij of W matrix, whole point of building a classification model is to find the matrix W which gives good estimates yHat_i for any given input x_i)


```
 𝑥ᵢ                                   𝑦̂ᵢ
⎡1⎤
⎢2⎥       ┌──────────────────┐       ⎡1⎤
⎢3⎥ ‒‒‒‒≻ │ Classifier Model │ ‒‒‒‒≻ ⎢2⎥
⎢4⎥       └──────────────────┘       ⎣3⎦
⎣5⎦
 

 𝑥ᵢ                 Softmax(𝑊·𝑥ᵢ)           𝑦̂ᵢ

⎡1⎤               ⎛             ⎡1⎤ ⎞
⎢2⎥               ⎜ ⎡1 2 3 4 5⎤ ⎢2⎥ ⎟       ⎡1⎤
⎢3⎥ ‒‒‒‒≻  Softmax⎜ ⎢1 2 3 4 5⎥·⎢3⎥ ⎟ ‒‒‒‒≻ ⎢2⎥
⎢4⎥               ⎜ ⎣1 2 3 4 5⎦ ⎢4⎥ ⎟       ⎣3⎦
⎣5⎦               ⎝             ⎣5⎦ ⎠
 

⎡1⎤
⎢2⎥                      ⎛ ⎡1⎤ ⎞            ⎡1⎤
⎢3⎥   ‒‒‒‒≻       Softmax⎜ ⎢2⎥ ⎟    ‒‒‒‒≻   ⎢2⎥
⎢4⎥                      ⎝ ⎣3⎦ ⎠            ⎣3⎦
⎣5⎦

               ⎡ ⎛       ℯ(1)         ⎞ ⎤
               ⎢ ⎜ ‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒ ⎟ ⎥
               ⎢ ⎝ ℯ(1) + ℯ(2) + ℯ(3) ⎠ ⎥
⎡1⎤            ⎢                        ⎥
⎢2⎥            ⎢ ⎛       ℯ(2)         ⎞ ⎥         ⎡1⎤
⎢3⎥   ‒‒‒‒≻    ⎢ ⎜ ‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒ ⎟ ⎥  ‒‒‒‒≻  ⎢2⎥
⎢4⎥            ⎢ ⎝ ℯ(1) + ℯ(2) + ℯ(3) ⎠ ⎥         ⎣3⎦
⎣5⎦            ⎢                        ⎥
               ⎢ ⎛       ℯ(3)         ⎞ ⎥
               ⎢ ⎜ ‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒ ⎟ ⎥
               ⎣ ⎝ ℯ(1) + ℯ(2) + ℯ(3) ⎠ ⎦ 

```