# Compute the Bigram LM, of the form “word1, wordenter code here2”, P(word2 | word1)
P(w2 | w1) = P(w1 w2) / P(w1) =
( count(w1 w2) / count(total bigram) ) / ( count(w1)/count(total unigram))

- punctuation does NOT count; so the words is ‘(1991)’ and ‘1991’are the same. You must parse your input: replace all characters not in this set: [a-z, A-Z, 0-9] with spaces.
- all text should be normalized to lowercase
- Ignore lines with less than 3 words.
- Input should be lines of text (separated by new line and/or carriage return)


## Background and Definition: N-Grams

A language models LM describes the probability of words appearing in a sentence or corpus. A unigram LM models the probability of a single word appearing in the corpus, but an n-gram LM models the probability of the n_th word appearing given the words n-1, n-2, … .
See: https://blog.xrds.acm.org/2017/10/introduction-n-grams-need/

Let P(w) be the probability of w:
As an example, given the following corpus: “The Cat in the Hat is the best cat in the hat”, a unigram LM language model would be: (using fractions for clarity)
P(the) = 4/12
P(cat) = 2/12
P(in) = 2/12
P(hat) = 2/12
P(is) = 1/12
P(best) = 1/12

For unigrams, the probability of ‘cat’ appearing anywhere in the corpus is 2/12 using maximum likelihood estimation MLE (a.k.a. word count) - note this is a very simplistic model – the closed universe model.

A bigram (n-gram, n=2) LM:
P(the cat) = 1/11
P(cat in) = 2/11
P(in the) = 2/11
P(the hat) = 2/11
P(hat is) = 1/11
P(is the) = 1/11
P(the best) = 1/11
P(best cat) = 1/11
However, most likely is that we are not interested in the probability of the phrase, but in the conditional probability of ‘cat’ given that the word ‘the’ has been seen. In our example, the probability of ‘cat’ given ‘the’, P(cat|the), is given by Bayes theorem:

P(B given A) = P(A and B) / P(A)

In this project , let’s approximate this using the closed-corpus assumption (no unseen words exist, so no smoothing for those statisticians in class): P(cat|the) = P(the cat) / P(the) = (1/11) / (4/12) = 0.273

#### Job 3 is about: united states x For P( x | united states) = p , find the x with the highest p
