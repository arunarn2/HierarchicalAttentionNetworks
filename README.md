# HierarchicalAttentionNetworks

Hierarchical Attention Networks [HAN](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf) consists of the following  parts:

1. Embedding layer
2. Word Encoder: word level bi-directional GRU to get rich representation of words
3. Word Attention:word level attention to get important information in a sentence
4. Sentence Encoder: sentence level bi-directional GRU to get rich representation of sentences
5. Sentence Attention: sentence level attention to get important sentence among sentences
6. Fully Connected layer + Softmax

These models have 2 levels of attention: one at the word level and one at the sentence level thereby allowing the model to pay less or more attention to individual words and sentences accordingly when constructing the represenation of a document.

![Hierarchical Attn Network](han.png)
