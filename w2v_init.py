from gensim.models import Word2Vec

vocab = Word2Vec.load_word2vec_format('/tmp/vectors.bin', binary=True)
vocab.init_sims(replace=True)

vocab[word] # raw numpy vector of a word

