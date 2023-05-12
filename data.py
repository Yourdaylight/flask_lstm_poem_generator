import numpy as np

datas = np.load("tang.npz",allow_pickle=True)
data = datas['data']
ix2word = datas['ix2word'].item()
print(ix2word)
word2ix = datas['word2ix'].item()

print(type(word2ix))
