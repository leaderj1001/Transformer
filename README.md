# Attention Is All You Need - Transformer
[Attention Is All You Need paper](https://arxiv.org/pdf/1706.03762.pdf)

## Network Architecture
![캡처](https://user-images.githubusercontent.com/22078438/58172292-2445ab00-7cd3-11e9-9835-27430e9a33b5.PNG)
- Training:
  - At training, all sentences except the last word of the target sentence are given as output.
- Evaluation:
  - At the time of evaluation, output's input is start "<BOS> token", and output's input is added every time a word comes out. Then "<EOS> token" appears or translate the sentence up to max_len.
  
- Example:
  - Example Sentence: Several women wait outside in a city. (English) -> Mehrere Frauen warten in einer Stadt im Freien. (German)
  - Training:
    - Source sentence: Several women wait outside in a city.
    - Output's input: Mehrere Frauen warten in einer Stadt im
    - Target sentence: Mehrere Frauen warten in einer Stadt im Freien.
  - Evaluation:
    - Source sentence: Several women wait outside in a city.
    - Output's input: ![CodeCogsEqn (3)](https://user-images.githubusercontent.com/22078438/58172791-7cc97800-7cd4-11e9-84fc-ab64f5d58057.gif)

## Positional Encoding
![positionalEncoding](https://user-images.githubusercontent.com/22078438/58095960-feef6900-7c0e-11e9-8f31-082dec0ee4ec.PNG)

- Input data shape: (batch_size, max_len)
- Input Embedding output shape: (batch_size, max_len, embedding_dim)
- Positional Encoding
  - Positional encoding method that sets the position of each word and embedding dimention regardless of input sentence
  - Positional encoding is performed for each sentence length.
- Formula
  - ![CodeCogsEqn](https://user-images.githubusercontent.com/22078438/58171563-42aaa700-7cd1-11e9-8cd7-cb16937e180d.gif)
  - ![CodeCogsEqn (1)](https://user-images.githubusercontent.com/22078438/58171560-41797a00-7cd1-11e9-923c-ac9ff5911217.gif)

## Scaled Dot-Product Attention
![scaled dot-product](https://user-images.githubusercontent.com/22078438/58099380-796fb700-7c16-11e9-9910-3cd7542ca1ad.PNG)
- Matrix multiplication and softmax of query and key, we can see how each word affects other words.

## Multi-Head Attention
![Multi-head](https://user-images.githubusercontent.com/22078438/58099407-855b7900-7c16-11e9-812d-9a699662d91f.PNG)

## Feed Forward
![feedforward](https://user-images.githubusercontent.com/22078438/58099419-8be9f080-7c16-11e9-828e-f015ae3ee575.PNG)

## Add & Norm
- Layer normalization was used in the paper.
- Formula
  - ![CodeCogsEqn (2)](https://user-images.githubusercontent.com/22078438/58171718-91584100-7cd1-11e9-9694-56539b8318c4.gif)
- [Layer Normalization Paper Reference](https://arxiv.org/pdf/1607.06450.pdf)
  
## Masking
- Encoder
  - I also used masking in the encoder section. Because the sentence is less than max_len, <pad> = 1 is inserted, so I masked it.
- Decoder
  - It masks the word because it can not predict the current word by looking at the future word.

## Todo
- example of execute
- code example
- code refactoring & translate code
