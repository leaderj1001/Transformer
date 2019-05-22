# Attention Is All You Need - Transformer

## Network Architecture
![network](https://user-images.githubusercontent.com/22078438/58095633-6c4eca00-7c0e-11e9-81c0-5895af56c566.PNG)
- Positional Encoding
- Multi-Head Attention
- Scaled Dot-Product Attention
- Masked Multi-Head Attention
- Feed Forward
- Add & Norm

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
  
## Masking
- Encoder
  - I also used masking in the encoder section. Because the sentence is less than max_len, <pad> = 1 is inserted, so I masked it.
- Decoder
  - It masks the word because it can not predict the current word by looking at the future word.

## Todo
- example of execute
- code example
- code refactoring & translate code
