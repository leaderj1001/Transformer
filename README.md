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

## Scaled Dot-Product Attention
![scaled dot-product](https://user-images.githubusercontent.com/22078438/58099380-796fb700-7c16-11e9-9910-3cd7542ca1ad.PNG)

## Multi-Head Attention
![Multi-head](https://user-images.githubusercontent.com/22078438/58099407-855b7900-7c16-11e9-812d-9a699662d91f.PNG)

## Feed Forward
![feedforward](https://user-images.githubusercontent.com/22078438/58099419-8be9f080-7c16-11e9-828e-f015ae3ee575.PNG)
