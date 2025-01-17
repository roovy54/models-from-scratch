# Conditional GAN (CGAN)

Conditional GAN was introduced to generate outputs according to our choice and provides flexibility for generation. If we have 10 numbers and we want to generate lets say 7 we can condition our model and generate 7.

They convert the labels to embeddings. For discriminator the embeddings are of size image_size * image_size so that the embeddings can be concantenated as a channel in the input. In generator we concatenate the embeddings to the noise. 
