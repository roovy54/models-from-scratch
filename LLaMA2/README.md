# LLaMA Architecture

<p align="center">
  <img src="LlaMA_Architecture.png" alt="Alt text">
</p>

## RMS Norm 

Instead of normal layer normalization that uses mean and variance to modify the distribution, In RMS Norm they claim that mean does not play a role in improving the model and it is just the variance that helps and they found a statistic that is devoid of mean which is RMS. Here gamma is a learnable parameter. 

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\large&space;\alpha_i=\frac{1}{\text{RMS}(\alpha)}g_{i}&space;\text{&space;where&space;}&space;\text{RMS}(\alpha)=\sqrt{\frac{1}{n}\sum_{i=1}^n&space;\alpha_i^2}" alt="RMS Alpha Equation">
</p>

## Rotary Positional Encoding 

They replace absolute positional encoding with RoPE embeddings to decrease the attention between two tokens if they are far apart. They do it by finding method to include the relative distance with the dot product of query and key while calculating attention values. 

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\large&space;f_q(\mathbf{x}_m,m)=(\mathbf{W}_q\mathbf{x}_m)e^{im\theta}" alt="Query Function">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\large&space;f_k(\mathbf{x}_n,n)=(\mathbf{W}_k\mathbf{x}_n)e^{in\theta}" alt="Key Function">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\large&space;g(\mathbf{x}_m,\mathbf{x}_n,m-n)=\text{Re}[(\mathbf{W}_q\mathbf{x}_m)(\mathbf{W}_k\mathbf{x}_n)^*e^{i(m-n)\theta}]" alt="Attention Score Function">
</p>


## Grouped Multi Query Attention

GPUs are fast in computing but are comparitively slow in transfer of memory. In regular multihead attention the ratio of computation to memory transfer is very low and it didnt affect the overall time taken but after applying KV Caching this significantly increased the ration thus increasing the time for computation. In order to prevent this we split only the query into n heads instead of keys and values (Multi Query Attention) thus significantly reducing the ratio. In Grouped Mult Query Attention we divide the keys and values in lesser groups than dividing than the number of heads the query is divided into providing significant balance between time and quality.

<p align="center">
  <img src="GMQA.png" alt="Alt text">
</p>

## SwiGLU Activation

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\large&space;\text{SwiGLU}(x,w,v)=\text{Swish}_{\beta}(xW)&space;\otimes&space;(xV)" alt="SwiGLU Equation">

They use SwiGLU activation function by observing performance through experimentation.
</p>
