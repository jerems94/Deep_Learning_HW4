r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 256
    hypers['seq_len'] = 100
    hypers['h_dim'] = 128
    hypers['n_layers'] = 5
    hypers['dropout'] = 0.1
    hypers['learn_rate'] = 0.01
    hypers['lr_sched_factor'] = 0.5
    hypers['lr_sched_patience'] = 1

    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = 'ACT I. '
    temperature = 0.03
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
1. First, we split the corpus into sequences because no sequence model can learn from such a long sequence as the corpus.
This first issue may be solved by Truncated BPTT since it solves the vanishing gradients issues.

2. Assuming we want to train the mode on the whole text, it would require to input the whole text sequentially and 
get hidden states for every and single char from the whole text. This means waiting from all previous outputs in order
to compute a specific output.

3. Ignoring the two issues above, training the model on the whole corpus may cause overfitting on the corpus. If the model
has enough parameters, it would be able to memorize the whole sequences or its beginning at least. 
"""

part1_q2 = r"""
**Your answer:**
The generated text shows longer memory than the sequence length because we pass the last model hidden state for sequence 
${i}$ to be the model hidden state at the beginning of sequence ${i+1}$ processing. 

Also, at inference time, we always pass hidden states between character predictions. Thus, the model is not influenced 
by the sequence length.

Moreover, we paid attention to create a batch sampler such that sequences at the same index in batches are consecutive 
in the original corpus.
"""

part1_q3 = r"""
**Your answer:**
We don't shuffle the batch order exactly because of the batch sampler we created.
Thanks to this Sampler, sequences at the same index in consecutive batches are consecutive in the original corpus.
Thus shuffling the batches would create consecutive sequences to be unrelated one to each other.
"""

part1_q4 = r"""
**Your answer:**
1. As we understood, the softmax T function emphasizes the char sampling distribution such that the model will generate 
chars which have the higher probability when T is low. In the other hand, when we train the model, we want the model
to handle more diffuse characters distribution and adapt its weights to these uncertain predictions, then we let a 
higher T.
  
  Let's look at the function : 
  $$
  \mathrm{softmax}_T(\vec{y}) = \frac{e^{\vec{y}/T}}{\sum_k e^{y_k/T}}
  $$
2. When T is very high, $y/T$ becomes a vector of zeros and then $e^(y/T)$  becomes vector of ones.
In the denominator, every element in the sum becomes 1, then every element in the softmax vector tends to the same value.
Then, every character has approximatively the  same probability to be chosen. 

3. When T is very small, the inverse happens. Only the very likely characters are chosen since their probabilities 
are being emphasized to have only one possible char to output.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=32,
        z_dim=256,
        data_label=1,
        label_noise=0.3,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            betas=(0.5, 0.999)

            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            betas=(0.5, 0.999)
            # You an add extra args for the optimizer here
        ),
    )    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
