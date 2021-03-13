import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

class Discriminator(nn.Module):
    def __init__(self, in_size, sn=False):
        """
        :param in_size: The size of on input image (without batch dimension).
        :param sn: Wether to use Spectral Normalization of Conv2d layers.
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======

        self.conv_params = dict(kernel_size=4, stride=2, padding=1)
        self.channels = [self.in_size[0], 64, 128, 256]

        feature_layers = []
        for i in range(len(self.channels) - 1):
            if sn == False:
                feature_layers.append(nn.Conv2d(self.channels[i], self.channels[i+1],
                                            		 **self.conv_params))
            else:
                feature_layers.append(nn.utils.spectral_norm(nn.Conv2d(self.channels[i], self.channels[i+1],
                                            		 **self.conv_params)))
            feature_layers.append(nn.BatchNorm2d(self.channels[i+1]))
            feature_layers.append(nn.LeakyReLU(negative_slope=0.1))


        ex = torch.randn([1, self.in_size[0], self.in_size[1], self.in_size[2]])
        self.feature_extractor = nn.Sequential(*feature_layers)
        in_features = self.feature_extractor(ex)

        in_features = in_features.view(1, -1)
        factor = 4
        classifier_layers = [nn.Linear(in_features.shape[1], in_features.shape[1] // factor),
                             nn.LeakyReLU(negative_slope=0.1),
                             nn.Dropout(p=0.2),
                             nn.Linear( in_features.shape[1] // factor, 1)]
        self.classifier = nn.Sequential(*classifier_layers)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        N = x.shape[0]

        features = self.feature_extractor(x)
        features = features.view(N, -1)
        y = self.classifier(features)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        self.channels = [self.z_dim, 1024, 512, 256, 128, out_channels]
        self.kernel = featuremap_size

        padding = 0
        gen_layers = []
        for i in range(len(self.channels) - 1):
            if i == 0:
                padding = 0
            else:
                padding = 1

            gen_layers.append(nn.ConvTranspose2d(self.channels[i], self.channels[i+1],
                                                 self.kernel, stride=2, padding=padding,
                                                 bias=False))
            if out_channels == self.channels[i+1]:
                gen_layers.append(nn.Tanh())
            else:
                gen_layers.append(nn.Dropout(p=0.2))
                gen_layers.append(nn.BatchNorm2d(self.channels[i + 1]))
                gen_layers.append(nn.ReLU())
        self.generator = nn.Sequential(*gen_layers)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        latent_space_samples = torch.randn([n, self.z_dim], requires_grad=with_grad, device=device)
        samples = self.forward(latent_space_samples)

        if with_grad == False:
            samples = samples.detach()
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        z = z.view(z.shape[0], -1, 1, 1)
        x = self.generator(z)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    loss = nn.BCEWithLogitsLoss()

    data_labels = data_label * torch.ones_like(y_data)
    gen_labels = (1 - data_label) * torch.ones_like(y_data)

    data_labels = data_labels + torch.rand_like(data_labels, device=y_data.device) * label_noise - label_noise / 2
    gen_labels = gen_labels + torch.rand_like(gen_labels, device=y_data.device) * label_noise - label_noise / 2

    loss_data = loss(y_data, data_labels)
    loss_generated = loss(y_generated, gen_labels)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    gen_target = torch.ones_like(y_generated) * data_label
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(y_generated, gen_target)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: DataLoader,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    #print(x_data.shape)
    dsc_optimizer.zero_grad()

    real_y = dsc_model(x_data)
    gen_y = dsc_model(gen_model.sample(real_y.shape[0], with_grad=False))

    dsc_loss = dsc_loss_fn(real_y, gen_y)

    dsc_loss.backward()

    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()

    gen_y = dsc_model(gen_model.sample(real_y.shape[0], with_grad=True))

    gen_loss = gen_loss_fn(gen_y)

    gen_loss.backward()

    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    assert len(dsc_losses) == len(gen_losses)
    l = len(dsc_losses)
    if l == 1:
        saved = True
        torch.save(gen_model.state_dict(), checkpoint_file)
    if l >= 2:
        if dsc_losses[-1] < dsc_losses[-2] and gen_losses[-1] < gen_losses[-2]:
            saved = True
            torch.save(gen_model, checkpoint_file)
    # ========================

    return saved

def discriminator_loss_fn_wasserstein(y_data, y_generated, label_noise=0.0):
    disc_loss = - y_data.mean() + y_generated.mean()
    return disc_loss


def generator_loss_fn_wasserstein(y_generated):
    """
    Computes the loss of the generator given generated data using a
    Earth mover's metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :return: The generator loss.
    """
    gen_loss = - y_generated.mean()
    return gen_loss
