import torch
from torchattacks.attack import Attack

class SPN(Attack):
    r"""
    Add Salt and Pepper Noise.
    altered from torchattack

    `eps` represents the probability of noise
    """
    def __init__(self, model, eps=0.05, **kwargs):
        super().__init__("SPN", model)
        self.eps = eps
        self._supported_mode = ['default']

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        
        # Creating a mask where the noise will be applied
        mask = torch.rand_like(images) < self.eps
        
        # Applying salt and pepper noise
        noise = torch.rand_like(images)
        salt_pepper_noise = torch.where(noise > 0.5, torch.ones_like(images), torch.zeros_like(images))
        adv_images = torch.where(mask, salt_pepper_noise, images)
        
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()  # Ensure pixel values are valid

        return adv_images