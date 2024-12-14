import torch
from torchattacks.attack import Attack

class IPN(Attack):
    r"""
    Add Impulse Noise (Random Value Noise).
    altered from torchattack

    `eps` represents the probability of noise
    """
    def __init__(self, model, eps=0.05, **kwargs):
        super().__init__("IPN", model)
        self.eps = eps
        self._supported_mode = ['default']

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        
        # Creating a mask where the noise will be applied
        mask = torch.rand_like(images) < self.eps
        
        # Generating random noise values across the full range of the image data
        random_values = torch.rand_like(images)
        
        # Applying the random value noise only where the mask is True
        adv_images = torch.where(mask, random_values, images)
        
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()  # Ensure pixel values are valid

        return adv_images