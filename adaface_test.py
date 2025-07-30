import torch
from head import AdaFace

# typical inputs with 512 dimension
B = 5
embbedings = torch.randn((B, 512)).float()  # latent code
norms = torch.norm(embbedings, 2, -1, keepdim=True)
normalized_embedding  = embbedings / norms
labels =  torch.randint(70722, (B,))

# instantiate AdaFace
adaface = AdaFace(embedding_size=512,
                  classnum=70722,
                  m=0.4,
                  h=0.333,
                  s=64.,
                  t_alpha=0.01,)

# calculate loss
cosine_with_margin = adaface(normalized_embedding, norms, labels)
loss = torch.nn.CrossEntropyLoss()(cosine_with_margin, labels)
