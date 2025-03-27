This an implementation of "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT) By Dosovitskiy et al., I implemented the full paper from scratch using only basic Pytorch building blocks. 


I followed a modular implementation approach, strating with image patchify/unpatchify, the transformer part (With learnable positional encoding and CLS token) and finally the full ViT with the classification head, you can find each under `ViT/architectures`.

I tested each part rigorously againt the already implemented pytorch versions, you can find the full tests under `notebooks`.


For the final test I trained my ViT on the CIFAR10 dataset, you can find the results and the  under `notebooks/ViT_final_test`.

The model performance was poor as expected due to the small dataset and the weak inductive bias of attention, which can be mitigated by larger datasets.  
