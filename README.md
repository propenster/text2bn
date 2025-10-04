# propenster/text2bn powerful GRN Language Model...that rocks!



## Setup...
```bash
conda create -n text2bn pytorch torchdata cudatoolkit -c pytorch
conda install PyPDF2, metapub, tqdm, tokenizers, transformers, pytorch

pip install torch tensorboard torchdata cudatoolkit PyPDF2 metapub tqdm tokenizers transformers SentencePiece

conda activate text2bin

```


* Day 3, I've entered parameter HELL... Time to turn these functions having like 25 arguments into receiving 
*CLI args via argparse... it's all getting messier at this point