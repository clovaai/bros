## Fine-tuning examples

#### FUNSD EE - BIO (BIES) tagger

##### Prepare data

We conducted the FUNSD EE experiment based on the FUNSD data preprocessed in LayoutLM.
Original code can be found in [this link](https://github.com/microsoft/unilm/tree/master/layoutlm/deprecated/examples/seq_labeling).
To run it, please follow the steps below:

1) move to `preprocess/funsd/`.
2) run `bash preprocess.sh`.
3) run `preprocess_2nd.py`. This scripts converts the preprocessed data in LayoutLM to fit this repo.
 
Data will be created in `datasets/funsd/`.

##### Perform fine-tuning

1) Run the command below:
```
CUDA_VISIBLE_DEVICES=0 python train.py --config=configs/finetune_funsd_ee_bies.yaml
```
2) Evaluate the model
```
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=configs/finetune_funsd_ee_bies.yaml --pretrained_model_file=finetune_funsd_ee_bies__bros-base-uncased/checkpoints/epoch=99-last.pt
```

---

#### FUNSD EE - SPADE decoder

##### Prepare data

1) move to `preprocess/funsd_spade/`.
2) run `preprocess.py`.

Data will be created in `datasets/funsd_spade`.

##### Perform fine-tuning

1) Run the command below:
```
CUDA_VISIBLE_DEVICES=0 python train.py --config=configs/finetune_funsd_ee_spade.yaml
```
2) Evaluate the model
```
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=configs/finetune_funsd_ee_spade.yaml --pretrained_model_file=finetune_funsd_ee_spade__bros-base-uncased/checkpoints/epoch=99-last.pt
```

---

#### FUNSD EL - SPADE decoder

##### Prepare data

Same as above.
1) move to `preprocess/funsd_spade/`.
2) run `preprocess.py`.

Data will be created in `datasets/funsd_spade`.

##### Perform fine-tuning

1) Run the command below:
```
CUDA_VISIBLE_DEVICES=0 python train.py --config=configs/finetune_funsd_el_spade.yaml
```
2) Evaluate the model
```
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=configs/finetune_funsd_el_spade.yaml --pretrained_model_file=finetune_funsd_el_spade__bros-base-uncased/checkpoints/epoch=99-last.pt
```

---

#### SROIE* EE - BIO tagger

In the original SROIE task, semantic contents (Company, Date, Address, and Total price) are generated without explicit connection to the text blocks. To convert SROIE into a EE task, we developed SROIE* by matching ground truth contents with text blocks. We also split the original training set into 526 training and 100 testing examples because the ground truths are not given in the original test set (now it is opened).

#### Preprae data

1) Download SROIE* ([sroie.tar.gz](https://drive.google.com/file/d/1D47qwhOki_NlUFufnE9Y7Rs0YH_VKlmH))
2) Extract `sroie.tar.gz` to `datasets/`. For the image files, you need to download from the [official website](https://rrc.cvc.uab.es/?ch=13).

##### Perform fine-tuning

1) Run the command below:
```
CUDA_VISIBLE_DEVICES=0 python train.py --config=configs/finetune_sroie_ee_bio.yaml
```
2) Evaluate the model
```
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=configs/finetune_sroie_ee_bio.yaml --pretrained_model_file=finetune_sroie_ee_bio__bros-base-uncased/checkpoints/epoch=29-last.pt
```
