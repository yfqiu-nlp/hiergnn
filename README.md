# Abstractive Summarization Guided by Latent Hierarchical Document Structure
Code and materials for the [paper](https://yfqiu.netlify.app/publication/hiergnn/hiergnn.pdf) "Abstractive Summarization Guided by Latent Hierarchical Document Structure". Part of our code is borrowed from [fairseq implementation for BART](https://github.com/facebookresearch/fairseq/tree/main/examples/bart). You can first run the baseline to get familiar with the whole pipeline.

## Basic installations

You first need to install the fairseq by,

    cd fairseq
    pip install --editable ./

You then need to download the official checkpoint for `bart.large` as the backbone for HierGNN-BART from [here](https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.md),

    wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
    tar -xzvf bart.large.tar.gz
    rm bart.large.tar.gz

Please make sure you are using `PyTorch==1.7`.

## Data

### Use our data
You can download our used data from [here](https://drive.google.com/drive/folders/1bUEO5AZ65zfGS8RqdbQrm8hlAv8cW13l?usp=sharing).

### Processing the data by yourself (For CNN/DailyMail as the example)

Alternatively, you can first download the original data (without splitting source article  into sentences) from [here](https://github.com/icml-2020-nlp/semsim). We then use the `sent_tokenize` from [nltk](https://www.nltk.org/api/nltk.tokenize.html) to split the source article into sentences, and add `<cls>` between sentences, with the following command, 

    python3 ssplt.py <input-source-file> <output-processed-file>

For example,

    python3 ssplt.py cnndm-raw/train.source cnndm-ssplit/train.source

Then you can BPE all texts using `hie_bpe.sh` from `cnndm-ssplit`,

      TASK=cnndm-ssplit
      PROG=fairseq/examples/roberta/multiprocessing_bpe_encoder.py

      for SPLIT in train val
      do
         for LANG in source target
         do
         python $PROG \
               --encoder-json hie_encoder.json \
               --vocab-bpe vocab.bpe \
               --inputs "$TASK/$SPLIT.$LANG" \
               --outputs "$TASK/$SPLIT.bpe.$LANG" \
               --workers 60 \
               --keep-empty;
         done
      done


then binarize the dataset with `hie_bin.sh`,

      TASK=cnndm-ssplit
      DICT=checkpoints/dict.source.txt
      fairseq-preprocess \
         --source-lang "source" \
         --target-lang "target" \
         --trainpref "${TASK}/train.bpe" \
         --validpref "${TASK}/val.bpe" \
         --destdir "${TASK}-bin/" \
         --workers 60 \
         --srcdict $DICT \
         --tgtdict $DICT;


## Train

The command for training is:

    sh hie_train.sh

## Valid/Test

The commands for inference is:

    sh hie_test.sh

## Evaluation

For evaluation, we use the ROUGE implementation from [google-research](https://github.com/google-research/google-research/tree/master/rouge), with the following command,

    sh hie_eval.sh

## Released Checkpoints and Outputs

<table>
   <tr>
      <td></td>
      <td></td>
      <td>ROUGE-1</td>
      <td>ROUGE-2</td>
      <td>ROUGE-L</td>
      <td>Checkpoints</td>
      <td>Outputs</td>
   </tr>
   <tr>
      <td>CNN/DailyMail</td>
      <td>BART</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td></td>
      <td>HierGNN-BART</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>XSum</td>
      <td>BART</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td></td>
      <td>HierGNN-BART</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>PubMed</td>
      <td>BART</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td></td>
      <td>HierGNN-BART</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
</table>

## Citation
    @inproceedings{qiu2022hiergnn,
        title={Abstractive Summarization Guided by Latent Hierarchical Document Structure},
        author={Yifu Qiu and Shay Cohen},
        booktitle={The 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP 2022)},
        year={2022}
    }

