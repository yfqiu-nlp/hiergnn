# Abstractive Summarization Guided by Latent Hierarchical Document Structure
Code and materials for the [paper](https://yfqiu.netlify.app/publication/hiergnn/hiergnn.pdf) "Abstractive Summarization Guided by Latent Hierarchical Document Structure". Part of our code is borrowed from [fairseq implementation for BART](https://github.com/facebookresearch/fairseq/tree/main/examples/bart). You can first run the baseline to get familiar with the whole pipeline.

## Data

You can download our used data from [here]().

## Train

The command for training is:

    sh train_hier.sh

## Valid/Test

The commands for inference is:

    sh test_hier.sh

## Evaluation

For evaluation, we use the ROUGE implementation from [Google-research](https://github.com/google-research/google-research/tree/master/rouge), with the following command,

    eval_hier.sh

## Released Checkpoints and Outputs

<table>
   <tr>
      <td></td>
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

