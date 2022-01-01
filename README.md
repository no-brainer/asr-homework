# ASR project

## Installation guide

1) Install Python packages
```shell
pip install -r ./requirements.txt
```
2) Download checkpoint and default config
```shell
mkdir default_test_model
gdown --id 1FMoIxP_rQA4gXQ9395FZupQ4juKOu7LZ -O default_test_model/checkpoint.pth  # checkpoint
gdown --id 1-VJb5kP2Pa7IL59WkQ38pLJkQLrxn6Bx -O default_test_model/config.json  # default config
```
3) Necessary resources are downloadable. If a class requires some external material, it downloads it. The list of such classes:
* `BackgroundNoise` from `hw_asr/augmentations/wave_augmentations` downloads noise for augmentations in [line 22](https://gitlab.com/no-brainer/asr/-/blob/hw_asr_2021/hw_asr/augmentations/wave_augmentations/background_noise.py#L22)
* `CTCBPETextEncoder` from `hw_asr/text_encoder` downloads pretrained BPE model in [line 19](https://gitlab.com/no-brainer/asr/-/blob/hw_asr_2021/hw_asr/text_encoder/ctc_bpe_text_encoder.py#L19)
* `CTCCharTextEncoder` from `hw_asr/text_encoder` downloads pretrained KenLM and a vocab for shallow fusion in [line 42](https://gitlab.com/no-brainer/asr/-/blob/hw_asr_2021/hw_asr/text_encoder/ctc_char_text_encoder.py#L42) and [line 68](https://gitlab.com/no-brainer/asr/-/blob/hw_asr_2021/hw_asr/text_encoder/ctc_char_text_encoder.py#L68)

# About

This repository includes implementations of:
1) [LSTM](./hw_asr/model/baseline_model.py)
2) [QuartzNet](./hw_asr/model/quartznet.py) [1]
3) [Deep Speech 2](./hw_asr/model/crnn.py) [2]

There are also implementations of multiple decoding strategies and vocabularies. Language model is KenLM.

# References
1) [Samuel Kriman et al. QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions. 2019](https://arxiv.org/abs/1910.10261)
2) [Dario Amodei et al. Deep Speech 2: End-to-End Speech Recognition in English and Mandarin. 2015](https://arxiv.org/abs/1512.02595)
