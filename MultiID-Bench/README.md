# MultiID-Bench
> [![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg)](https://arxiv.org/abs/)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://doby-xu.github.io/WithAnyone/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow.svg)](https://huggingface.co/WithAnyone/WithAnyone)
[![MultiID-Bench](https://img.shields.io/badge/MultiID-Bench-Green.svg)](https://huggingface.co/datasets/WithAnyone/MultiID-Bench)
[![MultiID-2M](https://img.shields.io/badge/MultiID_2M-Dataset-Green.svg)](https://huggingface.co/datasets/WithAnyone/MultiID-2M)

## Download

[HuggingFace Dataset](https://huggingface.co/datasets/WithAnyone/MultiID-Bench)

## Evaluation

### Environment Setup

Besides the `requirements.txt`, you need to install the following packages:

```bash
pip install aesthetic-predictor-v2-5 
pip install facexlib
pip install colorama
pip install pytorch_lightning
git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch

# in MultiID_Bench/
mkdir pretrained
```


You need the following models to run the evaluation:

- CLIP
- arcface
- aesthetic-v2.5
- adaface
- facenet

For the first three models, they will be automatically downloaded when you run the evaluation script for the first time. Most of the models will be cached in the `HF_HOME` directory, which is usually `~/.cache/huggingface`. About 5GB of disk space is needed.

For adaface, you need to download the model weights from [adaface_ir50_ms1mv2.ckpt](https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing) (This is the original link provided by the authors of AdaFace) and put it in the `pretrained` directory.

This repository includes code from [AdaFace](https://github.com/mk-minchul/AdaFace?tab=readme-ov-file). AdaFace is included in this codebase for merely easier import. You can also clone it separately from its original repository, and modify the import paths accordingly.


### Data to Evaluate

By running:
```
python hf2bench.py \
    --dataset WithAnyone/MultiID-Bench \
    --output <root directory to save the data> \
    --from_hub
```
you can arrange the generated images and the corresponding text prompts in the following structure:
```
root/
├── id1/
│   ├── out.jpg
│   ├── ori.jpg
│   ├── ref_1.jpg
│   ├── ref_2.jpg
│   ├── ref_3.jpg
│   ├── ref_4.jpg
│   └── meta.json
│
├── id2/
│   ├── out.jpg
│   ├── ori.jpg
│   ├── ref_1.jpg
│   ├── ref_2.jpg
│   ├── ref_3.jpg
│   ├── ref_4.jpg
│   └── meta.json
│
└── ...
``` 

Or you can manually download the data by
```
huggingface-cli download WithAnyone/MultiID-Bench --repo-type dataset --local-dir <root directory to save the data>
```
and arrange the files:
```
python hf2bench.py --dataset <root directory to save the data> --output <root directory to save the data>
```

If you run the `infer_withanyone.py` script in this repository, the output directory will be in the correct format.

The `meta.json` file should contain the prompt used to generate the image, in the following format:

```json
{
    "prompt": "a photo of a person with blue hair and glasses"
}
```

### Run Evaluation

You can run the evaluation script as follows:

```python
from eval import BenchEval_Geo

def run():
    evaler = BenchEval_Geo(
        target_dir=<root directory mentioned above>,
        output_dir=<output directory to save the evaluation results>,
        ori_file_name="ori.jpg", # the name of the ground truth image file
        output_file_name="out.jpg", # the name of the generated image file
        ref_1_file_name="ref_1.jpg", # the name of the first reference image file
        ref_2_file_name="ref_2.jpg", # the name of the second reference image file
        # ref_2_file_name=None, # if you only have one reference image, set ref_2_file_name to None
        # ref_3_file_name="ref_3.jpg", # the name of the third reference
        # ref_4_file_name="ref_4.jpg", # the name of the fourth reference,
        caption_keyword="prompt", # the keyword to extract the prompt from meta.json
        names_keyword=None
    )
    evaler()
if __name__ == "__main__":
    run()
```





