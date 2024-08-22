# EditSSS
This is the code attached to the paper published in ACL 2024 Findings: [SSS: Editing Factual Knowledge in Language Models Towards Semantic Sparse Space](https://aclanthology.org).

## Environment Setup
Follow the instructions in [EasyEdit](https://github.com/zjunlp/EasyEdit) to build the environment. 

```python
conda create -n edit python=3.9.7
...
pip install -r requirements.txt
```

## Quick Start
SSS supports three main training-based knowledge editing methods, including FT-L, MEND and SERAC. The base model is gpt2-xl-1.5B. You can also add new test model in "./hparams". 

Run the baseline methods using:

```bash
bash ./scripts/ft.sh
bash ./scripts/mend.sh
bash ./scripts/serac.sh
```

Run SSS method using:

```bash
bash ./scripts/ft_SSS.sh
bash ./scripts/mend_SSS.sh
bash ./scripts/serac_SSS.sh
```

To test the locality or portability, you can use the command:

```bash
# portability
bash ./scripts/test_inverse_relation.sh
bash ./scripts/test_one_hop.sh
bash ./scripts/test_subject.sh

# locality
bash ./scripts/test_locality.sh
```

## Acknowledgement
This repository is built using the [EasyEdit](https://github.com/zjunlp/EasyEdit) codebase.