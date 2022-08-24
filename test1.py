import torch



try:
    state_dict = torch.load("/Users/joebob/PycharmProjects/py37_torch17/bert_text_classification-main/pretrained_bert/pytorch_model.bin", map_location=torch.device('cpu'))
except Exception:
    raise OSError(
        "error"
    )