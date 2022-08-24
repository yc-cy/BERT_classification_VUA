# coding: UTF-8
import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from train import train
from config import Config
from preprocess import DataProcessor, get_time_dif
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

parser = argparse.ArgumentParser(description="Bert Chinese Text Classification")
parser.add_argument("--mode", type=str, default="train", help="train/demo/predict")
parser.add_argument("--data_dir", type=str, default="./data", help="training data and saved model path")
parser.add_argument("--pretrained_bert_dir", type=str, default="./pretrained_bert", help="pretrained bert model path")
parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")
parser.add_argument("--input_file", type=str, default="./data/input.txt", help="input file to be predicted")
args = parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    set_seed(args.seed)
    config = Config(args.data_dir)
    # 获取分词器
    # pretrained_bert_dir：./pretrained_bert
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_dir)
    # 获取预训练模型config配置
    bert_config = BertConfig.from_pretrained(args.pretrained_bert_dir, num_labels=config.num_labels)
    # 获取预训练模型
    model = BertForSequenceClassification.from_pretrained(
        os.path.join(args.pretrained_bert_dir, "pytorch_model.bin"),
        config=bert_config
    )
    model.to(config.device)

    if args.mode == "train":
        print("loading data...")
        start_time = time.time()
        train_iterator = DataProcessor(config.train_file, config.device, tokenizer, config.batch_size, config.max_seq_len, args.seed)
        dev_iterator = DataProcessor(config.dev_file, config.device, tokenizer, config.batch_size, config.max_seq_len, args.seed)
        time_dif = get_time_dif(start_time)
        print("time usage:", time_dif)

        # train
        train(model, config, train_iterator, dev_iterator)
    
    elif args.mode == "demo":
        model.load_state_dict(torch.load(config.saved_model))
        model.eval()
        while True:
            sentence = input("请输入文本:\n")
            inputs = tokenizer(
                sentence, 
                max_length=config.max_seq_len,
                truncation="longest_first",
                return_tensors="pt")
            inputs = inputs.to(config.device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs[0]
                label = torch.max(logits.data, 1)[1].tolist()
                print("分类结果:" + label)
            flag = str(input("continue? (y/n):"))
            if flag == "Y" or flag == "y":
                continue
            else:
                break
    else:
        model.load_state_dict(torch.load(config.saved_model))
        model.eval()

        text = []
        with open(args.input_file, mode="r", encoding="UTF-8") as f:
            for line in tqdm(f):
                sentence = line.strip()
                if not sentence:    continue
                text.append(sentence)

        num_samples = len(text)
        num_batches = (num_samples - 1) // config.batch_size + 1
        for i in range(num_batches):
            start = i * config.batch_size
            end = min(num_samples, (i + 1) * config.batch_size)
            inputs = tokenizer.batch_encode_plus(
                text[start: end],
                padding=True,
                max_length=config.max_seq_len,
                truncation="longest_first",
                return_tensors="pt")
            inputs = inputs.to(config.device)

            outputs = model(**inputs)
            logits = outputs[0]

            preds = torch.max(logits.data, 1)[1].tolist()
            labels = [config.label_list[_] for _ in preds]
            for j in range(start, end):
                print("%s\t%s" % (text[j], labels[j - start]))
                

if __name__ == '__main__':
    main()



# bert.embeddings.word_embeddings.weight True
# bert.embeddings.position_embeddings.weight True
# bert.embeddings.token_type_embeddings.weight True
# bert.embeddings.LayerNorm.weight True
# bert.embeddings.LayerNorm.bias True
# bert.encoder.layer.0.attention.self.query.weight True
# bert.encoder.layer.0.attention.self.query.bias True
# bert.encoder.layer.0.attention.self.key.weight True
# bert.encoder.layer.0.attention.self.key.bias True
# bert.encoder.layer.0.attention.self.value.weight True
# bert.encoder.layer.0.attention.self.value.bias True
# bert.encoder.layer.0.attention.output.dense.weight True
# bert.encoder.layer.0.attention.output.dense.bias True
# bert.encoder.layer.0.attention.output.LayerNorm.weight True
# bert.encoder.layer.0.attention.output.LayerNorm.bias True
# bert.encoder.layer.0.intermediate.dense.weight True
# bert.encoder.layer.0.intermediate.dense.bias True
# bert.encoder.layer.0.output.dense.weight True
# bert.encoder.layer.0.output.dense.bias True
# bert.encoder.layer.0.output.LayerNorm.weight True
# bert.encoder.layer.0.output.LayerNorm.bias True
# bert.encoder.layer.1.attention.self.query.weight True
# bert.encoder.layer.1.attention.self.query.bias True
# bert.encoder.layer.1.attention.self.key.weight True
# bert.encoder.layer.1.attention.self.key.bias True
# bert.encoder.layer.1.attention.self.value.weight True
# bert.encoder.layer.1.attention.self.value.bias True
# bert.encoder.layer.1.attention.output.dense.weight True
# bert.encoder.layer.1.attention.output.dense.bias True
# bert.encoder.layer.1.attention.output.LayerNorm.weight True
# bert.encoder.layer.1.attention.output.LayerNorm.bias True
# bert.encoder.layer.1.intermediate.dense.weight True
# bert.encoder.layer.1.intermediate.dense.bias True
# bert.encoder.layer.1.output.dense.weight True
# bert.encoder.layer.1.output.dense.bias True
# bert.encoder.layer.1.output.LayerNorm.weight True
# bert.encoder.layer.1.output.LayerNorm.bias True
# bert.encoder.layer.2.attention.self.query.weight True
# bert.encoder.layer.2.attention.self.query.bias True
# bert.encoder.layer.2.attention.self.key.weight True
# bert.encoder.layer.2.attention.self.key.bias True
# bert.encoder.layer.2.attention.self.value.weight True
# bert.encoder.layer.2.attention.self.value.bias True
# bert.encoder.layer.2.attention.output.dense.weight True
# bert.encoder.layer.2.attention.output.dense.bias True
# bert.encoder.layer.2.attention.output.LayerNorm.weight True
# bert.encoder.layer.2.attention.output.LayerNorm.bias True
# bert.encoder.layer.2.intermediate.dense.weight True
# bert.encoder.layer.2.intermediate.dense.bias True
# bert.encoder.layer.2.output.dense.weight True
# bert.encoder.layer.2.output.dense.bias True
# bert.encoder.layer.2.output.LayerNorm.weight True
# bert.encoder.layer.2.output.LayerNorm.bias True
# bert.encoder.layer.3.attention.self.query.weight True
# bert.encoder.layer.3.attention.self.query.bias True
# bert.encoder.layer.3.attention.self.key.weight True
# bert.encoder.layer.3.attention.self.key.bias True
# bert.encoder.layer.3.attention.self.value.weight True
# bert.encoder.layer.3.attention.self.value.bias True
# bert.encoder.layer.3.attention.output.dense.weight True
# bert.encoder.layer.3.attention.output.dense.bias True
# bert.encoder.layer.3.attention.output.LayerNorm.weight True
# bert.encoder.layer.3.attention.output.LayerNorm.bias True
# bert.encoder.layer.3.intermediate.dense.weight True
# bert.encoder.layer.3.intermediate.dense.bias True
# bert.encoder.layer.3.output.dense.weight True
# bert.encoder.layer.3.output.dense.bias True
# bert.encoder.layer.3.output.LayerNorm.weight True
# bert.encoder.layer.3.output.LayerNorm.bias True
# bert.encoder.layer.4.attention.self.query.weight True
# bert.encoder.layer.4.attention.self.query.bias True
# bert.encoder.layer.4.attention.self.key.weight True
# bert.encoder.layer.4.attention.self.key.bias True
# bert.encoder.layer.4.attention.self.value.weight True
# bert.encoder.layer.4.attention.self.value.bias True
# bert.encoder.layer.4.attention.output.dense.weight True
# bert.encoder.layer.4.attention.output.dense.bias True
# bert.encoder.layer.4.attention.output.LayerNorm.weight True
# bert.encoder.layer.4.attention.output.LayerNorm.bias True
# bert.encoder.layer.4.intermediate.dense.weight True
# bert.encoder.layer.4.intermediate.dense.bias True
# bert.encoder.layer.4.output.dense.weight True
# bert.encoder.layer.4.output.dense.bias True
# bert.encoder.layer.4.output.LayerNorm.weight True
# bert.encoder.layer.4.output.LayerNorm.bias True
# bert.encoder.layer.5.attention.self.query.weight True
# bert.encoder.layer.5.attention.self.query.bias True
# bert.encoder.layer.5.attention.self.key.weight True
# bert.encoder.layer.5.attention.self.key.bias True
# bert.encoder.layer.5.attention.self.value.weight True
# bert.encoder.layer.5.attention.self.value.bias True
# bert.encoder.layer.5.attention.output.dense.weight True
# bert.encoder.layer.5.attention.output.dense.bias True
# bert.encoder.layer.5.attention.output.LayerNorm.weight True
# bert.encoder.layer.5.attention.output.LayerNorm.bias True
# bert.encoder.layer.5.intermediate.dense.weight True
# bert.encoder.layer.5.intermediate.dense.bias True
# bert.encoder.layer.5.output.dense.weight True
# bert.encoder.layer.5.output.dense.bias True
# bert.encoder.layer.5.output.LayerNorm.weight True
# bert.encoder.layer.5.output.LayerNorm.bias True
# bert.encoder.layer.6.attention.self.query.weight True
# bert.encoder.layer.6.attention.self.query.bias True
# bert.encoder.layer.6.attention.self.key.weight True
# bert.encoder.layer.6.attention.self.key.bias True
# bert.encoder.layer.6.attention.self.value.weight True
# bert.encoder.layer.6.attention.self.value.bias True
# bert.encoder.layer.6.attention.output.dense.weight True
# bert.encoder.layer.6.attention.output.dense.bias True
# bert.encoder.layer.6.attention.output.LayerNorm.weight True
# bert.encoder.layer.6.attention.output.LayerNorm.bias True
# bert.encoder.layer.6.intermediate.dense.weight True
# bert.encoder.layer.6.intermediate.dense.bias True
# bert.encoder.layer.6.output.dense.weight True
# bert.encoder.layer.6.output.dense.bias True
# bert.encoder.layer.6.output.LayerNorm.weight True
# bert.encoder.layer.6.output.LayerNorm.bias True
# bert.encoder.layer.7.attention.self.query.weight True
# bert.encoder.layer.7.attention.self.query.bias True
# bert.encoder.layer.7.attention.self.key.weight True
# bert.encoder.layer.7.attention.self.key.bias True
# bert.encoder.layer.7.attention.self.value.weight True
# bert.encoder.layer.7.attention.self.value.bias True
# bert.encoder.layer.7.attention.output.dense.weight True
# bert.encoder.layer.7.attention.output.dense.bias True
# bert.encoder.layer.7.attention.output.LayerNorm.weight True
# bert.encoder.layer.7.attention.output.LayerNorm.bias True
# bert.encoder.layer.7.intermediate.dense.weight True
# bert.encoder.layer.7.intermediate.dense.bias True
# bert.encoder.layer.7.output.dense.weight True
# bert.encoder.layer.7.output.dense.bias True
# bert.encoder.layer.7.output.LayerNorm.weight True
# bert.encoder.layer.7.output.LayerNorm.bias True
# bert.encoder.layer.8.attention.self.query.weight True
# bert.encoder.layer.8.attention.self.query.bias True
# bert.encoder.layer.8.attention.self.key.weight True
# bert.encoder.layer.8.attention.self.key.bias True
# bert.encoder.layer.8.attention.self.value.weight True
# bert.encoder.layer.8.attention.self.value.bias True
# bert.encoder.layer.8.attention.output.dense.weight True
# bert.encoder.layer.8.attention.output.dense.bias True
# bert.encoder.layer.8.attention.output.LayerNorm.weight True
# bert.encoder.layer.8.attention.output.LayerNorm.bias True
# bert.encoder.layer.8.intermediate.dense.weight True
# bert.encoder.layer.8.intermediate.dense.bias True
# bert.encoder.layer.8.output.dense.weight True
# bert.encoder.layer.8.output.dense.bias True
# bert.encoder.layer.8.output.LayerNorm.weight True
# bert.encoder.layer.8.output.LayerNorm.bias True
# bert.encoder.layer.9.attention.self.query.weight True
# bert.encoder.layer.9.attention.self.query.bias True
# bert.encoder.layer.9.attention.self.key.weight True
# bert.encoder.layer.9.attention.self.key.bias True
# bert.encoder.layer.9.attention.self.value.weight True
# bert.encoder.layer.9.attention.self.value.bias True
# bert.encoder.layer.9.attention.output.dense.weight True
# bert.encoder.layer.9.attention.output.dense.bias True
# bert.encoder.layer.9.attention.output.LayerNorm.weight True
# bert.encoder.layer.9.attention.output.LayerNorm.bias True
# bert.encoder.layer.9.intermediate.dense.weight True
# bert.encoder.layer.9.intermediate.dense.bias True
# bert.encoder.layer.9.output.dense.weight True
# bert.encoder.layer.9.output.dense.bias True
# bert.encoder.layer.9.output.LayerNorm.weight True
# bert.encoder.layer.9.output.LayerNorm.bias True
# bert.encoder.layer.10.attention.self.query.weight True
# bert.encoder.layer.10.attention.self.query.bias True
# bert.encoder.layer.10.attention.self.key.weight True
# bert.encoder.layer.10.attention.self.key.bias True
# bert.encoder.layer.10.attention.self.value.weight True
# bert.encoder.layer.10.attention.self.value.bias True
# bert.encoder.layer.10.attention.output.dense.weight True
# bert.encoder.layer.10.attention.output.dense.bias True
# bert.encoder.layer.10.attention.output.LayerNorm.weight True
# bert.encoder.layer.10.attention.output.LayerNorm.bias True
# bert.encoder.layer.10.intermediate.dense.weight True
# bert.encoder.layer.10.intermediate.dense.bias True
# bert.encoder.layer.10.output.dense.weight True
# bert.encoder.layer.10.output.dense.bias True
# bert.encoder.layer.10.output.LayerNorm.weight True
# bert.encoder.layer.10.output.LayerNorm.bias True
# bert.encoder.layer.11.attention.self.query.weight True
# bert.encoder.layer.11.attention.self.query.bias True
# bert.encoder.layer.11.attention.self.key.weight True
# bert.encoder.layer.11.attention.self.key.bias True
# bert.encoder.layer.11.attention.self.value.weight True
# bert.encoder.layer.11.attention.self.value.bias True
# bert.encoder.layer.11.attention.output.dense.weight True
# bert.encoder.layer.11.attention.output.dense.bias True
# bert.encoder.layer.11.attention.output.LayerNorm.weight True
# bert.encoder.layer.11.attention.output.LayerNorm.bias True
# bert.encoder.layer.11.intermediate.dense.weight True
# bert.encoder.layer.11.intermediate.dense.bias True
# bert.encoder.layer.11.output.dense.weight True
# bert.encoder.layer.11.output.dense.bias True
# bert.encoder.layer.11.output.LayerNorm.weight True
# bert.encoder.layer.11.output.LayerNorm.bias True
# bert.pooler.dense.weight True
# bert.pooler.dense.bias True
# classifier.weight True
# classifier.bias True