# 기본
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim

# Modeling
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration

# meteor_score 
import nltk

class G2T_Module:
    def __init__(self,):
        nltk.download('wordnet')
        nltk.download('omw-1.4')

        # GPU check
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device('cpu')

        self.max_length=50
        
        model_name = 'paust/pko-t5-base'
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name, force_download=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

        PATH ='/home/stonemaeng/g2t/KoT5/checkpoints_base/checkpoint_epoch_18.pt'
        # checkpoint = torch.load(PATH, map_location=device)
        # model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
        self.model.eval()


    def predict(self, sentences):

        translations = []
        for sentence in sentences:
            inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            # 모델에 데이터 전달
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_length,
                num_beams=5,  # Beam search 설정
                early_stopping=True
            )

            # 디코딩
            for output in output_ids:
                translation = self.tokenizer.decode(output, skip_special_tokens=True)
                translations.append(translation)
        return translations

if __name__ == "__main__":
    # 테스트 문장
    test_sentences = ["날씨 덥다 화 난다", "목 마르다 물 마시다", '수영장 가다 놀다', '재미 있다 피곤하다 잠']

    # init
    G2T = G2T_Module()

    # predict
    translations = G2T.predict(test_sentences)

    # check
    print("Translations:", translations)