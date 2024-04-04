import requests
import spacy
import re
import pandas as pd
import numpy as np
import subprocess
import time
import transformers
import torch
import language_tool_python
from transformers import DistilBertModel, DistilBertTokenizerFast
from bs4 import BeautifulSoup
from spacy import displacy

try:
    # Try to load the 'en_core_web_lg' model
    spacy.load('en_core_web_lg')
    print("'en_core_web_lg' is already installed.")
except OSError:
    # If the model is not found, download it
    print("Downloading 'en_core_web_lg'...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"], check=True)
    print("'en_core_web_lg' has been successfully downloaded.")

# ----------------- Trying to detect sensitive information --------------------
def sensitive_infor_of_string(string):
    #regex_sg = r"(\+?65)?\s*?([\d]{4})\s*?([\d]{4})\b"   #sg
    regex_sg = r"(?<!\d)(\+65)?\s*?(\d{8})\b"   
    regex_us = r"(\(\d{3}\) |\d{3}-)?\d{3}-\d{4}"            #us
    regex_cn = r"1[34578]\d{9}"     #China
    #regex_cn = r"(\+86)?\s*?(\d{3,4})\s*?(\d{3})\b"  # China
    regex_em = r"(?P<user>[\w.-]+)@(?P<domain>[\w.-]+)"  #email
    
#     regex = [regex_sg,regex_us,regex_cn,regex_em]
#     for re in regex:
    pattern_sg = re.compile(regex_sg)
    pattern_us = re.compile(regex_us)
    pattern_cn = re.compile(regex_cn)
    pattern_em = re.compile(regex_em)
    
    # find match in the text
    matches_sg = pattern_sg.findall(string)
    matches_us = pattern_us.findall(string)
    matches_cn = pattern_cn.findall(string)
    matches_em = pattern_em.findall(string)
    
    sensitive_info = {'sg':[],'us':[],'cn':[],'email':[],'name':[],'address':[]}
    for match in matches_sg:
        sensitive_info['sg'].append('+65 '+' '.join(match[1:]))
    for match in matches_us:
        sensitive_info['us'].append(' '.join(match[1:]))
    for match in matches_cn:
        sensitive_info['cn'].append('+86 '+''.join(match))
    for match in matches_em:
        sensitive_info['email'].append('@'.join(match))
    
    NER = spacy.load('en_core_web_lg')
    ner_string = NER(string)
    name = []
    for word in ner_string.ents:
        if word.label_=='PERSON':
            name.append(word.text)
        if word.label_=='GPE':
            sensitive_info['address'].append(word.text)
    
    
    def check_name_on_wikipedia(name):
    # Using the GET request to check for the existence of a Wikipedia page for the given name
        response_get = requests.get(f"https://en.wikipedia.org/wiki/{name}")
        if response_get.status_code == 200:
            soup = BeautifulSoup(response_get.content,'html.parser')
            references = soup.find_all('li',id = lambda x: x and x.startswith('cite_note-'))
            num_reference = len(references)
            return num_reference>8, "GET", response_get.url
        else:
            # If GET request fails, we can try a POST request to the Wikipedia search page
            payload = {'search': name}
            response_post = requests.post("https://en.wikipedia.org/w/index.php", data=payload)
            if name.lower() in response_post.text.lower():
                return True, "POST", response_post.url
            else:
                return False, "POST", response_post.url
    
    for n in name:
        name_to_check = n.replace(' ','_')
        exists, method, url = check_name_on_wikipedia(name_to_check)
        if exists == True:
            continue
        else:
            sensitive_info['name'].append(n)
    return sensitive_info


# text = "Chris van is a great man and he lives in Singapore NUS,his phone number is 89438900 and email is e2313@wq.com,\
# he likes Einstein very much"
# result = sensitive_infor_of_string(text)
# print(result)

# ----------------- Load finetuned distillbert and do inference --------------------
device = 'cpu'
class DistillBERTClass(torch.nn.Module):
    def __init__(self, distillBERT):
        super(DistillBERTClass, self).__init__()
        self.l1 = distillBERT
        # Freeze DistilBERT model parameters
        for param in self.l1.parameters():
            param.requires_grad = False
        self.dropout = torch.nn.Dropout(0.2)
        self.pre_classifier = torch.nn.Linear(768, 256)
        self.classifier_1 = torch.nn.Linear(256, 32)
        self.classifier_2 = torch.nn.Linear(32, 1)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0] # Take the last layer
        pooler = hidden_state[:, 0] # Take the CLS token only

        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)

        pooler = self.classifier_1(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)

        output = self.classifier_2(pooler)
        return output
    
distillBERT_pretrained = DistilBertModel.from_pretrained("distilbert-base-uncased")
model = DistillBERTClass(distillBERT_pretrained)
model.to(device)
model.load_state_dict(torch.load('distillbert_toxic.pth',  map_location=torch.device('cpu')))

def is_toxic(text):
    # Testing with a new sample
    sample = text
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    encoded_dict = tokenizer.encode_plus(
        sample,                      # Input text
        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        max_length = 512,           # Pad & truncate all sentences
        padding='max_length',
        pad_to_max_length = True,  # Pad all to the max length of the model
        truncation = True,
        return_attention_mask = True,   # Construct attn. masks
        return_tensors = 'pt',     # Return pytorch tensors
    ).to(device)

    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    model.eval()  # Put the model in evaluation mode

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prob = torch.sigmoid(outputs.data)
        predicted_class = (prob > 0.5).int().item()
    print(f"Predicted class: {predicted_class}")
    return predicted_class

# text = "COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK"
# toxic_result = is_toxic(text)
# print(toxic_result)

# ----------------- Trying to detect other sensitive information --------------------
def detect_sentence(sentence):
    tool = language_tool_python.LanguageTool('en-US')
    text = sentence
    matches = tool.check(text)
    for match in matches:
        print(match)

def physcial_disease(sentence,ner_model,dir):
    result =[]
    if ner_model:
        loaded_nlp = spacy.load(dir)
        test_sentence = sentence
        doc = loaded_nlp(test_sentence)
        result = [(ent.text, ent.label_) for ent in doc.ents]
    else:
        disease_df = pd.read_csv(dir)
        disease_list = disease_df['name'].tolist()

        def find_diseases(text):
            found_diseases = []
            for disease in disease_list:
                if disease.lower() in text.lower():
                    found_diseases.append(disease)
            return found_diseases
        # # Example usage
        # sample_text = "I have been diagnosed with fever."
        identified_diseases = find_diseases(sentence)
        result = [(identified_diseases[0],'DISEASE')]
    return result

def mental_disease(sentence,dir):
    disease_df = pd.read_csv(dir)
    disease_list = disease_df['name'].tolist()

    def find_diseases(text):
        found_diseases = []
        for disease in disease_list:
            if disease.lower() in text.lower():
                found_diseases.append(disease)
        return found_diseases
    # # Example usage
    # sample_text = "I have been diagnosed with fever."
    identified_diseases = find_diseases(sentence)
    result = [(identified_diseases[0],'MENTAL DISEASE')]
    return result

def process_text(text):
    sensitive_info = sensitive_infor_of_string(text)
    is_sentence_toxic = is_toxic(text)
    sensitive_info_disease = physcial_disease(text,True,'model')
    return {"sensitive_info": sensitive_info, "toxic": is_sentence_toxic, "sensitive_info_disease": sensitive_info_disease}
    

if __name__ == '__main__':
    # text = "Chris van is a great man and he lives in Singapore NUS,his phone number is 89438900 and email is e2313@wq.com,\
    # he likes Einstein very much"
    # result = sensitive_infor_of_string(text)
    # print(result)

    # text = "COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK"
    # toxic_result = is_toxic(text)
    # print(toxic_result)

    # sentence = 'she has Allergy.'
    # print(physcial_disease(sentence,True,'model'))
    example_text = "Chris van is a great man and he lives in Singapore NUS,his phone number is 89438900 and email is e2313@wq.com,\
    # he likes Einstein very much"
    process_text(example_text)