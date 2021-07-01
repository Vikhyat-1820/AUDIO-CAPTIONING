import pandas as pd
import numpy as np
import yaml
from re import sub as re_sub
import pickle


def clean_sentence(sentence):
	sentence=sentence.lower()
	sentence = re_sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')
	sentence = re_sub('[,.!?;:\"]', '', sentence)
	return sentence


def caption_dict_creation(base_dir):
	file_path="../Settings/text_preprocessing.yaml"
	text_settings=None
	with open(file_path,'r') as f:
		text_settings=yaml.load(f, Loader=yaml.FullLoader)

	caption_csv=pd.read_csv(base_dir + '/' + text_settings['file_name'])
	caption_dict={}
	cap_prefix=text_settings['captions_fields_prefix']
	maxlen=0
	for i in range(caption_csv.shape[0]):
		audio_file_name=caption_csv.iloc[i:i+1]['file_name'].values[0]
		caption_dict[audio_file_name]=list()
		for cap_id in range(1,6):
			caption_no=cap_prefix.format(cap_id)
			caption=caption_csv.iloc[i:i+1][caption_no].values[0]
			caption=clean_sentence(caption)

			if text_settings['add_special_tokens']:
				caption='startseq ' + caption + ' endseq'
			maxlen=max(maxlen,len(caption.split(' ')))
			caption_dict[audio_file_name].append(caption)

	caption_dict_file=open('../features/caption_dict_file.pkl','wb')
	pickle.dump(caption_dict,caption_dict_file)
	return caption_dict,maxlen


def get_word_dict(caption_dict):
	words=list()
	for key in caption_dict:
		for caption in caption_dict[key]:
			[words.append(word) for word in caption.split(' ')]

	words=list(set(words))
	number_of_words=len(words)
	word2idx=dict()
	idx2word=dict()

	for i in range(number_of_words):
		word2idx[words[i]]=i+1
		idx2word[i+1]=words[i]
	word2idx_file=open('../features/word2idx_file.pkl','wb')
	pickle.dump(word2idx,word2idx_file)

	idx2word_file=open('../features/idx2word_file.pkl','wb')
	pickle.dump(idx2word,idx2word_file)
	return word2idx,idx2word


def get_caption_feature(caption_dict,word2idx,maxlen):
	caption_feature=dict()
	for key in caption_dict:
		caption_feature[key]=list()
		for caption in caption_dict[key]:
			cap_idx=[word2idx[word] for word in caption.split(' ')]
			cap_len=len(cap_idx)
			pad_len=maxlen-cap_len
			cap_idx=(pad_len-(pad_len//2))*[0] + cap_idx + [0]*(pad_len//2)
			caption_feature[key].append(cap_idx)

	caption_feature_file=open('../features/caption_feature_file.pkl','wb')
	pickle.dump(caption_feature,caption_feature_file)
	return caption_feature