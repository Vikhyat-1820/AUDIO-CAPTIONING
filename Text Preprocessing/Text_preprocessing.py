import numpy as np
import yaml
import pandas as pd
from text_utils import caption_dict_creation,get_word_dict,get_caption_feature



def main():
	base_dir="C:/Users/Vikhyat/Desktop/ML/audio captioning/AUDIO CAPTIONING/Data"
	file_path="../Settings/text_preprocessing.yaml"
	text_settings=None
	with open(file_path,'r') as f:
		text_settings=yaml.load(f, Loader=yaml.FullLoader)
	
	caption_dict,maxlen=caption_dict_creation(base_dir)
	print(maxlen)
	
	word2idx,idx2word=get_word_dict(caption_dict)

	caption_feature=get_caption_feature(caption_dict,word2idx,maxlen)
	
	# print(caption_feature)




if __name__=="__main__":
	main()