	import numpy as np
import sys
sys.path.append("C:/Users/Vikhyat/Desktop/ML/audio captioning/AUDIO CAPTIONING/utils")

import yaml
import pandas as pd
from text_utils import caption_list_creation,get_word_dict,get_caption_feature
from file_io import load_yaml_file,update_settings,save_pkl_file
from keras.preprocessing.text import Tokenizer


def main():
	main_settings_dir="C:/Users/Vikhyat/Desktop/ML/audio captioning/AUDIO CAPTIONING/Settings/dir_and_files.yaml"
	main_settings=load_yaml_file(main_settings_dir)

	text_settings=load_yaml_file(main_settings['text_proc_settings'])
	tokenizer_path=text_settings['tokenizer_path']
	print(tokenizer_path)
	
	caption_list,maxlen,lines_list=caption_list_creation(main_settings['Data_base_dir'])
	tokenizer=Tokenizer()
	tokenizer.fit_on_texts(lines_list)
	vocab_size=len(tokenizer.word_index)+1

	save_pkl_file(tokenizer,tokenizer_path)

	text_settings.update({'maxlen': maxlen})
	text_settings.update({'vocab_size':vocab_size})
	
	
	update_settings(main_settings['text_proc_settings'],text_settings)

	
	# word2idx,idx2word=get_word_dict(caption_dict)

	# caption_feature=get_caption_feature(caption_dict,word2idx,maxlen)
	
	



if __name__=="__main__":
	main()