import numpy as np
import os
import sys
sys.path.append("C:/Users/Vikhyat/Desktop/ML/audio captioning/AUDIO CAPTIONING/utils")

sys.path.append("C:/Users/Vikhyat/Desktop/ML/audio captioning/AUDIO CAPTIONING/Audio_Feature_Extraction")

import yaml
import pandas as pd
from file_io import load_yaml_file,update_settings,save_pkl_file,load_pkl_file
from keras.preprocessing.text import Tokenizer
from feature_extractor import audio_feature_extractor

from keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical




def dataset_creation(batch_size):
	main_settings_dir="C:/Users/Vikhyat/Desktop/ML/audio captioning/AUDIO CAPTIONING/Settings/dir_and_files.yaml"
	main_settings=load_yaml_file(main_settings_dir)

	dataset_settings=load_yaml_file(main_settings['data_set_creation'])
	text_settings=load_yaml_file(main_settings['text_proc_settings'])
	audio_settings=load_yaml_file(main_settings['audio_proc_settings'])

	last_idx=dataset_settings['last_index']

	caption_list=load_pkl_file(text_settings['caption_list_file_path'])
	tokenizer=load_pkl_file(text_settings['tokenizer_path'])
	length=len(caption_list)
	print(length)
	max_len=text_settings['maxlen']
	print(max_len)
	vocab_size=len(tokenizer.word_index)+1
	print(vocab_size)



	X1=list()
	X2=list()
	Y=list()
	audio_file_path=main_settings['Data_base_dir'] + '/' + audio_settings['folder_name']
	audio_name_list=os.listdir(audio_file_path)
	print(len(audio_name_list))
	for i in range(batch_size):
		last_idx+=1
		last_idx%=length
		audio_name = caption_list[last_idx][0]
		if audio_name in audio_name_list:
			caption=caption_list[last_idx][1]
			seq=tokenizer.texts_to_sequences([caption])[0]
			audio=audio_feature_extractor(audio_file_path,audio_name)
			# print(audio.shape)
			for i in range(1,len(seq)):
				inp=seq[:i]
				op=seq[i]
				inp=pad_sequences([inp],maxlen=max_len)[0]
				op=to_categorical([op],num_classes=vocab_size)[0]
				X1.append(audio)
				X2.append(inp)
				Y.append(op)
	dataset_settings.update({'last_index':last_idx})
	update_settings(main_settings['data_set_creation'],dataset_settings)
	X1=np.array(X1)
	X2=np.array(X2)
	Y=np.array(Y)
	print(X1.shape)
	print(X2.shape)
	print(Y.shape)
	yield [X1,X2],Y





# def main():
# 	dataset_creation(7)


# if __name__=="__main__":
# 	main()