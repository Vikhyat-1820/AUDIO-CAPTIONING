import librosa
import yaml
import numpy as np
import os
from feature_extractor import audio_feature_extractor
import pickle
import sys
sys.path.append("C:/Users/Vikhyat/Desktop/ML/audio captioning/AUDIO CAPTIONING/utils")

from file_io import load_yaml_file,update_settings

def Audio_Preprocessing():
	main_settings_dir="C:/Users/Vikhyat/Desktop/ML/audio captioning/AUDIO CAPTIONING/Settings/dir_and_files.yaml"
	main_settings=load_yaml_file(main_settings_dir)
	
	audio_settings=load_yaml_file(main_settings['audio_proc_settings'])
	base_dir=main_settings['Data_base_dir']

	base_dir=base_dir + '/' + audio_settings['folder_name']
	audio_file=os.listdir(base_dir)
	audio_features={}
	for file_name in audio_file:
		name=file_name.split('.')[0]
		features=audio_feature_extractor(base_dir,file_name)
		print(features.shape)
		audio_features[name]=features

	audio_file=open('../features/audio_features.pkl','wb')
	pickle.dump(audio_features,audio_file)
	audio_file.close()

	# audio_file=open('../features/audio_features.pkl','rb')
	# features=pickle.load(audio_file)
	# print(features.keys())








def main():
	Audio_Preprocessing()





if __name__=='__main__':
	main()

