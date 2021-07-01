import librosa
import yaml
import numpy as np
import os
from feature_extractor import audio_feature_extractor
import pickle


def Audio_Preprocessing():
	audio_settings=None
	file_path="../Settings/audio_Preprocessing.yaml"
	with open(file_path,'r') as f:
		audio_settings=yaml.load(f, Loader=yaml.FullLoader)
	
	base_dir="C:/Users/Vikhyat/Desktop/ML/audio captioning/AUDIO CAPTIONING/Data/TEST"
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

	audio_file=open('../features/audio_features.pkl','rb')
	features=pickle.load(audio_file)
	print(features.keys())








def main():
	Audio_Preprocessing()





if __name__=='__main__':
	main()

