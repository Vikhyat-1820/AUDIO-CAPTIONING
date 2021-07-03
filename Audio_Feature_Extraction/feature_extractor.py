import librosa
import numpy as np
import yaml
import sys
sys.path.append("C:/Users/Vikhyat/Desktop/ML/audio captioning/AUDIO CAPTIONING/utils")

from file_io import load_yaml_file,update_settings



def audio_feature_extractor(base_dir,file_name):
	main_settings_dir="C:/Users/Vikhyat/Desktop/ML/audio captioning/AUDIO CAPTIONING/Settings/dir_and_files.yaml"
	main_settings=load_yaml_file(main_settings_dir)


	audio,sample_rate=librosa.load(base_dir+'/' + file_name)

	audio_settings=load_yaml_file(main_settings['audio_proc_settings'])

	nframes=audio_settings['nframes']

	feature_type=audio_settings['feature_type']

	if feature_type=='mfcc':
		sample_rate=audio_settings['mfcc']['sr']
		n_mfcc=audio_settings['mfcc']['n_mfcc']
		features=librosa.feature.mfcc(audio,sr=sample_rate,n_mfcc=n_mfcc)
		features=features.T[:nframes]
		return features
	
	if feature_type=='mel':
		sample_rate=audio_settings['mel']['sr']
		nb_fft=audio_settings['mel']['nb_fft']
		hop_size=audio_settings['mel']['hop_size']
		nb_mels=audio_settings['mel']['nb_mels']
		window_function=audio_settings['mel']['window_function']
		features=librosa.feature.melspectrogram(audio,sr=sample_rate,n_fft=nb_fft,hop_length=hop_size,window=window_function,n_mels=nb_mels)
		features=features.T[:nframes]
		return features;

	
