import yaml
import pickle


def load_yaml_file(file_path):
	with open(file_path,'r') as f:
		settings=yaml.load(f, Loader=yaml.FullLoader)
	return settings



def update_settings(file_path,settings):
	with open(file_path,'w') as f:
		documents = yaml.dump(settings, f)
	return documents



def save_pkl_file(data,path):
	with open(path,'wb') as f:
		documents = pickle.dump(data, f)
	return documents



def load_pkl_file(path):
	with open(path,'rb') as f:
		data = pickle.load(f)
		return data