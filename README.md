# Audio captioning 

Welcome to the repository of the audio captioning.

----
## Setting up the code

To start using the audio captioning, firstly you
have to set-up the code. Please that the code in this repository
is tested with Python 3.9.

To set-up the code, you have to do the following: 

  1. Clone this repository.
  2. Use pip to install dependencies ```` pip install requirement.txt ````
  
Use the following command to clone this repository at your terminal:

````shell script
$ git clone https://github.com/Vikhyat-1820/AUDIO-CAPTIONING.git
````

All the required libraries are included in requirement.txt.

## Folder Structure

    AUDIO-CAPTIONING/
     | - Audio_Feature_Extraction/
     |   | - Audio_preprocessing.py/
     |   | - feature_extractor.py/
     | - Data_Handler/
     |   |- Dataset_creation.py/
     | - Models/
     |   | - model.py/
     | - Settings/
     |   | - audio_preprocessing.yaml/
     |   | - data_set_creation.yaml/
     |   | - dir_and_files.yaml/
     |   | - text_preprocessing.yaml/
     | - Text Preprocessing/
     |   | - Text_preprocessing.py/
     |   | - text_utils.py/
     | - features/
     |   | - caption_list_file.pkl/
     |   | - tokenizer.pkl/
     | - features/
     |   | - file_io.py/
     | - Data/
     |   | - TEST/
     |   | - TEST_CP/
     
     
  
## DataSet

The dataset used for audio captioning is clotho dataset. Clotho dataset is freely available online at the Zenodo platform. You can find Clotho at [here](https://zenodo.org/record/3490684#.YQFQkI4zYrM).

## Training

Step 1 :- Run ```` python Text Preprocessing/Text_preprocessing.py```` in terminal, this will create two files in features folder.

Step 2 :- Run ```` python Models/model.py ````. It will start training the model and create MODEL.H5 in models folder.


 


     
     
     
