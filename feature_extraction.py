import os
import pandas as pd
import numpy as np
import librosa
from pyAudioAnalysis import audioFeatureExtraction

print("Start")
directory = 'C:/Users/Atulya/Documents/GitHub/accent-classifier/' #change path to project folder

def main(directory):
    df=pd.DataFrame()
    for wav_file in os.listdir(directory+'/raw'):
        print(wav_file)
        s = wav_file
        s = s[:-4]
        name_lable = ''.join([i for i in s if not i.isdigit()])
        if name_lable not in ['english', 'french', 'mandarin', 'arabic', 'spanish', 'hindi']:
            name_lable = 'other'

        x, Fs = librosa.load(directory+'/raw/'+ wav_file)
        x = x[5*Fs:10*Fs]; #taking 5 seconds to 10 seconds
        x = np.concatenate((x, [0]* (Fs - x.shape[0])), axis=0);
        F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs);
        P = np.reshape(F.T, (1, F.shape[0]*F.shape[1]))
        P=P.tolist()[0]
        P=P+[name_lable]
        df=df.append([P])
    df.to_csv(directory + "/" + 'Features.csv')
    print('Done csv')

main(directory)
print("Completed")