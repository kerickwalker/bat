import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import wavio
import numpy as np
import wave
from scipy.io.wavfile import read as read_wav
import pylab
from numpy.lib import stride_tricks

""" NOTE: While a lot of this was self authored (lines 60-89), the spectrogram images I was producing were just not the correct colors. I couldn't find a way to make the
contrast between the noise caught by the microphone and the background more visible. The code between lines 16-30, 32-57, and 91-112 was made following this stack overflow
post https://stackoverflow.com/questions/44787437/how-to-convert-a-wav-file-to-a-spectrogram-in-python3. All it really is the template for the graph and the correct coloring
for it. The actual accesing of the files, processing of the wav data, and saving of the images was all pretty simple itself."""

#short time fourier transform of audio signal
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning, hopFactor=1):
    win = np.hamming(frameSize) + 1e-10
    hopSize = int(frameSize - np.floor(overlapFac * frameSize)) * hopFactor

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs




folders = ["Pipistrellus pygmaus with social sound", "Noctula nyctalus with noise", "Pipistrellus pygmaus wo social sound", "Noctula nyctalus with out social sound and noise"]
folders1 = ["test"]
folders1 = ["Noctula nyctalus with out social sound and noise"]
def wavToSpectro(folders):
    for folder in folders:
        for fN in os.listdir(f"/Users/elijahmendoza/OCS_Materials/Neural_Networks/NeuralNetworksProject/{folder}/to crop"):
            #print(fN)
            fileName = fN[:-4]
            if ".wav" in fN:
                fileToImport = f"/Users/elijahmendoza/OCS_Materials/Neural_Networks/NeuralNetworksProject/{folder}/to crop/{fileName}.wav"
                pngName = f"/Users/elijahmendoza/OCS_Materials/Neural_Networks/NeuralNetworksProject/{folder}/Bar Spectrograms/{fileName}"


                samp_rate, samp = wavfile.read(fileToImport)

                # our samp is 5_000_000 (for a given clip)
                # our samp rate is 500_000 (for a given clip)
                # if we divide our samp/samp_rate then we get the length of our clip (in this case 10)
                # adjust sample rate 
  
                frequencies, times, spectrogram = signal.spectrogram(samp, samp_rate)
                binsize = 2**10
                colormap = "jet"

                #hopfactor Max: 15
                #hopfactor min: ?

                s = stft(samp, binsize, hopFactor=2)
                sshow, freq = logscale_spec(s, factor=1, sr=samp_rate)
                ims = 20. * np.log10(np.where(np.abs(sshow) < 1e-10, 1e-10, np.abs(sshow))) # amplitude to decibel
                timebins, freqbins = np.shape(ims)
                
                plt.figure(figsize=(3.0, 2.0), dpi=100)
                plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="bilinear")
                #plt.colorbar()
                plt.axis('off')   # Turn off axis
                plt.margins(0, 0) # Set margins to zero
                #plt.gca().set_aspect('equal')

                #plt.xlabel("time (s)")
                #plt.ylabel("frequency (hz)")
                plt.xlim([0, timebins-1])
                plt.ylim([3, 250])

                #xlocs = np.float32(np.linspace(0, timebins-1, 5))
                #plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samp)/timebins)+(0.5*binsize))/samp_rate])
                #ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
                #plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

                plt.savefig(pngName, bbox_inches="tight", pad_inches=0.0)
                plt.clf()
                plt.close()



wavToSpectro(folders)
