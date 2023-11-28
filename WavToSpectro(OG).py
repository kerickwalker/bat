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
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

#scale frequency axis logarithmically
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


folders1 = ["Pipistrellus pygmaus with social sound", "Noctula nyctalus with noise", "Pipistrellus pygmaus wo social sound", "Noctula nyctalus with out social sound and noise"]
folders = ["test"]
folders1 = ["Noctula nyctalus with out social sound and noise"]
def wavToSpectro(folders):
    for folder in folders:
        for fN in os.listdir(f"/Users/elijahmendoza/OCS_Materials/Neural_Networks/NeuralNetworksProject/{folder}"):
            #print(fN)
            fileName = fN[:-4]
            if ".wav" in fN:
                #fileName = fileName[:-4]
                #print(f"/Users/elijahmendoza/OCS_Materials/Neural_Networks/NeuralNetworksProject/{folder}/{fileName}")
                fileToImport = f"/Users/elijahmendoza/OCS_Materials/Neural_Networks/NeuralNetworksProject/{folder}/{fileName}.wav"
                pngName = f"/Users/elijahmendoza/OCS_Materials/Neural_Networks/NeuralNetworksProject/{folder}/{fileName}(test1)"

                #currentfile is the data obj of the new wavfile
                #currentFile = wave.open(fileToImport, 'r')
                samp_rate, samp = wavfile.read(fileToImport)
                #sample_rate, samples = read_wav(fileToImport)
                #sample_rate = currentFile.getnframes()
                #print(samp_rate)
                #samples = currentFile.getframerate()
                #print(samp)
                
                frequencies, times, spectrogram = signal.spectrogram(samp, samp_rate)
                binsize = 2**10
                colormap = "twilight"

                """ plt.pcolormesh(times, frequencies, np.log(spectrogram))
                #plt.imshow(spectrogram, cmap="jet")
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.savefig(f'{pngName}') """

                s = stft(samp, binsize)
                sshow, freq = logscale_spec(s, factor=1.0, sr=samp_rate)
                ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
                timebins, freqbins = np.shape(ims)
                print(timebins)
                
                """ plt.figure(figsize=(15, 7.5))
                plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
                plt.colorbar()

                plt.xlabel("time (s)")
                plt.ylabel("frequency (hz)")
                plt.xlim([0, timebins-1])
                plt.ylim([0, freqbins])

                xlocs = np.float32(np.linspace(0, timebins-1, 5))
                plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samp)/timebins)+(0.5*binsize))/samp_rate])
                ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
                plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

                plt.savefig(pngName, bbox_inches="tight")

                plt.clf() """

                # Define the frequency range to plot
                start_freq_pipi = 50
                end_freq_pipi = 220

                start_freq_noct = 0
                end_freq_noct = 45


                plt.figure(figsize=(9.3, 5))
                plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
                #plt.colorbar()

                #plt.xlabel("time (s)")
                #plt.ylabel("frequency (hz)")
                if folder == "Pipistrellus pygmaus wo social sound":
                    plt.xlim([0, timebins-1])
                    plt.ylim([start_freq_pipi, end_freq_pipi])  # Setting y-axis limits for the specified frequency range
                    plt.axis('off')  # Turn off axis
                    plt.margins(0, 0)  # Set margins to zero
                    plt.gca().set_aspect(18.60000)
                elif folder == "Pipistrellus pygmaus with social sound":
                    plt.xlim([0, timebins-1])
                    plt.ylim([start_freq_pipi, end_freq_pipi])  # Setting y-axis limits for the specified frequency range
                    plt.axis('off')  # Turn off axis
                    plt.margins(0, 0)  # Set margins to zero
                    plt.gca().set_aspect(18.60000)
                elif folder == "Noctula nyctalus with noise":
                    plt.xlim([0, timebins-1])
                    plt.ylim([start_freq_noct, end_freq_noct])  # Setting y-axis limits for the specified frequency range
                    plt.axis('off')  # Turn off axis
                    plt.margins(0, 0)  # Set margins to zero
                    plt.gca().set_aspect(18.60000)
                elif folder == "Noctula nyctalus with out social sound and noise":
                    plt.xlim([0, timebins-1])
                    plt.ylim([start_freq_pipi, end_freq_pipi])  # Setting y-axis limits for the specified frequency range
                    plt.axis('off')  # Turn off axis
                    plt.margins(0, 0)  # Set margins to zero
                    plt.gca().set_aspect(18.60000)
                elif folder == "test":
                    if fileName == "Noctula Nyctaus Example" or fileName == "Noctula Nyctaus Example w/ Noise" or fileName == "noctulawithnoise97":
                        plt.xlim([0, timebins-1])
                        plt.ylim([start_freq_noct, end_freq_noct])  # Setting y-axis limits for the specified frequency range
                        plt.axis('off')  # Turn off axis
                        plt.margins(0, 0)  # Set margins to zero
                        plt.gca().set_aspect(18.60000)
                    else:
                        plt.xlim([0, timebins-1])
                        plt.ylim([start_freq_pipi, end_freq_pipi])  # Setting y-axis limits for the specified frequency range
                        plt.axis('off')  # Turn off axis
                        plt.margins(0, 0)  # Set margins to zero
                        plt.gca().set_aspect(18.60000)
                

                #xlocs = np.float32(np.linspace(0, timebins-1, 5))
                #plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samp)/timebins)+(0.5*binsize))/samp_rate])
                #ylocs = np.int16(np.round(np.linspace(start_freq, end_freq, 10)))
                #plt.yticks(ylocs, ["%.02f" % f for f in ylocs])

                plt.savefig(pngName, bbox_inches='tight',pad_inches=0.0)
                plt.clf()



wavToSpectro(folders)