import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import triang
from scipy.fftpack import fft
import stft as DFT
from scipy.signal import get_window
import sys,math,os
sys.path.append('/Users/mac/git/sms-tools/software/models')
import utilFunctions as UF

from scipy import signal
from scipy.signal import butter, lfilter, freqz


M = 2048
N = 2048
fs = 44100
w = get_window('hamming', M)
x = get_window('blackmanharris', M)
#x = signal.chebwin(M, at=100)
#x = x/sum(x)

#plt.plot(x)

m, p = DFT.dftAnal(x, w, N)
m = 10**(m/20); 
print np.shape(m)
plt.plot(m)
plt.show()

#y = DFT.dftSynth(m, p, w.size)*sum(w) #sum(w) is to normalize
