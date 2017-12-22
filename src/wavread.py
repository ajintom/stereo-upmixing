import numpy as np
from scipy.signal import resample, blackmanharris, triang
from scipy.fftpack import fft, ifft, fftshift
import math, copy, sys, os
from scipy.io.wavfile import write, read
from sys import platform
import subprocess
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), './utilFunctions_C/'))
try:
	import utilFunctions_C as UF_C
except ImportError:
	print "\n"
	print "-------------------------------------------------------------------------------"
	print "Warning:"
	print "Cython modules for some of the core functions were not imported."
	print "Exiting the code!!"
	print "-------------------------------------------------------------------------------"
	print "\n"
	sys.exit(0)
	
winsound_imported = False	
if sys.platform == "win32":
	try:
		import winsound
		winsound_imported = True
	except:
		print "You won't be able to play sounds, winsound could not be imported"

def isPower2(num):
	"""
	Check if num is power of two
	"""
	return ((num & (num - 1)) == 0) and num > 0

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

def wavread(filename):
	"""
	Read a sound file and convert it to a normalized floating point array
	filename: name of file to read
	returns fs: sampling rate of file, x: floating point array
	"""

	if (os.path.isfile(filename) == False):                  # raise error if wrong input file
		raise ValueError("Input file is wrong")

	fs, x = read(filename)

	if (len(x.shape) !=2):                                   # raise error if more than one channel
		raise ValueError("Audio file should be stereo")

	if (fs !=44100):                                         # raise error if more than one channel
		raise ValueError("Sampling rate of input sound should be 44100")

	#scale down and convert audio into floating point number in range of -1 to 1
	x = np.float32(x)/norm_fact[x.dtype.name]
	return fs, x