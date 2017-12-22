import numpy as np
import math, copy, sys, os
import matplotlib.pyplot as plt

import scipy.io.wavfile as wf
from scipy.io.wavfile import write
from scipy.signal import get_window
from scipy import signal

import low_pass as crossover
import stft as DFT

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

#--------------------------------------------------------------------------------
#defining paramteres for STFT

M = 4096	 	#frame size
N = 32768	 	#FFT size
H = int(M/4) 	#hop size
maxFreq = 20000.0
W = 'hamming'
w = get_window(W, M)
#--------------------------------------------------------------------------------
#defining paramters for ambience extraction 

u1 = 1.0
u0 = 0.001
sigma = 8
phi0 = 0.5
lamda = 0.1 #forgetting factor 

#defining paramteres for unmixing
v = 0.00001		 #low floor
E = 0.125		 #gaussian window width for left and right channels
E_C = E 		 #guassian window width of center channel
				 #individual window widths can be controlled depending on desired center image
si_l = -1.0 	 #panning index for left
si_c = 0.0		 #panning index for center
si_r = 1.0		 #panning index for right

smoothingKernelSize = 101  #odd
#--------------------------------------------------------------------------------
#reading from wavfile, allows only stereo

#fs, signal_read = wf.read('W_SP.wav') # bursts of 5 tones panned left to right
fs, signal_read = wf.read('sine_pan.wav') # sine tone panned continuously left to right
#fs, signal_read = wf.read('W_sine_pan.wav') # sine tone panned continuously left to right with white noise

signal = np.float32(signal_read)/norm_fact[signal_read.dtype.name]
s1 = signal[:, 0] #left channel
s2 = signal[:, 1] #right channel

#Testing simple amplitude panned signals
#s2 = 0.2*s1
#s1 = .8* s1

# plt.plot(s1)
# plt.show()
#--------------------------------------------------------------------------------
#applying low pass filter for subwoofer
# sig = crossover.butter_lowpass_filter(signal, 0.01, 100.0, 10)
# crossover = np.int16(sig/np.max(np.abs(sig)) * 32767)
# write('SUB.wav', 44100, 2 * crossover[:,0])

# plt.plot(crossover)
# plt.show()
#--------------------------------------------------------------------------------

def c_add(m1, p1, m2, p2):	#complex addition
	r = np.multiply(m1,np.cos(p1)) + np.multiply(m2,np.cos(p2))
	i = np.multiply(m1,np.sin(p1)) + np.multiply(m2,np.sin(p2))
	m = np.sqrt(np.multiply(r,r) + np.multiply(i,i))
	p = np.arctan(np.divide(i,r))
	return m, p
#--------------------------------------------------------------------------------

def coherence(mX, pX, mY, pY, m, p, lamda=0.1): #lamda = forgetting factor
	m_11, p_11 = c_add( (1 - lamda) * m[0], p[0], (lamda * np.multiply(mX,mX)), 0 )
	m_22, p_22 = c_add( (1 - lamda) * m[1], p[1], (lamda * np.multiply(mY,mY)), 0 )
	m_12, p_12 = c_add( (1 - lamda) * m[2], p[2], (lamda * np.multiply(mX,mY)), (pX - pY))
	return np.transpose(np.c_[m_11, m_22, m_12]),np.transpose(np.c_[p_11, p_22, p_12])

#--------------------------------------------------------------------------------
def movingAvg (arr, kSize = smoothingKernelSize):#moving average filter
	# odd kSize
	arr2 = np.append(arr[0]*np.ones((kSize-1)/2),arr)
	arr2 = np.append(arr2,arr[np.size(arr)-1]*np.ones((kSize-1)/2))
	arr2 = np.convolve(arr2, np.ones((kSize,))/kSize,mode='valid')
	return arr2

#------STFT--------------------------------------------------------------------------

def stft_anal_synth(s1,s2, fs, w, N, H, 
	m_phi=[np.zeros(1+N/2), np.zeros(1+N/2), np.zeros(1+N/2)], p_phi=[np.zeros(1+N/2), np.zeros(1+N/2), np.zeros(1+N/2)],
	m_sim=[np.zeros(1+N/2), np.zeros(1+N/2), np.zeros(1+N/2)], p_sim=[np.zeros(1+N/2), np.zeros(1+N/2), np.zeros(1+N/2)]):
	"""
	STFT analysis-synthesis for Ambience Extraction
	s1: stereo_left
	s2: stereo_right

	w: analysis window, N: FFT size, H: hop size
	returns y: output sound
	"""
	M = w.size                                    	 # size of analysis window
	hM1 = int(math.floor((M+1)/2))                	 # half analysis window size by rounding
	hM2 = int(math.floor(M/2))                    	 # half analysis window size by floor
	s1 = np.append(np.zeros(hM2),s1)                 # add zeros at beginning to center first window at sample 0
	s1 = np.append(s1,np.zeros(hM1))                 # add zeros at the end to analyze last sample
	s2 = np.append(np.zeros(hM2),s2)                 # add zeros at beginning to center first window at sample 0
	s2 = np.append(s2,np.zeros(hM1))                 # add zeros at the end to analyze last sample
	pin = hM1                                        # initialize sound pointer in middle of analysis window       
	pend = s1.size-hM1                               # last sample to start a frame
	w = w / sum(w)                                   # normalize analysis window
	yL = np.zeros(s1.size)                           # initialize output array
	yR = np.zeros(s2.size)                          
	yL_F = np.zeros(s2.size) 
	yC = np.zeros(s2.size) 
	yR_F = np.zeros(s2.size) 

	max_L = np.zeros(1)
	max_R = np.zeros(1)
	max_C = np.zeros(1)
	max_1 = np.zeros(1)
	max_2 = np.zeros(1)
	while pin<=pend:                              	 # while sound pointer is smaller than last sample      
	#-----------------analysis---------------------------------  
		x1 = s1[pin-hM1:pin+hM2]                   	 # select one frame of input sound
		mX1, pX1 = DFT.dftAnal(x1, w, N)             # compute dft
		x2 = s2[pin-hM1:pin+hM2]
		mX2, pX2 = DFT.dftAnal(x2, w, N)

		mX1 = 10**(mX1/20); 
		mX2 = 10**(mX2/20);

	#----------------spectral transformations------------------

		#-----caluclating inter-channel short-time coherence------
		m_phi, p_phi = coherence(mX1, pX1, mX2, pX2, m_phi, p_phi, lamda)
		phi = np.divide(m_phi[2],np.sqrt(np.multiply(m_phi[0],m_phi[1])))
		
		if (np.sum(p_phi[0])!=0	or np.sum(p_phi[1])!=0):
			print("coh_phases not cancelling")

		tau = ((u1-u0)/2)*np.tanh(sigma*np.pi*(phi0-phi)) + ((u1+u0)/2)
		
		mY1 = np.multiply(mX1,tau)
		mY2 = np.multiply(mX2,tau)

		#--caluclating similarity for identifying panned sources and unmixing them--	
		#-copmute coherence with lamda = 1.0
		m_sim, p_sim = coherence(mX1, pX1, mX2, pX2, m_sim, p_sim, 1.0) 
		sim = 2 * np.divide( m_sim[2], np.add(m_sim[0],m_sim[1])) #similarity function

		if (np.sum(p_sim[0])!=0	or np.sum(p_sim[1])!=0):
			print("sim_phases not cancelling")	

		sim_0 = np.divide(m_sim[2], m_sim[0]) #partial similarity function for left channel
		sim_1 = np.divide(m_sim[2], m_sim[1]) #partial similarity function for right channel

		diff = np.subtract(sim_0, sim_1) #equation 8 in the report
		pos = (diff>0).astype(int) * 1	
		neg = (diff<0).astype(int) * -1
		delta = np.add(pos, neg) 		 #equation 9	
		
		pan_ind = np.multiply ( np.subtract(np.ones(np.size(sim)), sim), delta) #equation 10
		#moving average filter to smoothen panning indices across frequency points
		pan_ind = movingAvg(pan_ind) 	  

		gwf_l = v + (1-v) * np.exp(np.multiply((-1/(2*E)), np.square(np.subtract(pan_ind, si_l)) ) )
		gwf_c = v + (1-v) * np.exp(np.multiply((-1/(2*E)), np.square(np.subtract(pan_ind, si_c)) ) )
		gwf_r = v + (1-v) * np.exp(np.multiply((-1/(2*E)), np.square(np.subtract(pan_ind, si_r)) ) )
		
		#normalizing the windows
		gwf_sum = gwf_l + gwf_c + gwf_r
		gwf_l = np.divide(gwf_l, gwf_sum)
		gwf_c = np.divide(gwf_c, gwf_sum)
		gwf_r = np.divide(gwf_r, gwf_sum)
		#complex addition
		mX_sumLR, pX_sumLR = c_add(mX1, pX1, mX2, pX2)	
		#equation 13
		mY_L = np.multiply(gwf_l, mX_sumLR)
		mY_C = np.multiply(gwf_c, mX_sumLR)
		mY_R = np.multiply(gwf_r, mX_sumLR)

		#-----Power Compensation---------RMS(stereo) = RMS(Surround)-----------	

		total_pow = np.sum(np.square(mX1))+ np.sum(np.square(mX2))   # Stereo RMS Power

		new_pow = np.sum(np.square(mY1))+np.sum(np.square(mY2))+np.sum(np.square(mY_L)) + np.sum(np.square(mY_C))+np.sum(np.square(mY_R))  # 5.1 RMS Power

		pow_ratio = np.sqrt(total_pow/new_pow);						 # Ratio of power 

		# normalizing output spectrum 
		mY1 = pow_ratio*mY1;		 #rear-left								
		mY2 = pow_ratio*mY2;		 #read-right
		mY_L = pow_ratio*mY_L;		 #front-left
		mY_C = pow_ratio*mY_C;		 #front-center	
		mY_R = pow_ratio*mY_R;		 #front-right

		max_L = np.append(max_L,np.sum(np.square(mY_L)))
		max_C = np.append(max_C,np.sum(np.square(mY_C)))
		max_R = np.append(max_R,np.sum(np.square(mY_R)))
		max_1 = np.append(max_1,np.sum(np.square(mX1)))
		max_2 = np.append(max_2,np.sum(np.square(mX2)))

				#print np.max(mY_L),np.max(mY_C),np.max(mY_R)
		#print np.max(mX1),np.max(mX2)
		

		#-----spectral plots------	

		#print(np.min(mY1),np.max(mY1),np.min(mY2),np.max(mY2))
		#print(np.min(mX1),np.max(mX1),np.min(mX2),np.max(mX2))
		#plt.plot(mY1)
		#plt.plot(mX1)
		#plt.show()

		#plt.plot(mY1)
		# plt.plot(gwf_l)
		# plt.plot(gwf_c)
		# plt.plot(gwf_r)
		# plt.show()

		#plt.plot(mX1)
		#plt.plot(mY_C)
		#plt.show()

		mY1 = 20*np.log10(mY1); 
		mY2 = 20*np.log10(mY2);
		mY_L = 20*np.log10(mY_L); 
		mY_C = 20*np.log10(mY_C);
		mY_R = 20*np.log10(mY_R); 

	#-------------------------synthesis-----------------------------

		#----ambience: rear left and right speakers-----
		y1 = DFT.dftSynth(mY1, pX1, M)               	# compute idft
		yL[pin-hM1:pin+hM2] += H*y1                 	# overlap-add to generate output sound  
		y2 = DFT.dftSynth(mY2, pX2, M)               
		yR[pin-hM1:pin+hM2] += H*y2                 
		
		#----front image: Left, Center, Right speakers-----
		yl = DFT.dftSynth(mY_L , pX_sumLR, M)              
		yL_F[pin-hM1:pin+hM2] += H*yl                

		yc = DFT.dftSynth(mY_C , pX_sumLR, M)               
		yC[pin-hM1:pin+hM2] += H*yc                 		

		yr = DFT.dftSynth(mY_R, pX_sumLR, M)               
		yR_F[pin-hM1:pin+hM2] += H*yr                 

		#----hopping----
		pin += H                                   	 # advance sound pointer
	                              
	yL = np.delete(yL, range(hM2))                   # delete half of first window which was added in dftAnal
	yL = np.delete(yL, range(yL.size-hM1, yL.size))  # add zeros at the end to analyze last sample
	yR = np.delete(yR, range(hM2)) 
	yR = np.delete(yR, range(yR.size-hM1, yR.size))   
	yL_F = np.delete(yL_F, range(hM2)) 
	yL_F = np.delete(yL_F, range(yL_F.size-hM1, yL_F.size))  
	yR_F = np.delete(yR_F, range(hM2)) 
	yR_F = np.delete(yR_F, range(yR_F.size-hM1, yR_F.size)) 
	yC = np.delete(yC, range(hM2)) 
	yC = np.delete(yC, range(yC.size-hM1, yC.size)) 

	# label_1, = plt.plot(max_L, label='max_L')
	# label_2, = plt.plot(max_C, label='max_C')
	# label_3, = plt.plot(max_R, label='max_R')
	# label_4, = plt.plot(max_1, label='max_1')
	# label_5, = plt.plot(max_2, label='max_2')
	# plt.legend(handles=[label_1,label_2,label_3,label_4,label_5])
	# plt.show()

	return yL, yR, yL_F, yC, yR_F
#--------------------------------------------------------------------------------

#output
synth_aL, synth_aR, synthL_F, synth_C, synthR_F = stft_anal_synth(s1,s2, fs, w, N, H)
#synthL_F = s1 - synth_aL #subtracting ambienceLEFT from originalLEFT
#synthR_F = s2 - synth_aR #subtracting ambienceLEFT from originalLEFT

def raw_to_int16(synth):
	out = copy.deepcopy(synth)                     # copy array
	out *= INT16_FAC                               # scaling floating point -1 to 1 range signal to int16 range
	out = np.int16(out)                            # converting to int16 type
	return out

Rear_Left = raw_to_int16(synth_aL)      
Rear_Right = raw_to_int16(synth_aR)   
Center= raw_to_int16(synth_C)   
Front_Left = raw_to_int16(synthL_F)      
Front_Right = raw_to_int16(synthR_F)                 

write('_Rear_Left.wav', 44100, Rear_Left)
write('_Rear_Right.wav', 44100, Rear_Right)
write('_Center.wav', 44100, Center)
write('_Front_Left.wav', 44100, Front_Left)
write('_Front_Right.wav', 44100, Front_Right)
#plt.gcf().clear()
#plt.plot(s1 - s2)
#plt.show()


