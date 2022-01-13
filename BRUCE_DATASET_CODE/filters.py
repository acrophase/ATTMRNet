import scipy.signal
import numpy as np

def baseline_removal(raw_signal):
    '''
    input - raw_ecg_signal.
    output - signal after removing the baseline wander.

    Description - This function takes the raw ecg signal and then a median filter of 200ms and 600ms 
    is applied on raw_ecg signal. Applying the median filter twice will give the baseline wander, then subtract
    this baseline wander from raw_ecg to get the ecg signal without baseline wandeing.
    '''
    srate = 256
    l1 = int(0.2*srate)
    l2 = int(0.6*srate)
    if l1%2==0:
        l1 = l1+1
    if l2%2==0:
        l2 = l2+1       
    bw1 = scipy.signal.medfilt(raw_signal,l1)
    bw2 = scipy.signal.medfilt(bw1,l2)
    output = raw_signal-bw2
    return output

def band_pass(lcf,ucf,order):
    '''
    input -- lcf == lower cuttoff frequency
             ucf == upper cuttoff frequency
             order == order of the filter
    output-- filter kernal coefficients.

    Description -- This function creates a bandpass filter for smoothing the EDR signal. It takes
    lower cuttoff frequency, upper cuttoff frequency, order and return the filter coefficients of a bandpass 
    filter.
    '''
    fs = 4
    nyquist = fs/2
    if (lcf<ucf):
        frange = [lcf,ucf]
        fbpB,fbpA = scipy.signal.butter(order,np.array(frange)/nyquist,btype='bandpass')
        return fbpB,fbpA
    else:
        raise ValueError

def low_pass(order,ripple,flp):
    srate   = 25.6 # hz
    nyquist = srate/2
    fkernB,fkernA = scipy.signal.cheby2(order,ripple,np.array(flp)/nyquist,btype='lowpass')
    return fkernB,fkernA

def high_pass(order,ripple,flp):
    srate   = 25.6 # hz
    nyquist = srate/2
    fkernB,fkernA = scipy.signal.cheby2(order,ripple,np.array(flp)/nyquist,btype='highpass')
    return fkernB,fkernA


def cheby_lp (order , ripple , flp):
    '''
    inputs - order -- order of filter
             ripple -- ripple in filter
             flp -- lower cutoff frequency
    outputs - filter kernal A and B

    Description - Function design a chebyshev low pass filter
    '''
    srate = 256
    nyquist = srate/2
    fkernB , fkernA = scipy.signal.cheby2(order , ripple , flp/nyquist , btype='lowpass')
    return fkernB , fkernA

def cheby_hp (order , ripple , fhp):
    '''
    inputs - order -- order of filter
             ripple -- ripple in filter
             fhp -- upper cutoff frequency
    outputs - filter kernal A and B

    Description - Function design a chebyshev high pass filter
    '''
    srate = 256
    nyquist = srate/2
    fkernB , fkernA = scipy.signal.cheby2(order , ripple , fhp/nyquist , btype='highpass')
    return fkernB , fkernA