import scipy.signal
import numpy as np

srate = 700
def low_pass(order , ripple , flp):
    global srate
    nyquist = srate/2

    flpB , flpA = scipy.signal.cheby2(order , ripple,flp/nyquist , btype='lowpass')

    return flpB , flpA

def high_pass(order , ripple , fhp):
    global srate
    nyquist = srate/2

    fhpB , fhpA = scipy.signal.cheby2(order,ripple,fhp/nyquist , btype = 'highpass')
    return fhpB,fhpA

    