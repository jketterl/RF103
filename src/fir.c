/*
This software is part of libcsdr, a set of simple DSP routines for
Software Defined Radio.

Copyright (c) 2014, Andras Retzler <randras@sdr.hu>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ANDRAS RETZLER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "fir.h"

#define MFIRDES_GWS(NAME) \
    if(!strcmp( #NAME , input )) return WINDOW_ ## NAME;

window_t firdes_get_window_from_string(char* input)
{
    MFIRDES_GWS(BOXCAR);
    MFIRDES_GWS(BLACKMAN);
    MFIRDES_GWS(HAMMING);
    return WINDOW_DEFAULT;
}

#define MFIRDES_GSW(NAME) \
    if(window == WINDOW_ ## NAME) return #NAME;

char* firdes_get_string_from_window(window_t window)
{
    MFIRDES_GSW(BOXCAR);
    MFIRDES_GSW(BLACKMAN);
    MFIRDES_GSW(HAMMING);
    return "INVALID";
}

float firdes_wkernel_blackman(float rate)
{
    //Explanation at Chapter 16 of dspguide.com, page 2
    //Blackman window has better stopband attentuation and passband ripple than Hamming, but it has slower rolloff.
    rate=0.5+rate/2;
    return 0.42-0.5*cos(2*M_PI*rate)+0.08*cos(4*M_PI*rate);
}

float firdes_wkernel_hamming(float rate)
{
    //Explanation at Chapter 16 of dspguide.com, page 2
    //Hamming window has worse stopband attentuation and passband ripple than Blackman, but it has faster rolloff.
    rate=0.5+rate/2;
    return 0.54-0.46*cos(2*M_PI*rate);
}


float firdes_wkernel_boxcar(float rate)
{   //"Dummy" window kernel, do not use; an unwindowed FIR filter may have bad frequency response
    return 1.0;
}

float (*firdes_get_window_kernel(window_t window))(float)
{
    if(window==WINDOW_HAMMING) return firdes_wkernel_hamming;
    else if(window==WINDOW_BLACKMAN) return firdes_wkernel_blackman;
    else if(window==WINDOW_BOXCAR) return firdes_wkernel_boxcar;
    else return firdes_get_window_kernel(WINDOW_DEFAULT);
}

void normalize_fir_f(float* input, float* output, int length)
{
    //Normalize filter kernel
    float sum=0;
    for(int i=0;i<length;i++) //@normalize_fir_f: normalize pass 1
        sum+=input[i];
    for(int i=0;i<length;i++) //@normalize_fir_f: normalize pass 2
        output[i]=input[i]/sum;
}

void firdes_lowpass_f(float *output, int length, float cutoff_rate, window_t window)
{   //Generates symmetric windowed sinc FIR filter real taps
    //  length should be odd
    //  cutoff_rate is (cutoff frequency/sampling frequency)
    //Explanation at Chapter 16 of dspguide.com
    int middle=length/2;
    float temp;
    float (*window_function)(float)  = firdes_get_window_kernel(window);
    output[middle]=2*M_PI*cutoff_rate*window_function(0);
    for(int i=1; i<=middle; i++) //@@firdes_lowpass_f: calculate taps
    {
        output[middle-i]=output[middle+i]=(sin(2*M_PI*cutoff_rate*i)/i)*window_function((float)i/middle);
        //printf("%g %d %d %d %d | %g\n",output[middle-i],i,middle,middle+i,middle-i,sin(2*PI*cutoff_rate*i));
    }
    normalize_fir_f(output,output,length);
}
