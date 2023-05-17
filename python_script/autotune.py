#!/usr/bin/python3
from pathlib import Path
import argparse
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as sig
from scipy.fft import fft, fftfreq
import psola


SEMITONES_IN_OCTAVE = 12


def degrees_from(scale: str):
    """Return the pitch classes (degrees) that correspond to the given scale"""
    degrees = librosa.key_to_degrees(scale)
    # To properly perform pitch rounding to the nearest degree from the scale, we need to repeat
    # the first degree raised by an octave. Otherwise, pitches slightly lower than the base degree
    # would be incorrectly assigned.
    degrees = np.concatenate((degrees, [degrees[0] + SEMITONES_IN_OCTAVE]))
    return degrees


def closest_pitch(f0):
    """Round the given pitch values to the nearest MIDI note numbers"""
    midi_note = np.around(librosa.hz_to_midi(f0))
    # To preserve the nan values.
    nan_indices = np.isnan(f0)
    midi_note[nan_indices] = np.nan
    # Convert back to Hz.
    return librosa.midi_to_hz(midi_note)


def closest_pitch_from_scale(f0, scale, filter = False):
    """Return the pitch closest to f0 that belongs to the given scale"""
    # Preserve nan.
    if np.isnan(f0):
        return np.nan
    degrees = degrees_from(scale)
    midi_note = librosa.hz_to_midi(f0)
    # Subtract the multiplicities of 12 so that we have the real-valued pitch class of the
    # input pitch.
    degree = midi_note % SEMITONES_IN_OCTAVE
    # Find the closest pitch class from the scale.
    degree_id = np.argmin(np.abs(degrees - degree))
    # Calculate the difference between the input pitch class and the desired pitch class.
    degree_difference = degree - degrees[degree_id]
    # Shift the input MIDI note number by the calculated difference.
    midi_note -= degree_difference
    # Convert to Hz.
    return librosa.midi_to_hz(midi_note)


def moving_average_filter(signal, window_size=512):
    return sig.fftconvolve(signal, np.ones(window_size) / window_size, mode='same')


# Estimate fundamental frequency using FFT
def estimate_fundamental_frequency(signal, sample_rate, previous_res, fmin, fmax):
    # Compute FFT of audio data
    fft_data = np.abs(fft(signal))
    fft_data = fft_data[:len(fft_data)//2+1]
    fft_freq = np.abs(fftfreq(len(signal), 1 / sample_rate))
    fft_freq = fft_freq[:len(fft_freq)//2+1]

    index1 = index2 = -1
    max_ampl = second_ampl = 0
    for i in range(len(fft_data)-2):
        if fmin < fft_freq[i] < fmax:
            if fft_data[i] > max_ampl:
                index2 = index1
                second_ampl = max_ampl
                index1 = i
                max_ampl = fft_data[i]

            elif fft_data[i] > second_ampl:
                index2 = i
                second_ampl = fft_data[i]

    if abs(fft_freq[index1] - previous_res) > abs(fft_freq[index2] - previous_res)*2:
        index1 = index2

    return fft_freq[index1]

# Estimate fundamental frequencies
def estimate_fundamental_frequencies(audio, sr, hop_length, fmin, fmax):
    # Estimate fundamental frequency of signal on window of size 1 second
    size = sr//hop_length+1
    window_size = hop_length*size
    second_f0 = []
    previous = 0
    for i in range(0, len(audio) - window_size, window_size):
        window = audio[i:i + window_size]
        res = estimate_fundamental_frequency(window, sr, previous, fmin, fmax)
        second_f0.append(res)
        previous = res

    # reformat to minimal window size
    f0 = []
    for i in range(len(second_f0)):
        for _ in range(size):
            f0.append(second_f0[i])
    for i in range(len(audio)//hop_length-len(second_f0)*size+1):
        f0.append(second_f0[-1])
    
    return f0


def aclosest_pitch_from_scale(f0, scale, filter = 0):
    """Map each pitch in the f0 array to the closest pitch belonging to the given scale."""
    sanitized_pitch = np.zeros_like(f0)
    for i in np.arange(f0.shape[0]):
        sanitized_pitch[i] = closest_pitch_from_scale(f0[i], scale)
    # Perform median filtering to additionally smooth the corrected pitch.
    if filter == 1:
        smoothed_sanitized_pitch = sig.medfilt(sanitized_pitch, kernel_size=11)
    elif filter == 2:
        smoothed_sanitized_pitch = moving_average_filter(sanitized_pitch, window_size=512)
    else:
        smoothed_sanitized_pitch = sanitized_pitch
    # Remove the additional NaN values after median filtering.
    smoothed_sanitized_pitch[np.isnan(smoothed_sanitized_pitch)] = \
        sanitized_pitch[np.isnan(smoothed_sanitized_pitch)]
    return smoothed_sanitized_pitch


def autotune(audio, sr, correction_function, scale, plot, smoothing):
    # Set some basis parameters.
    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C5')

    # # Pitch tracking using the PYIN algorithm.
    # f00, voiced_flag, voiced_probabilities = librosa.pyin(audio,
    #                                                      frame_length=frame_length,
    #                                                      hop_length=hop_length,
    #                                                      sr=sr,
    #                                                      fmin=fmin,
    #                                                      fmax=fmax)

    f0 = estimate_fundamental_frequencies(audio, sr, hop_length, fmin, fmax)

    f0 = np.array(f0)
    # # Apply the chosen adjustment strategy to the pitch.
    corrected_f0_median = correction_function(f0, scale, smoothing)
    if plot:
        # Plot the spectrogram, overlaid with the original pitch trajectory and the adjusted
        # pitch trajectory.
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        time_points = librosa.times_like(stft, sr=sr, hop_length=hop_length)
        log_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        # ax = plt.plot()
        plt.figure(1)
        # img = librosa.display.specshow(log_stft, x_axis='time', y_axis='log', ax=ax, sr=sr, hop_length=hop_length, fmin=fmin, fmax=8000)
        # fig.colorbar(img, ax=ax, format="%+2.f dB")
        plt.plot(time_points, f0, label='fft original pitch', color='cyan', linewidth=2)
        plt.plot(time_points, corrected_f0_median, label='fft corrected pitch with median smoothing', color='orange', linewidth=1)
        plt.plot(time_points, corrected_f0_simple, label='fft corrected pitch simple', color='red', linewidth=1)
        plt.plot(time_points, corrected_f0_moving, label='fft corrected pitch with moving average', color='green', linewidth=1)
        plt.legend(loc='lower right')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [M:SS]')
        plt.title("Corrected speach of male short sounds by fft")
        plt.savefig('pitch_correctionfft.png', dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure(2)
        plt.plot(time_points, f00, label='yin original pitch', color='cyan', linewidth=2)
        plt.plot(time_points, corrected_f00_simple, label='yin corrected pitch simple', color='red', linewidth=1)
        plt.plot(time_points, corrected_f00_median, label='yin corrected pitch wirh median smoothing', color='orange', linewidth=1)
        plt.plot(time_points, corrected_f00_moving, label='yin corrected pitch with moving average', color='green', linewidth=1)
        
        plt.legend(loc='lower right')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [M:SS]')
        plt.title("Corrected speach of male short sounds by yin")
        plt.savefig('pitch_correctionf.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Pitch-shifting using the PSOLA algorithm.
    return psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0_moving, fmin=fmin, fmax=fmax)


def main(filepath, smoothing, plot=False, write=True, scale="D:min"):
    filepath = Path(filepath)

    # Load the audio file.
    y, sr = librosa.load(str(filepath), sr=None, mono=False)

    # Only mono-files are handled. If stereo files are supplied, only the first channel is used.
    if y.ndim > 1:
        y = y[0, :]

    pitch_corrected_y = autotune(y, sr, closest_pitch, scale, plot, int(smoothing))

    if write:
        # Write the corrected audio to an output file.
        filepath = filepath.parent / (filepath.stem + '_pitch_corrected' + filepath.suffix)
        sf.write(str(filepath), pitch_corrected_y, int(sr))


# if __name__ == '__main__':
#     main()
    