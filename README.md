README.txt 

Assignment: EE623 — Assignment 1  
Objective 1: To analyze recorded speech samples of selected Hindi vowels and consonants in terms of:  
• Pitch estimation using narrow-band spectrogram and AMD method  
• Formant analysis using wide-band spectrogram  

------------------------------------------------------------
Language Chosen:
Hindi
------------------------------------------------------------

Recording Details:
1. Recording Tool       : Praat
2. Sampling Frequency   : 44.1 kHz
3. Bit Resolution       : 16 bits/sample
4. Recording Type       : Mono
5. Speaker              : Adult Male
6. Average Pitch Range  : ~130–150 Hz

------------------------------------------------------------
Recorded Sounds:

Vowels (स्वर):
1. अ (a)
2. आ (aa)
3. इ (i)
4. ऊ (uu)
5. ए (e)

Consonants (व्यंजन):
1. क (ka) — Velar
2. ज (ja) — Palatal
3. ध (dha) — Retroflex
4. म (ma) — Nasal
5. ब (ba) — Labial

------------------------------------------------------------
File Naming Convention:
Each recording is saved as a separate .wav file using the pattern:

hindi_sound.wav

Examples:
hindi_a.wav  
hindi_aa.wav  
hindi_i.wav  

------------------------------------------------------------
Purpose of Collected Samples:
1. To generate narrowband and wideband spectrograms for pitch and formant analysis.  
2. To estimate pitch using the Average Magnitude Difference (AMD) method.

------------------------------------------------------------
Repository Structure:
Recordings/
├── amd_estimation.m
├── cepstral_analysis.m
├── hindi_a.wav
├── hindi_aa.wav
├── hindi_ee.wav
├── hindi_uu.wav
├── hindi_ae.wav
├── hindi_dha.wav
├── hindi_ka.wav
├── hindi_ja.wav
├── hindi_ma.wav
└── hindi_ba.wav

------------------------------------------------------------
Remarks:
• All samples were recorded in a quiet environment to reduce background noise.  
• Each sound was clearly pronounced and sustained for approximately 1 second.  
• These recordings serve as the dataset for subsequent spectrogram, AMD, and cepstral analyses.

------------------------------------------------------------
