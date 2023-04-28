# Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis

## Slides for hacker role demo
https://docs.google.com/presentation/d/1Fi-MWyfUKxw3K1TLeujFjCdjSk0Nunkez3h5-j_jySY/edit?usp=sharing

## Samples from previous repo
Samples could be found [here](./samples), where two kind of experiments were conducted:
1. Conditioning on reference audio:
    * BZ_440K.wav is an inference result from model trained on Blizzard2013 for 440K steps (batch_size=16), the conditioned referecne audio is picked from its testing set.
    * LJ_448K.wav is another inference result from model trained on LJ_Speech for 448K steps (batch_size=16), the conditioned referecne audio is also picked from its testing set.
2. Combinations of GSTs:
    * normal.wav and slow.wav are two inference results from model trained on LJ_Speech, the difference between the two is by picking difference style tokens for style embedding.
    * high.wav and low.wav is another pair of example.

## Requirements
Tensorflow 1.4 is used.

## Steps and Usages
1. Data Preprocess:
    - Prepare wavs and transcription
    - Example format:
<pre><code>Blizzard_2013|CA-MP3-17-138.wav|End of Mansfield Park by Jane Austin.
Blizzard_2013|CA-MP3-17-139.wav|Performed by Catherine Byers.
...</code></pre>
2. Make TFrecords for faster data loading:
    - Check parameters in hyperparams.py
        - path informations
        - TFrecords partition number
        - sample_rate
        - mel-filter banks number
    - Run:
<pre><code>python3 make_tfrecords.py</code></pre>
3. Train the whole network:
    - Check log directory and model, summary settings in hyperparams.py
    - Run:
<pre><code>python3 train.py</code></pre>
4. Evaluation while training:
    - (Currently only do evaluation on the first batch_size data)
    - (Decoder RNN are manually "FEED_PREVIOUS" now)
    - Run:
<pre><code>python3 eval.py</code></pre>
5. Inference:
    - Check Inference input text in hyperparams.py
    - Pass reference audio path as argument
    - Reference audio: an arbitary .wav file
    - Directly condition on combination of GSTs is now undergoing, set below flag in infer.py <code>condition_on_audio = False</code> and set the combination weight you like
    - Run:
<pre><code>python3 infer.py [ref_audio_path]</code></pre>
6. Inference input text example format:
<pre><code>0. Welcome to the speech recognition course.
1. Recognize speech
2. Wreck a nice beach
...</code></pre>

## Overall work done for Hacker role :
1. Downloaded and pre-processed LJSpeech-1.1 dataset, resulting in transcript file ready to be trained on.
2. Documented the network.py file which corresponds to the architecture of the model, including Style token layer and the tacotron modules
3. Trained on the subset of the LJSpeech dataset, for 1000 steps 
4. Experimented with hyperparams like number of tokens, etc to reduce loss
5. Ran eval and inference using the trained model, gets bad results, due to very less training time

## Further Changes Possible
1. The paper proposes a second method of inference which is directly conditioning on combination of GSTs. This can be implemented, in infer.py, allowing to style based on combination of weights of GSTs.
2. GSTs or weights can be used to automatically generate style annotations and this may substantially
reduce the human-in-the-loop efforts.
3. Subset selection of tokens can be done, using clustering techniques like K-means or by applying PCA. 

