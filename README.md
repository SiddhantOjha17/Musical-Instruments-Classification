

# Musical Instrument Classification

Classification of 3 Musical Instrument:
1. Sitar
2. Violin
3. Mohan Veena

Play the audio files for yourself :)

Sitar:

https://user-images.githubusercontent.com/96463139/201171593-ed4e40bf-09b7-43b7-b501-ea431adf82c2.mp4

Violin:

https://user-images.githubusercontent.com/96463139/201172050-44cef93d-f41b-44e0-9e27-1bf679a30341.mp4

Mohan Veena:

https://user-images.githubusercontent.com/96463139/201172530-26e02fd7-26de-4c4f-bf18-37f1aeba409d.mp4


## Data
The data used is raw and not processed and taken from friends and family. Most of the audio files are of different lengths and have different Sampling Rate as well.
The hardest part of this Project was the Pre-Processing.

1. Rechannel
  Signals can either be mono or stereo. This method is used to get all our signals in the same dimensions.
  It converts all mono signals to stereo by duplicating the first channel
  Link for difference between mono/stereo : https://www.rowkin.com/blogs/rowkin/mono-vs-stereo-sound-whats-the-big-difference 

2. Resample
  The audio signals have different sampling rates as well. Hence, We need to standardise the sampling rate.
  Different sampling rates result in different array sizes. Ex: sr - 40000Hz means array size of 400000 whereas 40010Hz means aaray size of 40010.
  After standardisation we get all arrays of the same size

3. Padding and Truncating
  The audio files are bound to be of different lengths of time. This also needs to be standardised.
  This method either extends the length by padding with silence (Zero Padding) or reduces the length by truncating.

4. Mel-Spectogram
  This is one of the most important part of our Pre-Processing.
  The spectrogram does not us a lot of information.
  This happens because of the way humans perceive sound. Most of what we are able to hear are concentrated in a narrow range of frequencies and amplitudes.
  The way we hear frequencies in sound is known as ‘pitch’. It is a subjective impression of the frequency. So a high-pitched sound has a higher frequency than a low-pitched sound. Humans do not perceive frequencies linearly. We are more sensitive to differences between lower frequencies than higher frequencies.
  Humans hear them on a logarithmic scale rather than a linear scale.
  To deal with sound in a realistic manner, it is important for us to use a logarithmic scale via the Mel Scale and the Decibel Scale when dealing with Frequencies and Amplitudes in our data.
  This is where mel spectrograms helps us.

## Model
The model uses Artificial Neural Network. It has 3 layers:

1. Dense Layer with 512 nodes and Activation function Leaky Relu, input shape is defined to be (66048,).
2. Dense Layer with 128 nodes and Activation function Relu.
3. Dense Output Layer with Activation function Softmax (Multi Class Data)


After performing One-Hot Encoding

The model used categorical crossentropy to check the goodness of fit and used Adam optmizer to update its weights and other parameters. The model achieved around 100% accuracy on the test set.




## Future Development
Although this project is simple and easy to make, I really wanted to apply Neural Network to my own data and work on it from scratch.
I plan to use Convolutional Neural Network in near future.
I plan to add more instruments to this project and we could also use this to guess the instrumentsby just playing a tune on an application/ tuner.



