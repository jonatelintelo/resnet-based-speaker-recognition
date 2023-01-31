## Automatic Speaker Recognition

Automatic Speaker Recognition (determining the identity of the person that is speaking in a recording) is a research topic in the area of Speech Processing, alongside of similar topics such as Speech Recognition (finding the words spoken), accent, language, and emotion recognition, but also speech synthesis (generating speech from text), speech coding, compression, or perhaps the most basic of all: speech activity detection (finding out the periods in an audio recording where there is any speech).  In speaker recognition we're interested in the technique of recognizing speakers, not in particular persons per se, e.g., Osama bin Laden or president Biden.  Therefore, the problem is often cast as a _speaker detection task_: given _two_ segments of audio, determine whether these were spoken by the same speaker or by different speakers.  Traditionally, one of these segments was called the _enrollment_, the other the _test_ segment.  A direct application of this task is _speaker verification_: is this person speaking the person who she claims she is?  Often this application is contrasted to _speaker identification_, where the task is: what is the actual identity of the person speaking?  A system that can perform well on the speaker detection task can quite easily be used effectively in the other applications, and it is therefore that in all speaker recognition research this task is the topic of study.  And so it is in this course. 

Any speaker recognition system internally works with a _score_ that expresses the similarity between the speakers of the two segments.  We traditionally work with a sense of the score that is bigger to indicate more similarity (as opposed to a distance score that would get smaller, and is often bounded by 0).  For an actual _decision_ such a score needs to be thresholded: scores higher than the threshold will get the decision "same speaker", whereas comparisons with a score lower than the threshold will receive the decision "different speaker".  Setting such a threshold is actually far from trivial, and in general depends on a specific application and priors.  The capability of setting thresholds well is called _calibration_, and is a research area in itself.  In this course, like in most of the speaker recognition research, we will not assess calibration. 

### Evaluation metrics

Systems are evaluated by giving them many pairs of speech segments.  Each pair is called a _trial_, and the task is to give a score that is higher when the system finds the speakers more likely to be the same speaker.  For the _evaluation test set_, the identities of the speakers in the test segments are not known by the system (or by the students of this course building the system).  When the scores are submitted for evaluation, the identities are used to compute the performance metric, the _Equal Error Rate_ (EER).  We will give a short explanation what this metric means and how it is computed. 

There are two kinds of trials:

 - _target_ trials, when both segments are spoken by the same speaker, 
 - _non-target_ trials, when the two segments are spoken by different speakers. 

When a system would make a decision by thresholding a score, two different types of errors can be made.  A target trial can be classified as "different speakers", this is called a _false negative_ or _missed speaker_.  Alternatively, a non-target trial can be classified as "same speaker", this is called a _false positive_ or _false alarm_.  The submitted scores can be grouped in "target scores" and "non-target scores", once the speaker identities are known, and their [distributions](./images/2distributions.pdf) will be different, as non-target scores tend to be lower than target scores, as this is the task of the speaker recognition system.

When, given a set of submitted scores, the threshold is swept from low to high, the corresponding false negative rate (the number of false negatives divided by the number of target trials) and false positive rate (the number of false positives divided by the number of non-target trials) will vary from the extreme (FNR = 0, FPR = 1), when the threshold is below the lowest submitted score, to the other extreme (FNR = 1, FPR = 0) when it is above the highest.  In-between these extremes, the false negative rate is traded off against the false positive rate---this is where the action is.  This trade-off can be appreciated in a parametrized plot of the false negative rate versus the false positive rate.  This essentially is a [Receiver Operating Characteristic](./images/eer-roc.pdf) (ROC), but in speaker recognition we're used to warping the axes a little and calling this plot a [Detection Error Trade-off](./images/lineup-dets.pdf) (DET) plot. 

The trick now is that, given the set of submitted trials, this whole process can be done by the evaluator, who knows the identities of the speakers in the evaluation trials.  Rather than the full trade-off curve, it is nice to have a single metric that is characteristic of the entire curve.  There are many candidates for this, but in speaker recognition it is customary to use the [Equal Error Rate](images/eer-roc.pdf), this is the place on the curve where the false negative rate is equal to the false positive rate.  Typically, when the equal error rate is lower, the system is better.  The range of the EER is from 0 (perfect separation of target and non-target trials) to 50% (random scores). 

During the development of a system it is useful to have a set of trials, the _development set_, with speakers that are different from the speaker that are used in training, and for which you know for each trial whether it is a target or non-target trial, so that you can test your system and inspect scores and compute the EER yourself.  Although it is not rocket science to compute an EER, there are some caveats here and there, so we will provide code that computes the EER efficiently.  The EER we compute actually is the _ROC Convex Hull EER_, the value you get by first computing the convex hull of the (steppy) [ROC](./images/eer-roc.pdf), and then intersecting this with the line FPR = FNR.  

## A very short summary of existing approaches

Modern approaches to speaker recognition are end-to-end, they directly start from the uncompressed audio waveform data, processing it using a neural network starting with a couple of CNN layers (creating features), followed by transformers and/or fully connected layers, and then pooling over the time-dimension, producing an _embedding_.  During training this is further classified using a fully connected layer with all speaker IDs from the train set as targets, and during inference the embeddings of the two sides of the test trial are compared using something like a cosine score.  

Most earlier approaches first do some kind of _feature extraction_, often computing Mel Frequency Cepstral Coefficients (MFCCs), representing the audio waveform as a sequence of fixed-length vectors.  These can then be further processed by, e.g., a neural net, as in the case of x-vectors, or directly modeled.  Very early approaches used Gaussian Mixture Models (GMMs), and later deviations from a Uniform Background Model (UBM)---a large GMM modeling all possible speech---were used to compute the comparison score.  Later Support Vector Models modeled these deviations, and then Joint Factor Analysis (JFA) managed to factorize directions in these deviations into components that stem from speaker variation and components that stem from _session variation_.  Session variation, incidentally, has always been seen as the hardest problem in automatic speaker recognition.  A clever continuation to this approach was to use JFA techniques, but not explicitly separating for session and speaker, producing a single vector representing the entire speech utterance.  These vectors were coined _i-vectors_, and they stood at the basis of virtually all speaker recognition systems until performance was surpassed by neural networks. 

Apart from working on basic disciriminability of speakers, a lot of performance can be gained by various forms of normalization (making score distributions from different sub-sets of speakers become very similar) and calibration (making it possible to make the correct decision same/different using a precomputed threshold).  

## Some literature

 - A recent overview, including deep learning models: [Speaker recognition based on deep learning: An overview](./papers/bai-overview.pdf) Zhongxin Bai, Xiao-Lei Zhang, [Neural Networks, Volume 140, 2021](https://www.sciencedirect.com/science/article/pii/S0893608021000848), Pages 65-99, ISSN 0893-6080, https://doi.org/10.1016/j.neunet.2021.03.004. 
 - X vectors, the first neural embeddings: [X-vectors: Robust DNN embeddings for speaker recogntion](./papers/xvectors.pdf) David Snyder, Daniel Garcia-Romero, Gregory Sell, Daniel Povey, Sanjeev Khudanpur, [proc. ICASSP 2018](https://ieeexplore.ieee.org/document/8461375), pp. 5329-5333, doi: 10.1109/ICASSP.2018.8461375. 
 - Neural embeddings from raw waweforms instead of MFCCs: [Wav2Spk: A Simple DNN Architecture for Learning Speaker Embeddings from Waveforms](./papers/wav2spk.pdf), Weiwei Lin and Man-Wai Mak, Proc. Interspeech 2020, 3211-3215, doi: 10.21437/Interspeech.2020-1287
 - Recent state-of-the-art model: [ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification](./papers/ecapa_tdnn.pdf), Brecht Desplanques, Jenthe Thienpondt, Kris Demuynck, Proc. Interspeech 2020, 3830-3834, doi: 10.21437/Interspeech.2020-2650
 - Recent comparison of loss functions: [In Defence of Metric Learning for Speaker Recognition](./papers/metric_learning.pdf), Joon Son Chung et al, Proc. Interspeech 2020, 2977-2981, doi: 10.21437/Interspeech.2020-1064
 - I-Vectors: the state of the art for many years, a form of embeddings avant-la-lettre: [Front-End Factor Analysis for Speaker Verification](./papers/najim-ivector-taslp-2009.pdf) Dehak, N. and Kenny, P. J. and Dehak, R. and Dumouchel, P. and Ouellet, P., [IEEE Trans. on Audio, Speech and Language Processing](http://ieeexplore.ieee.org/document/5545402), vol. 19, no. 4, pp. 788-798, May 2011, doi: 10.1109/TASL.2010.2064307.
 - An introduction to calibration: [An Introduction to Application-Independent Evaluation of Speaker Recognition Systems](./papers/appindepeval-lnai-2007.pdf) David A. van Leeuwen and Niko Brümmer, 2007, In: Müller C. (eds) [Speaker Classification I.](https://link.springer.com/chapter/10.1007%2F978-3-540-74200-5_19) Lecture Notes in Computer Science, vol 4343. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-74200-5_19
 - An older overview: [A Tutorial on Text-Independent Speaker Verification](./papers/bimbot-overview.pdf), Bimbot, F., Bonastre, JF., Fredouille, C. et al. [EURASIP J. Adv. Signal Process. 2004](https://asp-eurasipjournals.springeropen.com/articles/10.1155/S1110865704310024), 101962 (2004).
 - An even older overview, by Dough Reynolds, who came up with the GMM/UBM approach: [An overview of automatic speaker recognition technology](./papers/reynolds-overview.pdf), D. A. Reynolds, [2002 IEEE International Conference on Acoustics, Speech, and Signal Processing](https://ieeexplore.ieee.org/document/5745552), 2002, pp. IV-4072-IV-4075, doi: 10.1109/ICASSP.2002.5745552.
 - Speaker recognition through what people say, not how they sound, by the nestor of automatic speaker recognition: [Speaker recognition based on idiolectal differences between speakers](./papers/doddington-idiolect-interspeech2001.pdf) George Doddington, [Proc. Interspeech 2001](https://www.isca-speech.org/archive/eurospeech_2001/doddington01_eurospeech.html), 2521-2524. 
 - A very old overview by a very respectable speech scientist: [Recent advances in speaker recognition](./papers/furui-overview.pdf) Sadaoki Furui, [Pattern Recognition Letters, Volume 18, Issue 9, 1997](https://www.sciencedirect.com/science/article/abs/pii/S0167865597000731), Pages 859-872, ISSN 0167-8655.