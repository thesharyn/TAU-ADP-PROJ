=======================================
     README for Conformer Model
   Training and Evaluation Setup
=======================================

Setup
-----
1. Create a Conda Environment on Linux

	$ conda create -n conformer0 python=3.10 sox -c conda-forge
	$ conda activate conformer0
  
   Notes:
	a. While not strictly required, using Conda is advised for preventing conflicts and making sure you do not mess with
		other projects' environments.
	b. The 'sox' package supplies the necessary torchaudio backend for Linux. Python 3.10 is required.

2. Install necessary packages

	$ pip install -r requirements.txt

3. Download the LibriSpeech dataset (if missing)

	$ python download-librispeech.py
	

   Notes:
	a. By default, this downloads the 6 necessary splits of the dataset ('train-clean-100', 'train-clean-360', 
		'train-other-500', 'dev-clean', 'test-clean', 'test-other') into ./dataset/LibriSpeech using a European server
		(for faster download) and with asking confirmation for download size first.
	b. Observe the script's flags for more customized usage; it can be used to download parts of this dataset regardless
		of this specific project


Train and Evaluate
------------------

1. (Optional) Change HuggingFace default cache directory

	For example:	

	$ setenv HF_HOME /home/yandex/APDL2425a/group_8/bin/HuggingFace-cache

	This might not be necessary or desired; we point this out though, because default cache directory might not be enough for
	the training script to be able to download the pre-trained neural language model (LM).

2. Run train and evaluate script of the Conformer-model

	$ python train.py hparams/conformer_small.yaml --data_folder datasets/LibriSpeech/


   Notes:
	a. The conformer_small.yaml details the model's archtecture and hyperparameters. It is taken from the a receipe inside
		the SpeechBrain repository (last commit for documentation: d9fb58f56). 
	b. Our only changes in the YAML file here were decreasing the batch size from 16 to 12 and turning dynamic batching off
		(both due to memory constraints); we've included hparams/original_conformer_small.yaml if you'd like to compare or
		see.
	c. This single train.py handles both training and evaluation parts end-to-end. It also employs a very useful checkpoint
		mechanism, so even if the program is unexpectedly interrupted, you won't lose anything beyond the last saved
		checkpoint (epoch).
	d. In particular, this means you can simply run the above command with our fully-trained model inside the directory (as
		we've included it in the repo) and it will immediately move to evaluation.
	e. The stdout will include all details about current epoch, lr, steps done, train and valid loss as well as the valid
		acc. It would also print, at the end, WER of the test splits (both clean and other). It would use the directory
		./results/conformer_small/7775 for this model; its save/ subdirectory contains the fully-trained model or saved
		checkpoints. These files in it could also prove useful: train_log.txt, wer_test-clean.txt, wer_text-other.txt

Additional contents 
-------------------

1. analyze_training.py

	This training analyzes logs (saved stdout files) of train.py to produce useful plots such as training and validation
	loss plot, validation accuracy plot and more. It can receive multiple logs of the same model's training and evaluation
	sequence (which makes sense as the script utilizes checkpoints, thus can span multiple runs, as we have done), but the
	logs must be complete (cover the entire sequence, meaning no run's log is missing) and provided in order.
	
	You may run it with the --help flag to see what are the arguments.
	
	Lastly, we note that our plots can be easily reconstructed (from our log files which we've included in the repo) by the
	following command.
		
		$ python analyze_training.py results/awesome.out results/awesome2.out results/awesome3.out results/awesome4.out results/awesome5.out --epochs 110

2. img/

	Contains relevant plots and media. All the plots can be reconstructed as mentioned in the previous clause.

3. extract_recordings.py

	This file is used to extract sample audio recordings from the LibriSpeech dataset, randomly. It saves them as MP3 files 
	(LibirSpeech originally uses FLAC format so a conversation is performed). Use --help to see relevant flags.

4. sample-recordings/

	This contains smaple recordings from the dataset's splits (some clean, some other). They were fetched using the script
	mentioned in the previous clause; you can use it too (though you'd probably get other recordings, as the script picks
	them in random).


5. results/

	Besides having the model's checkpoints and additional results, as mentioned above, this is where we saved our logs
	(stdout from the train.py runs; we had to employ many runs, by the way, due to resource constraints,
	and this is another reason the checkpoint mechanism is so useful).

Submitters
----------
By Sharyn Sircovich Sassun and Eldad Cohen
Advanced Topics in Audio Processing using Deep Learning - Final Project
Lecturer - Tal Rosenwein
Semester 2025A
Submitted April 2025
