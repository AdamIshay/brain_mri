## Install requirements.
We used Python 3.6 to run the code.

You can use the following to create a suitable Conda environment. 

###CREATE ENVIRONMENT  
Create and activate Conda environment:  
'conda create --name clevrer python=3.6'  
'conda activate clevrer'  
###INSTALLATION  
Install Clingo version (5.3.0 for now) and make sure that it is in your PATH variable. You run the following for this:  
'conda install -c potassco clingo==5.3.0 '  
Additionally, you will have to install Clyngor for calling Clingo in Python. You can install with:  
'conda install -c conda-forge clyngor'  
Install the rest of the requirements with:  
'conda install -c conda-forge tqdm'  
'conda install -c anaconda ipython'  
'conda install -c anaconda nltk'  
'conda install -c anaconda numpy'  
'conda install -c conda-forge ipdb'  
'conda install -c conda-forge matplotlib'  
## Download data  

Download the parsed programs into the 'parsed_programs' folder, along with the train.json, validation.json, and test.json into the 'questions' folder. You can get the files from:
https://github.com/chuangg/CLEVRER/tree/master/executor/data

Download the results of the video frame parser and dynamic predictions here:
https://github.com/chuangg/CLEVRER

Specifically, two files are required which are linked in the "Get the data" section. (1) The dynamic predictions and (2) results of video frame parser (visual masks).

To run the experiments, the directory format should match the following:

main/  
├─ data/  
│  ├─ processed_proposals/  
│  ├─ convert.py  
│  ├─ converted/  
│  ├─ parsed_programs/  
│  ├─ questions/  
│  │  ├─ train.json  
│  │  ├─ validation.json  
│  │  ├─ test.json  
│  ├─ propnet_preds/  
│  │  ├─ with_edge_supervision_old/  
├─ descriptive/  
├─ explanatory/  
├─ predictive/  
├─ counterfactual/  
├─ merge.py  

## Format mask r-cnn results:

cd main/data'
python convert.py'

## (optional) run question parser in the 'question_parser' directory. If not, the parsed programs from the baseline will be used (in step 2).

## To run descriptive, explanatory, predictive, and counterfactual queries for validation set, run the following from their respective directories.
You can also pass arguments for explanatory (--ASP, --IOD), predictive (--IOD, --PM, --AC), and counterfactual (--IOPD, --AC) queries to turn on and off certain enhancements. --ALL will run with all enhancements.

'python run_descriptive.py'

'python run_explanatory.py --ALL'

'python run_predictive.py --ALL'

'python run_counterfactual.py --ALL'

## To run descriptive, explanatory, predictive, and counterfactual queries for test set to submit the result, run the following from their respective directories and merge the each result:

'python test_descriptive.py'

'python test_explanatory.py'

'python test_predictive.py'

'python test_counterfactual.py'

'python merge.py'
