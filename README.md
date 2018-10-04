## Stereotypical Gender Associations in Language Have Decreased Over Time

**__Current Version: 0.10__**

Release Date: September 28, 2018  

**Please contact moamin@cs.stonybrook.edu for quick response to resolve any bug or feature update.**

### Reproducing Results Discussed in Manuscript

Please follow the following steps to download source code: 
  * Download and install git and git's extenstion git-lfs
  * Clone source code: git clone https://github.com/ruhulsbu/StereotypicalGenderAssociationsInLanguage.git
  * Go to the StereotypicalGenderAssociationsInLanguage directory
  * Unzip the dataset.tar.gz using command: tar -xvzf dataset.tar.gz
  * Run Jupyter Notebook in current directory using command: jupyter notebook 

Open the following notebook files: 
  *  Significance Test on Gender Bias using WEAT Topics.ipynb
  *  Stereotypical Gender Association in Language Over Time.ipynb
Run each cell in the notebooks sequentially to reproduce the results

### Dependencies

#### GitHub dependencies:
  * git-lfs to download large file dataset.tar.gz

This dataset is originally hosted in [Histwords Project](https://nlp.stanford.edu/projects/histwords/). The dataset can be downloaded from this link: [Embeddings](http://snap.stanford.edu/historical_embeddings/eng-all_sgns.zip). Please extract the files into the directory "dataset" in project home directory.

#### Core dependencies:
  * python 3.6.0
  * numpy 
  * scipy 
  * sklearn
  * seaborn
  * jupyter



