# Running the code

## 1. Install Anaconda
Install Anaconda from https://www.anaconda.com/download/)

## 2. Install Python, OpenCV, and Sci-kit
Once Anaconda is installed, let’s create a “sandbox” (a.k.a. “conda environment”) named “fad” with all necessary packages for this practical:

    conda create --name fad python=3.9
and activate it

    conda activate fad
then install OpenCV and Sci-kit

    conda install -c conda-forge opencv
    conda install scikit-image
    conda install -c conda-forge dlib
    conda install -c conda-forge tensorflow
    conda install -c conda-forge scikit-learn

## 3. Download and run script

**Steps**
1. Clone the repository
	 `git clone https://github.com/gavinuhran/facial-attributes-detection`
2. Set the repository as your working directory
3. Run the main file
	 `python scripts/main.py`
4. Drag the windows apart so they may all be viewed simultaneously
5. Press the ESC-key to analyze the next face
	6. There are 500 faces in the current dataset, so use CTRL+C in to quit the process early
