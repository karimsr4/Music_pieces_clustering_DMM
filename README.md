# Music beyond Major and Minor

##### By Ahmed J., Karim H. and Ayman M.

This code is our work to EPFL's CS-433 Machine Learning course project 2.

The project is supervised by **Dr Robert Lieck** from the Digital and Cognitive Musicology Lab DCML. 
Contact: robert.lieck@epfl.ch

## Dataset
The dataset is private, however it can be obtained from the DCML. You can contact Dr Lieck for this.  
The project can be run without the dataset, as we save the .csv and .npy files needed at each step.
## Dependencies
- the `pitchscapes` library.
- `plotly` for the 3D figures.
- `PyTorch` for the training.
- The `conda` packages: `pandas`, `numpy`, `matplotlib`.
--
## 3D interactive plots
The 3D interactive plots can be viewed at: https://dcmlab.github.io/music_beyond_major_and_minor_jellouli_mezghani_hadidane/

## Folders
- `src/`: Contains Python .py files.
- `notebooks/`: Contains Jupyter notebooks that we used to pre-process the data, visualize, apply the model and plot results.
- `notebooks/figures/`: Contains the 3D interactive plots.

## Files
- `./outcomes/report_final.pdf`: report of the project.
- `src/DirichletMixtureModel.py`: contains the implementation of the DMM clustering model.
- `src/train.py`: contains the code of training the model.
- `notebooks/data_preparation.ipynb`: Jupyter notebook used to read the data, pre-process it, add relevant information using Pandas dataframes, save result in csv files

## Warning
The `data_preparation.ipynb` requires the data to be run. Unfortunately, we can not share the data. Therefore, we pre-executed the notebook.


## Contact
In case any help is needed:
- Karim Hadidane : karim.hadidane@epfl.ch
- Ahmed Jellouli : ahmed.jellouli@epfl.ch
- Ayman Mezghani : ayman.mezghani@epfl.ch

