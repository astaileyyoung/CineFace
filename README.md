# CineFace

The code here is for my project CineFace, which uses AI to detect and identify faces in film and television. 

The dataset can be downloaded [here](https://drive.google.com/file/d/1YO7jiCmMv66vZ18sBkWiJ40v4HrYQYJn/view?usp=sharing). 

The corresponding face encodings can be downloaded [here](https://drive.google.com/file/d/1D-Z5L9VWYcTciMjsyFIAu2ZWY5_amxzG/view?usp=sharing). 
The encodings are saved as .npy files corresponding to either an episode or a movie. The filenames are in the form of {imdb_id}\_{season}\_{episode} The encoded faces are in sequence, so the easiest way to join the encodings to the correct face data is simply to load the appropriate .csv file and add the encodings as a column.
