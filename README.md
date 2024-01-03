# Interactive Character Path-Following Using Long-Horizon Motion Matching With Revised Future Queries



This project demonstrates an interactive character path-following system using long-horizon motion matching with revised future queries. The goal of this project is to enable real-time character navigation while maintaining natural and smooth movement.



This project is based on the research paper titled "[Interactive Character Path-Following Using Long-Horizon Motion Matching With Revised Future Queries](https://doi.org/10.1109/ACCESS.2023.3240589)" by Jeongmin Lee, Taesoo Kwon, and Yoonsang Lee, published in IEEE Access in 2023.



## How to Use



### Step 1: Clone Repository and Generate Binary Files



First, clone this repository to your local machine, and checkout to a specific branch.

```bash

git clone https://github.com/jmmhappy/2dt_match.git
git checkout tracking

```

Next, generate a binary file from the dataset using Python3. You can do this by running the following two lines in Python3 shell:

```python

from util.bhvMenu import generate

generate("path_to_directory", "output.bin", True)

```

Note that the `True` option generates a motion matching database.



Lastly, if you want to, learn a future direction network(rnn). The below code will generate a binary:

```

python3 trainNetwork.py -d <data file> -o <output name> -w <window size>

```





### Step 2: Compile rotations.cpp (if necessary)



If you're using Linux, you can skip this step. However, if you're using another operating system, you'll need to compile `util/rotations.cpp`. This file is written with pybind11, so make sure you have it installed before proceeding. To compile the file, navigate to the `util` directory and run the following commands:

```go

cd util

mkdir build

cd build

cmake ..

make check -j 4

```

Check a pybind11 tutorial https://pybind11.readthedocs.io/en/latest/basics.html#first-steps for more details.



### Step 3: Run Application.py



Now that you have the binary file and the necessary dependencies set up, you can run the main program, `application.py`, by executing the following command in the terminal:

```terminal

python3 application.py --data [output.bin] --weights [rnn_network.bin]

```

Note that the RNN network is optional. If something like "No context" error appears, add `PYOPENGL_PLATFORM=egl` in from of the command.  



That's it! Follow these steps to get started with the interactive character path-following system using long-horizon motion matching with revised future queries.



## Dataset



This project uses the [Phase-Functioned Neural Networks for Character Control](http://siggraph.org/conference/archive/2017/program/presentations/holden-phase-functioned-neural-networks-character-control) dataset. This dataset was developed by Holden, D., Komura, T., & Saito, J. (2017), and can be freely used for academic or non-commercial purposes. For commercial use, please contact contact@theorangeduck.com.



In our demo, we exclude jumping, t-poses, and walking on uneven terrain.







## Contributing



To contribute to this project, first create a fork on GitHub and then submit changes as a pull request to the original repository.

 