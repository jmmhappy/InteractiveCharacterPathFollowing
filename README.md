# Interactive Character Path-Following Using Long-Horizon Motion Matching With Revised Future Queries



This project demonstrates an interactive character path-following system using long-horizon motion matching with revised future queries. The goal of this project is to enable real-time character navigation while maintaining natural and smooth movement.



This project is based on the research paper titled "[Interactive Character Path-Following Using Long-Horizon Motion Matching With Revised Future Queries](https://doi.org/10.1109/ACCESS.2023.3240589)" by Jeongmin Lee, Taesoo Kwon, and Yoonsang Lee, published in IEEE Access in 2023.


## Prerequisites

This project requires the below prerequisites:

```
pip3 install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/your_linux wxPython
pip3 install pybind11
sudo apt-get install libeigen3-dev
```


## How to Use



### Step 1: Clone Repository and Generate Binary Files



First, clone this repository to your local machine, and checkout to a specific branch.

```bash

git clone https://github.com/jmmhappy/2dt_match.git
git checkout tracking

```

Next, generate a data binary from the dataset using Python3. You can do this by running the following two lines in Python3 shell:

```python

from util.bvhMenu import generate
generate("path_to_directory", "output.bin", True)

```

Note that the `True` option generates a motion matching database.



Lastly, if you want to, learn a future direction network(rnn). The below code will generate a binary:

```
python3 trainNetwork.py -d <data binary> -o <output rnn binary> -w <window size>
```





### Step 2: Compile util/rotations.cpp

This file is written with pybind11, so make sure you have it installed before proceeding. To compile the file, run the following commands:

```
cd util
mkdir build
cd build
cmake ..
make check -j 4
mv <output> ..
```

Notes:

1. If your computer cannot `find_package(pybind11)`, try installing `pip3 install "pybind3[global]"`. Or you can manually add the package path as a cmake option. For example, `cmake .. -Dpybind11_DIR=/path/to/pybind11/share/cmake/pybind11`.

2. If your computer cannot find `Eigen/Core`, try making a softlink. `sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen`.



### Step 3: Run Application.py

Now that you have the binary file and the necessary dependencies set up, you can run the main program, `application.py`, by executing the following command in the terminal:

```terminal
python3 application.py --data <data binary> --weights <rnn binary>
```

Note that the RNN network is optional. 

Notes:
1. If something like "No context" error appears, add `PYOPENGL_PLATFORM=egl` in from of the command.  



## Dataset

This project uses the [Phase-Functioned Neural Networks for Character Control](http://siggraph.org/conference/archive/2017/program/presentations/holden-phase-functioned-neural-networks-character-control) dataset. This dataset was developed by Holden, D., Komura, T., & Saito, J. (2017), and can be freely used for academic or non-commercial purposes. For commercial use, please contact contact@theorangeduck.com.

In our demo, we exclude jumping, t-poses, and walking on uneven terrain.


## Contributing

To contribute to this project, first create a fork on GitHub and then submit changes as a pull request to the original repository.
