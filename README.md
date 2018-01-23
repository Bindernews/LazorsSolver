Lazor Solver
============

This is a solver for [the Lazors game](https://play.google.com/store/apps/details?id=net.pyrosphere.lazors).
It takes a screenshot of the game and attempts to deduce and solve the puzzle from just that screenshot.

## Installing
The program should work on Python 3.5, but has only been tested on 3.6.
Use `pip install -r requirements.txt` (preferably in a [virtual environment](https://docs.python.org/3/library/venv.html))
to install the required libraries.

## Running
Simply execute `python solve.py`. To change the file being processed, edit the `cv2.imread(<filename>)` line in
the `main()` function.

Note that the program will generate several `logXX.png` files in the directory where it's run. This is for debugging
purposes as the program is still a work in progress.

## History
I became bored with solving the puzzles on my own, so I decided to write a solver because I thought it would be fun.
Then I realized I didn't want to have to manually enter the game state, so I began tinkering with image processing
and computer vision in order to determine the state of the game given a screenshot.