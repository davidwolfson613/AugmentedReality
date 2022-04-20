# AugmentedReality

This is a project showing how augmented reality works. The repo contains an image of the American flag, and this image is projected onto a wall that contains markers showing where the flag should be placed. Instead of an image of a wall, a video of the wall incorporating rotations, zooming in and out, and panning of the camera are used to depict the perspective transform of the flag involved in the projection. The transform is calculated on a per frame basis.

# How to run
The code can be run locally by cloning this repo, going into this directory and running "python script.py" from the command line.

NOTE: The filepaths are all relative, so be sure to keep the "wall.mp4" and "flag.png" files in the same directory as "script.py".
