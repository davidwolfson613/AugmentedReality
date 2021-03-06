# AugmentedReality

This is a project showing how augmented reality works. The repo contains an image of the American flag, and this image is projected onto a wall that contains markers showing where the flag should be placed. Instead of an image of a wall, a video of the wall incorporating rotations, zooming in and out, and panning of the camera is used to depict the perspective transform of the flag involved in the projection. The positions of the markers in each frame are not hard-coded or stored anywhere. The markers are found in each frame using Computer Vision algorithms, and those locations are used in other CV algorithms to determine how to warp the flag in order to make it fit within the markers in the video. These processes are done dynamically, meaning if you were to create another video with these markers and attempt to project a different image onto the markers in that new video, it would work. Feel free to try this out. If this would like to be attempted, the only change that would need to be made is changing the names of the files used. These can be found at the bottom of the file "script.py", line 144 to specify the file name of the video to be used (replace the "to_use\wall.mp4" with "to_use\file_to_use.mp4") and line 163 to specify the file name of the image to be projected (replace the "to_use\flag.png" with "to_use\file_to_use.png").

# How to run locally
The code can be run locally by cloning this repo, going into this directory and running this command from the command line:

    python script.py

In order to install all the required packages, run the following command from the command line:

    pip install -r requirements.txt

NOTE: The filepaths are all relative, so be sure to keep the "wall.mp4" and "flag.png" files in the "to_use" directory. As mentioned above, if you would like to apply augmented reality to different files, please make the relevant changes to the file paths in the "script.py" file.
