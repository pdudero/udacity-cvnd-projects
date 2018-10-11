
Project instructions are found in the [Udacity README](README_Udacity.md).

## Method:

A summary of the foundation of the GraphSLAM method can be found in [this notebook](GraphSLAM_Derivation.ipynb).

[The Robot Moving and Sensing notebook](1.%20Robot%20Moving%20and%20Sensing.ipynb) defines a robot class. The student is tasked with implementing the sense function, given a set of randomly generated landmarks.

The implementation of the robot class is subsequently moved to the [robot_class.py](robot_class.py) file.

[The Omega and Xi notebook](2.%20Omega%20and%20Xi,%20Constraints.ipynb) introduces the implementation of the linear algebra solution to Graph SLAM.

In [the Landmark Detection and Tracking notebook](3.%20Landmark%20Detection%20and%20Tracking.ipynb) the full solution of offline Graph SLAM implemented. That is, the sense and move cycle is repeated numerous times as the robot randomly walks through its grid world until after all landmarks are sensed. Then the list of moves and measurements are used to determine the approximate locations of the landmarks and the robot itself. Success is determined by manually comparing the final estimates with the original truth information.

I chose to perform additional "extra credit" work implementing a version of online SLAM, using the concepts laid out by Sebastian Thrun in [this video](https://www.youtube.com/watch?v=jaeNlxhQL1I). [The MySLAM notebook](5.%20MySLAM.ipynb) shows that work. Changes include
1. Considering only the most recent pose.  
1. MyOnlineSlam is now implemented as a class.  
1. The data loop happens outside of the class.  
1. The x and y xi vectors are separated. Omega is the same for both.  
1. The robot attempts a solution with every pose, using the most recent pose. A snapshot of the world, the truth information and the robot's estimates is taken and then collected into a video.

The class still expects a fixed number of landmarks, and so the robot is not able to solve the equation until all landmarks are sensed, but once they are, the locations are updated with every pose. A potential improvement for the future would be to "get more real-world", by removing all prior knowledge of landmarks, such as the number of them, or which landmark was just sensed. This means allowing omega and xi to grow as new landmarks are detected. It also means determining whether two measurements represents the same landmark or two different ones.

## Results:

Notebook 3 contains its own results at the end. However, perhaps the most intuitive demonstration of the results would be the animation produced using MyOnlineSlam (notebook 5), which can be downloaded [here](animation.mp4). In the video, one can see the robot moving around in the grid world, and the ground truth landmark locations depicted as blue x's. The transparent red circle around the robot represents the detection radius of the robot. The red lines radiating from the robot represent measurements of landmarks in terms of distance and angle. That these lines don't point directly at the ground truth landmark locations is indicative of the noise injected into the measurements and movements. 48 seconds into the video one can see that all landmarks have been sensed and a solution reached; all estimated positions show up near the ground truth locations as colored red, and are updated henceforth.
