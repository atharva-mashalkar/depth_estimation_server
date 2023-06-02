## WalkBuddy Project Documentation

This documentation provides an overview of the WalkBuddy project and instructions for setting up the server and client components. Please follow the steps below to get started.

### Cloning the Repository

To clone the repository, please make sure to clone it into the `testing_branch` instead of the `main` branch. This ensures that you are working with the appropriate branch for testing purposes. You can use the following command to clone the repository:

git clone -b testing_branch https://github.com/atharva-mashalkar/depth_estimation_server


### Starting the Server

To start the server, execute the following command within the project directory:

python server.py



### Configuring the Client

Before running the client, you need to update the domain name to which it should connect. Open the relevant file and modify the appropriate section where the domain name is set. Replace the current domain name, `server.walkbuddy.in`, with the desired domain or IP address of the server.

### Files Overview

The project contains the following files with their respective functionalities:

- **hand_rec.py**: This file contains code for finding the finger tip coordinates.
- **depth_estimation.py**: Here, you can find code for estimating depth from the received image.
- **main.py**: This file combines the outputs from `hand_rec` and `depth_estimation` to determine the cropped image and find the minimum distance of the object within the cropped image.

Please refer to the source code of each file for more detailed information on their implementations.
