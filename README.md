# Car-classsifier
This code detects whether an image is lamborghini, porche or ferrari

## Step 1
Run 'retrain.bat'. This will download the inception model which is already trained on 
100k images on 1000 categories.

## Step 2
![Alt text](final_layer.png?raw=true "Optional Title")
After download is complete, the script will take the training images and start creating 'Bottleneck' files for the model. Once the script finishes generating all the bottleneck files, the actual training of the final layer begins. The training operates efficiently by feeding the cached value for each image into the Bottleneck layer. The true label for each image is also fed into the node labeled GroundTruth. Just these two inputs are enough to calculate the classification probabilities, training updates, and the various performance metrics.
As it trains you'll see a series of step outputs, each one showing training accuracy, validation accuracy, and the cross entropy:
The training accuracy shows the percentage of the images used in the current training batch that were labeled with the correct class.
Validation accuracy: The validation accuracy is the precision (percentage of correctly-labelled images) on a randomly-selected group of images from a different set.
Cross entropy is a loss function that gives a glimpse into how well the learning process is progressing (lower numbers are better here). 
After training is complete, the Model File is saved as '/tf_files/retrained_graph.pb'.

## Step 3
To see the labels array of the predicted values, run 'label_image.py'.

## Step 4
To predict the images in 'test_images' folder, run 'guess.py'.

## Step 5
To see the results of training, run 'tensorboard.bat'. To see the tensorboard, open the link '127.0.0.1:6006' on your browser. 
