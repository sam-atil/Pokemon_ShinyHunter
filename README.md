# Pokemon_ShinyHunter
This project is the accumulation of my semester in the course CS366: Computer Vision. 
Utilizing CV tasks such as Object classification and Object Localization, my program has learned from a custom-made dataset to recognize certain species of pokemon and determine whether the pokemon on screen is Shiny

The main program runs by locking onto an open window of the DeSmuME emulator and automates the player character's movement to encounter wild pokemon. Once in battle, TemplateMatching is used to slice out the region where the wild pokemon is and passes the image to two models. A Pokemon Classifier CNN Model and a Shiny Classifier Multi-Modal Model. 

## Dataset
The datasets used by my program have been created through data augmentation from the Albumentations Library
Gathering sprites from the Pokemon Games HeartGold, Platinum, and Black, I was able to create training, validation, and testing sets diverse enough to prevent data leakage. 
Each dataset contains both Shiny and Regular Pokemon, this is to help both models understand that even though there are color differences among species, both forms of pokemon should still be classified as the same pokemon. For example, a Shiny Geodude is still classified as a Geodude instead of a yellow colored pokemon. 

## Pokemon Classifier CNN
The Pokemon Classifier Model is a custom made CNN model with 4 Convolutional layers. Using CrossEntropyLoss and an AdamW optimizer, my model was able to achieve an accuracy of 90.90% on the Training set and 78.33% on the Validation set after training for 25 epochs. After performance evaluation, my model achieved 80.30% accuracy on the Test Set. 

## Shiny Classifier Multi-Modal
The Shiny Classifier uses a Multi-Modal model made from scratch. The model recieves both an Image input and a classification label input. The classification input is the output from the Pokemon Classification model
For Example: Abra is associated with the label 0, and that value is then passed into the Multi-Modal model. 

Using the same CNN structure from the Pokemon Classifier model and a fusion layer to combine image features and label features, my model was able to isolate the pokemon species and begin classifying whether the input image was a shiny pokemon or a normal pokemon. 

After training for 5 epochs, my model was able to achieve a Training accuracy of 89.69% and a Validation accuracy of 87.17%. After performance evaluation my model achieved a Test accuracy of 85.41%

## Main.py
This is the main file, it creates the Shiny_Hunt class. The main functions of this class is the hunt_loop() and summary_loop()

The hunt_loop() function is how the program automates player movement and escape battle when it finds a non-shiny pokemon. If a shiny pokemon is found, the program stops and allows me to take over and catch the pokemon. 
using template matching to detect battle UI, the program knows it's time to stop looping walking movement and began image classification. By using template matching again, the program slices out a small image containing the pokemon sprite and passes that image to both the Pokemon Classification model and Shiny Classification model.

The summary_loop() function is meant to act as a debugger function. Since Shiny pokemon are rare to find, in my version of the emulator I was able to hack in a shiny pokemon. So this allowed me to test that my program successfully classifies a pokemon as Shiny from within the summary screen in the game.

## Future Implementations
As of right now, the program only prints out a statement alerting the user that a Shiny is found. My next goal is to update the program to send alerts to my phone that a Shiny was found. This way I can continue to work on my laptop without needing to constantly look back at the Emulator. 
