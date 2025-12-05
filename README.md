# Pokemon_ShinyHunter
This project is the accumulation of my semester in the course CS366: Computer Vision. 
Utilizing CV tasks such as Object classification and Object Localization, my program has learned from a custom-made dataset to recognize certain species of pokemon and determine whether the pokemon on screen is Shiny

The main program runs by locking onto an open window of the DeSmuME emulator and automates the player character's movement to encounter wild pokemon. Once in battle, TemplateMatching is used to slice out the region where the wild pokemon is and passes the image to two models. A Pokemon Classifier CNN Model and a Shiny Classifier Multi-Modal Model. 

##Dataset
The datasets used by my program have been created through data augmentation from the Albumentations Library
Gathering sprites from the Pokemon Games HeartGold, Platinum, and Black, I was able to create training, validation, and testing sets diverse enough to prevent data leakage. 
Each dataset contains both Shiny and Regular Pokemon, this is to help both models understand that even though there are color differences among species, both forms of pokemon should still be classified as the same pokemon. For example, a Shiny Geodude is still classified as a Geodude instead of a yellow colored pokemon. 


