#Import Libaries
import cv2
import os
from PIL import Image
import numpy as np
from pywinauto import Application
import ctypes
import time
import torch
import torchvision.transforms as transforms
from models import CNN, ShinyCNN, ShinyModal
from rembg import remove

class shiny_hunt:
    #Constructor
    def __init__(self, win_str, cnn_model, shiny_model, cnn_transform, shiny_transform, class_dict, device):
        '''
        Parameters:
        -----------
        win_str: String regex to connect to Emulator Window
        '''

        self.win_str = win_str #Window String
        self.cnn = cnn_model
        self.shiny = shiny_model
        self.cnn_transform = cnn_transform
        self.shiny_transform = shiny_transform
        self.class_dict = class_dict
        self.device = device

        # Keyboard event flags
        self.KEYEVENTF_KEYDOWN = 0x0000
        self.KEYEVENTF_KEYUP = 0x0002

        # Virtual key codes for keyboard
        self.VK_X = 0x58  # X key = A button
        self.VK_Z = 0x5A # Z Key = B button
        self.VK_UP = 0x26 #Up Arrow
        self.VK_DOWN = 0x28 #Down Arrow
        self.VK_LEFT = 0x25 #Left Arrow
        self.VK_RIGHT = 0x27 #Right Arrow

        #Battle State (False == Overworld, True == Wild Encounter)
        self.battle = False

    def hunt_loop(self):
        #Creating Window Application
        win = self.win_setup()
        print("Window Found! Starting Shiny Hunt")
        time.sleep(0.3)

        #Extracting templates
        poke_templates = self.get_poke_template()

        while True:
            #Taking a screenshot
            screenshot = win.capture_as_image()
            frame = np.array(screenshot)[:, :, ::-1] #Converts PIL screenshot to BGR

            #Passing frame to battle_match
            if self.battle_match(frame): #If True, run battle mode loop
                print("Wild Pokemon Encountered!")
                shiny_found = self.battle_loop(win, poke_templates)
                
                if shiny_found:
                    print("ENDING HUNT!")
                    break
                continue

            #Overworld keyboard loop
            self.world_loop()
            time.sleep(0.05)

    def battle_loop(self, win, templates):
        self.battle = True
        shiny_found = False
        while self.battle:
            raw_img = win.capture_as_image()
            frame = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)


            #Call poke_template to extract region of pokemon sprite
            screen_sprite = self.poke_match(frame, templates)
            screen_sprite = np.array(screen_sprite)

            #Passing to models
            model_pred = self.is_shiny(screen_sprite)

            if model_pred == 1: #Found a shiny
                print("SHINY FOUND!")
                self.battle = False
                return True
 
            else:
                self.battle = False
                self.battle_NotShiny() #Escape Encounter
                return False

    #Used when in summary of pokemon
    def summary_loop(self):
        win = self.win_setup()
        print("Window Found! Starting Shiny Hunt")
        time.sleep(0.3)

        #Extracting templates
        poke_templates = self.get_poke_template()

        while True:
            screenshot = win.capture_as_image()
            frame = np.array(screenshot)[:, :, ::-1] #Converts PIL screenshot to BGR

            screen_sprite = self.poke_sum(frame, poke_templates)
            time.sleep(0.3)

            screen_sprite = np.array(screen_sprite)

            #Passing to models
            model_pred = self.is_shiny(screen_sprite)

            if model_pred == 1: #Found a shiny
                print("SHINY FOUND!")

            else:
                print("Pokemon is Not a Shiny")


    #Helper Functions
    def win_setup(self):
        app = Application().connect(title_re = self.win_str)
        win = app.top_window()
        win.set_focus()
        print("Window Successfully Created")
        return win
    

    #Classifier Function
    def is_shiny(self, frame):
        
        img = frame

        #Adding Batch Dimension and converting to Tensor
        img_tensor = self.cnn_transform(img).unsqueeze(0)

        #moving device to same as model
        print("Receiving device")
        device = next(self.cnn.parameters()).device
        img_tensor = img_tensor.to(device)

        #Passing img through model 1
        print("Passing img to model1")
        with torch.no_grad():
            cnn_output = self.cnn(img_tensor)
            probs = torch.softmax(cnn_output, dim = 1)

        #model1 prediction
        pred_prob, pred_class = torch.max(probs, dim=1)
        pred_class = pred_class.item() #index number
        pred_prob = pred_prob.item()
        class_name = self.class_dict[pred_class]
        print(f"Predicted Probability: {pred_prob}")
        print(f"Predicted Pokemon: {class_name}")


        #model2 prediction

        id_tensor = torch.tensor([pred_class], dtype=torch.long, device=self.device)


        with torch.no_grad():
            shiny_output = self.shiny(img_tensor, id_tensor)
            shiny_probs = torch.softmax(shiny_output, dim=1)

        shiny_prob, shiny_class = torch.max(shiny_probs, dim=1)
        shiny_class = shiny_class.item() #1 or 0
        
        shiny_bool = ''
        if shiny_class == 1:
            shiny_bool = "Shiny!"
        else:
            shiny_bool = "Not A Shiny"

        print(f"Predicted Shiny Probability: {shiny_prob}")
        print(f"Predicted Pokemon Shiny: {shiny_bool}")


        return shiny_class

    #Template match functions

    #Template Match for when battle starts
    def battle_match(self, raw_img):
        folder_path = "templates/battle_UI"
        file_path = os.path.join(folder_path, "UI.png")

        img_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        #Specifying Threshold
        thres = 0.8

        #performing match operaitons
        res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        #Reassigning battle state
        return max_val > thres

    #Template Match for Pokemon
    def poke_match(self, raw_img, templates):
        #Variables to store which sprite is more likely the raw_img
        best_conf = 0
        best_pt = None

        #converting raw_img to Grayscale
        height, width, chan = raw_img.shape
        t_height, t_width = 0,0

        for sprite in templates:
            res = cv2.matchTemplate(raw_img, sprite, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_conf:
                best_conf = max_val
                best_pt = max_loc
                t_height, t_width, _ = sprite.shape

        #Extracting sprite region from raw_img
        x,y = best_pt

        offset_x = 100
        offset_y = -280
        center_x = x + t_width // 2 + offset_x
        center_y = y + t_height // 2 + offset_y
        crop_size = 144

        # clamp to image boundaries

        crop_x1 = max(0, center_x - crop_size // 2)
        crop_y1 = max(0, center_y - crop_size // 2)
        crop_x2 = min(width, crop_x1 + crop_size)
        crop_y2 = min(height, crop_y1 + crop_size)


        cropped = raw_img[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        #Removing Background and White Background
        cropped_removed = remove(Image.fromarray(cropped))
        alpha = cropped_removed.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", cropped_removed.size, (255,255,255,255))
        bg.paste(cropped_removed, mask=alpha)
        cropped = bg.convert('RGB')
        
        return cropped
    
    def poke_sum(self, raw_img, templates):
        #Variables to store which sprite is more likely the raw_img
        best_conf = 0
        best_pt = None

        #converting raw_img to Grayscale
        height, width, chan = raw_img.shape
        t_height, t_width = 0,0

        for sprite in templates:
            res = cv2.matchTemplate(raw_img, sprite, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_conf:
                best_conf = max_val
                best_pt = max_loc
                t_height, t_width, _ = sprite.shape

        #Extracting sprite region from raw_img
        x,y = best_pt

        offset_x = 30
        offset_y = 0
        center_x = x + t_width // 2 + offset_x
        center_y = y + t_height // 2 + offset_y
        crop_size = 144

        # clamp to image boundaries

        crop_x1 = max(0, center_x - crop_size // 2)
        crop_y1 = max(0, center_y - crop_size // 2)
        crop_x2 = min(width, crop_x1 + crop_size)
        crop_y2 = min(height, crop_y1 + crop_size)


        cropped = raw_img[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        #Removing Background and White Background
        cropped_removed = remove(Image.fromarray(cropped))
        alpha = cropped_removed.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", cropped_removed.size, (255,255,255,255))
        bg.paste(cropped_removed, mask=alpha)
        cropped = bg.convert('RGB')
        
        return cropped

    #Extracts pokemon sprite templates from folders
    def get_poke_template(self):
        folder_path = "templates/poke_templates"

        temp = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            cur_file_img = cv2.imread(file_path)
            scaled_img = cv2.resize(cur_file_img, (96,96), interpolation=cv2.INTER_AREA)
            temp.append(scaled_img)

        return temp



    #Player Control Loop Functions
    def world_loop(self):
        
        #Move left 3 spaces
        for i in range(3):
            #Left Key
            ctypes.windll.user32.keybd_event(self.VK_LEFT, 0, self.KEYEVENTF_KEYDOWN, 0)
            time.sleep(0.4)
            ctypes.windll.user32.keybd_event(self.VK_LEFT, 0, self.KEYEVENTF_KEYUP, 0)

        #Move right 3 spaces
        for i in range(3):
            #Right Key
            ctypes.windll.user32.keybd_event(self.VK_RIGHT, 0, self.KEYEVENTF_KEYDOWN, 0)
            time.sleep(0.4)
            ctypes.windll.user32.keybd_event(self.VK_RIGHT, 0, self.KEYEVENTF_KEYUP, 0)

    #Called when program decided Pokemon is Not Shiny
    def battle_NotShiny(self):
        #Down Key
        ctypes.windll.user32.keybd_event(self.VK_DOWN, 0, self.KEYEVENTF_KEYDOWN, 0)
        time.sleep(0.4)
        ctypes.windll.user32.keybd_event(self.VK_DOWN, 0, self.KEYEVENTF_KEYUP, 0)

        #Left Key
        ctypes.windll.user32.keybd_event(self.VK_LEFT, 0, self.KEYEVENTF_KEYDOWN, 0)
        time.sleep(0.4)
        ctypes.windll.user32.keybd_event(self.VK_LEFT, 0, self.KEYEVENTF_KEYUP, 0)

        #Right Key
        ctypes.windll.user32.keybd_event(self.VK_RIGHT, 0, self.KEYEVENTF_KEYDOWN, 0)
        time.sleep(0.4)
        ctypes.windll.user32.keybd_event(self.VK_RIGHT, 0, self.KEYEVENTF_KEYUP, 0)

        #X Key
        ctypes.windll.user32.keybd_event(self.VK_X, 0, self.KEYEVENTF_KEYDOWN, 0)
        time.sleep(0.4)
        ctypes.windll.user32.keybd_event(self.VK_X, 0, self.KEYEVENTF_KEYUP, 0)



def main():
    #Creating the Pokemon Class Dictionary
    idx_to_species = {
        0: "Abra",
        1: "Bellsprout",
        2: "Caterpie",
        3: "Clefairy",
        4: "Ditto",
        5: "Drowzee",
        6: "Ekans",
        7: "Geodude",
        8: "Growlithe",
        9: "Hoothoot",
        10: "Hoppip",
        11: "Kakuna",
        12: "Ledyba",
        13: "Mareep",
        14: "Meowth",
        15: "Metapod",
        16: "Nidoran_F",
        17: "Nidoran_M",
        18: "Pidgey",
        19: "Pikachu",
        20: "Rattata",
        21: "Sentret",
        22: "Spearow",
        23: "Spinarak",
        24: "Stantler",
        25: "Vulpix",
        26: "Weedle",
        27: "Wooper",
        28: "Yanma",
        29: "Zubat"
    }

    #Setting up project
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Loading Pokemon Classifier Model
    mon_classifier = torch.load("Model/mon_classifier.pt", map_location=device)
    mon_classifier.eval()


    #Loading Shiny Classifier Model
    shiny_classifier = torch.load("Model/shiny_classifier.pt", map_location=device)
    shiny_classifier.eval()


    mon_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.8300, 0.8160, 0.8006],
            std =[0.2662, 0.2793, 0.2973]
        ),
    ])

    shiny_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.82539384, 0.82051394, 0.81272638],
            std = [0.28565448, 0.28943319, 0.29837072]
        ),
    ])

    program = shiny_hunt(
        win_str="DeSmuME 0.9.13 x64 SSE2 | Pokémon HeartGold",
        cnn_model=mon_classifier,
        shiny_model=shiny_classifier,
        cnn_transform=mon_transform,
        shiny_transform=shiny_transform,
        class_dict=idx_to_species,
        device = device)
    #program.hunt_loop()
    program.summary_loop()

if __name__ == "__main__":
    main()