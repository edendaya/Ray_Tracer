class Img:
    def __init__(self, img_width, img_height, screen_width):
       self.img_width = img_width       
       self.img_height = img_height
       self.screen_width = screen_width
       self.ratio = self.screen_width / self.img_width
    

