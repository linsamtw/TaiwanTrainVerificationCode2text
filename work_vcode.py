
from PIL import Image, ImageDraw, ImageFont
from random import randint
import os
import sys
import platform
if 'Windows' in platform.platform():
    PATH = "\\".join( os.path.abspath(__file__).split('\\')[:-1])
    FONTPATH = ["{}\\Times Bold.ttf".format(PATH),
            "{}\\Courier-BoldRegular.ttf".format(PATH)]
else:
    PATH = "/".join( os.path.abspath(__file__).split('/')[:-1])
    FONTPATH = ["{}/Times Bold.ttf".format(PATH),
            "{}/Courier-BoldRegular.ttf".format(PATH)]
sys.path.append(PATH)


#FONTPATH = ["Times Bold.ttf"]
import random

class rect:
    def __init__(self):
        self.size = (randint(5, 21), randint(5, 21))
        self.location = (randint(1, 199), randint(1, 59))
        self.luoverlay = True if randint(1, 10) > 6 else False
        self.rdoverlay = False if self.luoverlay else True if randint(1, 10) > 8 else False
        self.lucolor = 0 if randint(0, 1) else 255
        self.rdcolor = 0 if self.lucolor == 255 else 255
        self.ludrawn = False
        self.rddrawn = False
        self.pattern = randint(0, 1)


    def draw(self, image, overlay):
        if((overlay or not self.luoverlay) and not self.ludrawn):
            self.ludrawn = True
            stp = self.location
            transparent = int(255 * 0.45 if self.lucolor == 0 else 255 * 0.8)
            color = (self.lucolor, self.lucolor, self.lucolor, transparent)
            uline = Image.new("RGBA", (self.size[0], 1), color)
            lline = Image.new("RGBA", (1, self.size[1]), color)
            image.paste(uline, stp, uline)
            image.paste(lline, stp, lline)
        if((overlay or not self.rdoverlay) and not self.rddrawn):
            self.rddrawn = True
            dstp = (self.location[0], self.location[1] + self.size[1])
            rstp = (self.location[0] + self.size[0], self.location[1])
            transparent = int(255 * 0.45 if self.rdcolor == 0 else 255 * 0.8)
            color = (self.rdcolor, self.rdcolor, self.rdcolor, transparent)
            dline = Image.new("RGBA", (self.size[0], 1), color)
            rline = Image.new("RGBA", (1, self.size[1]), color)
            image.paste(dline, dstp, dline)
            image.paste(rline, rstp, rline)

A_Za_z = []
for i in range(65, 91):
    A_Za_z.append( chr(i) )
for i in range(10):
    A_Za_z.append(i)
    
# self = captchatext(1,0)
class captchatext:# priority = 1; offset = 0
    def __init__(self, priority, offset):
        
        self.number = random.sample(A_Za_z,1)[0]
        #self.number = randint(1,10)
        self.color = [randint(10, 140) for _ in range(3)]
        self.angle = randint(-55, 55)
        self.priority = priority
        self.offset = 0
        self.next_offset = 0

    def draw(self, image):
        
        fontpath = FONTPATH[ random.sample(range(2),1)[0] ] 
        color = (self.color[0], self.color[1], self.color[2], 255)
        font = ImageFont.truetype( fontpath , randint(25, 27) * 10)
        text = Image.new("RGBA", (250, 300), (0, 0, 0, 0))
        textdraw = ImageDraw.Draw(text)
        
        textdraw.text((0, 0), str(self.number), font=font, fill=color)
        #textdraw.text((0, 0), 'j', font=font, fill=color)

        text = text.rotate(self.angle, expand=True)
        text = text.resize((int(text.size[0] / 10), int(text.size[1] / 10)))
        base = int(self.priority * (200 / 6))
        rand_min = (self.offset - base - 2) if (self.offset - base - 2) >= -15 else -15
        rand_min = 0 if self.priority == 0 else rand_min
        rand_max = (33 - text.size[0]) if self.priority == 5 else (33 - text.size[0] + 10)
        try:
            displace = randint(rand_min, rand_max)
        except:
            displace = rand_max
        location = (base + displace, randint(3, 23))
        self.next_offset = location[0] + text.size[0]
        image.paste(text, location, text)
        # plt.imshow(image)


def work_vcode_fun(amount,file_path,amount2):# amount = 5 ; file_path = 'test_data'
    
    os.chdir(PATH)
    if file_path not in os.listdir():
        os.makedirs(file_path)
        
    #numberlist = []
    #status = 1
    for index in range(amount):
        if index % 100==0: print(index)
        #print(index)
        # index = 1
        numberstr = ""
        bgcolor = [randint(180, 250) for _ in range(3)]
        captcha = Image.new('RGBA', (200, 60), (bgcolor[0], bgcolor[1], bgcolor[2], 255))
        rectlist = [rect() for _ in range(32)]
        for obj in rectlist:
            obj.draw(image=captcha, overlay=False)
    
        offset = 0
        #vcode = ''
        #amount2 = random.sample([5,6],1)[0]
        for i in range(amount2):
            newtext = captchatext(i, offset)
            newtext.draw(image=captcha)
            offset = newtext.next_offset
            numberstr += str(newtext.number)
        if 'Windows' in platform.platform():
            path = '{}\\{}\\{}.jpg'.format(PATH,file_path,numberstr)
            captcha.convert("RGB").save(path, "JPEG")
        else:
            path = '{}/{}/{}.jpg'.format(PATH,file_path,numberstr)
            captcha.convert("RGB").save(path, "JPEG")

