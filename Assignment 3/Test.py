import sys
import os
import uuid
from pdf2image import convert_from_path
from PIL import Image
import tempfile
import subprocess
from flair.data import Sentence
from flair.models import SequenceTagger
import re
import json
import pytesseract
from PIL import Image

def main():
    PanCardFilePath = sys.argv[1]
    FileName = str(uuid.uuid4()).replace('-', '')

    if(os.path.exists(PanCardFilePath) == True):
        Names = []
        Date = []
    
        if(str(PanCardFilePath).endswith(('.png','.jpg','.jpeg'))):
            StringText = pytesseract.image_to_string(Image.open(PanCardFilePath))
            print(StringText.replace('\n', ' , ').title())

        elif(str(PanCardFilePath).endswith(('.pdf'))):
            convert_pdf(PanCardFilePath, FileName + '.png')
            StringText = pytesseract.image_to_string(Image.open(FileName + '.png'))
            print(StringText.replace('\n', ' , ').title())
            subprocess.run('rm -f {}'.format(FileName + '.png',), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if(len(StringText) <= 6):
            ReturnJSON = {
                "Error" : "Cannot read the file.",
                "Status" : False
            }
            sys.stderr.write(json.dumps(ReturnJSON, indent=3))
        
        else:
            tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast")
            sentence = Sentence(StringText.replace('\n', ' , ').title())
            tagger.predict(sentence)
            for entity in sentence.get_spans('ner'):
                if(entity.get_label("ner").value == 'PERSON'):
                    Names.append(entity.text)
                if(entity.get_label("ner").value == 'DATE'):
                    Date.append(entity.text)
            
            if(len(Names) < 1):
                ReturnJSON = {
                "Error" : "No data found",
                "Status" : False
                }   
                sys.stderr.write(json.dumps(ReturnJSON, indent=3))

            else:
                re_exp = r"[A-Z]{5}[0-9]{4}[A-Z]{1}"
                PANCARDID = re.findall(re_exp, StringText)
                if(len(Names) == 1):
                    ReturnJSON = {
                        "Name" : Names[0],
                        "Date" : Date[0],
                        "PanCardNumber" : PANCARDID[0],
                        "Status" : True
                    }
                    sys.stdout.write(json.dumps(ReturnJSON))
                else:
                    ReturnJSON = {
                        "Name" : Names[0],
                        "FatherName" : Names[1],
                        "DateOfBirth" : Date[0],
                        "PanCardNumber" : PANCARDID[0],
                        "Status" : True
                    }
                    sys.stdout.write(json.dumps(ReturnJSON, indent=3))

    else:
        ReturnJSON = {
                "Error" : "File Not Found",
                "Status" : False
            }
        sys.stderr.write(json.dumps(ReturnJSON, indent=3))

def convert_pdf(file_path, output_path):

    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_path(file_path, output_folder=temp_dir)
        temp_images = []
        for i in range(len(images)):
            image_path = f'{temp_dir}/{i}.png'
            width, height = images[i].size
            images[i].save(image_path, 'PNG')
            temp_images.append(image_path)
        imgs = list(map(Image.open, temp_images))

    min_img_width = min(i.width for i in imgs)
    total_height = 0
    for i, img in enumerate(imgs):
        total_height += imgs[i].height
    merged_image = Image.new(imgs[0].mode, (min_img_width, total_height))

    y = 0
    for img in imgs:
        merged_image.paste(img, (0, y))
        y += img.height
    merged_image.save(output_path)


if(__name__) == "__main__":
    main()