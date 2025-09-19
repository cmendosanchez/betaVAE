import matplotlib.pyplot as plt
import os 
from math import isqrt
from PIL import Image
from PIL import ImageDraw, ImageFont

def image_grid(imgs, rows, cols):
    #assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


res = '2mm'
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=100)
date = '2025-07-22/15-05-22'
model_path = f'/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Output/{date}'
if not os.path.exists(f'{model_path}/snapshots_grid/'):
    os.makedirs(f'{model_path}/snapshots_grid/')

ROI  = 'S.C.-sylv.'
SUBJECTS = [f for f in os.listdir(f'{model_path}/subjects') if f.endswith('.nii.gz')]
SUBJECTS = [sub.split('_')[0] for sub in SUBJECTS]
SUBJECTS = list(set(SUBJECTS))
print(SUBJECTS,len(SUBJECTS))

for idx_i,i in enumerate(list(range(0,len(SUBJECTS),100))):
    print(idx_i)

    images =[]
    
    for idx,sub in enumerate(SUBJECTS[i:i+100]):
        for figure_img in ['_input.png','_output.png']:


            img = Image.open(f'{model_path}/snapshots/{sub}{figure_img}').convert("RGBA")
            # Set your text and font
            text = sub+figure_img.split('.png')[0]
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=40)

            # Create temporary image to draw rotated text
            text_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_layer)

            # Draw the text on the temp image
            # Start drawing from top-left corner (x=10, y=10), rotated counterclockwise
            draw.text((10, 10), text, font=font, fill=(0, 0, 0, 255))

            # Rotate the entire temp image with text
            text_layer = text_layer.rotate(90, expand=0)

            # Composite the text over the base image
            img = Image.alpha_composite(img, text_layer)
            images.append(img)




    plt.figure(figsize=(30,30))
    grid = image_grid(images,5,8)
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(f'{model_path}/snapshots_grid/batch_{idx_i}.png',dpi=300)
    #plt.show()