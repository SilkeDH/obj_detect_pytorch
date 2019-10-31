"""
Methods to put together all interesting images and data
in order to be able to transfer all of it as a single file.
"""

import os
from PIL import Image
from fpdf import FPDF
import obj_detect_pytorch.config as cfg

# Input and result images from the segmentation
files = ['{}/Input_image_patch.png'.format(cfg.DATA_DIR),
         #'{}/Groundtruth.png'.format(cfg.DATA_DIR),
         '{}/Classification_map.png'.format(cfg.DATA_DIR)]
         #'{}/Error_map.png'.format(cfg.DATA_DIR)]


# Merge images and add a color legend
def merge_images():
    result = Image.new("RGB", (620, 840))  
    for index, path in enumerate(files):
        img = Image.open(path)
        img.thumbnail((620, 420), Image.ANTIALIAS)
        x = index // 2 * 620
        y = index % 2 * 420
        w, h = img.size
        result.paste(img, (x, y, x + w, y + h))
    
    merged_file = '{}/merged_maps.png'.format(cfg.DATA_DIR)
    result.save(merged_file)

    return merged_file


# Put images and accuracy information together in one pdf file
def create_pdf(image, boxes, labels, probs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Results:', ln=1)
    pdf.set_font('Arial', 'B', 15)
    pdf.cell(0, 10, 'Labelwise Accuracy:', ln=1)
    pdf.set_font('Arial', size=14)

    for i in range(len(boxes)):
        pdf.cell(0, 10, '\t\t{}: \t\t {}'.format(labels[i], probs[i]), ln=1)
        pdf.cell(0, 10, '\t\t\t coordinates: \t\t {} , {} '.format(boxes[i][0], boxes[i][1]), ln=1)

        
    pdf.set_font('Arial', 'I', size=14)
    pdf.cell(1, 10, '-> Images are in the next page.', ln=2)
    
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Original vs. Classified Image', ln=1)
    pdf.image(image, 20, 25, w= 170)
    results = '{}/prediction_results.pdf'.format(cfg.DATA_DIR)
    pdf.output(results,'F')

    return results