import argparse
import json 
import numpy as np
import os
import pickle
import re


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--imagefolder', help='Path to folder for images', default=".\Data\\resized")
    parser.add_argument('-c', '--captions', help='Path to captions JSON file', default=".\Data\captioning_dataset.json")
    parser.add_argument('-w','--writefolder', help='Path to folder to write to', default=os.getcwd())
    args = parser.parse_args()

    # Load the captions dictionary
    with open(args.captions) as f:
        captions_dict = json.load(f) 

    # Load the image IDs
    image_IDs = os.listdir(args.imagefolder)
    image_IDs = [id.split('.')[0] for id in image_IDs]
    image_IDs = np.array([id.split('_') for id in image_IDs])
    
    #num_batches = math.ceil(len(image_IDs)/batch_size)

    # Compute the mask for image_IDs with captions
    mask = np.in1d(image_IDs[:, 0], list(captions_dict.keys()))
    mask_path = os.path.join(args.writefolder, "imagecaptionmask.pickle")
    if not os.path.exists(mask_path):
        with open(mask_path, 'wb') as handle:
            pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Image details
    all_details = []
    for id in image_IDs[mask]:
        link = captions_dict[id[0]]['article_url']
        details = re.findall("[0-9]{4}/[0-9]{2}/[0-9]{2}/", link)
        if len(details) == 1:
            details = details[0].split('/')[:-1]
            details = ["{}-{}".format(details[0], details[1])]
        elif link=="https://www.nytimes.com/interactive/2015/health/stillbirth-reader-stories.html":
            details = ["2015-00"]
        elif link=="https://www.nytimes.com/interactive/2015/world/nobel-peace-prize-timeline.html":
            details = ["2015-00"]
        else:
            raise Exception("LINK CAN'T BE HANDLED: {}".format(link))
        details.append(id[0])
        details.append(id[1])
        details.append(link)
        all_details.append(details) # YYYY-MM, article ID, image ID, article Link
    all_details = np.array(all_details)

    # Pickle the file
    details_path = os.path.join(args.writefolder, "imagedetails.pickle")
    if not os.path.exists(details_path):
        with open(details_path, 'wb') as handle:
            pickle.dump(all_details, handle, protocol=pickle.HIGHEST_PROTOCOL)
