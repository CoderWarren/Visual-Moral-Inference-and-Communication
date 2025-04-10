import argparse
import clip
import math
import numpy as np
import os
import pickle
from PIL import Image
import torch




def preprocess_batch(preprocessor, imagefolder, filenames, writefolder, batch_number):
    """ Preprocesses each batch so they can be embedded
    """
    # setting up paths
    preprocessedfolder = os.path.join(writefolder, "preprocessed-imgs")
    if not os.path.exists(preprocessedfolder):
        os.makedirs(preprocessedfolder)
    write_path = os.path.join(preprocessedfolder, "preprocessed_{}.pickle".format(batch_number))

    # Consider the case if the batch has already been preprocessed
    if os.path.exists(write_path):
        print("Batch {} ({} images) already preprocessed. Loading batch.".format(batch_number, len(filenames)))
        with open(write_path, 'rb') as handle:
            images = pickle.load(handle)
    # Consider the case the batch hasn't been preprocessed
    # Preallocate empty torch to save space
    # https://discuss.pytorch.org/t/appending-to-a-tensor/2665/6
    else:
        i=0
        images = None
        for fn in filenames:
            # Preprocess the images
            PIL_img = Image.open(os.path.join(imagefolder, fn))
            img = preprocessor(PIL_img)
            PIL_img.close()
            # If PIL_images is empty at this point initialize it properly
            if images is None:
                output_cat_size = list(img.size())
                output_cat_size.insert(0, len(filenames))
                images = torch.empty(*output_cat_size, dtype=img.dtype, device=img.device)
            # Slot in the images
            images[i] = img
            i+=1
        # Save preprocessed images
        with open(write_path, 'wb') as handle:
            pickle.dump(images, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Batch {} ({} images) preprocessed and saved.".format(batch_number, len(filenames)))
    # Return the images so they can be embedded
    return images




def embed_batch(model, device, batch, writefolder, batch_number):
    """ Embed a preprocessed batch
    """
    # setting up paths
    embeddingfolder = os.path.join(writefolder, "embedded-imgs")
    if not os.path.exists(embeddingfolder):
        os.makedirs(embeddingfolder)
    write_path = os.path.join(embeddingfolder, "embeddings_{}.pickle".format(batch_number))
    # Consider the case if the batch has already been embedded
    if os.path.exists(write_path):
        print("Batch {} ({} images) already embedded. Loading batch.".format(batch_number))
        with open(write_path, 'rb') as handle:
            img_embeds = pickle.load(handle)
    # Consider the case if the batch hasn't been embedded    
    else:
        mini_batch_size = 1000
        num_mini_batches = math.ceil(batch.shape[0]/mini_batch_size)
        img_embeds = []
        for mini_batch_number in range(num_mini_batches):
            with torch.no_grad():
                mini_batch = batch[mini_batch_size*mini_batch_number : mini_batch_size*(mini_batch_number+1)]
                mini_img_embeds = model.encode_image(mini_batch.to(device))
                mini_img_embeds = mini_img_embeds.to("cpu")
            mini_img_embeds = np.array(mini_img_embeds)
            mini_img_embeds = (mini_img_embeds.transpose()/np.linalg.norm(mini_img_embeds, axis=1)).transpose()
            img_embeds.append(mini_img_embeds)
        img_embeds = np.concatenate(img_embeds, axis=0)
        with open(write_path, 'wb') as handle:
            pickle.dump(img_embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Batch {} ({} images) embedded and saved.".format(batch_number, batch.shape[0]))

    # Return the embeds so predictions can be made
    return img_embeds




if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--imagefolder', help='Path to folder for images', default=".\Data\\resized")
    parser.add_argument('-w','--writefolder', help='Path to folder to write to', default=os.getcwd())
    args = parser.parse_args()

    # Load the clip model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ViT_model, ViT_preprocess = clip.load("ViT-B/32", device=device) # Load ViT Encoder

    #print(torch.cuda.memory_summary())

    imagefiles = os.listdir(args.imagefolder)
    batch_size = 10000
    num_batches = math.ceil(len(imagefiles)/batch_size)

    # Preprocess
    for batch_number in range(num_batches):
        preprocessed = preprocess_batch(preprocessor=ViT_preprocess, 
                                        imagefolder=args.imagefolder, 
                                        filenames=imagefiles[batch_size*batch_number : batch_size*(batch_number+1)],
                                        writefolder=args.writefolder,
                                        batch_number=batch_number)
        embeds = embed_batch(model=ViT_model,
                             device=device,
                             batch=preprocessed,
                             writefolder=args.writefolder,
                             batch_number=batch_number)
