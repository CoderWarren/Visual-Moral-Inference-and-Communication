import argparse
import clip
import json 
import math
import numpy as np
import os
import pickle
import torch

def tokenize_batch(captions_dict, batch_IDs, writefolder, batch_number):
    """ Retrieve and tokenize captions for the image IDs
    """
    # setting up paths
    tokenizedfolder = os.path.join(writefolder, "tokenized-captions")
    if not os.path.exists(tokenizedfolder):
        os.makedirs(tokenizedfolder)
    write_path = os.path.join(tokenizedfolder, "tokenized_{}.pickle".format(batch_number))
    mask_path = os.path.join(tokenizedfolder, "tokenized_mask_{}.pickle".format(batch_number))
    
    # Consider the case if the batch has already been tokenized
    if os.path.exists(write_path):
        print("Batch {} ({} captions) already tokenized. Loading batch.".format(batch_number, len(batch_IDs)))
        with open(write_path, 'rb') as handle:
            tokenized = pickle.load(handle)
        with open(mask_path, 'rb') as handle:
            mask = pickle.load(handle)
    # Consider the case the batch hasn't been tokenized
    else:
        mask = []
        tokenized = []
        for id in batch_IDs:
            try:
                t = clip.tokenize(captions_dict[id[0]]['images'][id[1]])
            except:
                mask.append(False)
            else:
                tokenized.append(t)
                mask.append(True)
        tokenized = torch.cat(tokenized)
        # Save tokenized batch
        with open(write_path, 'wb') as handle:
            pickle.dump(tokenized, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(mask_path, 'wb') as handle:
            pickle.dump(np.array(mask), handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Batch {} ({} out of {} valid) tokenized and saved.".format(batch_number, len(tokenized), len(batch_IDs)))
    return tokenized, mask


def embed_batch(model, device, batch, writefolder, batch_number):
    """ Embed a batch
    """
    # setting up paths
    embeddingfolder = os.path.join(writefolder, "embedded-captions")
    if not os.path.exists(embeddingfolder):
        os.makedirs(embeddingfolder)
    write_path = os.path.join(embeddingfolder, "embeddings_{}.pickle".format(batch_number))
    # Consider the case if the batch has already been embedded
    if os.path.exists(write_path):
        print("Batch {} already embedded. Loading batch.".format(batch_number))
        with open(write_path, 'rb') as handle:
            caption_embeds = pickle.load(handle)
    # Consider the case if the batch hasn't been embedded    
    else:
        mini_batch_size = 1000
        num_mini_batches = math.ceil(batch.shape[0]/mini_batch_size)
        caption_embeds = []
        for mini_batch_number in range(num_mini_batches):
            with torch.no_grad():
                mini_batch = batch[mini_batch_size*mini_batch_number : mini_batch_size*(mini_batch_number+1)]
                mini_cap_embeds = model.encode_text(mini_batch.to(device))
                mini_cap_embeds = mini_cap_embeds.to("cpu")
            mini_cap_embeds = np.array(mini_cap_embeds)
            mini_cap_embeds = (mini_cap_embeds.transpose()/np.linalg.norm(mini_cap_embeds, axis=1)).transpose()
            caption_embeds.append(mini_cap_embeds)
        caption_embeds = np.concatenate(caption_embeds, axis=0)
        with open(write_path, 'wb') as handle:
            pickle.dump(caption_embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Batch {} ({} captions) embedded and saved.".format(batch_number, batch.shape[0]))
    return caption_embeds



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
    batch_size = 10000
    num_batches = math.ceil(len(image_IDs)/batch_size)

    # Compute the mask for image_IDs with captions
    mask = np.in1d(image_IDs[:, 0], list(captions_dict.keys()))
    with open(os.path.join(args.writefolder, "imagecaptionmask.pickle"), 'wb') as handle:
        pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load the clip model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ViT_model, ViT_preprocess = clip.load("ViT-B/32", device=device) # Load ViT Encoder

    # Preprocess
    token_mask = []
    for batch_number in range(num_batches):
        batch_mask = mask[batch_size*batch_number : batch_size*(batch_number+1)]
        tokenized, successful_mask = tokenize_batch(captions_dict=captions_dict,
                                                    batch_IDs=image_IDs[batch_size*batch_number : batch_size*(batch_number+1)][batch_mask],
                                                    writefolder=args.writefolder,
                                                    batch_number=batch_number)
        token_mask.append(successful_mask)
        embeds = embed_batch(model=ViT_model,
                             device=device,
                             batch=tokenized,
                             writefolder=args.writefolder,
                             batch_number=batch_number)
    token_mask=np.concatenate(token_mask)
    # Record the token_mask
    write_path = os.path.join(args.writefolder, "token_mask.pickle")
    # Consider the case if the batch has already been embedded
    if not os.path.exists(write_path):
        with open(write_path, 'wb') as handle:
            pickle.dump(token_mask, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
