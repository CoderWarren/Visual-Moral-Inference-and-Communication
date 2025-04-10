import argparse
import math
import numpy as np
import os
import pickle

if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--writefolder', help='Path to folder to write to', default=os.getcwd())
    parser.add_argument('-mp','--modelpath', help='Path to the model', default=".\Finalized-Models\CLIP-JOINT.pickle")
    args = parser.parse_args()

    # Load the model
    with open(args.modelpath, 'rb') as handle:
        model = pickle.load(handle)

    # At this point the mask should already be computed. Load it.
    mask_path = os.path.join(args.writefolder, "imagecaptionmask.pickle")
    with open(mask_path, 'rb') as handle:
        mask = pickle.load(handle)

    results = []
    # Iterate over batches
    batch_size = 10000
    num_batches = math.ceil(len(mask)/batch_size)
    imgembeddingfolder = os.path.join(args.writefolder, "embedded-imgs")
    capembeddingfolder = os.path.join(args.writefolder, "embedded-captions")
    captokenfolder = os.path.join(args.writefolder, "tokenized-captions")
    for batch_number in range(num_batches):
        batch_mask = mask[batch_size*batch_number : batch_size*(batch_number+1)]
        with open(os.path.join(imgembeddingfolder, "embeddings_{}.pickle".format(batch_number)), 'rb') as handle:
            img_batch = pickle.load(handle)
        with open(os.path.join(capembeddingfolder, "embeddings_{}.pickle".format(batch_number)), 'rb') as handle:
            cap_batch = pickle.load(handle)
        with open(os.path.join(captokenfolder, "tokenized_mask_{}.pickle".format(batch_number)), 'rb') as handle:
            token_mask = pickle.load(handle)
        results.append(model.predict((img_batch[batch_mask])[token_mask]+cap_batch))
    results = np.concatenate(results, axis=0)
    
    # Save results
    prediction_path = os.path.join(args.writefolder, "JointPredictions.pickle")
    with open(prediction_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
