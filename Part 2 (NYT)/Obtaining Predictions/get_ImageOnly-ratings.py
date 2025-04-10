import argparse
import math
import numpy as np
import os
import pickle


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--writefolder', help='Path to folder to write to', default=os.getcwd())
    parser.add_argument('-mp','--modelpath', help='Path to the model', default=".\Finalized-Models\CLIP-IMG.pickle")
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
    embeddingfolder = os.path.join(args.writefolder, "embedded-imgs")
    for batch_number in range(num_batches):
        batch_mask = mask[batch_size*batch_number : batch_size*(batch_number+1)]
        with open(os.path.join(embeddingfolder, "embeddings_{}.pickle".format(batch_number)), 'rb') as handle:
            batch = pickle.load(handle)
        results.append(model.predict(batch[batch_mask]))
    results = np.concatenate(results, axis=0)
    
    # Save results
    prediction_path = os.path.join(args.writefolder, "imageOnlyPredictions.pickle")
    with open(prediction_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
