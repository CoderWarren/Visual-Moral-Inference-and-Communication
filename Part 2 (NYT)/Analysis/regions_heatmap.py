import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import scipy.stats as sp

def load_data(args):
    """ Read the data and prepare it accordingly
    """
    # Load the data
    with open(args.predictions, 'rb') as handle:
        predictions = pickle.load(handle)
    with open(args.details, 'rb') as handle:
        details = pickle.load(handle)
    with open(args.tmask, 'rb') as handle:
        token_mask = pickle.load(handle)
    details = details[token_mask]
    # Two links aren't associated with dates. Get rid of them.
    # The links are:
    # https://www.nytimes.com/interactive/2015/health/stillbirth-reader-stories.html
    # https://www.nytimes.com/interactive/2015/world/nobel-peace-prize-timeline.html
    mask = (details[:,0]!='2015-00')
    details = details[mask]
    predictions = predictions[mask]
    years = ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018"]

    unique_mask = np.unique(details[:,2:], return_index=True, axis=0)[1]
    details = details[unique_mask]
    predictions = predictions[unique_mask]

    return predictions, details, years


def get_subset(predictions, details, category, year):
    """ Returns the subset of predictions and details that match the category and year
    """
    # Code below taken from:
    # https://stackoverflow.com/questions/38974168/finding-entries-containing-a-substring-in-a-numpy-array
    mask = np.flatnonzero(np.core.defchararray.find(details[:,3], category)!=-1)
    subset_details = details[mask]
    subset_predictions = predictions[mask]
    mask = np.flatnonzero(np.core.defchararray.find(subset_details[:,0], year)!=-1)
    subset_details = subset_details[mask]
    subset_predictions = subset_predictions[mask]
    return subset_details, subset_predictions


if __name__ == '__main__':
    # Parse Arguments and load data
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--predictions', help='Path to the predictions file', default=".\JointPredictions.pickle")
    parser.add_argument('-d','--details', help='Path to the details file', default=".\imagedetails.pickle")
    parser.add_argument('-t','--tmask', help='Path to the token mask file', default=".\\token_mask.pickle")
    args = parser.parse_args()
    predictions, details, years = load_data(args)

    #########################################################################################################
    # CATEGORIES ############################################################################################
    #########################################################################################################
    CATEGORIES = ["/nyregion/", "/us/", "/world/europe/", "/world/asia/","/world/africa/", "/world/middleeast/"]
    # NOTE: "/australia/" and "/world/canada/" will not span the entire section so they've been excluded
    #########################################################################################################

    plt_names = ["Valence", "Arousal", "Morality", "Authority (Relevance)",
                 "Fairness (Relevance)", "Harm (Relevance)","Ingroup (Relevance)", "Purity (Relevance)"]
    for i in range(len(plt_names)): 
        means = []
        labels = []
        for c in CATEGORIES:
            mean_row = []
            label_row = []
            for y in years:
                ratings = get_subset(predictions, details, c, y)[1][:,i]
                mean_row.append(np.mean(ratings))
                label_row.append(r"$\bf{"+str(np.around(np.mean(ratings), 3))+"}$"+
                                 "\n ({}{})\n [n={}]".format(u"\u00B1",np.around(sp.sem(ratings), 3),len(ratings)))
            means.append(mean_row)
            labels.append(label_row)

        fig, ax = plt.subplots()
        sns.heatmap(means, 
                    annot=labels, 
                    fmt="",
                    cmap="coolwarm_r",
                    xticklabels=years,
                    yticklabels=[c[1:-1] for c in CATEGORIES],
                    vmin=2, 
                    vmax=4,
                    ax=ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Category")
        ax.set_title("{} of NYT images pertaining to different regions over time".format(plt_names[i]))
        ax.collections[0].colorbar.set_label("Mean Predicting Ratings of Images in category/year")
        plt.show()