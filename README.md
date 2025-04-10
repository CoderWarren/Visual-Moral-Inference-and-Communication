# Visual Moral Inference and Communication from Image-Text Fusion
This repository provides code for the analyses from the following work:

Zhu, W., Ramezani, A. and Xu, Y. (2025) Visual moral inference and
communication. In Proceedings of the 47th Annual Meeting of the Cognitive
Science Society.

## Data
* Download SMID Data from [here](https://osf.io/2rqad/), specifically, the images and the file ```SMID_norms.csv```.
  Store these in ```Data``` directory.
* Download GoodNews Data from [here](https://github.com/furkanbiten/GoodNews), specifically, the images and the files ```captioning_dataset.json``` and ```news_dataset.json```. Store these in ```Data``` directory. The images folder (which should be named ```resized```) is initially of format ```.tar```---it should be extracted and placed into the ```Data``` directory.

### Citation
* Crone DL, Bode S, Murawski C, Laham SM. The Socio-Moral Image Database (SMID): A novel stimulus set for the study of social, moral and affective processes. PLoS One. 2018 Jan 24;13(1):e0190954. doi: 10.1371/journal.pone.0190954. PMID: 29364985; PMCID: PMC5783374.
* A. F. Biten, L. Gomez, M. Rusiñol and D. Karatzas, "Good News, Everyone! Context Driven Entity-Aware Captioning for News Images," 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA, 2019, pp. 12458-12467, doi: 10.1109/CVPR.2019.01275.

## How to use Scripts/Notebooks
The notebooks found in ```Part 1 (SMID)``` covers all analysis that was done on SMID (including model training). Within this directory, all code chunks in ```train.ipynb``` should be run prior to running any code in ```Part 2 (NYT)```. Some tables and plots produced will feature additional information regarding Valence and Arousal--while this was cut from the original paper, these additonal plots/rows/columns were not removed from the code as we wanted to leave people the option of exploring these extra variables if they would like to do so.

Scripts and notebooks found in ```Part 2 (NYT)``` covers all analysis that was done on GoodNews. Code inside of ```Obtaining Predictions``` must be run in the following order to obtain predictions.
```
python ./"Part 2 (NYT)"/"Obtaining Predictions"/get_image_details.py
python ./"Part 2 (NYT)"/"Obtaining Predictions"/get_text_embeds.py
python ./"Part 2 (NYT)"/"Obtaining Predictions"/get_image_embeds.py
python ./"Part 2 (NYT)"/"Obtaining Predictions"/get_ImageOnly_ratings.py
python ./"Part 2 (NYT)"/"Obtaining Predictions"/get_Joint_ratings.py
python ./"Part 2 (NYT)"/"Obtaining Predictions"/get_CLIPtext_ratings.py
```

After doing so, code in  ```Analysis``` could be run to produce plots and to find the image captions as featured in the paper. Note that the produced plots do not feature captions---we added these captions onto the plots manually after the plots themselves were created to optimize spacing. To run ```regions_heatmap.py```, simply do the following:
```
python ./"Part 2 (NYT)"/Analysis/regions_heatmap.py
```
