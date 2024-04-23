"""

Grabbing fibrosis score information from each slide (JSON format)

"""

import os
import sys

import pandas as pd
import json

def main():

    base_dir = '/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/020524_DUET_Patches/'
    slides = os.listdir(base_dir)

    quant_path = '/Collagen_Quantification/Stitched_Features.json'

    all_features = []
    for slide in slides:
        print(slide)
        slide_quant_path = base_dir+slide+quant_path
        if os.path.exists(slide_quant_path):
            with open(slide_quant_path,'r') as f:
                slide_features = json.load(f)
                f.close()
            
            slide_features['Image Name'] = slide
            all_features.append(slide_features)

    
    combined_df = pd.DataFrame.from_records(all_features)
    print(combined_df)
    combined_df.to_csv(base_dir+'Collagen_Stitched_Features.csv')

if __name__=="__main__":
    main()



