from util.utils import *
from transforms.transforms import *
from dataset.dataset import INRDataset
from inr import train_inr
from inr_cpu import parallel_training
import numpy as np

def priority_merge_prot(dataframes, target_df):
    """
    Merge 'Prot' values from multiple dataframes into target dataframe in priority order,
    using 'KID' as the merge key. Much more efficient implementation.
    
    Args:
        dataframes: List of dataframes, each containing 'Prot' and 'KID' columns (in decreasing priority order)
        target_df: Target dataframe to merge 'Prot' values into (must contain 'KID' column)
    
    Returns:
        DataFrame with aggregated 'Prot' values merged into target_df
    """
    # Create a copy of the target dataframe
    result = target_df.copy()
    
    # Create an empty dataframe with just KID and Prot columns
    prot_values = pd.DataFrame({'KID': [], 'Prot': [], 'Prot_ref': []})
    
    # Process dataframes in priority order
    for df in dataframes:
        print(f"Processing dataframe with {len(df)} rows. currently have {len(prot_values)} prot values")
        # Extract just the KID and Prot columns
        current = df[['KID', 'Prot', 'Prot_ref']].copy()
        
        # Only add keys that aren't already in our prot_values dataframe
        missing_keys = current[~current['KID'].isin(prot_values['KID'])]
        
        # Concatenate with existing values
        prot_values = pd.concat([prot_values, missing_keys])
    
    # Merge the aggregated Prot values into the result dataframe
    result = result.merge(prot_values, on='KID', how='left')
    
    return result

def creat_data():
    local_rank, world_size, gpus_per_node = setup()
    # local_rank = torch.device('cpu')

    kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
    lightpred_df = pd.read_csv('/data/lightPred/tables/kepler_predictions_clean_seg_0_1_2_median.csv')
    lightpred_df['Prot_ref'] = 'lightpred'
    lightpred_df.rename(columns={'predicted period': 'Prot'}, inplace=True)

    santos_df = pd.read_csv('/data/lightPred/tables/santos_periods_19_21.csv')
    santos_df['Prot_ref'] = 'santos'
    mcq14_df = pd.read_csv('/data/lightPred/tables/Table_1_Periodic.txt')
    mcq14_df['Prot_ref'] = 'mcq14'
    reinhold_df = pd.read_csv('/data/lightPred/tables/reinhold2023.csv')
    reinhold_df['Prot_ref'] = 'reinhold'

    p_dfs = [lightpred_df, santos_df, mcq14_df, reinhold_df]
    kepler_df = priority_merge_prot(p_dfs, kepler_df).dropna(subset=['Prot'])
    kepler_df = kepler_df[kepler_df['Prot'] > 0]
    print("total number of stars: ", len(kepler_df))

    kois = pd.read_csv('/data/lightPred/tables/kois.csv')
    kepler_df = kepler_df[~kepler_df['KID'].isin(kois['KID'])]

    # revrese the order of the dataframe
    # kepler_df = kepler_df.sort_values(by='Prot', ascending=False)
    kepler_df = kepler_df.sample(frac=1)

 
    light_transforms = Compose([
                                MovingAvg(3*48),
                                ToTensor(), ])

    inr_dataset = INRDataset(df=kepler_df, npy_path='/data/lightPred/data/raw_npy',
                            transforms=light_transforms, scale_flux=True
                            )
    train_inr(inr_dataset, local_rank, num_samples=np.inf, num_iters=1500, plot_every=1)
    # parallel_training(inr_dataset, num_processes=2, num_samples=10, num_iters=2000)


if __name__ == '__main__':
    creat_data()