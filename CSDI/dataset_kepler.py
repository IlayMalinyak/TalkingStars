from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transforms import *
from torch.utils.data.distributed import DistributedSampler


class KeplerDataset():
    """
    A dataset for Kepler data.
    """
    def __init__(self,
                df:pd.DataFrame=None,
                prot_df:pd.DataFrame=None,
                npy_path:str=None,
                transforms:object=None,
                seq_len:int=4800,
                target_len:int=30*48,
                label_len:int=30*48,
                masked_transforms:bool=False,
                use_acf:bool=False,
                use_fft:bool=False,
                scale_flux:bool=True,
                labels:object=None,
                 dims:int=1
                ):
        """
        dataset for Kepler data
        Args:
            df (pd.DataFrame): DataFrame containing Kepler paths
            prot_df (pd.DataFrame): DataFrame containing rotation periods
            npy_path (str): Path to numpy files
            transforms (object): Transformations to apply to the data
            seq_len (int): Sequence length
            target_transforms (object): Transformations to apply to the target
            masked_transforms (bool): Whether to apply masking transformations

        """
        self.df = df
        self.transforms = transforms
        self.npy_path = npy_path
        self.prot_df = prot_df
        self.seq_len = seq_len
        self.target_len = target_len
        self.label_len = label_len
        if df is not None and 'predicted period' not in df.columns:
            if prot_df is not None:
                self.df = pd.merge(df, prot_df[['KID', 'predicted period']], on='KID')
        self.mask_transform = RandomMasking(mask_prob=0.2) if masked_transforms else None
        self.use_acf = use_acf
        self.use_fft = use_fft
        self.scale_flux = scale_flux
        self.labels = labels
        self.dims = 1 + self.use_acf + self.use_fft


    def __len__(self):
        return len(self.df)

    def read_lc(self, filename: str):
        """
        Reads a FITS file and returns the PDCSAP_FLUX and TIME columns as numpy arrays.

        Args:
            filename (str): The path to the FITS file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The PDCSAP_FLUX and TIME columns as numpy arrays.
        """
        with fits.open(filename) as hdulist:
                binaryext = hdulist[1].data
                meta = hdulist[0].header
        df = pd.DataFrame(data=binaryext)
        x = df['PDCSAP_FLUX']
        time = df['TIME'].values
        return x,time, meta

    def read_row(self, idx):
        """
        Reads a row from the DataFrame.
        """
        row = self.df.iloc[idx]

        # read row from npy file
        if self.npy_path is not None:
            try:
                file_path = os.path.join(self.npy_path, f"{int(row['KID'])}.npy")
                x = np.load(file_path)            
                if not isinstance(x, np.ndarray) or x.size == 0:
                    print(f"Warning: Empty or invalid numpy array for {row['KID']}")
                    x = np.zeros((self.seq_len, 1))
            except FileNotFoundError:
                print(f"Error: File not found for {row['KID']}")
                x = np.zeros((self.seq_len, 1))
            except Exception as e:
                print(f"Error loading file for {row['KID']}: {str(e)}")
                x = np.zeros((self.seq_len, 1))
            meta = dict()
            for key in row.keys():
                if key not in ['data_file_path']:
                    meta[key] = row[key]
            return x, meta

        # read row from FITS file
        else:
            paths = row['data_file_path']
            meta = {}
            for i in range(len(paths)):
                x, time, meta = self.read_lc(paths[i])
                meta = dict(meta.items())
                if i == 0:
                    x_tot = x.copy()
                else:
                    border_val = np.nanmean(x) - np.nanmean(x_tot)
                    x -= border_val
                    x_tot = np.concatenate((x_tot, np.array(x)))
                self.cur_len = len(x)
            return x_tot, meta

    def fill_nan_np(self, x:np.ndarray, interpolate:bool=True):
        """
        fill nan values in a numpy array

        Args:
                x (np.ndarray): array to fill
                interpolate (bool): whether to interpolate or not

        Returns:
            np.ndarray: filled array
        """
        non_nan_indices = np.where(~np.isnan(x))[0]
        nan_indices = np.where(np.isnan(x))[0]
        if len(nan_indices) and len(non_nan_indices):
            if interpolate:
                # Interpolate NaN values using linear interpolation
                interpolated_values = np.interp(nan_indices, non_nan_indices, x[non_nan_indices])
                # Replace NaNs with interpolated values
                x[nan_indices] = interpolated_values
            else:
                x[nan_indices] = 0	
        return x
    
    def random_chop(self, x, seq_len, pred_len, label_len=None):
        """
        Chop the input sequence into encoder input and decoder input for Autoformer.
        
        Args:
            x: Input tensor of shape [dims, length]
            seq_len: Length of encoder input sequence (e.g., 4800 for 100 days)
            pred_len: Length of prediction horizon (e.g., 1440 for 30 days)
            label_len: Length of known context for decoder (overlap with encoder sequence)
                       If None, defaults to 20% of seq_len
                       
        Returns:
            x_enc: Encoder input sequence [dims, seq_len]
            x_dec: Decoder input sequence [dims, label_len + pred_len]
            x_mark_enc: Time features for encoder with Kepler's resolution (1/48 days)
            x_mark_dec: Time features for decoder with Kepler's resolution (1/48 days)
            start: Starting index of the chopped sequence
        """
        # Default label_len to 20% of seq_len if not provided
        if label_len is None:
            label_len = int(seq_len * 0.2)
            
        # Ensure we have enough data
        total_required = seq_len
        if x.shape[0] < total_required:
            x = torch.nn.functional.pad(x, (0,0,0, total_required - x.shape[0]), mode='constant', value=0)
            # raise ValueError(f"Input sequence length {x.shape[0]} is shorter than required length {total_required}")
        
        # Random starting point
        max_start = x.shape[0] - total_required
        start = np.random.randint(0, max(1, max_start))
        
        # Encoder sequence
        end_enc = start + seq_len
        x_enc = x.clone()[start:end_enc, :]
        mask_enc = torch.ones((seq_len, 1))

        
        t_enc = torch.arange(start, end_enc, dtype=torch.float32) / 48.0
        
        # Convert to time features format similar to Autoformer
        # For simplicity, we'll create a single feature (time in days)
        # In a real implementation, you might want to expand this to include
        # month, day, weekday, hour features from these timestamps
        # t_enc = t_enc.unsqueeze(-1)  # [seq_len, 1]
        
        return x_enc, mask_enc, t_enc, start
    
    def transform_data(self, x, transforms, info, idx):
        try:
            if transforms is not None:
                x, mask, info = self.transforms(x, mask=None, info=info)
            else:
                x = torch.tensor(x)
                mask = torch.zeros(1, self.seq_len)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            elif (len(x.shape) == 2) and (x.shape[0] == 1):
                x = x.squeeze(0).unsqueeze(-1)
            if self.use_acf:
                acf = torch.tensor(info['acf']).squeeze().unsqueeze(-1).nan_to_num(0)
                x = torch.cat((x, acf), dim=-1)
            if self.use_fft:
                fft = torch.tensor(info['fft']).squeeze().unsqueeze(-1).nan_to_num(0)
                x = torch.cat((x, fft), dim=-1)
        except Exception as e:
            print(f"Error in transforms for index {idx}: {str(e)}")
            traceback.print_exc()
            x = torch.zeros(self.seq_len, self.dims)
            mask = torch.zeros(self.seq_len, 1)
        return x, mask, info



    def __getitem__(self, idx):
        tic = time.time()
        x, info = self.read_row(idx)
        x = self.fill_nan_np(x, interpolate=True)
        info['idx'] = idx
        
        if self.scale_flux:
            min_val = np.nanmin(x)
            max_val = np.nanmax(x)
            range_val = max_val - min_val
            
            if range_val > 1e-10:  # Using a small epsilon value
                x = (x - min_val) / range_val
                x = x * 2 - 1
            else:
                x = np.zeros_like(x)

        
        # Apply transformations if any
        x, _, info = self.transform_data(x, self.transforms, info, idx)
        
        # Get inputs
        try:
            x_enc, mask_enc, x_mark_enc, start = self.random_chop(
                x, self.seq_len, self.target_len, label_len=self.label_len)
        except ValueError as e:
            print(f"Error in random_chop for index {idx}: {str(e)}")
            start = 0
            x_enc, x_mark_enc = torch.zeros(self.seq_len, self.dims), torch.zeros((self.seq_len))
            mask_enc = torch.ones((self.seq_len,1))
        
        target_mask = mask_enc.clone()
        target_mask[-self.target_len:] = 0.
        info['chop_start'] = start
        info['label_len'] = self.label_len

        if torch.isnan(x_enc).any():
            print(f"NaN values found in x_enc for index {idx}")
    
        
        info = info if isinstance(info, dict) else {}

        s = {
            'observed_data': x_enc,
            'observed_mask': mask_enc,
            'gt_mask': target_mask,
            'timepoints': x_mark_enc,
            'feature_id': np.arange(self.dims) * 1.0, 
        }

        return s


def get_dataloader(datatype,device, ddp=False, world_size=1, batch_size=64):
    transforms = Compose([
                            MovingAvg(49),
                            StandardScaler(),
                            # Normalize(['std']),
                            ToTensor(),
                             ])

    kepler_df = pd.read_csv('/data/lightPred/tables/kepler_dr25_meta_data.csv')
    train_df, test_df = train_test_split(kepler_df, test_size=0.2, random_state=1234)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=1234)

    if ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=device)
        valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=device)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=device)
        shuffle = 0
    else:
        train_sampler = None
        valid_sampler = None
        test_sampler = None
        shuffle = 1
    dataset = KeplerDataset(df=train_df,
                        transforms=transforms,
                        npy_path = '/data/lightPred/data/raw_npy',
                        seq_len=4800,
                        target_len=30*48,
                        masked_transforms = False,
                        use_acf=False,
                        use_fft=False,
                        scale_flux=False,
                        labels=[],
                        dims=1,
                                )
    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, shuffle=shuffle)
    valid_dataset = KeplerDataset(df=val_df,
                        transforms=transforms,
                        npy_path = '/data/lightPred/data/raw_npy',
                        seq_len=4800,
                        target_len=30*48,
                        masked_transforms = False,
                        use_acf=False,
                        use_fft=False,
                        scale_flux=True,
                        labels=[],
                        dims=1,
                                )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler, shuffle=0)
    test_dataset = KeplerDataset(df=test_df,
                        transforms=transforms,
                        npy_path = '/data/lightPred/data/raw_npy',
                        seq_len=4800,
                        target_len=30*48,
                        masked_transforms = False,
                        use_acf=False,
                        use_fft=False,
                        scale_flux=True,
                        labels=[],
                        dims=1,
                                )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=0)

    scaler = 0
    mean_scaler = 1


    

    return train_loader, valid_loader, test_loader, scaler, mean_scaler

def test_dataset():
    train_dl, valid_loader, test_loader, scaler, mean_scaler = get_dataloader('kepler', 'cuda:0', batch_size=64)
    for i, batch in enumerate(train_dl):
        obs_data, observed_mask, gt_mask, timepoints, feature_id = batch['observed_data'], batch['observed_mask'], batch['gt_mask'], batch['timepoints'], batch['feature_id']
        print("zeos in obs mask:", torch.sum(observed_mask == 0))
        print("zeros in gt mask:", torch.sum(gt_mask == 0))
        for b in range(obs_data.shape[0]):
            plt.plot(timepoints[b].cpu().numpy(), obs_data[b, :, 0].cpu().numpy(), label='observed data')
            # cond_data = obs_data[b, :, 0].cpu().numpy() * observed_mask[b, :, 0].cpu().numpy()
            gt_data = obs_data[b, :, 0].cpu().numpy() * gt_mask[b, :, 0].cpu().numpy()
            plt.plot(timepoints[b].cpu().numpy(), gt_data, label='masked data')
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('flux')
            plt.savefig(f'/data/TalkingStars/figs/csdi_kepler_{b}.png')
            plt.close()


        break


if __name__ == "__main__":
    test_dataset()