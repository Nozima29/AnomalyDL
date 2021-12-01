from torch.utils.data import Dataset
import pickle
import torch
import conf


class SWaTDataset(Dataset):
    def __init__(self, pickle_jar, attack=False):
        self.attack = attack
        with open(pickle_jar, 'rb') as f:
            self.picks = pickle.load(f)

    def __len__(self):
        return len(self.picks)

    def __getitem__(self, idx):
        items = {
            'ts': self.picks[idx][0].strftime(conf.SWaT_TIME_FORMAT),
            'given': torch.from_numpy(self.picks[idx][1]),
            'predict': torch.from_numpy(self.picks[idx][2]),
            'answer': torch.from_numpy(self.picks[idx][3])
        }
        # print(idx)
        if self.attack:
            items['attack'] = torch.tensor(1 if self.picks[idx][4] else 0,
                                           dtype=torch.uint8)
        return items


# if __name__ == '__main__':
#     dataset = SWaTDataset(
#         'data/dat/swat-train-P{}.dat'.format(1), attack=True)
#     print(dataset[0]['given'].shape)
