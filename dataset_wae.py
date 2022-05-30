from random import randrange, shuffle
from typing import Iterator, T_co

import numpy as np
from torch.utils.data import IterableDataset

import torch

class TrainDataset(IterableDataset):
    def __init__(
        self,
        data_path: str,
        crops_per_slice: int = 1,
    ) -> None:
        super().__init__()

        self.crops_per_slice = crops_per_slice

        self._process_data(np.load(data_path))

    def _process_data(self, data):
        #self.osgb_data = np.stack(
        #    [
        ##        data["x_osgb"],
        #        data["y_osgb"],
        #    ]
        #)
        self.cached_items = []
        data_array = data["data"]
        *_, t, y, x = data_array.shape
        for day in data_array:
            # change 4 (20 min) to whichever skip you like
            # this might depend on your memory constraints
            for i in range(0, t-1, 1):
                slice = day[i : i + 1, :, :]
                #target_slice = day[i + 12 : i + 36, :, :]
                crops = 0
                while crops < self.crops_per_slice:
                    crop = self._get_crop(slice, y, x)
                    if crop is not None:
                        self.cached_items.append(crop)
                        crops += 1
        shuffle(self.cached_items)


    def _get_crop(self, slice, y, x):
        rand_x = randrange(0, x - 128)
        rand_y = randrange(0, y - 128)
        #osgb_data = self.osgb_data[:, rand_y : rand_y + 128, rand_x : rand_x + 128]
        input_data = slice[:, rand_y : rand_y + 128, rand_x : rand_x + 128]
        # target_data = target_slice[
        #     :, rand_y + 32 : rand_y + 96, rand_x + 32 : rand_x + 96
        # ]
        if input_data.shape != (1, 128, 128):
            return None

        return (input_data/1023 - 0.5)*2

    def __iter__(self) -> Iterator[T_co]:
        for item in self.cached_items:
            yield item



class TestDataset(IterableDataset):
    def __init__(
        self,
        data_path: str,
        crops_per_slice: int = 1, 
        image_gen: bool = False
    ) -> None:
        super().__init__()

        self.crops_per_slice = crops_per_slice

        self._process_data(np.load(data_path), image_gen)

    def _process_data(self, data, image_gen=False):
        #self.osgb_data = np.stack(
        #    [
        ##        data["x_osgb"],
        #        data["y_osgb"],
        #    ]
        #)
        self.cached_items = []
        data_array = data["data"]
        *_, t, y, x = data_array.shape
        for day in data_array:
            # change 4 (20 min) to whichever skip you like
            # this might depend on your memory constraints
            for i in range(t-1, 1, -4):
                slice = day[i : i + 1, :, :]
                #target_slice = day[i + 12 : i + 36, :, :]
                crops = 0
                while crops < self.crops_per_slice:
                    crop = self._get_crop(slice, y, x)
                    if crop is not None:
                        self.cached_items.append(crop)
                        crops += 1
        shuffle(self.cached_items)

        if image_gen:
            self.cached_items = np.array(self.cached_items[:10])
            with open('./ground_truth_images.npy', 'wb') as f:
                np.save(f, self.cached_items)


    def _get_crop(self, slice, y, x):
        rand_x = randrange(0, x - 128)
        rand_y = randrange(0, y - 128)
        #osgb_data = self.osgb_data[:, rand_y : rand_y + 128, rand_x : rand_x + 128]
        input_data = slice[:, rand_y : rand_y + 128, rand_x : rand_x + 128]
        # target_data = target_slice[
        #     :, rand_y + 32 : rand_y + 96, rand_x + 32 : rand_x + 96
        # ]
        #input_data = np.expand_dims(input_data, axis=0)
        if input_data.shape != (1, 128, 128):
            return None

        return (input_data/1023 - 0.5)*2

    def __iter__(self) -> Iterator[T_co]:
        for item in self.cached_items:
            yield item

